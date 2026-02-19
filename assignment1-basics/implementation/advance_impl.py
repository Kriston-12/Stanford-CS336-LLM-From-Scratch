
import collections
import heapq
from multiprocessing import Pool
from typing import BinaryIO, Optional, Tuple, Iterable
# from dataclasses import dataclass
import os
import regex as re
from collections import Counter
# 1. 用doubly linked list 表示corpus, 在merge的时候只需修改响铃两个token的指针，O(1) 
    # strucutre: [Node(val=byte1, prev=None, next=Node(val=byte2, prev=Node(val=byte1), next=None))]
# 2. 维护一个maxHeap, log(n)时间找到出现频率最高的pair of tokens, merge后更新heap中相关pair的频率
# 3. 一个position_map dict[tuple[int, int], set[int]]. Key是一个pair (A, B), value是当前pair在corpus中的对应node.
    # set[int] = set(Node1, Node2, Node3). merge可以直接用 map[pair].prev = map[pair].next来更新node


# 下面两个classes用于维护BPE的merge逻辑，O(1)时间修改语段
# dataclass会自带__repr__, __init__, __eq__, __hash__等方法，slots=True可以节省内存，加快属性访问，eq=False表示不需要比较对象的相等性（默认是比较对象的属性值），因为我们只需要比较对象的id来判断是否是同一个node，所以不需要比较属性值
# @dataclass(slots=True, eq=False)
# class _Node:
#     val: int
#     prev: Optional["_Node"] = None
#     next: Optional["_Node"] = None

#     def __hash__(self):
#         return id(self)

# No need to implementa __hash__ and __equal__, by default, python objects are hashable and equal by their id 
class _Node:
    __slots__ = ['val', 'prev', 'next'] # 节省内存，加快属性访问
    
    def __init__(self, val: int, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class BPEDoublyLinkedList:
    __slots__ = ("_node", "_size") # 节省内存，加快属性访问
    def __init__(self, tokens: Optional[Iterable[int]] = None):
        node = _Node(0)
        node.prev = node.next = node
        self._node = node
        self._size = 0
        
        if tokens is not None:
            for token in tokens:
                self.append(token)
    
    def __len__(self) -> int:
        return self._size

    def append(self, val: int) -> _Node:
        node = self._node
        return self._insert_between(val, node.prev, node)
    
    def _insert_between(self, val: int, left: _Node, right: _Node) -> _Node:
        node = _Node(val=val, prev=left, next=right)
        left.next = node
        right.prev = node
        return node
    
    def _unlink(self, node: _Node) -> None:
        assert node.prev is not None and node.next is not None, "Node must be linked to be unlinked"
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None 
    
    def merge_with_next(self, left: _Node, new_id: int) -> _Node:
        assert left.next is not None, "Left node must have a next node to merge with"
        right = left.next
        left.val = new_id
        self._unlink(right)
        return left


class _ReverseLexOrderPair:
    """Heap tie-break helper.

    In a min-heap, this makes the lexicographically *greater* (bytes, bytes) pair
    compare as "smaller" so it is popped first when frequencies tie.
    """

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_ReverseLexOrderPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseLexOrderPair) and self.pair == other.pair


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[bytes]
) -> list[int]:
    assert desired_num_chunks > 0, "disired num of chunks must be greater than 0"

    file.seek(0, os.SEEK_END)
    file_len = file.tell()
    
    chunk_size = file_len // desired_num_chunks

    # if we want 2 chunks, boundaries should be [0, first chunk end, file_end]
    boundaries: list[int] = [i * chunk_size for i in range(0, desired_num_chunks + 1)]
    boundaries[-1] = file_len

    transition_size = 2048
    # Adjust each tentative boundary forward until it lands on a special-token boundary.
    # Important: .find() returns an *offset within the buffer* (or -1), so we must convert
    # to an *absolute file offset* and ignore missing tokens.
    for i in range(1, desired_num_chunks):
        initial_pos = boundaries[i]
        # Clamp to file length and avoid negative seeks.
        initial_pos = max(0, min(initial_pos, file_len))
        file.seek(initial_pos)

        while True:
            read_buffer = file.read(transition_size)
            if read_buffer == b"":
                boundaries[i] = file_len
                break

            found_offsets = [read_buffer.find(tok) for tok in special_tokens]
            found_offsets = [off for off in found_offsets if off != -1]
            if found_offsets:
                boundaries[i] = initial_pos + min(found_offsets)
                break

            initial_pos += transition_size
            if initial_pos >= file_len:
                boundaries[i] = file_len
                break
            file.seek(initial_pos)

    boundaries[-1] = file_len
    # Ensure monotonicity (defensive for small files / many chunks).
    for i in range(1, len(boundaries)):
        if boundaries[i] < boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1]
    return boundaries

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _pre_tokenize_worker(
    corpus_chunk: str,
    special_tokens: list[str],
    ) -> dict[tuple[int, ...], int]:
    # Split on special tokens *before* applying PAT, so merges cannot occur across the
    # text boundaries delimited by special tokens.
    # Important: special tokens may contain regex metacharacters (e.g., '|'), so escape them.
    # Reference-style: split on special tokens first, then apply PAT within each piece.
    # Important: use a *capturing* group so special tokens are preserved as separate pieces.
    pieces: list[str]
    special_to_id = {tok: i for i, tok in enumerate(special_tokens)}
    if special_tokens:
        special_pat = re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")
        pieces = special_pat.split(corpus_chunk)
    else:
        pieces = [corpus_chunk]

    str_freqs: Counter[str] = Counter()
    special_freqs: Counter[str] = Counter()

    for piece in pieces:
        if not piece:
            continue
        if piece in special_to_id:
            special_freqs[piece] += 1
            continue
        for m in PAT.finditer(piece):
            str_freqs[m.group(0)] += 1

    # Normal PAT tokens are returned as tuples of *token ids*.
    # Initial vocab ids are:
    #   0..len(special_tokens)-1 : special tokens
    #   len(special_tokens)..len(special_tokens)+255 : single-byte tokens
    byte_id_offset = len(special_tokens)
    out: dict[tuple[int, ...], int] = {}
    for s, c in str_freqs.items():
        b = s.encode("utf-8")
        ids = tuple(byte_id_offset + x for x in b)
        out[ids] = out.get(ids, 0) + c
    # Special tokens should be atomic IDs, *not* their raw byte sequences.
    # In the reference implementation, special tokens are assigned ids first (0..),
    # followed by the 256 single-byte tokens.
    for tok, c in special_freqs.items():
        out[(special_to_id[tok],)] = out.get((special_to_id[tok],), 0) + c
    return out

def _preprocess(
    num_processes: int, 
    corpus: BinaryIO, 
    split_tokens: list[bytes]
    ) -> dict[tuple[int, ...], int]:
    chunk_boundaries = None
    chunk_boundaries = find_chunk_boundaries(corpus, num_processes, split_tokens)
    
    total_workers = len(chunk_boundaries)
    token_chunks: list[str] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        corpus.seek(start)
        token_chunks.append(corpus.read(end - start).decode("utf-8", errors="ignore"))

    word_freqs: dict[tuple[int, ...], int] = Counter()
    special_token_strs = [t.decode("utf-8") for t in split_tokens]
    with Pool(processes=total_workers) as pool:
        for token_map in pool.starmap(_pre_tokenize_worker, [(chunk, special_token_strs) for chunk in token_chunks], chunksize=1):
            word_freqs.update(token_map)
    return dict(word_freqs)

class BPETrainer:

    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str], *, disired_num_chunks: int = 8):
        self.corpus = None
        # Reference ordering: special tokens first, then the 256 byte tokens.
        self.vocab: dict[int, bytes] = {}
        # Store merges internally as token-id pairs; convert to bytes at return time.
        self.merge_ids: list[tuple[int, int]] = []
        self.input_path: BinaryIO = open(input_path, "rb")
        self.vocab_size = vocab_size
        self.special_tokens: list[bytes] = [special_token.encode("utf-8") for special_token in special_tokens]
        self.desired_num_chunks = disired_num_chunks

        # Add special tokens deterministically at the start.
        self.special_token_to_id: dict[bytes, int] = {}
        next_id = 0
        for tok in self.special_tokens:
            self.vocab[next_id] = tok
            self.special_token_to_id[tok] = next_id
            next_id += 1

        # Then add all single-byte tokens.
        for i in range(256):
            self.vocab[next_id] = bytes([i])
            next_id += 1

    def pretokenize(self) -> dict[tuple[int, ...], int]:
        # Worker already splits on special tokens before PAT tokenization.
        return _preprocess(self.desired_num_chunks, self.input_path, self.special_tokens)

    def train(self, num_merges: int) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        # Populate word_freqs, word_templates, wordId_freqs, pair_freqs, heap
        word_freqs: dict[tuple[int, ...], int] = self.pretokenize()
        word_templates: list[BPEDoublyLinkedList] = [BPEDoublyLinkedList(word) for word in word_freqs.keys()]
        wordId_freqs: list[int] = [freq for freq in word_freqs.values()] # wordId_freqs[i] is the frequency of word_templates[i]
        pair_freqs: dict[tuple[int, int], int] = Counter()
        pair_position_map: dict[tuple[int, int], set[tuple[int, _Node]]] = collections.defaultdict(set)
        for i, word_template in enumerate(word_templates):
            node = word_template._node.next # [sentinel] -> [token1] -> [token2] -> ... -> [sentinel]
            while node.next != word_template._node:
                pair_freqs[(node.val, node.next.val)] += wordId_freqs[i]
                pair_position_map[(node.val, node.next.val)].add((i, node))
                node = node.next
        # Heap key: (-freq, reverse_lex_pair(bytes,bytes), pair).
        # Per handout, deterministically break ties by preferring the lexicographically
        # *greater* pair (like Python's max on (bytes,bytes)).
        heap: list[tuple[int, _ReverseLexOrderPair, tuple[int, int]]] = [
            (-freq, _ReverseLexOrderPair((self.vocab[p[0]], self.vocab[p[1]])), p)
            for p, freq in pair_freqs.items()
        ]
        heapq.heapify(heap)


        # The adapter passes num_merges=os.cpu_count(); ignore that and derive merges from vocab_size.
        # Contract: vocab_size includes special tokens.
        target_merges = max(0, self.vocab_size - len(self.vocab))


        sentinel_val = 0
        merges_done = 0
        while merges_done < target_merges and heap:
            # Pop until we find a non-stale entry.
            while heap:
                neg_f, _, pair = heapq.heappop(heap)
                freq = -neg_f
                if pair_freqs.get(pair, 0) == freq and freq > 0:
                    break
            else:
                break

            merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]


            # Allocate the next available vocab id contiguously.
            new_id = len(self.vocab)
            self.vocab[new_id] = merged_bytes
            self.merge_ids.append(pair)
            merges_done += 1

            # Snapshot positions to avoid mutating a set while iterating it.
            positions = list(pair_position_map[pair])

            to_remove: dict[tuple[int, int], list[tuple[int, _Node]]] = collections.defaultdict(list)
            to_add: dict[tuple[int, int], list[tuple[int, _Node]]] = collections.defaultdict(list)
            freq_delta: Counter[tuple[int, int]] = Counter()

            a, b = pair
            for wordId, node in positions:
                # Stale check: the node may have been affected by a previous merge within this same outer loop.
                if node.next is None:
                    continue
                if node.val != a or node.next.val != b:
                    continue

                wfreq = wordId_freqs[wordId]
                word = word_templates[wordId]

                left_node = node.prev
                right_node = node.next  # this node will be unlinked by merge
                right_next = right_node.next

                # Remove the AB occurrence at (wordId, node)
                to_remove[(a, b)].append((wordId, node))
                freq_delta[(a, b)] -= wfreq

                # Left neighbor: (L,A) -> (L,AB) at left_node
                if left_node is not None and left_node != word._node and left_node.val != sentinel_val:
                    la = (left_node.val, a)
                    lz = (left_node.val, new_id)
                    to_remove[la].append((wordId, left_node))
                    to_add[lz].append((wordId, left_node))
                    freq_delta[la] -= wfreq
                    freq_delta[lz] += wfreq

                # Right neighbor: (B,R) -> (AB,R)
                if right_next is not None and right_next != word._node and right_next.val != sentinel_val:
                    br = (b, right_next.val)
                    zr = (new_id, right_next.val)
                    # The (B,R) occurrence is stored at the left node of that pair, i.e., right_node.
                    to_remove[br].append((wordId, right_node))
                    to_add[zr].append((wordId, node))
                    freq_delta[br] -= wfreq
                    freq_delta[zr] += wfreq

                # Perform the actual merge in the linked list.
                word.merge_with_next(node, new_id)

            # Apply batched position updates.
            for p, items in to_remove.items():
                s = pair_position_map.get(p)
                if not s:
                    continue
                for it in items:
                    s.discard(it)
            for p, items in to_add.items():
                s = pair_position_map[p]
                for it in items:
                    s.add(it)

            # Apply frequency deltas.
            for p, d in freq_delta.items():
                pair_freqs[p] += d
                if pair_freqs[p] < 0:
                    pair_freqs[p] = 0


            # Push all pairs whose frequencies may have changed back onto the heap.
            for p in freq_delta.keys():
                freq = pair_freqs.get(p, 0)
                if freq > 0:
                    heapq.heappush(heap, (-freq, _ReverseLexOrderPair((self.vocab[p[0]], self.vocab[p[1]])), p))

        merges_bytes: list[tuple[bytes, bytes]] = [(self.vocab[a], self.vocab[b]) for a, b in self.merge_ids]
        return self.vocab, merges_bytes


        





