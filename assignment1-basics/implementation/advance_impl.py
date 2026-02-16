
import collections
import heapq
from multiprocessing import Pool
from typing import BinaryIO, Optional, Tuple, Iterable
from dataclasses import dataclass
import os
import regex as re
from collections import Counter
# 1. 用doubly linked list 表示corpus, 在merge的时候只需修改响铃两个token的指针，O(1) 
    # strucutre: [Node(val=byte1, prev=None, next=Node(val=byte2, prev=Node(val=byte1), next=None))]
# 2. 维护一个maxHeap, log(n)时间找到出现频率最高的pair of tokens, merge后更新heap中相关pair的频率
# 3. 一个position_map dict[tuple[int, int], set[int]]. Key是一个pair (A, B), value是当前pair在corpus中的对应node.
    # set[int] = set(Node1, Node2, Node3). merge可以直接用 map[pair].prev = map[pair].next来更新node


# 下面两个classes用于维护BPE的merge逻辑，O(1)时间修改语段
@dataclass(slots=True)
class _Node:
    val: int
    prev: Optional["_Node"] = None
    next: Optional["_Node"] = None

class BPEDoublyLinkedList:
    __slots__ = ("_val")
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
    initial_pos = boundaries[1]
    for i in range(1, desired_num_chunks + 1):
        initial_pos = max(initial_pos, boundaries[i])
        file.seek(initial_pos)
        while True:
            read_buffer = file.read(transition_size)
            if read_buffer == b"":
                boundaries[i] = file_len
                return boundaries[:i + 1]

            found_at = [read_buffer.find(special_token) for special_token in special_tokens]
            if found_at:
                boundaries[i] = min(found_at)
                break

            initial_pos += transition_size
    return boundaries

def _pre_tokenize_worker(
    corpus_chunk: str,
    word_pattern: str
    ) -> dict[tuple[int, ...], int]:
    tokens: list[str] = re.findall(word_pattern, corpus_chunk)
    str_freqs: dict[str, int] = Counter(tokens)
    return {tuple(s.encode("utf-8")): c for s, c in str_freqs.items()}

def _preprocess(
    num_processes: int, 
    corpus: BinaryIO, 
    split_tokens: bytes
    ) -> dict[tuple[int, ...], int]:
    chunk_boundaries = None
    chunk_boundaries = find_chunk_boundaries(corpus, num_processes, split_tokens)
    
    total_workers = len(chunk_boundaries)
    token_chunks: list[str] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        corpus.seek(start)
        token_chunks.append(corpus.read(end - start).decode("utf-8", errors="ignore"))

    word_freqs: dict[tuple[int, ...], int] = Counter()
    with Pool(processes=total_workers) as pool:
        for token_map in pool.imap_unordered(_pre_tokenize_worker, token_chunks, chunksize=1):
            word_freqs.update(token_map)
    return dict(word_freqs)

class BPETrainer:

    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str], *, disired_num_chunks: int = 8):
        self.corpus = None
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)} 
        self.merge: list[tuple[bytes, bytes]] = [] # [[token1, token2], ....]
        self.input_path: BinaryIO = open(input_path, "rb")
        self.vocab_size = vocab_size
        self.special_tokens: list[bytes] = [special_token.encode("utf-8") for special_token in special_tokens]
        self.desired_num_chunks = disired_num_chunks

    def pretokenize(self) -> dict[tuple[int, ...], int]:
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
        heap: list[tuple[int, bytes, tuple[int, int]]] = [(-freq, [-b for b in (self.vocab[pair[0]] + self.vocab[pair[1]])], pair) for pair, freq in pair_freqs.items()]
        heapq.heapify(heap)

        for i in range(num_merges):
            if not heap:
                break
            neg_f, _, pair = heapq.heappop(heap)
            if -neg_f != pair_freqs[pair]: # stale pair, skip. 比如说开始 右 (X A B Y), (A, B)频率是100. merge (A, B) -> Z. AB 频率最高是(100), (X A) 频率(50), AB被merge之后，heap中(X, A) 变成了 (X, AB). 但是heap中还有 (X, A), 所以我们需要判断如果pair_freqs[pair] != -neg_f, 说明这个pair已经被merge过了，频率已经更新了，但是heap中还没有更新，所以这个pair是stale的，我们需要跳过它。
                continue
            self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merge.append(pair)
            pair_freqs[pair] = 0
            new_pair_counter: dict[tuple[int, int], int] = Counter()
            for wordId, node in pair_position_map[pair]:
                left_old, right_old = pair
                word_templates[wordId].merge_with_next(node, 256 + i)
                pair_freq = wordId_freqs[wordId]
                if node.prev != word_templates[wordId]._node: # node.prev is not the sentinel
                    left, right = left_pair = (node.prev.val, node.val) # node.val has already been updated here 
                    pair_freqs[left_pair] -= pair_freq
                    pair_position_map[(left_old, right_old)].remove((wordId, node.prev)) 
                    pair_position_map[(left, right)].add((wordId, node.prev))
                    pair_freqs[(left, right)] += pair_freq
                    new_pair_counter[(left, right)] += pair_freq
                if node.next != word_templates[wordId]._node: # node.next is not the sentinel
                    left, right = right_pair = (node.val, node.next.val)
                    pair_freqs[right_pair] -= pair_freq
                    pair_position_map[(left_old, right_old)].remove((wordId, node))
                    pair_position_map[(left, right)].add((wordId, node))
                    pair_freqs[(left, right)] += pair_freq
                    new_pair_counter[(left, right)] += pair_freq
            for new_pair, freq in new_pair_counter.items():
                heapq.heappush(heap, (-freq, [-b for b in (self.vocab[new_pair[0]] + self.vocab[new_pair[1]])], new_pair))


        





