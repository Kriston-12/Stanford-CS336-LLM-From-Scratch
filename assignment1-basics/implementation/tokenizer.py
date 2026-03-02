import heapq
import json
from typing import Iterable, Iterator, Tuple, Optional

import regex as re


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class _Node: # vocab[vocab_id] = val, vocab_id是为了我们找到merge之后保存对应的token id, 方便后续encode成token id list
    __slots__ = ['val', 'prev', 'next']
    def __init__(self, val: int,  prev = None, next = None):
        self.val = val
        self.prev = prev
        self.next = next

class EncoderDoublyLinkedList:
    __slots__ = ['_node' ]
    def __init__(self, init_list: Optional[Iterable[int]] = None):
        Node = _Node(0)
        Node.prev = Node.next = Node
        self._node = Node

        for x in init_list:
            self.append(x)

    # self._node.prev stays unchanged, it's always itself
    # self._node.right changes to the newly init node
    # the []
    def append(self, val: int):
        new_node = _Node(val,  self._node.prev, self._node)
        last_added_node = self._node.prev
        last_added_node.next = new_node
        self._node.prev = new_node # update the last added node to be the new node
        return new_node
        
    def unlink(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None # help GC
    
    def merge_with_next(self, left: _Node, merged_val:int):
        left.val = merged_val
        self.unlink(left.next)

    def __iter__(self):
        cur = self._node.next
        while cur != self._node:
            yield cur
            cur = cur.next

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[Tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_offset = len(self.special_tokens)
        self.merge_offset = self.bytes_offset + 256

        # In the tests, vocab is dict[token_id, bytes], where the bytes are the
        # *original bytes* (the test fixture already inverted GPT-2's bytes<->unicode
        # remapping). So our tokenizer should operate directly on raw bytes.
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Important: handle overlapping special tokens by matching the longest first.
        # Example: "<|endoftext|>" vs "<|endoftext|><|endoftext|><|endoftext|>".
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_patterns = re.compile(
                "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            )
            self.special_to_id: dict[str, int] = {
                tok: self.bytes_to_id[tok.encode("utf-8")]
                for tok in self.special_tokens
            }
        else:
            self.special_patterns = None
            self.special_to_id = {}

        # rank[(id_left,id_right)] = merge_index
        self.rank: dict[Tuple[int, int], int] = {}
        for i, (b1, b2) in enumerate(self.merges):
            self.rank[(self.bytes_to_id[b1], self.bytes_to_id[b2])] = i

    @classmethod
    def from_files(cls, vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        vocab: dict[int, bytes] = {}
        merges: list[Tuple[bytes, bytes]] = []
        with open(vocab_filepath, "rb") as f:
            # vocab is a json file
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            for line in f:
                cleaned = line.rstrip()
                if cleaned and len(cleaned.split(" ")) == 2:
                    # Keep as bytes (tests provide merges as bytes; if you load raw GPT2 merges here, you must convert)
                    a, b = cleaned.split(" ")
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))
        return cls(vocab, merges, special_tokens)
    
    # 首先要明确在我的bpe的实现中，vocab[0...len(special_tokens)-1]是special tokens, 它们对应b'0', b'1', ..., b'{len(special_tokens)-1}'
    # vocab[256 + i] = merge for merge in merges[i], 这意味着merges[i]对应的token id是vocab[256 + i]
    # naive的方法是将text转成list[bytes], 然后for merge in merges逐步更新list[bytes]. O(n * m) where n is the length of text and m is the number of merges.
    # 当然不能这么naive, doubly linkedlist + heap 方法如下:
    # 1. words = self.special_patterns.split(text) -> list[str]
    # 2. for i, word in enumerate(words): list[DLList][i] = [] 空标记此位置是specialtokens
            # 否则 list[DLList][i] = DLList(list of byte ids for word)
            # 注意 list[DLList][i] 位置对应text 位置，顺序不变，方便后续还原
    # 3. assume merge 有序, for merge in merges: 将list[DLList]中所有包含merge的Dlist内部合并
        # 持续这个过程直到所有merges都处理完
    # 4. 最后得到的list[DLList]中每个DLList内部的node.val就是我们要返回的token id, 直接遍历list[DLList]中的每个DLList, 遍历DLList中的node.val, 就得到最终的token id list 
    # 上面这种方法相比naive更高效，但是实现的时候要instantiate和text长度相当的data structures，容易爆memory
    # 下面是我想的更好的方法，并且容易parallelize:
    # 1. words = self.special_patterns.split(text) -> list[str]
    # 2. for single_word in PAT.finditer()
    # 3. 在for循环中然后对每个single做如下处理:
    #   3.1 将single_word encode 成 EncoderDoublyLinkedList.
    #   3.2 将这个EncoderDoublyLinkedList中的每个相邻的node pair加入heap, heap的key是这个pair对应的merge的rank, value是这个pair所在的node
    # 4. while heap is not empty:
    #   4.1 pop heap, 得到rank, node，如果rank不在self.rank中，说明这个pair不是一个merge，continue
    #   4.2 否则说明这个pair是一个merge，找到对应
    def encode(self, text: str) -> list[int]:
        if self.special_patterns:
            pieces = self.special_patterns.split(text)
        else:
            pieces = [text]
        
        out: list[int] = []
        for piece in pieces:
            if not piece:
                continue
            # Special tokens are appended to vocab (by tests) if missing.
            if piece in self.special_tokens:
                out.append(self.special_to_id[piece])
                continue

            for m in PAT.finditer(piece):
                raw = m.group(0).encode("utf-8")
                out.extend(self._encode_single(raw))
        return out
    
    def _encode_single(self, text: bytes) -> list[int]:
        if not text:
            return []

        # Base encoding: one token per raw byte.
        # IMPORTANT: do not assume any particular id layout; look up ids by bytes.
        token_ids = [self.bytes_to_id[bytes([b])] for b in text]
        if len(token_ids) == 1:
            return token_ids
        
        dll = EncoderDoublyLinkedList(token_ids)
        # (rank, tie_breaker, node). tie_breaker avoids comparing _Node objects.
        heap: list[Tuple[int, int, _Node]] = []
        def try_push(node: _Node):
            if node is None or node is dll._node:
                return
            next_node = node.next
            if next_node is None or next_node is dll._node:
                return
            pair = (node.val, next_node.val)
            rank = self.rank.get(pair)
            if rank is None:
                return
            heapq.heappush(heap, (rank, id(node), node))

        cur_node = dll._node.next
        while cur_node.next != dll._node:
            try_push(cur_node)
            cur_node = cur_node.next
        
        while heap:
            rank, _, node = heapq.heappop(heap)
            # stale: node was unlinked
            if node.prev is None or node.next is None:
                continue
            next_node = node.next
            if next_node is None or next_node is dll._node:
                continue
            pair = (node.val, next_node.val)
            if self.rank.get(pair) != rank:
                continue

            # Compute merged token id by bytes: vocab[new] = vocab[left] + vocab[right]
            merged_bytes = self.vocab[node.val] + self.vocab[next_node.val]
            new_id = self.bytes_to_id[merged_bytes]
            dll.merge_with_next(node, new_id)
            if node.prev != dll._node:
                try_push(node.prev)
            if node.next != dll._node:
                try_push(node)
        return [node.val for node in dll]
            


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # The vocab stores raw bytes; concatenating them reproduces the original
        # UTF-8 byte stream for normal text.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")