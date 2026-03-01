import heapq
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
        
    def unlink(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None # help GC
    
    def merge_with_next(self, left: _Node, merged_val:int):
        left.val = merged_val
        self.unlink(left.next)

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[Tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_offset = len(self.special_tokens)
        self.merge_offset = self.bytes_offset + 256

        self.special_patterns = re.compile("(" + '|'.join(re.escape(tok) for tok in self.special_tokens) + ")") if self.special_tokens else None

        bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.rank: dict[Tuple[int, int], int] = {}
        for i, merge in enumerate(self.merges):
            token_id1 = bytes_to_id[merge[0]]
            token_id2 = bytes_to_id[merge[1]]
            self.rank[(token_id1, token_id2)] = i

    @classmethod
    def from_files(cls, vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        pass
    
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
            words = self.special_patterns.split(text)
        else:
            words = [text]
        
        output_token_ids: list[int] = []
        for word in words:
            if word in self.special_tokens:
                output_token_ids.append(self.special_tokens.index(word))
            else:
                output_token_ids.extend(self._encode_single(word.encode("utf-8")))
        return output_token_ids
    
    def _encode_single(self, text: bytes) -> list[int]:
        token_ids = [self.bytes_offset + b for b in text]
        if not token_ids:
            return []
        
        dll = EncoderDoublyLinkedList(token_ids)
        heap: list[Tuple[int, _Node]] = [] # (rank, id(node), node)
        def try_push(node: _Node):
            next_node = node.next
            pair = (node.val, next_node.val)
            if pair in self.rank:
                rank = self.rank[pair]
                heapq.heappush(heap, (rank, node))

        cur_node = dll._node.next
        while cur_node.next != dll._node:
            try_push(cur_node)
            cur_node = cur_node.next
        
        while heap:
            rank, node = heapq.heappop(heap)
            next_node = node.next
            pair = (node.val, next_node.val)
            if pair not in self.rank or next_node == dll._node or next_node is None: 
                continue # stale
            dll.merge_with_next(node, rank + self.merge_offset)
            if node.prev != dll._node:
                try_push(node.prev)
            if node.next != dll._node:
                try_push(node)
        return [node.val for node in dll]
            


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")