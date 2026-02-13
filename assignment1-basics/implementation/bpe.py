import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizerParams():
    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[int, int], int]):
        self.vocab = vocab
        self.merges = merges

def pre_tokenize(corpus_chunk: str, split_pattern: str) -> dict[bytes, int]:
    tokens = re.findall(split_pattern, corpus_chunk)
    str_freqs = Counter(tokens)
    return {s.encode("utf-8"): c for s, c in str_freqs.items()}

def freq_tok_seq_in_pretokenization_dict(tok1: bytes, tok2: bytes, 
                                          pretok_dict: dict[list[int], int]) -> int:
    for token_sequence in pretok_dict:
        for x, y in zip(token_sequence, token_sequence[1:]):
            if tok1 == x and tok2 == y:
                return pretok_dict[token_sequence]
    return 0

def merge(indices: list[int], pair: tuple[int, int], new_index) -> list[int]:
    index1, index2 = pair
    i = 0
    new_indices = []
    while i < len(indices):
        if indices[i] == index1 and i < len(indices) - 1 and indices[i + 1] == index2:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def train_bpe(text: str, num_merges: int) -> BPETokenizerParams:
    indices: list[int] = list(text.encode("UTF-8")) # or list(map(int, text.encode("UTF-8")))
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: dict[tuple[int, int], int] = None

    pretok_dict = pre_tokenize(text)
    for i in range(num_merges):
        counts: dict[tuple[int, int], int] # map adacent token to freqs
        for tok1, tok2 in zip(indices, indices[1:]):
            freq_appear_in_pre_token = freq_tok_seq_in_pretokenization_dict(vocab[tok1], vocab[tok2], pretok_dict) 
            if freq_appear_in_pre_token > 0:
                counts[tok1, tok2] += freq_appear_in_pre_token
            else:
                counts[tok1, tok2] += 1
    
        most_freq_pair = max(counts, key=counts.get)
        index1, index2 = most_freq_pair

        new_index = 256 + i
        merges[most_freq_pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, most_freq_pair, new_index)

    return BPETokenizerParams(vocab, merges)


            

