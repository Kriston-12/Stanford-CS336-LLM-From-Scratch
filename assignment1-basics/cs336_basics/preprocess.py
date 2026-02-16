import os
from typing import BinaryIO, Iterable
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import Tuple

# does not always return a list with length = dired_num_chunks
# might return 2 when disired_num_chunks = 5 when it finds only 2 special_tokens in the file
def find_chunk_boundaries(
    input_file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: bytes
) -> list[int]: 
    assert isinstance(special_tokens, bytes), "split_tokens has to be bytes"
    assert desired_num_chunks >= 1, "desired_num_chunks has to be at least 1"
    input_file.seek(0, os.SEEK_END) #seek(offset, whence) offset=0：偏移 0 个字节, whence=os.SEEK_END：从“文件末尾”作为基准位置开始算
    file_len = input_file.tell()
    input_file.seek(0)

    desired_chunk_length = file_len // desired_num_chunks

    # Boundary list includes start (0) and end (file_len).
    # We may return fewer than desired_num_chunks if boundaries overlap.
    chunk_boundaries = [i * desired_chunk_length for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_len

    mini_chunk_len = 2048
    initial_boundary = 0
    for i in range(1, desired_num_chunks):
        # Start from the initial guess for this boundary; if earlier boundaries moved forward,
        # ensure boundaries never go backwards.
        initial_boundary = max(chunk_boundaries[i], initial_boundary)
        input_file.seek(initial_boundary)
        while True:
            mini_chunk = input_file.read(mini_chunk_len)
            if mini_chunk == b"":
                chunk_boundaries[i] = file_len
                return chunk_boundaries[0:i + 1] # all later chunk_boundaries will have same boundary = file_len

            # Search within the bytes we just read.
            boundary = mini_chunk.find(special_tokens)  # -1 if not found
            if boundary != -1:  # found boundary
                chunk_boundaries[i] = initial_boundary + boundary
                break
            initial_boundary += mini_chunk_len

    return chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(
    corpus_chunk: str,
    split_pattern: str
    ) -> dict[bytes, int]:
    tokens: list[str] = re.findall(split_pattern, corpus_chunk)
    str_freqs: dict[str, int] = Counter(tokens)

    return {s.encode("utf-8"): c for s, c in str_freqs.items()}

def run_worker_pre_tokenize(
    corpus_chunk: str,
    ) -> dict[bytes, int]:
    return pre_tokenize(corpus_chunk, PAT)

def preprocess(
    num_processes: int, 
    file_path: str, 
    split_tokens: bytes
    ) -> dict[bytes, int]:
    chunk_boundaries = None
    file: BinaryIO = open(file_path, "rb")
    chunk_boundaries = find_chunk_boundaries(file, num_processes, split_tokens)
    
    total_workers = len(chunk_boundaries)
    token_chunks: list[str] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        file.seek(start)
        token_chunks.append(file.read(end - start).decode("utf-8", errors="ignore"))

    total_token_map: dict[bytes, int] = Counter()
    with Pool(processes=total_workers) as pool:
        for token_map in pool.imap_unordered(run_worker_pre_tokenize, token_chunks, chunksize=1):
            total_token_map.update(token_map)
    
    return dict(total_token_map)




    

