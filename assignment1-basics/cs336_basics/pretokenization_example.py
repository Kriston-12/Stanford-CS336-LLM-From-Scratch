import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # Move to end of file
    file_size = file.tell() # Get current position (which is the file size as we are at the end). unit is bytes
    file.seek(0)            # Move back to the start of the file

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # [0, chunk_size, 2*chunk_size, ..., desired_num_chunks*chunk_size]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi] # start at chunk_size, 2*chunk_size, ..., (desired_num_chunks-1)*chunk_size
        file.seek(initial_position)  # Start at boundary guess -- end of previous chunk. starts at end of chunk 0
        while True:
            # for the first chunk, it reads at near the end of chunk 0
            # which means we aim to find the special token that's near the end of chunk 0
            # here we read if the special token is in the next 4k bytes, if not, we read the next 4k bytes until we find the special token
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            # We just read the next 4k bytes, so if we find the special token, we know that the boundary should be at the position of the special token
            found_at = mini_chunk.find(split_special_token)

            # we decide to split the chunk at the position of the special token
            # the worst uneven case is chunksize = 10 (disired num chunks = 10, file size = 100), min_chunk_size = 4096
            # we did not find the special token in the entire file (100 bytes) at all
            # even tho we want 10 chunks, we will end up with 1 chunk, which is the entire file, because we never find the special token to split the chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
