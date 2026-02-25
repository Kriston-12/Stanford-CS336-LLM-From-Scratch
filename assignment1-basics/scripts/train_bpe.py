from __future__ import annotations

import argparse
import base64
import json
import os
import resource
import time
from pathlib import Path

from implementation.advance_impl import BPETrainer
from contextlib import contextmanager

def _rss_gb() -> float:
    # ru_maxrss: kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def decode_b64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _train(path: str, vocab_size: int, num_chunks: int) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        trainer = BPETrainer(
            input_path=str(path),
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            disired_num_chunks=num_chunks,
        )
        return trainer.train(num_merges=vocab_size)

@contextmanager
def profile(out_folder: Path, sort_by: str = "cumulative", top_n: int = 50):
    """
    Profile the code inside the context, write .prof and a readable .txt report.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / "profile.prof"
    import cProfile
    import pstats

    prof = cProfile.Profile()
    prof.enable()

    try: 
        yield #这里会执行with块中的代码--也就是真正被profile的函数
    finally: # 函数的执行已经完毕，profile完成，执行下面的代码
        prof.disable() 

        prof.dump_stats(str(out_path)) # 将profile结果以二进制格式写入out_path指定的文件中，通常以.prof结尾

        txt_path = out_path.with_suffix(out_path.suffix + ".txt") # 一般传入的output_path 以.prof结尾，这里会生成一个.prof.txt的文件
        with open(txt_path, "w", encoding="utf-8") as f:
            stats = pstats.Stats(prof, stream=f)
            stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)
        
        print(f"[cProfile] wrote: {out_path}")
        print(f"[cProfile] wrote: {txt_path}")                        


# starting script
'''
uv run scripts/train_tinystories_bpe.py   --input tests/fixtures/tinystories_sample_5M.txt   --vocab-size 10000
'''
def main_coarse_time_mem_with_summary() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to TinyStories training text (must contain <|endoftext|> delimiters)",
    )
    ap.add_argument("-v", "--vocab-size", type=int, default=10_000)
    ap.add_argument("-n", "--num-chunks", type=int, default=os.cpu_count() or 8)
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("artifacts") / "tinystories_bpe")
    
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rss0 = _rss_gb()

    vocab, merges = _train(str(args.input), args.vocab_size, args.num_chunks)
    
    t1 = time.time()
    rss1 = _rss_gb()

    # Longest token by byte length.
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))

    # Serialize vocab + merges in an inspection-friendly (round-trippable) format.
    vocab_path = args.out_dir / "vocab.b64.json"
    merges_path = args.out_dir / "merges.b64.json"
    summary_path = args.out_dir / "summary.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(i): _b64(b) for i, b in vocab.items()}, f)
    
    print(f"Learned {len(vocab)} vocab items and {len(merges)} merges.")

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump([[_b64(a), _b64(b)] for a, b in merges], f)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(args.input),
                "vocab_size": args.vocab_size,
                "num_chunks": args.num_chunks,
                "train_seconds": t1 - t0,
                "rss_start_gb": rss0,
                "rss_end_gb": rss1,
                "rss_max_gb": max(rss0, rss1),
                "longest_token_id": int(longest_id),
                "longest_token_len_bytes": int(len(longest_bytes)),
                "longest_token_bytes_b64": _b64(longest_bytes),
                "longest_token_bytes_repr": repr(longest_bytes),
            },
            f,
            indent=2,
        )

    print(f"Done. Train time: {t1 - t0:.2f}s")
    print(f"Peak RSS (approx, process only): {max(rss0, rss1):.2f} GB")
    print(f"Longest token id={longest_id}, len={len(longest_bytes)} bytes")
    print(f"Wrote: {vocab_path}")
    print(f"Wrote: {merges_path}")
    print(f"Wrote: {summary_path}")


'''
uv run python scripts/train_bpe.py -i tests/fixtures/tinystories_sample_5M.txt -v 10000

use tuna to visualize:
uv run tuna artifacts/tinystories_bpe_smoke/profile.prof
'''
def main_profiled() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to TinyStories training text (must contain <|endoftext|> delimiters)",
    )
    ap.add_argument("-v", "--vocab-size", type=int, default=10_000)
    ap.add_argument("-n", "--num-chunks", type=int, default=os.cpu_count() or 8)
    ap.add_argument("-p", "--profile", type=Path, default=Path("artifacts") / "tinystories_bpe_smoke", help="Write cProfile stats to this .prof path")
    ap.add_argument("-ps", "--profile-sort", default="cumulative", help="cumulative | tottime | calls | ...")
    ap.add_argument("-pt", "--profile-top", type=int, default=50, help="How many rows to print in the text report")

    args = ap.parse_args()

    with profile(args.profile, args.profile_sort, args.profile_top):
        _train(args.input, args.vocab_size, args.num_chunks)

if __name__ == "__main__":
    main_profiled()
