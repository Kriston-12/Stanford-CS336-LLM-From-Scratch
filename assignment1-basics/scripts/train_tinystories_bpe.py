from __future__ import annotations

import argparse
import base64
import json
import os
import resource
import time
from pathlib import Path

from implementation.advance_impl import BPETrainer

def _rss_gb() -> float:
    # ru_maxrss: kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def decode_b64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# starting script
'''
uv run scripts/train_tinystories_bpe.py   --input tests/fixtures/tinystories_sample_5M.txt   --vocab-size 10000

'''

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to TinyStories training text (must contain <|endoftext|> delimiters)",
    )
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--num-chunks", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts") / "tinystories_bpe")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rss0 = _rss_gb()

    trainer = BPETrainer(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        disired_num_chunks=args.num_chunks,
    )
    vocab, merges = trainer.train(num_merges=args.vocab_size)

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


if __name__ == "__main__":
    main()
