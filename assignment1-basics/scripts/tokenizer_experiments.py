"""Tokenizer experiments + Scalene-friendly benchmarks.

This script is meant for assignment writeups:

(a) Sample N docs from TinyStories + OpenWebText, encode with their *own* tokenizers,
    and report compression ratio (bytes/token).
(b) Tokenize OpenWebText with TinyStories tokenizer; compare compression.

Design goals:
- Deterministic sampling (seeded).
- Minimal overhead in the measured loops (so Scalene is meaningful).
- Works with the repo's BPE artifacts written by `scripts/train_bpe.py`.

Typical usage (optional):
- Compression ratios only:
    uv run python scripts/tokenizer_experiments.py --mode ratios
- Full scalene run (encode/decode):
    uv run scalene --html --outfile artifacts/scalene_tokenizers.html \
        scripts/tokenizer_experiments.py --mode scalene

Notes:
- Scalene is an external profiler; we simply provide a stable main loop.
- We keep the encode/decode work in pure Python loops so Scalene can attribute time.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from implementation.tokenizer import Tokenizer


DEFAULT_TINYSTORIES_VOCAB = Path("artifacts") / "tinystories_bpe" / "vocab.b64.json"
DEFAULT_TINYSTORIES_MERGES = Path("artifacts") / "tinystories_bpe" / "merges.b64.json"

# OpenWebText artifacts may not exist in this repo snapshot; keep defaults but allow override.
DEFAULT_OPENWEBTEXT_VOCAB = Path("artifacts") / "openwebtext_bpe" / "vocab.b64.json"
DEFAULT_OPENWEBTEXT_MERGES = Path("artifacts") / "openwebtext_bpe" / "merges.b64.json"

DEFAULT_TINYSTORIES_TXT = Path("tests") / "fixtures" / "tinystories_sample_5M.txt"
DEFAULT_OPENWEBTEXT_TXT = Path("data") / "openwebtext_50k.txt"

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def _decode_b64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def load_bpe_from_artifacts(vocab_b64_json: Path, merges_b64_json: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load (vocab, merges) from the artifacts format produced in `scripts/train_bpe.py`."""

    with open(vocab_b64_json, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    vocab: dict[int, bytes] = {int(k): _decode_b64(v) for k, v in raw_vocab.items()}

    with open(merges_b64_json, "r", encoding="utf-8") as f:
        raw_merges = json.load(f)

    merges: list[tuple[bytes, bytes]] = [(_decode_b64(a), _decode_b64(b)) for a, b in raw_merges]
    return vocab, merges


def load_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> Tokenizer:
    vocab, merges = load_bpe_from_artifacts(vocab_path, merges_path)
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def iter_documents(path: Path) -> Iterable[str]:
    """Yield documents split by the <|endoftext|> delimiter."""

    delim = "<|endoftext|>"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buf: list[str] = []
        for line in f:
            # The delimiter may be on its own line or embedded; handle both.
            while delim in line:
                before, _, after = line.partition(delim)
                buf.append(before)
                doc = "".join(buf).strip("\n")
                if doc:
                    yield doc
                buf = []
                line = after
            buf.append(line)
        # trailing chunk
        tail = "".join(buf).strip("\n")
        if tail:
            yield tail


def sample_documents(path: Path, n_docs: int, seed: int, max_chars: int | None = 20_000) -> list[str]:
    """Reservoir sample N docs (deterministic) without loading the whole file."""

    rng = random.Random(seed)
    samples: list[str] = []
    seen = 0

    for doc in iter_documents(path):
        if max_chars is not None and len(doc) > max_chars:
            doc = doc[:max_chars]
        seen += 1
        if len(samples) < n_docs:
            samples.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n_docs:
                samples[j] = doc

    if not samples:
        raise ValueError(f"No documents found in {path}")
    return samples


@dataclass
class RatioResult:
    dataset: str
    tokenizer: str
    bytes_total: int
    tokens_total: int

    @property
    def bytes_per_token(self) -> float:
        return self.bytes_total / max(1, self.tokens_total)


def compression_ratio(docs: list[str], tok: Tokenizer) -> RatioResult:
    bytes_total = 0
    tokens_total = 0
    for d in docs:
        b = d.encode("utf-8", errors="ignore")
        ids = tok.encode(d)
        bytes_total += len(b)
        tokens_total += len(ids)
    return RatioResult(dataset="", tokenizer="", bytes_total=bytes_total, tokens_total=tokens_total)


def _format_ratio(name: str, ratio: float) -> str:
    return f"{name}: {ratio:.3f} bytes/token"


def run_ratios(
    *,
    tinystories_tok: Tokenizer,
    openwebtext_tok: Tokenizer,
    tinystories_docs: list[str],
    openwebtext_docs: list[str],
) -> None:
    # (a)
    ts_self = compression_ratio(tinystories_docs, tinystories_tok).bytes_per_token
    ow_self = compression_ratio(openwebtext_docs, openwebtext_tok).bytes_per_token

    # (b)
    ow_with_ts = compression_ratio(openwebtext_docs, tinystories_tok).bytes_per_token

    print(_format_ratio("TinyStories tokenizer on TinyStories", ts_self))
    print(_format_ratio("OpenWebText tokenizer on OpenWebText", ow_self))
    print(_format_ratio("TinyStories tokenizer on OpenWebText", ow_with_ts))

    print("\nWriteup-friendly sentences:")
    print(
        f"(a) TinyStories tokenizer compresses TinyStories to ~{ts_self:.2f} bytes/token, "
        f"and OpenWebText tokenizer compresses OpenWebText to ~{ow_self:.2f} bytes/token (10 docs each)."
    )
    print(
        f"(b) Using the TinyStories tokenizer on OpenWebText is worse (higher bytes/token, ~{ow_with_ts:.2f}); "
        "it falls back to shorter subwords/bytes more often because the merges/vocab are mismatched to the domain."
    )


def run_scalene(
    *,
    tinystories_tok: Tokenizer,
    openwebtext_tok: Tokenizer,
    tinystories_docs: list[str],
    openwebtext_docs: list[str],
    rounds: int,
    warmup: int,
) -> None:
    """Hot loops for Scalene to profile.

    We measure 4 workloads:
    - encode TinyStories with TinyStories tokenizer
    - decode those ids
    - encode OpenWebText with OpenWebText tokenizer
    - decode those ids

    And optionally encode OpenWebText with TinyStories tokenizer (domain mismatch).

    Output is deliberately minimal; Scalene HTML report will show hotspots.
    """

    # Warmup (helps avoid measuring one-time regex compilation / caches)
    for _ in range(warmup):
        _ = [tinystories_tok.encode(d) for d in tinystories_docs]
        _ = [openwebtext_tok.encode(d) for d in openwebtext_docs]

    # Pre-encode once so decode workload isn't dominated by encode.
    ts_ids = [tinystories_tok.encode(d) for d in tinystories_docs]
    ow_ids = [openwebtext_tok.encode(d) for d in openwebtext_docs]
    ow_ids_with_ts = [tinystories_tok.encode(d) for d in openwebtext_docs]

    # Encode loops
    for _ in range(rounds):
        for d in tinystories_docs:
            tinystories_tok.encode(d)

    for _ in range(rounds):
        for d in openwebtext_docs:
            openwebtext_tok.encode(d)

    for _ in range(rounds):
        for d in openwebtext_docs:
            tinystories_tok.encode(d)

    # Decode loops
    for _ in range(rounds):
        for ids in ts_ids:
            tinystories_tok.decode(ids)

    for _ in range(rounds):
        for ids in ow_ids:
            openwebtext_tok.decode(ids)

    for _ in range(rounds):
        for ids in ow_ids_with_ts:
            tinystories_tok.decode(ids)

    # Print small sanity summary so the run isn't "silent".
    ts_lens = [len(x) for x in ts_ids]
    ow_lens = [len(x) for x in ow_ids]
    ow_ts_lens = [len(x) for x in ow_ids_with_ts]
    print(
        "done; tokens/doc medians: "
        f"TS->TS={statistics.median(ts_lens):.0f}, OW->OW={statistics.median(ow_lens):.0f}, OW->TS={statistics.median(ow_ts_lens):.0f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ratios", "scalene"], default="ratios")

    ap.add_argument("--tinystories-txt", type=Path, default=DEFAULT_TINYSTORIES_TXT)
    ap.add_argument("--openwebtext-txt", type=Path, default=DEFAULT_OPENWEBTEXT_TXT)

    ap.add_argument("--tinystories-vocab", type=Path, default=DEFAULT_TINYSTORIES_VOCAB)
    ap.add_argument("--tinystories-merges", type=Path, default=DEFAULT_TINYSTORIES_MERGES)

    ap.add_argument("--openwebtext-vocab", type=Path, default=DEFAULT_OPENWEBTEXT_VOCAB)
    ap.add_argument("--openwebtext-merges", type=Path, default=DEFAULT_OPENWEBTEXT_MERGES)

    ap.add_argument("--special-token", action="append", dest="special_tokens", default=list(DEFAULT_SPECIAL_TOKENS))

    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=20_000)

    ap.add_argument("--rounds", type=int, default=30, help="For --mode scalene: how many rounds over the sampled docs")
    ap.add_argument("--warmup", type=int, default=2, help="For --mode scalene: warmup rounds")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    for p in [args.tinystories_txt, args.openwebtext_txt, args.tinystories_vocab, args.tinystories_merges]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    if not args.openwebtext_vocab.exists() or not args.openwebtext_merges.exists():
        if args.mode == "ratios":
            raise FileNotFoundError(
                "OpenWebText tokenizer artifacts not found. "
                "Point --openwebtext-vocab/--openwebtext-merges to your trained 32k tokenizer artifacts."
            )
        # For scalene mode, still allow profiling TinyStories tokenizer on both datasets.

    tinystories_docs = sample_documents(args.tinystories_txt, args.n_docs, args.seed, max_chars=args.max_chars)
    openwebtext_docs = sample_documents(args.openwebtext_txt, args.n_docs, args.seed, max_chars=args.max_chars)

    ts_tok = load_tokenizer(args.tinystories_vocab, args.tinystories_merges, args.special_tokens)

    if args.openwebtext_vocab.exists() and args.openwebtext_merges.exists():
        ow_tok = load_tokenizer(args.openwebtext_vocab, args.openwebtext_merges, args.special_tokens)
    else:
        ow_tok = ts_tok

    if args.mode == "ratios":
        run_ratios(
            tinystories_tok=ts_tok,
            openwebtext_tok=ow_tok,
            tinystories_docs=tinystories_docs,
            openwebtext_docs=openwebtext_docs,
        )
    else:
        run_scalene(
            tinystories_tok=ts_tok,
            openwebtext_tok=ow_tok,
            tinystories_docs=tinystories_docs,
            openwebtext_docs=openwebtext_docs,
            rounds=args.rounds,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
