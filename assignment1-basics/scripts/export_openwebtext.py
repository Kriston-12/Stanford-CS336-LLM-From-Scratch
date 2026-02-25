import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

EOT = "<|endoftext|>"

'''
uv run python scripts/export_openwebtext.py --num-docs 200000
'''
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data") / "openwebtext_200k.txt")
    ap.add_argument("--num-docs", type=int, default=200_000)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    ds = load_dataset(
        "openwebtext",
        split=args.split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    n = min(args.num_docs, len(ds))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=f"writing {n} docs"):
            text = ds[i]["text"]
            if not text:
                continue
            text = text.strip()
            if not text:
                continue
            f.write(text)
            f.write("\n")
            f.write(EOT)
            f.write("\n")

    print(f"Wrote: {args.out} ({n} docs)")


if __name__ == "__main__":
    main()