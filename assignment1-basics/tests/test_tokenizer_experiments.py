from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tokenizer_experiments import (
    load_bpe_from_artifacts,
    load_tokenizer,
    sample_documents,
)


def test_sample_documents_smoke() -> None:
    docs = sample_documents(Path("tests/fixtures/tinystories_sample_5M.txt"), n_docs=3, seed=0, max_chars=1000)
    assert len(docs) == 3
    assert all(isinstance(d, str) and d for d in docs)


@pytest.mark.parametrize(
    "vocab_path,merges_path",
    [
        (Path("artifacts/tinystories_bpe/vocab.b64.json"), Path("artifacts/tinystories_bpe/merges.b64.json")),
    ],
)
def test_load_bpe_from_artifacts_smoke(vocab_path: Path, merges_path: Path) -> None:
    vocab, merges = load_bpe_from_artifacts(vocab_path, merges_path)
    assert isinstance(vocab, dict) and len(vocab) > 256
    assert isinstance(merges, list) and len(merges) > 0


def test_encode_decode_roundtrip_smoke() -> None:
    tok = load_tokenizer(
        Path("artifacts/tinystories_bpe/vocab.b64.json"),
        Path("artifacts/tinystories_bpe/merges.b64.json"),
        special_tokens=["<|endoftext|>"],
    )
    text = "Hello world! <|endoftext|>"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    # decode uses errors="replace" so it should be safe; for normal ASCII we expect exact match.
    assert out == text
