"""Online perplexity prediction → predictions.jsonl.

Reads a CSV of sentences (with optional `cefr`, `l1`, `dataset`,
`item_id` columns), runs a HuggingFace causal LM, computes per-sentence
perplexity, and emits canonical JSONL records via
`eval_scripts.emit.write_records`.

Required CSV columns:
    text                                          (the sentence)

Optional CSV columns (passed through to JSONL when present):
    item_id                                       (defaults to row index)
    dataset, cefr, l1                             (record context)

Usage:
    python -m eval_scripts.predict_online_ppl \\
        --model gpt2 \\
        --data data/celva-2-sample.csv \\
        --column text \\
        --model_name_label B0 \\
        --dataset CELVA-SP \\
        --out predictions.jsonl

Requires `transformers` and `torch` at runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

from eval_scripts.emit import build_record, write_records


def read_items(path: Path, *, text_col: str = "text") -> list[dict]:
    if not path.exists():
        raise SystemExit(f"data file not found: {path}")
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise SystemExit(f"required column {text_col!r} missing from {path}; have {reader.fieldnames}")
        return [dict(row) for row in reader]


def _load_hf(model_id: str, device: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "predict_online_ppl requires `transformers` and `torch`. "
            "Install with: pip install transformers torch"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return model, tokenizer


def _hf_loss_fn(model, tokenizer, device: str) -> Callable[[str], float | None]:
    import torch

    def score(sentence: str) -> float | None:
        s = sentence.strip()
        if not s:
            return None
        enc = tokenizer(s, return_tensors="pt", truncation=True).to(device)
        if enc["input_ids"].shape[1] < 2:
            return None
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        return float(out.loss.item())

    return score


def predict_records(
    rows: list[dict],
    *,
    score_loss: Callable[[str], float | None],
    model_label: str,
    text_col: str = "text",
    default_dataset: str | None = None,
):
    """Yield canonical records, skipping items the model can't score."""
    for i, row in enumerate(rows):
        loss = score_loss(row.get(text_col, ""))
        if loss is None:
            continue
        ppl = math.exp(loss)
        try:
            item_id = int(row.get("item_id", i))
        except (TypeError, ValueError):
            item_id = i
        yield build_record(
            model=model_label,
            item_id=item_id,
            dataset=row.get("dataset") or default_dataset,
            cefr=(row.get("cefr") or None) or None,
            l1=(row.get("l1") or None) or None,
            ppl=ppl,
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", type=Path, required=True, help="CSV with one sentence per row")
    ap.add_argument("--column", default="text", help="text column name (default: 'text')")
    ap.add_argument("--model_name_label", default=None, help="defaults to --model")
    ap.add_argument("--dataset", default=None, help="dataset label (overridden by row's dataset column if present)")
    ap.add_argument("--out", type=Path, required=True, help="output predictions.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    rows = read_items(args.data, text_col=args.column)
    model, tokenizer = _load_hf(args.model, args.device)
    score = _hf_loss_fn(model, tokenizer, args.device)
    label = args.model_name_label or args.model

    n = write_records(
        args.out,
        predict_records(
            rows,
            score_loss=score,
            model_label=label,
            text_col=args.column,
            default_dataset=args.dataset,
        ),
    )
    print(f"predict_online_ppl: wrote {n} predictions for {label} to {args.out}")


if __name__ == "__main__":
    main()
