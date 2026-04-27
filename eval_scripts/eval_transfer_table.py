"""Standalone eval: zero-shot transfer PPL by (model, dataset).

Columns: model | dataset | n | ppl_mean | ppl_std

Used for the cross-corpus table: same model, multiple datasets
(EFCAMDAT in-domain + andrew100k / CELVA-SP / KUPA-KEYS transfer).

Records lacking `ppl` are skipped.

Usage:
    python -m eval_scripts.eval_transfer_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/transfer.csv
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "n", "ppl_mean", "ppl_std"]


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    grouped = group_by(records, "model", "dataset")
    for (model, dataset), items in sorted(grouped.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        ppls = [r["ppl"] for r in items if isinstance(r.get("ppl"), (int, float))]
        if not ppls:
            continue
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "n": len(ppls),
            "ppl_mean": statistics.fmean(ppls),
            "ppl_std": std,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = build_rows(load_records(args.input))
    n = write_csv(args.out, _FIELDS, rows)
    print(f"transfer table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
