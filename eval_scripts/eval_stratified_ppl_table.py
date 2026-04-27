"""Standalone eval: stratified PPL by (model, dataset, CEFR, L1) cells.

Columns: model | dataset | cefr | l1 | n | ppl_mean | ppl_median | ppl_std

Records lacking a `ppl` value are skipped.

Usage:
    python -m eval_scripts.eval_stratified_ppl_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/stratified_ppl.csv
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "cefr", "l1", "n", "ppl_mean", "ppl_median", "ppl_std"]
_CEFR_ORDER = {lvl: i for i, lvl in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}


def _sort_key(key: tuple) -> tuple:
    model, dataset, cefr, l1 = key
    return (model or "", dataset or "", _CEFR_ORDER.get(cefr or "", 99), l1 or "")


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    grouped = group_by(records, "model", "dataset", "cefr", "l1")
    for (model, dataset, cefr, l1), items in sorted(grouped.items(), key=lambda kv: _sort_key(kv[0])):
        ppls = [r["ppl"] for r in items if isinstance(r.get("ppl"), (int, float))]
        if not ppls:
            continue
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "cefr": cefr or "",
            "l1": l1 or "",
            "n": len(ppls),
            "ppl_mean": statistics.fmean(ppls),
            "ppl_median": statistics.median(ppls),
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
    print(f"stratified PPL table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
