"""Standalone eval: cloze accuracy by (model, dataset, CEFR, L1).

Columns: model | dataset | cefr | l1 | n | correct | accuracy

Records lacking `native_gold_filler` or `predicted_filler` are skipped.

Usage:
    python -m eval_scripts.eval_cloze_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/cloze.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "cefr", "l1", "n", "correct", "accuracy"]
_CEFR_ORDER = {lvl: i for i, lvl in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}


def _eq(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip() == b.strip()


def _sort_key(key: tuple) -> tuple:
    model, dataset, cefr, l1 = key
    return (model or "", dataset or "", _CEFR_ORDER.get(cefr or "", 99), l1 or "")


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    grouped = group_by(records, "model", "dataset", "cefr", "l1")
    for (model, dataset, cefr, l1), items in sorted(grouped.items(), key=lambda kv: _sort_key(kv[0])):
        scored = [
            r for r in items
            if isinstance(r.get("predicted_filler"), str)
            and isinstance(r.get("native_gold_filler"), str)
        ]
        if not scored:
            continue
        n = len(scored)
        correct = sum(1 for r in scored if _eq(r["predicted_filler"], r["native_gold_filler"]))
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "cefr": cefr or "",
            "l1": l1 or "",
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = build_rows(load_records(args.input))
    n = write_csv(args.out, _FIELDS, rows)
    print(f"cloze table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
