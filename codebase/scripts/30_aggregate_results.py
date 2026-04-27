#!/usr/bin/env python3
"""Aggregate per-run evaluation artefacts into a single results table.

Walks every ``runs/<run_name>/last/`` directory and pulls out
``ppl_*.json`` + ``cloze.json``, producing ``runs/aggregate.json`` +
``runs/aggregate.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    rows: list[dict] = []
    for ckpt in args.runs_root.glob("*/last"):
        run_name = ckpt.parent.name
        row: dict = {"run": run_name}
        for fname in ("ppl_in_domain.json", "ppl_transfer.json", "cloze.json"):
            fpath = ckpt / fname
            if fpath.exists():
                row[fname.replace(".json", "")] = json.loads(fpath.read_text())
        rows.append(row)

    if not rows:
        print("[warn] no runs found")
        return

    out_json = args.output or (args.runs_root / "aggregate.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))

    # A flat CSV with the headline numbers for quick inspection.
    flat = []
    for r in rows:
        record = {"run": r["run"]}
        if "ppl_in_domain" in r and "efcamdat_test" in r["ppl_in_domain"]:
            record["ppl_indomain_overall"] = r["ppl_in_domain"]["efcamdat_test"]["overall"]
        if "ppl_transfer" in r:
            for corpus_name, report in r["ppl_transfer"].items():
                record[f"ppl_{corpus_name}"] = report["overall"]
        if "cloze" in r:
            record["cloze_top1"] = r["cloze"].get("top1")
            record["cloze_top5"] = r["cloze"].get("top5")
        flat.append(record)
    pd.DataFrame(flat).to_csv(out_json.with_suffix(".csv"), index=False)
    print(f"[done] -> {out_json} + CSV")


if __name__ == "__main__":
    main()
