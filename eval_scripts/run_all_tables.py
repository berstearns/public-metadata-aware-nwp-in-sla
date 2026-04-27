"""Convenience wrapper: produce every standalone metadata-aware-nwp eval table.

Usage:
    python -m eval_scripts.run_all_tables \\
        --input runs/<id>/predictions.jsonl \\
        --out_dir tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts import (
    eval_cloze_table,
    eval_stratified_ppl_table,
    eval_transfer_table,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out_dir", type=Path, required=True, help="directory to write CSVs into")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_stratified_ppl_table.main(["--input", str(args.input), "--out", str(args.out_dir / "stratified_ppl.csv")])
    eval_cloze_table.main(["--input", str(args.input), "--out", str(args.out_dir / "cloze.csv")])
    eval_transfer_table.main(["--input", str(args.input), "--out", str(args.out_dir / "transfer.csv")])


if __name__ == "__main__":
    main()
