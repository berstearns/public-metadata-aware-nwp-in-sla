#!/usr/bin/env python3
"""Prepare EFCAMDAT splits for training and evaluation.

Loads the CSVs, canonicalises columns, writes a per-split summary, and
dumps the canonical L1 inventory (for use by the G1 gate config).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gated_nwp.config import resolve_paths
from gated_nwp.data.efcamdat import load_efcamdat_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--output_dir", default=None, help="Defaults to cache_root/efcamdat")
    args = p.parse_args()

    paths = resolve_paths(args.paths_config)
    out_dir = Path(args.output_dir) if args.output_dir else paths.cache_root / "efcamdat"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": paths.efcamdat_train,
        "remainder": paths.efcamdat_remainder,
        "test": paths.efcamdat_test,
    }

    summary: dict[str, dict] = {}
    all_l1s: set[str] = set()
    for name, path in splits.items():
        print(f"[prepare] loading {name}: {path}")
        df = load_efcamdat_csv(path)
        summary[name] = {
            "n_rows": len(df),
            "cefr_counts": df["cefr"].value_counts(dropna=False).to_dict(),
            "l1_counts": df["l1"].value_counts(dropna=False).head(30).to_dict(),
        }
        all_l1s.update(str(v) for v in df["l1"].dropna().unique())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    l1_inventory = sorted(all_l1s)
    (out_dir / "l1_inventory.json").write_text(json.dumps(l1_inventory, indent=2))
    print(f"[prepare] summary -> {out_dir / 'summary.json'}")
    print(f"[prepare] L1 inventory ({len(l1_inventory)} values) -> {out_dir / 'l1_inventory.json'}")


if __name__ == "__main__":
    main()
