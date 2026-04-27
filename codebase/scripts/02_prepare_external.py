#!/usr/bin/env python3
"""Prepare transfer corpora (andrew100k, CELVA-SP, KUPA-KEYS).

Sanity-checks that each file is readable, has a text column (after
canonicalisation), and records row counts + available metadata columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gated_nwp.config import resolve_paths
from gated_nwp.data.external import load_transfer_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    paths = resolve_paths(args.paths_config)
    out_dir = Path(args.output_dir) if args.output_dir else paths.cache_root / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    # All three transfer corpora carry row-level `l1` and CEFR labels.
    targets = [
        ("andrew100k", paths.andrew100k_remainder),
        ("celva_sp", paths.celva_sp),
        ("kupa_keys", paths.kupa_keys),
    ]

    summary: dict[str, dict] = {}
    for name, path in targets:
        print(f"[prepare] loading {name}: {path}")
        df = load_transfer_csv(path)
        summary[name] = {
            "path": str(path),
            "n_rows": len(df),
            "has_cefr_labels": bool(df["cefr"].notna().any()),
            "has_l1_labels": bool(df["l1"].notna().any()),
            "cefr_counts": df["cefr"].value_counts(dropna=False).head(20).to_dict(),
            "l1_counts": df["l1"].value_counts(dropna=False).head(20).to_dict(),
        }

    (out_dir / "transfer_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[prepare] -> {out_dir / 'transfer_summary.json'}")


if __name__ == "__main__":
    main()
