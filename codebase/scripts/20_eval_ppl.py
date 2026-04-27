#!/usr/bin/env python3
"""Evaluate held-out perplexity, stratified by CEFR and L1.

Works on three slices:

* ``--split in_domain``  — EFCAMDAT held-out test set.
* ``--split transfer``   — andrew100k + CELVA-SP + KUPA-KEYS.
* ``--split all``        — both of the above.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from gated_nwp.config import resolve_paths
from gated_nwp.data.efcamdat import EfcamdatDataset, load_efcamdat_csv
from gated_nwp.data.external import TransferDataset, load_transfer_csv
from gated_nwp.evaluation.perplexity import compute_stratified_ppl


def main() -> None:
    from _eval_common import load_checkpoint_for_eval

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", choices=("in_domain", "transfer", "all"), default="in_domain")
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--eval_config", default="configs/eval_transfer.yaml")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    paths = resolve_paths(args.paths_config)
    # eval_config is loaded to surface schema errors early, then ignored.
    yaml.safe_load(Path(args.eval_config).read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, encoder, run_config = load_checkpoint_for_eval(args.checkpoint)
    max_seq_len = int(run_config.get("max_seq_len", 1024))

    results: dict[str, dict] = {}

    def eval_on(name: str, dataset) -> None:
        print(f"[ppl] evaluating {name} (n={len(dataset)})")
        report = compute_stratified_ppl(
            model, dataset, encoder=encoder, batch_size=args.batch_size, device=device
        )
        results[name] = asdict(report)

    if args.split in ("in_domain", "all"):
        df = load_efcamdat_csv(paths.efcamdat_test)
        ds = EfcamdatDataset.from_dataframe(df, tokenizer, max_seq_len=max_seq_len, encoder=encoder)
        eval_on("efcamdat_test", ds)

    if args.split in ("transfer", "all"):
        # All three transfer corpora carry both `l1` and `cefr_level`
        # (or `cefr_label` for andrew100k) at row level, so we don't
        # override anything at load time.
        spec = {
            "andrew100k": paths.andrew100k_remainder,
            "celva_sp": paths.celva_sp,
            "kupa_keys": paths.kupa_keys,
        }
        for name, path in spec.items():
            df = load_transfer_csv(path)
            ds = TransferDataset.from_dataframe(
                df, tokenizer, max_seq_len=max_seq_len, encoder=encoder
            )
            eval_on(name, ds)

    out_path = args.output or (args.checkpoint / f"ppl_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
