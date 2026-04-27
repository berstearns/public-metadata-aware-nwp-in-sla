#!/usr/bin/env python3
"""Materialise the B0 (stock GPT-2) checkpoint so it drops into the
existing eval pipeline without any special cases.

Writes to ``runs/b0_native/last/``:

* HuggingFace GPT-2 weights + tokenizer (via ``save_pretrained``)
* ``encoder.json``     — metadata inventory (needed by eval scripts)
* ``config.json``      — our ExperimentConfig dump (``model_variant=b0_native``)
* ``manifest.json``    — git/python/torch environment snapshot

Running this script is the full B0 "training" step: it performs no
gradient updates — B0 is literally the OpenAI release evaluated on L2
corpora.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gated_nwp.config import ExperimentConfig, load_config, resolve_paths
from gated_nwp.data.metadata import MetadataEncoder
from gated_nwp.utils.io import write_run_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/b0_native.yaml")
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--run_dir", default=None, help="Override runs_root/run_name")
    args = p.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    paths = resolve_paths(args.paths_config)
    run_dir = Path(args.run_dir) if args.run_dir else paths.runs_root / cfg.run_name
    final_dir = run_dir / "last"
    final_dir.mkdir(parents=True, exist_ok=True)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[b0] downloading stock {cfg.base_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(cfg.base_model)

    print(f"[b0] saving to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Share the G1 metadata inventory so stratified PPL / cloze stratify
    # the B0 numbers along the same (CEFR, L1) axes used by the other
    # variants. Load inventory from the shipped g1_gated.yaml.
    g1_cfg = load_config("configs/g1_gated.yaml")
    encoder = MetadataEncoder.from_config(g1_cfg.gate.cefr_classes, g1_cfg.gate.l1_classes)
    (final_dir / "encoder.json").write_text(
        json.dumps(
            {
                "cefr_classes": list(encoder.cefr_classes),
                "l1_classes": list(encoder.l1_classes),
            },
            indent=2,
        )
    )
    (final_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    write_run_manifest(final_dir, config=cfg, extra={"note": "no training performed"})
    print(f"[done] B0 checkpoint ready at {final_dir}")


if __name__ == "__main__":
    main()
