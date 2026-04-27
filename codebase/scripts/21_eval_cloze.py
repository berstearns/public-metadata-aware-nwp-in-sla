#!/usr/bin/env python3
"""Evaluate cloze / fill-in-the-gap accuracy on held-out EFCAMDAT."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from gated_nwp.config import resolve_paths
from gated_nwp.data.cloze import build_cloze_examples
from gated_nwp.data.efcamdat import load_efcamdat_csv
from gated_nwp.evaluation.cloze import score_cloze


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--eval_config", default="configs/eval_transfer.yaml")
    p.add_argument("--max_examples", type=int, default=2000)
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    paths = resolve_paths(args.paths_config)
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())

    import spacy

    nlp = spacy.load(args.spacy_model, disable=["ner", "lemmatizer"])

    from _eval_common import load_checkpoint_for_eval

    model, tokenizer, encoder, _ = load_checkpoint_for_eval(args.checkpoint)

    df = load_efcamdat_csv(paths.efcamdat_test).head(args.max_examples)
    examples = build_cloze_examples(
        df["text"].astype(str).tolist(),
        mask_pos=tuple(eval_cfg["cloze"]["mask_pos"]),
        num_masks_per_sentence=eval_cfg["cloze"]["num_masks_per_sentence"],
        cefr_values=df["cefr"].where(df["cefr"].notna(), None).tolist(),
        l1_values=df["l1"].where(df["l1"].notna(), None).tolist(),
        nlp=nlp,
    )
    print(f"[cloze] {len(examples)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = score_cloze(model, tokenizer, examples, encoder=encoder, device=device)

    out_path = args.output or (args.checkpoint / "cloze.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
