"""Shared training entry-point used by 10_/11_/12_ train scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import GPT2Tokenizer

from gated_nwp.config import ExperimentConfig, load_config, resolve_paths
from gated_nwp.data.efcamdat import EfcamdatDataset, load_efcamdat_csv
from gated_nwp.data.metadata import MetadataEncoder
from gated_nwp.training.trainer import train_one_variant


def build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--run_dir", default=None, help="Override runs_root/run_name")
    p.add_argument("--limit", type=int, default=None, help="Debug: cap training rows")
    return p


def build_datasets_and_encoder(
    cfg: ExperimentConfig,
    paths,
    *,
    limit: int | None = None,
) -> tuple[EfcamdatDataset, EfcamdatDataset, MetadataEncoder, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.concat(
        [
            load_efcamdat_csv(paths.efcamdat_train),
            load_efcamdat_csv(paths.efcamdat_remainder),
        ],
        ignore_index=True,
    )
    test_df = load_efcamdat_csv(paths.efcamdat_test)
    if limit is not None:
        train_df = train_df.head(limit)
        test_df = test_df.head(max(limit // 10, 10))

    encoder = MetadataEncoder.from_config(cfg.gate.cefr_classes, cfg.gate.l1_classes)
    train_ds = EfcamdatDataset.from_dataframe(
        train_df,
        tokenizer,
        max_seq_len=cfg.max_seq_len,
        encoder=encoder,
        metadata_mode=cfg.metadata_mode,
        prefix_template=cfg.metadata_prefix_template,
    )
    eval_ds = EfcamdatDataset.from_dataframe(
        test_df,
        tokenizer,
        max_seq_len=cfg.max_seq_len,
        encoder=encoder,
        metadata_mode=cfg.metadata_mode,
        prefix_template=cfg.metadata_prefix_template,
    )
    return train_ds, eval_ds, encoder, tokenizer


def run_training(description: str) -> None:
    args = build_argparser(description).parse_args()
    cfg = load_config(args.config)
    paths = resolve_paths(args.paths_config)
    train_ds, eval_ds, encoder, _ = build_datasets_and_encoder(cfg, paths, limit=args.limit)
    run_dir = Path(args.run_dir) if args.run_dir else paths.runs_root / cfg.run_name
    final = train_one_variant(cfg, train_ds, eval_ds, encoder, run_dir)
    print(f"[done] final checkpoint: {final}")
