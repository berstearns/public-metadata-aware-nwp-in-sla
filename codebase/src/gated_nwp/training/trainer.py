"""Training loop shared across baselines and the flagship.

Thin wrapper around HF Trainer so that every variant goes through the
same accelerator, optimizer, scheduler, and logging path. The one
non-standard bit is the collator that surfaces ``cefr_id`` / ``l1_id``
to the model's forward call.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, Trainer, TrainingArguments

from gated_nwp.config import ExperimentConfig
from gated_nwp.data.metadata import MetadataEncoder
from gated_nwp.models.gpt2_with_gate import build_model_for_variant
from gated_nwp.utils.io import write_run_manifest
from gated_nwp.utils.seeding import set_global_seed


def _collate_with_metadata(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    cefr_id = torch.tensor([b["cefr_id"] for b in batch], dtype=torch.long)
    l1_id = torch.tensor([b["l1_id"] for b in batch], dtype=torch.long)
    # Standard causal-LM label shift handled by HF when we pass labels = input_ids.
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
        "cefr_id": cefr_id,
        "l1_id": l1_id,
    }


def _build_tokenizer(cfg: ExperimentConfig) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no native pad token
    return tokenizer


def train_one_variant(
    cfg: ExperimentConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    encoder: MetadataEncoder,
    run_dir: str | Path,
) -> str:
    """Train one variant end-to-end; return path to the final checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(run_dir, config=cfg)

    set_global_seed(cfg.seed, deterministic=cfg.deterministic)

    model = build_model_for_variant(cfg, encoder)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=False,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_schedule,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.gradient_clip,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        eval_strategy=("steps" if eval_dataset is not None else "no"),
        eval_steps=cfg.eval_every,
        save_total_limit=3,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_drop_last=True,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collate_with_metadata,
    )
    trainer.train()

    final_dir = run_dir / "last"
    trainer.save_model(str(final_dir))
    # Capture the tokenizer + encoder + config alongside weights so the
    # checkpoint is self-contained.
    _build_tokenizer(cfg).save_pretrained(str(final_dir))
    (final_dir / "encoder.json").write_text(
        _json_dump(
            {
                "cefr_classes": list(encoder.cefr_classes),
                "l1_classes": list(encoder.l1_classes),
            }
        )
    )
    (final_dir / "config.json").write_text(_json_dump(asdict(cfg)))
    return str(final_dir)


def _json_dump(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)
