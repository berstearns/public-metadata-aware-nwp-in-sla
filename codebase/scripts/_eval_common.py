"""Checkpoint-loading helpers shared by 20_/21_/22_ eval scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gated_nwp.data.metadata import MetadataEncoder


def load_checkpoint_for_eval(ckpt_dir: Path) -> tuple[Any, Any, MetadataEncoder, dict]:
    """Load a checkpoint (trained by this codebase) for evaluation.

    Returns (model, tokenizer, metadata_encoder, saved_run_config).
    """
    from transformers import AutoTokenizer, GPT2LMHeadModel

    ckpt_dir = Path(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.pad_token = tokenizer.eos_token

    encoder_path = ckpt_dir / "encoder.json"
    cfg_path = ckpt_dir / "config.json"
    if not (encoder_path.exists() and cfg_path.exists()):
        raise FileNotFoundError(
            f"{ckpt_dir} missing encoder.json / config.json; was it trained by this codebase?"
        )
    meta = json.loads(encoder_path.read_text())
    run_config = json.loads(cfg_path.read_text())
    encoder = MetadataEncoder(
        cefr_classes=tuple(meta["cefr_classes"]),
        l1_classes=tuple(meta["l1_classes"]),
    )

    if run_config.get("model_variant") == "g1_metadata_gated":
        from transformers import GPT2Config

        from gated_nwp.config import GateConfig
        from gated_nwp.models.gpt2_with_gate import MetadataAwareGPT2LMHeadModel

        hf_config = GPT2Config.from_pretrained(ckpt_dir)
        gate_cfg = GateConfig(
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in run_config["gate"].items()}
        )
        model = MetadataAwareGPT2LMHeadModel(hf_config, gate_cfg, encoder)
        state_path = ckpt_dir / "pytorch_model.bin"
        if not state_path.exists():
            state_path = ckpt_dir / "model.safetensors"
        if state_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(state_path))
        else:
            state = torch.load(str(state_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        model = GPT2LMHeadModel.from_pretrained(ckpt_dir)

    return model, tokenizer, encoder, run_config
