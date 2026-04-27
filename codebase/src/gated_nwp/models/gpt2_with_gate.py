"""GPT-2 wrapper that plugs the metadata-aware gate at the SDPA output
of every layer.

We subclass HuggingFace's ``GPT2Attention`` to intercept
``_attn`` → reshape → ``c_proj`` and inject
:py:class:`MetadataAwareGatedAttention` between the per-head SDPA output
and the final output projection (Qwen position ``G_1``).

Only ``site=g1`` is currently implemented. Other sites are left as
``NotImplementedError`` and tracked in ``experiments-ideas/v02-*``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from gated_nwp.config import ExperimentConfig, GateConfig
from gated_nwp.data.metadata import MetadataEncoder
from gated_nwp.models.gated_attention import MetadataAwareGatedAttention


class GatedGPT2Attention(GPT2Attention):
    """Drop-in replacement for GPT2Attention that applies a
    metadata-aware gate at the SDPA output (``G_1``)."""

    def __init__(
        self,
        config: GPT2Config,
        gate_module: MetadataAwareGatedAttention,
        *,
        is_cross_attention: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.gate = gate_module

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_output, attn_weights = super()._attn(
            query, key, value, attention_mask=attention_mask, head_mask=head_mask
        )
        return attn_output, attn_weights

    # We override forward to have access to hidden_states inside the block,
    # so we can feed them into the gate. We call the parent's internal
    # primitives directly.
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        layer_past: tuple[torch.Tensor, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Project to Q, K, V (reuse HF internals).
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        # attn_output shape: (B, num_heads, T, head_dim)

        # Move heads to dim=-2 so we have (B, T, num_heads, head_dim),
        # which is the shape MetadataAwareGatedAttention expects.
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()

        # === Gate at G_1 ===
        attn_output = self.gate.apply_gate(attn_output, hidden_states)

        # Merge heads: (B, T, num_heads * head_dim)
        attn_output = attn_output.view(
            attn_output.size(0),
            attn_output.size(1),
            self.num_heads * self.head_dim,
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs: tuple = (attn_output, present)
        if output_attentions:
            outputs = (*outputs, attn_weights)
        return outputs


class MetadataAwareGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 LM head + learner-metadata embeddings + gate in every layer."""

    def __init__(
        self,
        config: GPT2Config,
        gate_config: GateConfig,
        encoder: MetadataEncoder,
    ) -> None:
        super().__init__(config)
        self.gate_config = gate_config
        self.encoder = encoder

        self.cefr_emb = nn.Embedding(encoder.num_cefr, gate_config.d_cefr)
        self.l1_emb = nn.Embedding(encoder.num_l1, gate_config.d_l1)

        if gate_config.site != "g1":
            raise NotImplementedError(
                f"Only gate.site='g1' is implemented; got {gate_config.site}. "
                "Other sites live in experiments-ideas/v02-*.md."
            )

        # Replace each block's attention with the gated version.
        for i, block in enumerate(self.transformer.h):
            gate_module = MetadataAwareGatedAttention.from_config(
                d_model=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                gate_config=gate_config,
            )
            new_attn = GatedGPT2Attention(
                config, gate_module, is_cross_attention=False, layer_idx=i
            )
            # Copy over pre-trained Q/K/V/O projections.
            new_attn.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = new_attn

    def _set_metadata(self, cefr_ids: torch.Tensor, l1_ids: torch.Tensor) -> None:
        e_cefr = self.cefr_emb(cefr_ids)
        e_l1 = self.l1_emb(l1_ids)
        for block in self.transformer.h:
            block.attn.gate.set_metadata_ctx(e_cefr, e_l1)

    def _clear_metadata(self) -> None:
        for block in self.transformer.h:
            block.attn.gate.clear_metadata_ctx()

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cefr_id: torch.Tensor | None = None,
        l1_id: torch.Tensor | None = None,
        **kwargs,
    ):
        if cefr_id is not None and l1_id is not None:
            self._set_metadata(cefr_id, l1_id)
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            self._clear_metadata()


def build_model_for_variant(
    cfg: ExperimentConfig,
    encoder: MetadataEncoder,
) -> nn.Module:
    """Factory: dispatch on ``cfg.model_variant`` and build the matching model.

    Variants:
      * ``b1_learner`` — vanilla ``GPT2LMHeadModel``, metadata ignored.
      * ``b2_metadata_concat`` — same, metadata prepended in the data pipeline.
      * ``g1_metadata_gated`` — our flagship with the gated attention.
    """
    if cfg.model_variant in {"b1_learner", "b2_metadata_concat"}:
        return GPT2LMHeadModel.from_pretrained(cfg.base_model)

    if cfg.model_variant == "g1_metadata_gated":
        hf_config = GPT2Config.from_pretrained(cfg.base_model)
        model = MetadataAwareGPT2LMHeadModel(hf_config, cfg.gate, encoder)
        # Load pre-trained weights into the transformer backbone (attention
        # Q/K/V/O already re-loaded inside GatedGPT2Attention.__init__).
        pretrained = GPT2LMHeadModel.from_pretrained(cfg.base_model)
        missing, unexpected = model.load_state_dict(pretrained.state_dict(), strict=False)
        # New params (gate.gate_proj.*, cefr_emb, l1_emb) show up as missing.
        for m in missing:
            if not (m.startswith(("cefr_emb", "l1_emb")) or ".gate." in m):
                raise RuntimeError(f"Unexpected missing param while loading: {m}")
        if unexpected:
            raise RuntimeError(f"Unexpected params while loading: {unexpected}")
        return model

    raise ValueError(f"Unknown model_variant: {cfg.model_variant}")
