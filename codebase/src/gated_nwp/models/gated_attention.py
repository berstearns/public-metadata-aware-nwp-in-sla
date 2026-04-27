"""Metadata-aware gated attention module.

Implements the Qwen-style SDPA-output gate (Qiu et al., 2025, arXiv:2505.06708)
extended with learner-metadata conditioning:

    Y' = Y ⊙ σ([X ; e_CEFR ; e_L1] W_θ)

where

- ``Y`` is the multi-head SDPA output of shape ``(B, T, q, d_k)`` (pre-concat
  across heads),
- ``X`` is the pre-normalised hidden state of shape ``(B, T, d_model)``,
- ``e_CEFR ∈ R^{d_c}`` and ``e_L1 ∈ R^{d_l}`` are learner-metadata
  embeddings, broadcast across the sequence dimension,
- ``W_θ ∈ R^{(d_model + d_c + d_l) × (q · d_k)}`` parameterises the gate,
- ``σ`` is sigmoid (configurable).

The module is constructed to compose with HF GPT-2's ``GPT2Attention``:
call ``apply_gate(merged_attn_output, hidden_state)`` on the
post-merge-heads tensor just before ``c_proj``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gated_nwp.config import GateConfig


class MetadataAwareGatedAttention(nn.Module):
    """Stateless module that produces a gate tensor from
    ``[hidden_state ; e_CEFR ; e_L1]`` and applies it to the SDPA output.

    The metadata embeddings are stored on the parent model and injected
    into this module per-batch via :py:meth:`set_metadata_ctx`. The module
    reads them back through ``self._metadata_ctx``, which the caller must
    set before every forward pass (see ``MetadataAwareGPT2LMHeadModel``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_cefr: int,
        d_l1: int,
        *,
        granularity: str = "elementwise",
        activation: str = "sigmoid",
        form: str = "multiplicative",
        init: str = "passthrough",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_cefr = d_cefr
        self.d_l1 = d_l1
        self.granularity = granularity
        self.activation = activation
        self.form = form

        gate_input_dim = d_model + d_cefr + d_l1
        if granularity == "elementwise":
            gate_output_dim = num_heads * head_dim
        elif granularity == "headwise":
            gate_output_dim = num_heads
        else:
            raise ValueError(f"Unknown granularity {granularity}")

        self.gate_proj = nn.Linear(gate_input_dim, gate_output_dim, bias=True)
        self._init_weights(init)

        # Filled in by the parent model just before forward().
        self._metadata_ctx: tuple[torch.Tensor, torch.Tensor] | None = None

    def _init_weights(self, init: str) -> None:
        if init == "passthrough":
            nn.init.zeros_(self.gate_proj.weight)
            if self.activation == "sigmoid" and self.form == "multiplicative":
                # sigmoid(10) ≈ 1.0 → pass-through at step 0
                nn.init.constant_(self.gate_proj.bias, 10.0)
            else:
                # additive / silu identity-ish: bias to 0 (additive of 0.5 ≈ soft id)
                nn.init.zeros_(self.gate_proj.bias)
        elif init == "zero":
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
        else:
            raise ValueError(f"Unknown init scheme {init}")

    def set_metadata_ctx(self, e_cefr: torch.Tensor, e_l1: torch.Tensor) -> None:
        """Stash per-batch metadata embeddings (shape ``(B, d_c)`` and
        ``(B, d_l)``) for use in the next forward call."""
        self._metadata_ctx = (e_cefr, e_l1)

    def clear_metadata_ctx(self) -> None:
        self._metadata_ctx = None

    def _gate_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute σ or SiLU of the gate projection on
        ``[hidden ; e_CEFR ; e_L1]``.

        Returns shape ``(B, T, num_heads, head_dim)`` for elementwise gates
        or ``(B, T, num_heads, 1)`` for headwise gates — ready to broadcast
        against the SDPA output.
        """
        if self._metadata_ctx is None:
            # No metadata set → fall back to zeros, keeping the gate
            # degenerate (route to 'unk' at inference for unknown metadata).
            bsz = hidden.size(0)
            e_cefr = hidden.new_zeros(bsz, self.d_cefr)
            e_l1 = hidden.new_zeros(bsz, self.d_l1)
        else:
            e_cefr, e_l1 = self._metadata_ctx

        _, seq_len, _ = hidden.shape
        # Broadcast metadata across sequence length: (B, d) → (B, T, d).
        e_cefr_b = e_cefr.unsqueeze(1).expand(-1, seq_len, -1)
        e_l1_b = e_l1.unsqueeze(1).expand(-1, seq_len, -1)
        gate_input = torch.cat([hidden, e_cefr_b, e_l1_b], dim=-1)

        raw = self.gate_proj(gate_input)
        if self.activation == "sigmoid":
            gate = torch.sigmoid(raw)
        elif self.activation == "silu":
            gate = F.silu(raw)
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        if self.granularity == "elementwise":
            gate = gate.view(*gate.shape[:-1], self.num_heads, self.head_dim)
        else:
            gate = gate.view(*gate.shape[:-1], self.num_heads, 1)
        return gate

    def apply_gate(
        self,
        y: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the gate to ``y``.

        Args:
            y: SDPA output shaped ``(B, T, num_heads, head_dim)`` — i.e.
               post-attention, pre-concat, pre-W_O.
            hidden: pre-normalised hidden state, ``(B, T, d_model)``.

        Returns:
            The gated output, same shape as ``y``.
        """
        gate = self._gate_scores(hidden)
        if self.form == "multiplicative":
            return y * gate
        elif self.form == "additive":
            return y + gate
        else:
            raise ValueError(f"Unknown gate form {self.form}")

    @classmethod
    def from_config(
        cls,
        *,
        d_model: int,
        num_heads: int,
        head_dim: int,
        gate_config: GateConfig,
    ) -> MetadataAwareGatedAttention:
        return cls(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            d_cefr=gate_config.d_cefr,
            d_l1=gate_config.d_l1,
            granularity=gate_config.granularity,
            activation=gate_config.activation,
            form=gate_config.form,
            init=gate_config.init,
        )
