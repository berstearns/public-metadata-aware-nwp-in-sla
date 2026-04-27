import pytest
import torch

from gated_nwp.config import GateConfig
from gated_nwp.models.gated_attention import MetadataAwareGatedAttention


@pytest.fixture
def gate() -> MetadataAwareGatedAttention:
    return MetadataAwareGatedAttention.from_config(
        d_model=64,
        num_heads=4,
        head_dim=16,
        gate_config=GateConfig(
            d_cefr=8,
            d_l1=8,
            granularity="elementwise",
            activation="sigmoid",
            form="multiplicative",
            init="passthrough",
        ),
    )


def test_gate_output_shape_matches_sdpa_output(gate: MetadataAwareGatedAttention) -> None:
    bsz, seq_len, n_heads, head_dim = 2, 5, 4, 16
    y = torch.randn(bsz, seq_len, n_heads, head_dim)
    hidden = torch.randn(bsz, seq_len, 64)
    e_cefr = torch.randn(bsz, 8)
    e_l1 = torch.randn(bsz, 8)
    gate.set_metadata_ctx(e_cefr, e_l1)
    out = gate.apply_gate(y, hidden)
    assert out.shape == y.shape


def test_passthrough_init_is_approx_identity(gate: MetadataAwareGatedAttention) -> None:
    """With passthrough init and sigmoid σ(10) ≈ 1, applying the gate
    should leave Y nearly unchanged at step 0."""
    y = torch.randn(2, 4, 4, 16)
    hidden = torch.randn(2, 4, 64)
    gate.set_metadata_ctx(torch.zeros(2, 8), torch.zeros(2, 8))
    out = gate.apply_gate(y, hidden)
    assert torch.allclose(out, y, atol=1e-3)


def test_missing_metadata_ctx_routes_to_zero_embeddings(gate: MetadataAwareGatedAttention) -> None:
    """With no metadata set, the gate falls back to zero-metadata
    embeddings — the gate should still apply deterministically."""
    y = torch.randn(2, 4, 4, 16)
    hidden = torch.randn(2, 4, 64)
    gate.clear_metadata_ctx()
    out1 = gate.apply_gate(y, hidden)
    out2 = gate.apply_gate(y, hidden)
    assert torch.allclose(out1, out2)


def test_headwise_granularity_shape() -> None:
    gate = MetadataAwareGatedAttention.from_config(
        d_model=64,
        num_heads=4,
        head_dim=16,
        gate_config=GateConfig(
            d_cefr=8,
            d_l1=8,
            granularity="headwise",
            activation="sigmoid",
            form="multiplicative",
            init="passthrough",
        ),
    )
    y = torch.randn(2, 5, 4, 16)
    hidden = torch.randn(2, 5, 64)
    gate.set_metadata_ctx(torch.zeros(2, 8), torch.zeros(2, 8))
    out = gate.apply_gate(y, hidden)
    assert out.shape == y.shape


def test_additive_form_shifts_output() -> None:
    """Additive form: Y' = Y + σ(...); should differ from Y in general."""
    gate = MetadataAwareGatedAttention.from_config(
        d_model=64,
        num_heads=4,
        head_dim=16,
        gate_config=GateConfig(
            d_cefr=8,
            d_l1=8,
            granularity="elementwise",
            activation="sigmoid",
            form="additive",
            init="zero",  # non-pass-through so output differs
        ),
    )
    # Manually set a nonzero weight so the gate isn't identically zero.
    torch.nn.init.normal_(gate.gate_proj.weight, std=0.1)
    y = torch.zeros(2, 4, 4, 16)
    hidden = torch.randn(2, 4, 64)
    gate.set_metadata_ctx(torch.zeros(2, 8), torch.zeros(2, 8))
    out = gate.apply_gate(y, hidden)
    assert not torch.allclose(out, y)
