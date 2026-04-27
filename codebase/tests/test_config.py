from pathlib import Path

import pytest

from gated_nwp.config import GateConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize(
    "cfg_name", ["b1_learner.yaml", "b2_metadata_concat.yaml", "g1_gated.yaml"]
)
def test_every_shipped_config_loads(cfg_name: str) -> None:
    cfg = load_config(CONFIGS_DIR / cfg_name)
    assert cfg.run_name
    assert cfg.base_model
    assert cfg.seed == 42


def test_g1_gated_has_populated_gate() -> None:
    cfg = load_config(CONFIGS_DIR / "g1_gated.yaml")
    assert cfg.gate.site == "g1"
    assert cfg.gate.granularity == "elementwise"
    assert cfg.gate.activation == "sigmoid"
    assert cfg.gate.form == "multiplicative"
    assert cfg.gate.init == "passthrough"
    assert "unk" in cfg.gate.cefr_classes
    assert "unk" in cfg.gate.l1_classes
    assert cfg.gate.d_cefr == 16
    assert cfg.gate.d_l1 == 32


def test_invalid_gate_site_rejected() -> None:
    with pytest.raises(ValueError, match=r"gate\.site must be one of"):
        GateConfig(site="g99")


def test_invalid_gate_granularity_rejected() -> None:
    with pytest.raises(ValueError, match=r"gate\.granularity invalid"):
        GateConfig(granularity="per_sample")
