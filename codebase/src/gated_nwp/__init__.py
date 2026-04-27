"""Metadata-aware gated attention for L2 next-word prediction."""

from gated_nwp.config import (
    ExperimentConfig,
    GateConfig,
    PathsConfig,
    load_config,
)

__all__ = [
    "ExperimentConfig",
    "GateConfig",
    "PathsConfig",
    "load_config",
]
__version__ = "0.1.0"
