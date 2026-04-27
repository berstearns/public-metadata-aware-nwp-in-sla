"""Dataclass-backed config loading.

Every script takes ``--config path/to/foo.yaml`` and resolves it here.
Nested keys are mapped to nested dataclasses. Unknown keys raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    efcamdat_train: Path
    efcamdat_remainder: Path
    efcamdat_test: Path
    andrew100k_remainder: Path
    celva_sp: Path
    kupa_keys: Path
    cache_root: Path
    runs_root: Path


@dataclass(frozen=True)
class GateConfig:
    site: str = "g1"
    granularity: str = "elementwise"
    head_sharing: str = "specific"
    activation: str = "sigmoid"
    form: str = "multiplicative"
    init: str = "passthrough"
    d_cefr: int = 16
    d_l1: int = 32
    cefr_classes: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2", "unk")
    l1_classes: tuple[str, ...] = ("unk",)

    def __post_init__(self) -> None:
        valid_sites = {"g1", "g2", "g3", "g4", "g5"}
        if self.site not in valid_sites:
            raise ValueError(f"gate.site must be one of {valid_sites}, got {self.site}")
        if self.granularity not in {"elementwise", "headwise"}:
            raise ValueError(f"gate.granularity invalid: {self.granularity}")
        if self.form not in {"multiplicative", "additive"}:
            raise ValueError(f"gate.form invalid: {self.form}")


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    model_variant: str
    base_model: str = "gpt2"

    max_seq_len: int = 1024
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 5.0e-5
    lr_schedule: str = "cosine"
    warmup_steps: int = 200
    num_epochs: int = 3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    seed: int = 42
    deterministic: bool = True

    log_every: int = 50
    save_every: int = 1000
    eval_every: int = 1000

    train_split: str = "efcamdat_train_plus_remainder"
    eval_split: str = "efcamdat_test"

    use_metadata: bool = False
    metadata_mode: str = "none"  # none | prefix_tokens | gate_input
    metadata_prefix_template: str = "<cefr={cefr}><l1={l1}>"
    gate: GateConfig = field(default_factory=GateConfig)


def _require_keys(actual: dict[str, Any], dataclass_type: type) -> None:
    known = {f.name for f in fields(dataclass_type)}
    extras = set(actual) - known
    if extras:
        raise ValueError(
            f"Unknown keys {sorted(extras)} for {dataclass_type.__name__}; "
            f"expected a subset of {sorted(known)}"
        )


def _construct(dataclass_type: type, raw: dict[str, Any]) -> Any:
    _require_keys(raw, dataclass_type)
    kwargs: dict[str, Any] = {}
    for f in fields(dataclass_type):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[f.name] = _construct(f.type, value or {})
        elif f.type is GateConfig or f.name == "gate":
            kwargs[f.name] = GateConfig(
                **{k: (tuple(v) if isinstance(v, list) else v) for k, v in (value or {}).items()}
            )
        elif f.type is tuple or (hasattr(f.type, "__origin__") and f.type.__origin__ is tuple):
            kwargs[f.name] = tuple(value) if value is not None else ()
        else:
            kwargs[f.name] = value
    return dataclass_type(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML into a validated ExperimentConfig."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(raw)}")
    return _construct(ExperimentConfig, raw)


def resolve_paths(paths_yaml: str | Path = "configs/paths.yaml") -> PathsConfig:
    """Load configs/paths.yaml into a PathsConfig, resolving relative entries
    against data_root."""
    path = Path(paths_yaml)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_root = Path(raw["data_root"]).expanduser().resolve()

    def resolve(key_path: str) -> Path:
        node: Any = raw
        for k in key_path.split("."):
            node = node[k]
        p = Path(str(node)).expanduser()
        return p if p.is_absolute() else (data_root / p)

    return PathsConfig(
        data_root=data_root,
        efcamdat_train=resolve("efcamdat.train"),
        efcamdat_remainder=resolve("efcamdat.remainder"),
        efcamdat_test=resolve("efcamdat.test"),
        andrew100k_remainder=resolve("transfer.andrew100k.remainder"),
        celva_sp=resolve("transfer.celva_sp"),
        kupa_keys=resolve("transfer.kupa_keys"),
        cache_root=Path(raw.get("cache_root", "./data-cache")).expanduser().resolve(),
        runs_root=Path(raw.get("runs_root", "./runs")).expanduser().resolve(),
    )
