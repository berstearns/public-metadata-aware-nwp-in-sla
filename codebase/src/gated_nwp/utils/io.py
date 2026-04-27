"""Small I/O helpers: JSONL read/write + run-manifest writer."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_run_manifest(
    run_dir: str | Path,
    *,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest capturing exact environment state for the run.

    Records: git sha, Python, torch, CUDA, platform, and the resolved
    config. Call once at the start of each training / eval run.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(config) if is_dataclass(config) else config,
    }
    if extra:
        manifest.update(extra)

    out_path = run_dir / "manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out_path
