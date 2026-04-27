import random

import numpy as np
import torch

from gated_nwp.utils.io import load_jsonl, save_jsonl
from gated_nwp.utils.seeding import set_global_seed


def test_set_global_seed_is_reproducible() -> None:
    set_global_seed(1234, deterministic=False)
    a = random.random(), np.random.rand(), torch.randn(1).item()
    set_global_seed(1234, deterministic=False)
    b = random.random(), np.random.rand(), torch.randn(1).item()
    assert a == b


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    records = [{"a": 1}, {"a": 2, "b": "hi"}, {"c": [1, 2, 3]}]
    save_jsonl(records, path)
    loaded = list(load_jsonl(path))
    assert loaded == records
