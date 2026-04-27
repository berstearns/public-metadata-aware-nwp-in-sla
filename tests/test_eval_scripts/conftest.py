"""Synthetic fixture for metadata-aware-nwp eval_scripts tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def predictions_jsonl_path(tmp_path: Path) -> Path:
    """Three-model, two-dataset synthetic predictions covering 3 CEFR cells."""
    records = [
        # B0 native GPT-2 on EFCAMDAT (in-domain)
        {"model": "B0", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 50.0, "predicted_filler": "the", "native_gold_filler": "school"},
        {"model": "B0", "item_id": 1, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 60.0, "predicted_filler": "go", "native_gold_filler": "go"},
        {"model": "B0", "item_id": 2, "dataset": "EFCAMDAT", "cefr": "B1", "l1": "French",
         "ppl": 45.0, "predicted_filler": "make", "native_gold_filler": "do"},
        # B0 on CELVA-SP (transfer)
        {"model": "B0", "item_id": 3, "dataset": "CELVA-SP", "cefr": "B1", "l1": "French",
         "ppl": 70.0, "predicted_filler": "do", "native_gold_filler": "do"},
        # B1 learner LM
        {"model": "B1", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 30.0, "predicted_filler": "school", "native_gold_filler": "school"},
        {"model": "B1", "item_id": 1, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 32.0, "predicted_filler": "go", "native_gold_filler": "go"},
        # G1 (ours)
        {"model": "G1", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 28.0, "predicted_filler": "school", "native_gold_filler": "school"},
        {"model": "G1", "item_id": 1, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish",
         "ppl": 29.5, "predicted_filler": "go", "native_gold_filler": "go"},
    ]
    p = tmp_path / "predictions.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p
