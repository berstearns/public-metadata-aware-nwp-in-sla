"""Tests for predict_online_ppl: pure-Python parts via dependency injection."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from eval_scripts import predict_online_ppl


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_read_items_basic(tmp_path: Path):
    p = tmp_path / "x.csv"
    _write_csv(
        p,
        [{"text": "a", "cefr": "B1"}, {"text": "b c d", "cefr": "A2"}],
        ["text", "cefr"],
    )
    rows = predict_online_ppl.read_items(p)
    assert len(rows) == 2
    assert rows[0]["text"] == "a"


def test_read_items_missing_column(tmp_path: Path):
    p = tmp_path / "x.csv"
    _write_csv(p, [{"sentence": "a"}], ["sentence"])
    with pytest.raises(SystemExit):
        predict_online_ppl.read_items(p)


def test_predict_records_emits_ppl(monkeypatch):
    rows = [
        {"text": "a", "item_id": "0", "cefr": "B1", "l1": "French", "dataset": "EFCAMDAT"},
        {"text": "b", "item_id": "1", "cefr": "A2", "l1": "Spanish"},
        {"text": "", "item_id": "2"},  # empty → skipped
    ]
    losses = {"a": 0.5, "b": 1.0}
    score = lambda s: losses.get(s.strip())  # noqa: E731

    out = list(predict_online_ppl.predict_records(
        rows, score_loss=score, model_label="B0", default_dataset="default-ds",
    ))
    assert len(out) == 2
    assert out[0]["model"] == "B0"
    assert out[0]["item_id"] == 0
    assert out[0]["ppl"] == pytest.approx(math.exp(0.5))
    assert out[0]["cefr"] == "B1"
    assert out[0]["l1"] == "French"
    assert out[0]["dataset"] == "EFCAMDAT"  # row dataset wins
    assert out[1]["dataset"] == "default-ds"  # falls back to flag


def test_full_cli_with_monkeypatched_hf(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "x.csv"
    _write_csv(
        csv_path,
        [
            {"text": "alpha", "cefr": "B1"},
            {"text": "bravo", "cefr": "A2"},
        ],
        ["text", "cefr"],
    )
    out = tmp_path / "predictions.jsonl"

    losses = {"alpha": 0.3, "bravo": 0.7}
    monkeypatch.setattr(predict_online_ppl, "_load_hf", lambda m, d: (None, None))
    monkeypatch.setattr(
        predict_online_ppl, "_hf_loss_fn", lambda m, t, d: lambda s: losses.get(s.strip()),
    )

    predict_online_ppl.main([
        "--model", "test/dummy",
        "--data", str(csv_path),
        "--out", str(out),
        "--model_name_label", "B0-fake",
        "--dataset", "EFCAMDAT",
    ])

    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(records) == 2
    assert records[0]["model"] == "B0-fake"
    assert records[0]["dataset"] == "EFCAMDAT"
    assert records[0]["ppl"] == pytest.approx(math.exp(0.3))


def test_predict_then_eval_round_trip(tmp_path: Path, monkeypatch):
    """predict_online_ppl → predictions.jsonl → eval_stratified_ppl_table."""
    from eval_scripts import eval_stratified_ppl_table

    csv_path = tmp_path / "x.csv"
    _write_csv(
        csv_path,
        [
            {"text": "first", "cefr": "B1", "l1": "French"},
            {"text": "second", "cefr": "B1", "l1": "French"},
            {"text": "third", "cefr": "A2", "l1": "Spanish"},
        ],
        ["text", "cefr", "l1"],
    )
    jsonl = tmp_path / "predictions.jsonl"

    fake = {"first": 0.5, "second": 0.7, "third": 0.9}
    monkeypatch.setattr(predict_online_ppl, "_load_hf", lambda m, d: (None, None))
    monkeypatch.setattr(
        predict_online_ppl, "_hf_loss_fn", lambda m, t, d: lambda s: fake.get(s.strip()),
    )
    predict_online_ppl.main([
        "--model", "x", "--data", str(csv_path), "--out", str(jsonl),
        "--model_name_label", "B0", "--dataset", "EFCAMDAT",
    ])

    out_csv = tmp_path / "p.csv"
    eval_stratified_ppl_table.main(["--input", str(jsonl), "--out", str(out_csv)])
    rows = list(csv.DictReader(out_csv.open()))
    by_key = {(r["model"], r["cefr"], r["l1"]): r for r in rows}
    # Two B1/French rows → mean of exp(0.5), exp(0.7)
    r = by_key[("B0", "B1", "French")]
    assert int(r["n"]) == 2
    assert float(r["ppl_mean"]) == pytest.approx((math.exp(0.5) + math.exp(0.7)) / 2)
