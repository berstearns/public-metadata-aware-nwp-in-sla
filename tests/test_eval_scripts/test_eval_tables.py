"""Smoke tests for metadata-aware-nwp eval-table scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eval_scripts import (
    eval_cloze_table,
    eval_stratified_ppl_table,
    eval_transfer_table,
    run_all_tables,
)


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_stratified_ppl(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "p.csv"
    eval_stratified_ppl_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_key = {(r["model"], r["dataset"], r["cefr"], r["l1"]): r for r in rows}
    # B0 EFCAMDAT A2 Spanish: 2 items (50, 60) → mean 55
    r = by_key[("B0", "EFCAMDAT", "A2", "Spanish")]
    assert int(r["n"]) == 2
    assert float(r["ppl_mean"]) == pytest.approx(55.0)
    # B0 CELVA-SP B1 French: 1 item
    r = by_key[("B0", "CELVA-SP", "B1", "French")]
    assert int(r["n"]) == 1
    assert float(r["ppl_mean"]) == pytest.approx(70.0)
    assert float(r["ppl_std"]) == 0.0


def test_cloze(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "c.csv"
    eval_cloze_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_key = {(r["model"], r["dataset"], r["cefr"], r["l1"]): r for r in rows}
    # B0 EFCAMDAT A2 Spanish: predicted "the","go" vs gold "school","go" → 1/2
    r = by_key[("B0", "EFCAMDAT", "A2", "Spanish")]
    assert int(r["correct"]) == 1
    assert float(r["accuracy"]) == pytest.approx(0.5)
    # B1 EFCAMDAT A2 Spanish: both correct
    r = by_key[("B1", "EFCAMDAT", "A2", "Spanish")]
    assert int(r["correct"]) == 2
    assert float(r["accuracy"]) == pytest.approx(1.0)


def test_transfer(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "t.csv"
    eval_transfer_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_md = {(r["model"], r["dataset"]): r for r in rows}
    # B0 across all 4 records: EFCAMDAT (3 items) and CELVA-SP (1 item)
    r = by_md[("B0", "EFCAMDAT")]
    assert int(r["n"]) == 3
    assert float(r["ppl_mean"]) == pytest.approx((50 + 60 + 45) / 3)
    r = by_md[("B0", "CELVA-SP")]
    assert int(r["n"]) == 1
    assert float(r["ppl_mean"]) == pytest.approx(70.0)


def test_run_all_tables_writes_three_csvs(predictions_jsonl_path: Path, tmp_path: Path):
    out_dir = tmp_path / "tables"
    run_all_tables.main(["--input", str(predictions_jsonl_path), "--out_dir", str(out_dir)])
    expected = {"stratified_ppl.csv", "cloze.csv", "transfer.csv"}
    assert {p.name for p in out_dir.iterdir()} == expected
    for name in expected:
        assert (out_dir / name).read_text().count("\n") > 1


def test_records_with_only_ppl_skip_cloze_table(tmp_path: Path):
    """Records lacking native_gold_filler / predicted_filler don't break cloze table."""
    p = tmp_path / "ppl_only.jsonl"
    p.write_text(
        '{"model": "B0", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "Spanish", "ppl": 50.0}\n'
    )
    out = tmp_path / "c.csv"
    eval_cloze_table.main(["--input", str(p), "--out", str(out)])
    # Header + zero rows
    text = out.read_text()
    assert text.count("\n") == 1


def test_load_records_rejects_missing_model(tmp_path: Path):
    from eval_scripts._io import load_records
    p = tmp_path / "bad.jsonl"
    p.write_text('{"item_id": 0, "ppl": 10.0}\n')
    with pytest.raises(SystemExit):
        load_records(p)
