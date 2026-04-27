"""Tests for the JSONL schema validator + emit helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_scripts import schema
from eval_scripts.emit import build_record, write_records


# ---- validate_record -------------------------------------------------------

def test_minimal_with_ppl():
    assert schema.validate_record({"model": "B0", "item_id": 0, "ppl": 50.0}) == []


def test_minimal_with_filler():
    assert schema.validate_record({"model": "B0", "item_id": 0, "predicted_filler": "x"}) == []


def test_useless_record_rejected():
    issues = schema.validate_record({"model": "B0", "item_id": 0})
    assert any("neither 'ppl' nor 'predicted_filler'" in i for i in issues)


def test_negative_ppl_rejected():
    issues = schema.validate_record({"model": "B0", "item_id": 0, "ppl": -1.0})
    assert any("ppl" in i and "< 0" in i for i in issues)


def test_positive_logprob_rejected():
    issues = schema.validate_record({
        "model": "B0", "item_id": 0, "predicted_filler": "x", "predicted_logprob": 0.5,
    })
    assert any("predicted_logprob" in i and "> 0" in i for i in issues)


@pytest.mark.parametrize("level", ["A1", "A2", "B1", "B2", "C1", "C2"])
def test_cefr_levels(level):
    assert schema.validate_record({
        "model": "B0", "item_id": 0, "ppl": 1.0, "cefr": level,
    }) == []


def test_invalid_cefr_rejected():
    issues = schema.validate_record({"model": "B0", "item_id": 0, "ppl": 1.0, "cefr": "ZZ"})
    assert any("cefr" in i for i in issues)


def test_extra_fields_allowed():
    assert schema.validate_record({
        "model": "B0", "item_id": 0, "ppl": 1.0, "future": {"x": 1},
    }) == []


# ---- validate_file ---------------------------------------------------------

def test_validate_file_passes_clean(tmp_path: Path):
    p = tmp_path / "ok.jsonl"
    p.write_text(
        '{"model": "B0", "item_id": 0, "ppl": 50.0}\n'
        '{"model": "B1", "item_id": 0, "predicted_filler": "x"}\n'
    )
    assert schema.validate_file(p) == {}


def test_validate_file_flags_per_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"model": "B0", "item_id": 0, "ppl": 50.0}\n'
        '{"model": "B1", "item_id": 0}\n'  # useless
        'not json\n'
    )
    issues = schema.validate_file(p)
    assert "line:1" not in issues
    assert "line:2" in issues
    assert "line:3" in issues


# ---- build_record / write_records -----------------------------------------

def test_build_record_normalises_cefr():
    r = build_record(model="G1", item_id=0, ppl=10.0, cefr="b1")
    assert r["cefr"] == "B1"


def test_build_record_raises_on_negative_ppl():
    with pytest.raises(ValueError):
        build_record(model="m", item_id=0, ppl=-5.0)


def test_write_records_round_trip(tmp_path: Path):
    out = tmp_path / "p.jsonl"
    rs = [build_record(model="B0", item_id=i, ppl=10.0 + i, cefr="B1") for i in range(3)]
    n = write_records(out, rs)
    assert n == 3
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["item_id"] for r in lines] == [0, 1, 2]


def test_write_records_refuses_useless(tmp_path: Path):
    out = tmp_path / "p.jsonl"
    bad = {"model": "m", "item_id": 0}  # carries neither ppl nor predicted_filler
    with pytest.raises(ValueError):
        write_records(out, [bad])
