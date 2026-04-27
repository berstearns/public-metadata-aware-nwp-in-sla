from pathlib import Path

import pandas as pd
import pytest

from gated_nwp.data.efcamdat import _canonicalise, load_efcamdat_csv


def test_canonicalise_renames_known_aliases() -> None:
    df = pd.DataFrame({"Text": ["hi"], "CEFR": ["A1"], "nationality": ["Spanish"]})
    out = _canonicalise(df)
    assert "text" in out.columns
    assert "cefr" in out.columns
    assert "l1" in out.columns
    assert out.loc[0, "text"] == "hi"
    assert out.loc[0, "cefr"] == "A1"
    assert out.loc[0, "l1"] == "Spanish"


def test_canonicalise_fills_missing_optional_columns() -> None:
    df = pd.DataFrame({"text": ["hi"]})
    out = _canonicalise(df)
    assert "cefr" in out.columns
    assert "l1" in out.columns
    assert out["cefr"].isna().all()


def test_canonicalise_rejects_missing_text_column() -> None:
    df = pd.DataFrame({"CEFR": ["A1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _canonicalise(df)


def test_load_efcamdat_csv_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "text": ["hello world", "foo bar baz"],
            "CEFR": ["A1", "B2"],
            "L1": ["Spanish", "German"],
        }
    ).to_csv(csv_path, index=False)
    out = load_efcamdat_csv(csv_path)
    assert len(out) == 2
    assert set(["text", "cefr", "l1"]).issubset(out.columns)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_efcamdat_csv(tmp_path / "nope.csv")
