"""Reference emitter: turn a `gated_nwp` inference result into a JSONL record.

Use this in any prediction script that wants to feed `eval_scripts/`:

    from eval_scripts.emit import build_record, write_records

    records = (
        build_record(
            model="G1",                       # B0 / B1 / B2 / G1
            item_id=item.id,
            dataset="EFCAMDAT",
            cefr=item.cefr,
            l1=item.l1,
            ppl=hypothesis.ppl,
            predicted_filler=hypothesis.token,
            predicted_logprob=hypothesis.logprob,
            native_gold_filler=item.gold,
        )
        for item, hypothesis in run(...)
    )
    write_records(Path("predictions.jsonl"), records)

Both helpers raise `ValueError` on schema-invalid output — fail at the
producer rather than ship bad JSONL downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from eval_scripts.schema import validate_record


def build_record(
    *,
    model: str,
    item_id: int,
    dataset: str | None = None,
    cefr: str | None = None,
    l1: str | None = None,
    ppl: float | None = None,
    predicted_filler: str | None = None,
    predicted_logprob: float | None = None,
    native_gold_filler: str | None = None,
    **extra: object,
) -> dict:
    rec: dict[str, object] = {"model": model, "item_id": item_id}
    if dataset is not None:
        rec["dataset"] = dataset
    if cefr is not None:
        rec["cefr"] = cefr.upper() if isinstance(cefr, str) else cefr
    if l1 is not None:
        rec["l1"] = l1
    if ppl is not None:
        rec["ppl"] = float(ppl)
    if predicted_filler is not None:
        rec["predicted_filler"] = predicted_filler
    if predicted_logprob is not None:
        rec["predicted_logprob"] = float(predicted_logprob)
    if native_gold_filler is not None:
        rec["native_gold_filler"] = native_gold_filler
    rec.update(extra)

    issues = validate_record(rec)
    if issues:
        raise ValueError(f"build_record produced an invalid record: {issues}")
    return rec


def write_records(out: Path, records: Iterable[dict]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for rec in records:
            issues = validate_record(rec)
            if issues:
                raise ValueError(f"refusing to write invalid record at line {n + 1}: {issues}")
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
