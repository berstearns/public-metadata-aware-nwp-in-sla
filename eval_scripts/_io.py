"""Shared I/O for standalone metadata-aware-nwp eval scripts.

Predictions JSONL schema (one record per (model, item)):

    {
      "model": "<name>",                 # required
      "item_id": <int>,                  # required
      "dataset": "<corpus_label>",       # optional
      "cefr": "<A1|A2|B1|B2|C1|C2>",     # optional
      "l1": "<L1_label>",                # optional
      "ppl": <float|null>,               # for stratified PPL / transfer
      "predicted_filler": "<str>",       # for cloze tables
      "predicted_logprob": <float|null>, # for cloze tables
      "native_gold_filler": "<str|null>" # for cloze tables
    }

Records contribute only to tables that need fields they actually carry —
a record with `ppl` but no `predicted_filler` shows up in PPL/transfer
tables but not cloze. This makes the schema additive: B0/B1/B2/G1
prediction scripts can each emit only the fields they have.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise SystemExit(f"jsonl not found at {path}")
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def load_records(path: Path) -> list[dict]:
    records = list(iter_jsonl(path))
    if not records:
        raise SystemExit(f"{path}: no records")
    bad = [i for i, r in enumerate(records) if not isinstance(r.get("model"), str) or not r["model"]]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records missing 'model' (e.g. line {bad[0]+1})")
    return records


def group_by(records: Iterable[Mapping[str, Any]], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in records:
        k = tuple(r.get(field) for field in keys)
        out.setdefault(k, []).append(dict(r))
    return out


def write_csv(out: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
