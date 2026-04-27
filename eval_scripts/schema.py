"""Canonical predictions.jsonl schema for metadata-aware-nwp eval scripts.

Required fields:
    model              (str, non-empty)
    item_id            (int)

Optional fields (validated when present):
    dataset            (str)
    cefr               (one of A1..C2 — case-insensitive)
    l1                 (str)
    ppl                (number ≥ 0)
    predicted_filler   (str)
    predicted_logprob  (number ≤ 0)
    native_gold_filler (str)

A record must carry at least one of `ppl` or `predicted_filler`,
otherwise it contributes to no eval table and is rejected as useless.
Unknown fields are allowed (forward-compat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED: tuple[tuple[str, type], ...] = (
    ("model", str),
    ("item_id", int),
)

OPTIONAL_TYPES: dict[str, type | tuple[type, ...]] = {
    "dataset": str,
    "cefr": str,
    "l1": str,
    "ppl": (int, float),
    "predicted_filler": str,
    "predicted_logprob": (int, float),
    "native_gold_filler": str,
}

ALLOWED_CEFR: frozenset[str] = frozenset({"A1", "A2", "B1", "B2", "C1", "C2"})


def validate_record(rec: Any) -> list[str]:
    """Return a list of human-readable issues. Empty list = valid record."""
    issues: list[str] = []
    if not isinstance(rec, dict):
        return [f"record is not a dict: {type(rec).__name__}"]

    for name, typ in REQUIRED:
        if name not in rec:
            issues.append(f"missing required field {name!r}")
            continue
        v = rec[name]
        if not isinstance(v, typ) or (typ is str and not v):
            issues.append(f"{name}: expected non-empty {typ.__name__}, got {v!r}")

    for name, typ in OPTIONAL_TYPES.items():
        if name not in rec or rec[name] is None:
            continue
        if not isinstance(rec[name], typ):
            issues.append(f"{name}: expected {typ}, got {type(rec[name]).__name__}")

    if isinstance(rec.get("cefr"), str) and rec["cefr"] and rec["cefr"].upper() not in ALLOWED_CEFR:
        issues.append(f"cefr: {rec['cefr']!r} not in {sorted(ALLOWED_CEFR)}")

    if isinstance(rec.get("ppl"), (int, float)) and rec["ppl"] < 0:
        issues.append(f"ppl: {rec['ppl']} < 0 (perplexity must be non-negative)")

    if isinstance(rec.get("predicted_logprob"), (int, float)) and rec["predicted_logprob"] > 0:
        issues.append(f"predicted_logprob: {rec['predicted_logprob']} > 0 (log-probability must be ≤ 0)")

    has_ppl = isinstance(rec.get("ppl"), (int, float))
    has_filler = isinstance(rec.get("predicted_filler"), str) and rec["predicted_filler"]
    if not (has_ppl or has_filler):
        issues.append("record carries neither 'ppl' nor 'predicted_filler' — contributes to no table")

    return issues


def validate_file(path: Path) -> dict[str, list[str]]:
    """Validate every line of a JSONL. Returns {f'line:{N}': [issues]} for non-conforming lines."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        raise SystemExit(f"jsonl not found at {path}")
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            key = f"line:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                out[key] = [f"invalid JSON: {exc}"]
                continue
            issues = validate_record(rec)
            if issues:
                out[key] = issues
    return out


def main(argv: list[str] | None = None) -> None:
    """`python -m eval_scripts.schema --input predictions.jsonl`"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl to validate")
    args = ap.parse_args(argv)

    bad = validate_file(args.input)
    if not bad:
        print(f"OK: {args.input} conforms to the predictions schema")
        return
    for key, issues in bad.items():
        print(f"{args.input}:{key}")
        for issue in issues:
            print(f"  - {issue}")
    raise SystemExit(f"FAIL: {len(bad)} non-conforming line(s) in {args.input}")


if __name__ == "__main__":
    main()
