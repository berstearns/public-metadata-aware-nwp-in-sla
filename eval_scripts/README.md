# eval_scripts/

Standalone evaluation scripts for the metadata-aware NWP paper. Each
script consumes one `predictions.jsonl` and emits one CSV table
corresponding to one paper-ready figure.

These scripts depend only on the Python standard library, so they can
be run anywhere the JSONL is available, without installing the
`gated_nwp` package or any ML dependencies.

## Predictions JSONL schema

One record per (model, item):

```jsonl
{"model": "<name>", "item_id": <int>, "dataset": "<corpus>",
 "cefr": "<A1|A2|B1|B2|C1|C2>", "l1": "<L1>",
 "ppl": <float|null>,
 "predicted_filler": "<str|null>", "predicted_logprob": <float|null>,
 "native_gold_filler": "<str|null>"}
```

Required: `model`, `item_id`. Other fields are optional; eval scripts
skip records gracefully when a needed field is missing.

## Tables

| Script                            | Output CSV          | Purpose |
|-----------------------------------|---------------------|---------|
| `eval_stratified_ppl_table.py`    | `stratified_ppl.csv` | per-(model, dataset, CEFR, L1) PPL summary — the core flagship table comparing B0 / B1 / B2 / G1 |
| `eval_cloze_table.py`             | `cloze.csv`         | cloze top-1 accuracy by (model, dataset, CEFR, L1) |
| `eval_transfer_table.py`          | `transfer.csv`      | zero-shot transfer PPL by (model, dataset) — andrew100k / CELVA-SP / KUPA-KEYS |

## Running one table

```bash
python -m eval_scripts.eval_stratified_ppl_table \
    --input runs/<id>/predictions.jsonl \
    --out tables/stratified_ppl.csv
```

## Running every table at once

```bash
python -m eval_scripts.run_all_tables \
    --input runs/<id>/predictions.jsonl \
    --out_dir tables/
```

## Adding a new table

1. Create `eval_scripts/eval_<name>_table.py`.
2. Implement `build_rows(records)` and a `main(argv=None)` entry point.
3. Use `from eval_scripts._io import load_records, group_by, write_csv`.
4. Add a smoke test under `tests/test_eval_scripts/`.
5. Add the script to the orchestrator in `run_all_tables.py`.
6. Update the table in this README.
