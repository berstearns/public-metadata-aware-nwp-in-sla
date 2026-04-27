# runs/

Per-run output artifacts: `predictions.jsonl` plus the CSV tables
emitted by `eval_scripts/`. One subdirectory per (model, dataset, regime)
tuple.

Convention:

    runs/<model_label>-<dataset_label>[-<note>]/

Each subdirectory contains:

    predictions.jsonl       # one record per (model, item)
    tables/
      stratified_ppl.csv
      cloze.csv
      transfer.csv

Real corpora (EFCAMDAT splits, CELVA-SP, KUPA-KEYS, andrew100k) are
not committed; full-corpus runs pull from the private rclone remote.

## Existing runs

| Subdirectory          | Model         | Data              | n  | What it shows |
|-----------------------|---------------|-------------------|----|---------------|
| `B0-celva-smoke/`     | `gpt2` (124M, no fine-tuning) | CELVA-SP 2-sample (French L1, A2 + B1) | 2  | end-to-end harness on real model + real (small) data; B0 PPL = 177 on A2 vs 47 on B1 — native gpt2 is dramatically worse on lower-CEFR learner text, the gradient the paper's metadata-aware gate aims to flatten |
