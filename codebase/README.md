# `gated_nwp` — Metadata-Aware Gated Attention for L2 Next-Word Prediction

Reference implementation for the paper *Metadata-Aware Gated Attention
for Next-Word Prediction on Second-Language Learner Text*.

The flagship model (`G1`) extends Qwen-style SDPA-output gating
(Qiu et al. 2025) by conditioning the gate score on learner metadata
(CEFR proficiency + L1), applied layer-wise inside GPT-2 Base.

See `../experiments-ideas/` for the full variant catalogue. This codebase
supports every variant through config changes alone.

---

## Setup

```bash
# Clone and enter
cd codebase

# Python 3.10+ and a CUDA-capable GPU are recommended for training
python -m venv .venv && source .venv/bin/activate

# Install package + dev tools
pip install -e ".[dev]"

# (Optional) pre-commit hooks
pre-commit install
```

Expected data layout (adjust in `configs/paths.yaml` if different):

```
./data/splits/
├── norm-EFCAMDAT-train.csv
├── norm-EFCAMDAT-remainder.csv
├── norm-EFCAMDAT-test.csv
├── norm-andrew100k-remainder.csv
├── norm-CELVA-SP.csv
└── norm-KUPA-KEYS.csv
```

All six CSVs expose `l1` (learner's first language) and a CEFR column
(`cefr_level` for EFCAMDAT / CELVA-SP / KUPA-KEYS, `cefr_label` for
andrew100k) at row level. The loader normalises both to `l1` and `cefr`.

## Pipeline

Every step is a standalone script under `scripts/`, numbered by pipeline
stage. Each script loads its config from `configs/` (override with
`--config`). Re-running a script with the same config + seed reproduces
its output.

```bash
# 0x — one-time setup
python scripts/00_download_models.py                       # pulls GPT-2 Base
python scripts/01_prepare_efcamdat.py                      # sanity-check EFCAMDAT splits
python scripts/02_prepare_external.py                      # sanity-check transfer corpora
python scripts/03_prepare_b0_checkpoint.py                 # stock GPT-2 → runs/b0_native/last/ (no training)

# 1x — training (one script per trained variant; requires GPU)
python scripts/10_train_b1_learner.py --config configs/b1_learner.yaml
python scripts/11_train_b2_metadata_concat.py --config configs/b2_metadata_concat.yaml
python scripts/12_train_g1_gated.py --config configs/g1_gated.yaml

# 2x — evaluation (PPL + cloze)
python scripts/20_eval_ppl.py --checkpoint runs/g1_gated/last --split in_domain
python scripts/20_eval_ppl.py --checkpoint runs/g1_gated/last --split transfer
python scripts/21_eval_cloze.py --checkpoint runs/g1_gated/last

# 3x — aggregation and reporting
python scripts/30_aggregate_results.py
```

### Running the no-training baseline (B0)

B0 is stock GPT-2 evaluated on L2 corpora — no gradient updates. From a
clean checkout:

```bash
pip install -e ".[dev]"
python scripts/00_download_models.py
python scripts/01_prepare_efcamdat.py
python scripts/02_prepare_external.py
make eval-b0                     # runs prep → PPL → cloze → aggregate
```

Results land in:

* `runs/b0_native/last/ppl_in_domain.json`   — PPL on EFCAMDAT test, stratified by (CEFR, L1)
* `runs/b0_native/last/ppl_transfer.json`    — PPL on andrew100k / CELVA-SP / KUPA-KEYS
* `runs/b0_native/last/cloze.json`           — cloze top-1 / top-5
* `runs/aggregate.csv`                       — headline numbers across every run

### Running the full pipeline

```bash
make all
```

## Code layout

```
codebase/
├── src/gated_nwp/         # importable library
│   ├── config.py          # dataclass-backed configs
│   ├── data/              # dataset loaders + metadata encoding
│   ├── models/            # metadata-aware gated attention, GPT-2 wrapper
│   ├── training/          # HF Trainer wrapper with deterministic defaults
│   ├── evaluation/        # stratified PPL + cloze
│   └── utils/             # seeding, I/O, signature-aware forward
├── scripts/               # standalone entry points, one per pipeline stage
├── configs/               # YAML configs, one per baseline / variant
├── tests/                 # unit + smoke tests
└── Makefile               # reproducible pipeline orchestration
```

## Evaluation metrics

Two quantitative metrics are in the pilot:

1. **Stratified perplexity** — token-level PPL on held-out EFCAMDAT +
   zero-shot transfer to andrew100k / CELVA-SP / KUPA-KEYS, broken out
   by (CEFR, L1) cells.
2. **Cloze accuracy** — top-1 / top-5 accuracy on content-word cloze
   constructed from the same held-out sets.

Error-distribution alignment (CoEdIT + ERRANT + Jensen-Shannon divergence
vs authentic learner error histograms) is deferred to future work. See
the paper's Limitations / Future Work section.

## Reproducibility

- Every script accepts `--seed` (default set in config; we use 42
  throughout the paper).
- Configs are version-controlled YAML; no hidden environment state.
- Training logs include exact git SHA, Python version, torch version,
  CUDA version, and the full merged config.
- `utils.seeding.set_global_seed` seeds Python, NumPy, PyTorch (CPU +
  CUDA), and sets `torch.use_deterministic_algorithms(True)` guarded by
  `CUBLAS_WORKSPACE_CONFIG`.
- Saved checkpoints include the training config + RNG state for exact
  resume.

## Testing

```bash
pytest                     # unit tests
pytest -m smoke            # smoke tests (require tokeniser + tiny GPT-2)
pytest --cov=gated_nwp     # coverage
```

CI runs lint (ruff) + type check (mypy, non-strict) + unit tests on
every push — see `.github/workflows/ci.yaml`.

## Citation

See the paper at `../main.tex`.

## License

MIT — see `LICENSE`.
