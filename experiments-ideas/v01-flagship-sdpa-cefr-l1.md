# v01 — Flagship: SDPA-output gate conditioned on CEFR + L1

**Status:** primary variant; the one written up in `main.tex`.

## Hypothesis

Injecting a sigmoid gate at the scaled dot-product attention output (`G_1`
in Qiu et al. 2025), with its gate score computed from
`[hidden_state ; e_CEFR ; e_L1]`, will lower held-out perplexity on
EFCAMDAT test sentences stratified by (CEFR, L1) cell, and this advantage
will transfer zero-shot to andrew100k, CELVA-SP (Spanish L1), and
KUPA-KEYS (Hungarian L1).

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` — after SDPA output, before concat / output projection |
| Granularity | Elementwise (shape `n × q × d_k`) |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative: `Y' = Y ⊙ σ([X; e_CEFR; e_L1] W_θ)` |

This matches the Qwen team's best configuration (Table 1 row 5 of the PDF)
and extends its gate input with learner-metadata embeddings.

## Metadata encoding

- CEFR: learned embedding over {A1, A2, B1, B2, C1, C2}, dim `d_c = 16`
- L1: learned embedding over EFCAMDAT L1 inventory, dim `d_l = 32`
- Broadcast across the sequence length before concatenation with hidden
  state
- Unknown / missing metadata → dedicated "unk" slots (needed for transfer
  data that lacks CEFR or has out-of-inventory L1)

## Task

Next-word prediction (autoregressive causal LM).

## Train corpus

EFCAMDAT split at
`./data/splits/norm-EFCAMDAT-train.csv`
+ `norm-EFCAMDAT-remainder.csv`. Held-in test set:
`norm-EFCAMDAT-test.csv`.

Base model: GPT-2 Base (117M), continual pre-training with gate initialised
so `σ(·) ≈ 1` at step 0 (zero-init the gate head so training starts from
the un-gated baseline).

## Eval protocol

In-domain:

- Token-level PPL on `norm-EFCAMDAT-test.csv`, stratified by CEFR and by L1.
- Top-1 / top-5 cloze accuracy on POS-masked content words, stratified
  the same way.

Zero-shot transfer (no gradient updates):

- `norm-andrew100k-remainder.csv` — multi-L1, same target (English),
  different register / curriculum. L1 inventory overlaps fully with
  EFCAMDAT.
- `norm-CELVA-SP.csv` — predominantly French-L1 (in EFCAMDAT
  inventory), so the gate's L1 embedding applies directly.
- `norm-KUPA-KEYS.csv` — mixed-L1 corpus with several out-of-inventory
  L1s (Polish, Greek, Hungarian, Dutch, Bengali, Slovene) that route
  to the learned `unk` slot. The hardest generalisation test.

Metrics: stratified PPL and cloze top-k. ERRANT-based error-distribution
alignment is deferred to future work.

## What this variant isolates

The joint contribution of proficiency (CEFR) and transfer (L1) signals
routed through a gating mechanism at the Qwen-optimal site. Baselines
(un-gated GPT-2, CMCL's learner GPT-2, un-gated metadata-concatenated
model) isolate the *gate* specifically from the *feature injection*.

## Expected outcome

1. In-domain PPL lower than every baseline, with largest gains on
   CEFR × L1 cells where the baseline is most uncertain (A1–B1 rows).
2. Cloze top-1 accuracy higher for G1 than every baseline, with the
   widest margin on A1 rows (most proficiency signal to exploit).
3. Transfer gain holds on andrew100k (fully overlapping L1 inventory);
   partial on CELVA-SP (French is in-inventory); weakens on KUPA-KEYS
   (several out-of-inventory L1s), motivating v07 typological features.
