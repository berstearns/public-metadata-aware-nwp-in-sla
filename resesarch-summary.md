# Summary: *Metadata-Aware Gated Attention for NWP on L2 Learner Text*

## Research Questions
1. **Can a small architectural knob make next-word prediction *controllable* by explicit learner metadata** (CEFR proficiency + L1), rather than the corpus-average implicit profile a continually-trained learner LM picks up?
2. **Is the SDPA-output gate (Qwen's $G_1$) also the optimal injection site for learner-metadata conditioning?** Testable prediction: the $G_1 \succ G_2 \succ G_3, G_4$ ordering Qwen reports on native LM benchmarks should reproduce — with a *larger* margin — on L2 NWP, since L2 effects are about which contextual information survives softmax aggregation.
3. **Does architectural gating outperform naive prompt-level metadata injection** at matched data exposure?

## Method (one equation)
Replace Qwen's gate $Y' = Y \odot \sigma(X W_\theta)$ with
$Y' = Y \odot \sigma([X;\,e_\text{CEFR};\,e_\text{L1}] W_\theta)$ — head-specific, elementwise, sigmoid, multiplicative; CEFR embedding $d_c=16$, L1 embedding $d_l=32$; bias-init so $\sigma \approx 1$ at step 0 (pass-through). Adds <2% params over GPT-2 Base.

## Datasets
- **Training:** EFCAMDAT (sentence-level release; `norm-EFCAMDAT-train/remainder/test.csv`). L1 inventory = 10 L1s (Arabic, French, German, Italian, Japanese, Mandarin, Portuguese, Russian, Spanish, Turkish); CEFR A1–C1 in test set.
- **Zero-shot transfer (3 corpora):**
  - `andrew100k` — Caines 100k, fully-overlapping L1 inventory (closest test).
  - `CELVA-SP` — predominantly French-L1 (in inventory).
  - `KUPA-KEYS` — mixed-L1, partially OOV (Polish, Greek, Hungarian, Dutch, Bengali, Slovene → `unk`); hardest L1-generalisation test.

## Baselines (matched data exposure, GPT-2 Base, ctx 1024)
- **B0** Native GPT-2 (no further training).
- **B1** Learner GPT-2 (CMCL companion: continual PT on EFCAMDAT, no metadata).
- **B2** Learner GPT-2 + metadata concat (CEFR/L1 tokens prepended to sequences) — the prompt-level baseline.
- **G1 (ours)** Learner GPT-2 with metadata-aware SDPA-output gate.

B0–B2 isolate the two claims: gate beats data-only (vs B1) and gate beats prompt injection (vs B2).

## Experiments / Metrics
- **Stratified perplexity** on EFCAMDAT held-out, broken down by (CEFR, L1) cells; transfer corpora by CEFR bucket.
- **Cloze accuracy** (top-1, top-5) on POS-selected masked content words (noun/verb/adj/det), stratified by (CEFR, L1).
- **Zero-shot PPL** on the three transfer corpora.
- Result tables (CEFR ppl, cloze, transfer) are pre-registered with `TBD` cells.

## Ablation Catalogue (`experiments-ideas/v01–v10`)
Nine+ complementary variants ablating one factor each: gate site ($G_1$ vs $G_2$, v02), granularity (elementwise vs headwise, v03), mode (multiplicative vs additive-SiLU, v04), metadata encoding (CEFR-only v05, L1-only v06, typological/WALS v07), task (cloze, v08), training regime (frozen backbone, v09), and data scale (EFCAMDAT + C4-200M, v10).

## Compute
~5 A100-days for the flagship (B2 + G1 runs); +18 A100-days for the full ablation sweep (or ~6 A100-days under the v09 frozen-backbone regime).
