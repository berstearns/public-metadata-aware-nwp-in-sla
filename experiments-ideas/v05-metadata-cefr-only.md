# v05 — Metadata ablation: CEFR only

## Hypothesis

Conditioning the gate on CEFR proficiency alone captures the dominant
share of v01's gain, because CEFR compresses the developmental trajectory
that drives most of the held-out PPL variance in learner corpora.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` |
| Granularity | Elementwise |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative |

Gate input: `[X ; e_CEFR]` only (no L1 embedding).

## Metadata encoding

- CEFR learned embedding, `d_c = 16`
- No L1 signal at all

## Task

Next-word prediction.

## Train corpus

EFCAMDAT.

## Eval protocol

Same as v01, including all three transfer corpora. On andrew100k, CEFR
labels are available (A1–C1); on CELVA-SP / KUPA-KEYS, CEFR may be noisy
or unavailable — in that case the gate falls back to the "unk" embedding
and we report PPL separately for known vs unknown CEFR cells.

## What this variant isolates

The CEFR signal's contribution independent of L1. Compared to v06 (L1
only), v05 tests which axis carries more predictive content for L2 NWP.

## Expected outcome

Most of v01's in-domain PPL gain but weaker L1-specific generalisation
patterns (e.g. missing the pro-drop→missing-pronoun tendency of Spanish
L1 learners). On transfer, v05 should do *better* than v01 on corpora
where L1 inventory diverges (e.g. KUPA-KEYS Hungarian), because there's
no out-of-inventory L1 mismatch.
