# v06 — Metadata ablation: L1 only

## Hypothesis

Conditioning the gate on L1 alone captures the transfer-effect axis
(article use, pro-drop, SVO rigidity) but leaves the proficiency
trajectory unmodelled. Smaller in-domain gain than v05, but potentially
useful as a complement in compositional variants.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` |
| Granularity | Elementwise |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative |

Gate input: `[X ; e_L1]` only (no CEFR embedding).

## Metadata encoding

- L1 learned embedding, `d_l = 32`
- No CEFR signal

## Task

Next-word prediction.

## Train corpus

EFCAMDAT.

## Eval protocol

Same as v01. On KUPA-KEYS (Hungarian) expect severe degradation — L1 ID
is out-of-inventory, gate sees only "unk" → degenerate. This is the
point: v06 should motivate v07 (typological features) as the
out-of-inventory fallback.

## What this variant isolates

The L1 transfer signal alone, without proficiency. Compared to v05,
answers: does L1 carry more or less signal than CEFR for L2 NWP?

## Expected outcome

Smaller in-domain PPL gain than v05. Per-L1 cloze accuracy should be
highest for L1s with well-documented English transfer effects (e.g.
Spanish L1 → determiner cloze; Mandarin L1 → verb-morphology cloze).
A follow-up analysis under the Future-Work ERRANT pipeline would then
check whether v06 also more closely matches per-L1 error profiles of
authentic EFCAMDAT data.
