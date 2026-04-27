# v03 — Granularity ablation: headwise gate

## Hypothesis

A headwise (scalar-per-head) gate is a much cheaper modulation than the
elementwise gate in v01 (Qwen: ~0.2M added params vs ~201M). On L2 NWP,
the coarser gate should capture most of the proficiency/L1 signal while
sacrificing little PPL — because L2 effects are plausibly
*head-level* (certain heads specialise on verb morphology, others on
determiners) rather than requiring per-dimension control.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` |
| Granularity | **Headwise** (shape `n × q`, single scalar per head) |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative |

## Metadata encoding

Same as v01: `[X; e_CEFR; e_L1]`.

## Task

Next-word prediction.

## Train corpus

EFCAMDAT.

## Eval protocol

Same as v01, plus an analysis: per-head gate activation heatmap across
layers × (CEFR, L1) cells — to test whether specific heads activate
selectively for specific learner profiles.

## What this variant isolates

Whether elementwise control is *necessary* for metadata modulation, or
whether the signal routes through a sparse subset of heads. If v03 comes
close to v01, the paper's engineering recommendation shifts toward
headwise (drastically fewer added parameters).

## Expected outcome

~0.7–0.9× of v01's gain at ~0.1% of v01's parameter cost. The per-head
heatmap should show structured activation patterns — certain heads
firing for A1–A2 texts, others for C-level texts — providing a
*mechanistic* story that complements the performance numbers.
