# v04 — Mode ablation: additive SiLU gate

## Hypothesis

Replacing the multiplicative sigmoid gate with an additive SiLU gate
(`Y' = Y + σ_SiLU([X; e_CEFR; e_L1] W_θ)`) under-performs v01, matching
Qwen's finding that multiplicative gating dominates additive (Table 1
row 14 vs row 5 in the PDF). Included as a control to show the mode
choice matters on the L2 task.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` |
| Granularity | Elementwise |
| Head sharing | Head-specific |
| Activation | SiLU |
| Form | **Additive**: `Y' = Y + σ_SiLU([X; e_CEFR; e_L1] W_θ)` |

## Metadata encoding

Same as v01.

## Task

Next-word prediction.

## Train corpus

EFCAMDAT.

## Eval protocol

Same as v01.

## What this variant isolates

Whether the gain from v01 comes from *filtering* (multiplicative) or from
*injection* (additive). Additive gating adds a metadata-conditioned bias
to each hidden state; multiplicative gating scales it down selectively.
The two mechanisms have different expressivity stories.

## Expected outcome

Small positive effect over the un-gated baseline (matches Qwen's
observation that additive still helps a bit), but clearly below v01.
Reporting both makes the method section's choice of multiplicative more
defensible and connects back to the "input-dependent sparsity" analysis
in §4.2 of the Qwen paper.
