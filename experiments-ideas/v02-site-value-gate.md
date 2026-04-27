# v02 — Site ablation: value-layer gate (`G_2`)

## Hypothesis

Moving the metadata-conditioned gate from the SDPA output (`G_1`) to the
value projection (`G_2`) will yield a smaller PPL gain, consistent with
Qiu et al.'s finding that `G_1 > G_2` in both perplexity and downstream
benchmarks. Confirming this on the learner-NWP task strengthens the
method-section justification for choosing `G_1`.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_2` — after value projection (before attention) |
| Granularity | Elementwise (`n × k × d_k`) |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative |

## Metadata encoding

Identical to v01: `[X; e_CEFR; e_L1]` (so only the *site* varies).

## Task

Next-word prediction.

## Train corpus

EFCAMDAT (same as v01).

## Eval protocol

Same as v01 — in-domain PPL + cloze + zero-shot transfer.

## What this variant isolates

Gate *site* under fixed metadata encoding. Any delta against v01 is
attributable to where in the attention block the metadata-conditioned
filter is applied.

## Expected outcome

Smaller PPL improvement than v01 but still positive. Consistent with Qwen
Table 1 row 6: `G_2` elementwise gains ~0.2 PPL where `G_1` gains ~0.3.
If v02 matches or beats v01 on this task, that's a negative surprise
worth reporting — it would mean learner-metadata gating interacts
differently with value-level vs output-level modulation.
