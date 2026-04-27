# v09 — Training regime: frozen backbone, gate-only training

## Hypothesis

Freezing GPT-2 and training only the gate parameters + metadata
embeddings matches most of v01's in-domain gain with a fraction of the
trainable-parameter budget and no risk of catastrophic forgetting of the
native-English backbone. This is the practical path to deploying a
single base model with a swappable "learner adapter".

## Gate configuration

Same as v01: `G_1`, elementwise, head-specific, sigmoid, multiplicative.

## Metadata encoding

Same as v01.

## Task

Next-word prediction.

## Train corpus

EFCAMDAT. **Base GPT-2 weights frozen.** Trainable: gate matrices `W_θ`
per layer (~2M params total for GPT-2 Base) + metadata embeddings (`d_c`
+ `d_l` × inventory size ≈ few thousand params).

## Eval protocol

Same as v01. Additional comparison: evaluate the gated model on a
**native English** held-out set (wikitext-103) and check PPL degradation
vs base GPT-2. The un-frozen v01 is expected to drift away from native
English; the frozen v09 should stay close.

## What this variant isolates

Whether the metadata-aware gate is sufficient on its own for learner
adaptation, or whether the backbone also needs to shift. Also tests
whether "learner mode" can be a toggle on top of an otherwise-native
base.

## Expected outcome

v09 recovers ~80–90% of v01's in-domain gain with ~1% of the trainable
parameter budget and no degradation on native English. This is the
clean-deployment story: ship one native GPT-2 + a tiny learner-adapter
gate and activate it when needed.
