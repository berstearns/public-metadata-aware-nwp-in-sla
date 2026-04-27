# v07 — Metadata ablation: typological features instead of L1 ID

## Hypothesis

Replacing the learned L1 embedding with a fixed typological feature
vector (WALS / Grambank) gives up some in-domain fit against v01 but
gains substantially on out-of-inventory L1 transfer — because typology
generalises while L1 IDs don't. This is the critical variant for
claiming broad applicability across languages not represented in
EFCAMDAT's L1 inventory.

## Gate configuration

| Knob | Setting |
|---|---|
| Site | `G_1` |
| Granularity | Elementwise |
| Head sharing | Head-specific |
| Activation | Sigmoid |
| Form | Multiplicative |

Gate input: `[X ; e_CEFR ; v_typology]`.

## Metadata encoding

- CEFR embedding, `d_c = 16` (same as v01)
- Typology: binarised WALS features (article presence, pro-drop, basic
  word order, past-tense morphology, plural marking, etc.) projected to
  `d_l = 32` via a learned linear layer

A principled subset (~20 features) rather than the full WALS matrix,
chosen for (i) availability across corpus L1s and (ii) SLA-theoretical
relevance to the error categories observed in the CMCL companion paper.

## Task

Next-word prediction.

## Train corpus

EFCAMDAT.

## Eval protocol

Same as v01 with one addition: a **leave-one-L1-out** eval. Retrain on
EFCAMDAT with Spanish L1 held out; evaluate on the held-out Spanish
portion. Compare v07 (typology-aware) vs v01 (L1-ID-aware). v01 must
fall back to "unk"; v07 has a real feature vector for Spanish.

Transfer to KUPA-KEYS (Hungarian, not in EFCAMDAT's top inventory) is the
cleanest natural out-of-inventory test.

## What this variant isolates

Whether the L1 signal generalises via typology or stays language-locked.

## Expected outcome

v07 ≈ v01 in-domain; v07 > v01 on leave-one-L1-out and on KUPA-KEYS.
A clean typology-transfer story would be the paper's strongest
generalisation claim.
