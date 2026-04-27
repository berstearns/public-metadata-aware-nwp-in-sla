# v08 — Task variant: cloze / fill-in-the-gap

## Hypothesis

The same metadata-aware gate helps on constrained cloze tasks (where a
specific word is masked and must be predicted) in addition to open-ended
NWP. Cloze is arguably a more faithful proxy for L2 testing formats
(C-test, gap-fill) than free generation, and aligns with the
production-vs-constrained distinction explored in `SLA-AL-EVAL/`.

## Gate configuration

Same as v01: `G_1`, elementwise, head-specific, sigmoid, multiplicative,
gate input `[X; e_CEFR; e_L1]`.

## Metadata encoding

Same as v01.

## Task

**Cloze.** At evaluation, content words (nouns, verbs, adjectives,
determiners — picked deterministically from POS tags) are replaced with
a `<MASK>` sentinel; the model predicts the masked token given bilateral
context. Two sub-settings:

1. *Bidirectional-style*: prepend the suffix + mask to the prefix so the
   causal model sees both sides in one concatenated prompt (reuse
   `align-humantokens-to-llmsubtokens` to keep target-word alignment
   precise across tokenisations).
2. *Left-context-only*: standard NWP framing — same metric but without
   the suffix.

Training objective combines NWP loss + masked-span loss (span of the
target word inserted mid-sequence), weighted 1:1.

## Train corpus

EFCAMDAT, with cloze targets sampled across POS per sentence.

## Eval protocol

- In-domain cloze accuracy on held-out EFCAMDAT, broken out by POS and
  by (CEFR, L1)
- Zero-shot transfer: same cloze protocol on andrew100k / CELVA-SP /
  KUPA-KEYS
- Shared eval with v01: report NWP PPL of the same model, to test
  whether joint training on cloze + NWP hurts open-ended generation

## What this variant isolates

Task-generalisation of the gating mechanism beyond open-ended
generation. Bridges the paper to the L2 testing literature and to
`SLA-AL-EVAL/`.

## Expected outcome

Gate helps more on cloze than on NWP (bilateral context lets the gate
exploit proficiency signal more effectively). CEFR stratification
strongly predicts cloze accuracy — A1 cloze accuracy rises most when the
gate is active.
