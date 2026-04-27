# v10 — Deferred: EFCAMDAT + C4200M synthetic mix

**Status:** deferred / not priority per user direction. Kept here for
design completeness.

## Hypothesis

Mixing EFCAMDAT (authentic learner text) with C4200M (synthetic
corruptions of clean web text) during continual pre-training gives the
gate more diverse error contexts to condition on. PPL on EFCAMDAT held-out
should hold or improve; transfer to out-of-inventory corpora should
improve more than EFCAMDAT-only training (v01), because C4200M's
typology of synthetic errors covers cases EFCAMDAT under-represents.

## Gate configuration

Same as v01.

## Metadata encoding

For EFCAMDAT rows: real CEFR + real L1.

For C4200M rows: no real metadata. Options:

1. Map C4200M's corruption severity (edit distance / error density) to a
   pseudo-CEFR label via a threshold on error rate per sentence.
2. Leave CEFR / L1 as "unk" for C4200M rows; the gate learns a third
   mode (synthetic) implicitly.

Option (1) is cleaner for the loss landscape; option (2) avoids fake
labels. Recommend running both and reporting.

## Task

Next-word prediction.

## Train corpus

~50/50 mix of EFCAMDAT (`norm-EFCAMDAT-train+remainder.csv`) and C4200M
synthetic (external, not yet downloaded). Small C4200M held-out test
used for in-distribution synthetic PPL.

## Eval protocol

Same as v01 + a synthetic-held-out PPL row. The interesting cell is
**transfer to KUPA-KEYS / CELVA-SP** — does a more diverse training mix
improve generalisation, or does it just dilute the EFCAMDAT signal?

## What this variant isolates

Whether data diversity (authentic + synthetic) helps the gate generalise
across unseen L1s / CEFR cells.

## Expected outcome

Small in-domain regression vs v01 on EFCAMDAT; meaningful gain on
KUPA-KEYS transfer. If the regression is severe, EFCAMDAT-only (v01)
remains the recommended configuration.

## Why deferred

User priority: EFCAMDAT-only first. Also, C4200M requires an external
download + preprocessing pipeline not yet in place. Revisit after v01–v09
are running.
