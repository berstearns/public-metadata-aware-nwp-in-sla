# Models to experiment — inventory

Every base model mentioned in the paper, with its role and the set of
training conditions (B0, B1, B2, G1) that apply to it.

## Notation

| Condition | What it is | Training required? |
|---|---|---|
| **B0** | stock base model, no fine-tune, no metadata | no |
| **B1** | continual pre-training on EFCAMDAT, no metadata | yes (~2 A100-days at GPT-2-Base scale) |
| **B2** | continual pre-training on EFCAMDAT with `<cefr=X><l1=Y>` prepended to each sequence | yes (same cost as B1) |
| **G1** | metadata-aware SDPA-output gated attention (ours) | yes (same cost as B1) |

---

## Flagship — every condition runs here

| Model | HF id | Params | Conditions | Status | Paper reference |
|---|---|---|---|---|---|
| GPT-2 Base | `gpt2` | 117M | **B0 · B1 · B2 · G1** | primary | `04-experiments.tex §Models and baselines`, `03-method.tex §Metadata-aware gate` |

This is the only model for which the paper commits to running the full
2×2 (condition × trained/not-trained) design. B0 already exists in the
smoke benchmark (`sample-benchmark-607a30d7-gpt2-20260421-094610/`). B1
is the CMCL-companion checkpoint (already trained, reused). B2 and G1
are the two new runs needed for the flagship.

---

## Scale-sweep — deferred to Future Work

Listed in `07-limitations.tex §Single base model, single scale` and
`§Scaling the gate with the backbone`. Priority order if the flagship
works:

| Model | HF id | Params | Conditions | Status | Rationale |
|---|---|---|---|---|---|
| Pythia 160M | `EleutherAI/pythia-160m` | 160M | B0 · G1 | future work | smallest scale-sweep point; validates gate on a different architecture family |
| Pythia 410M | `EleutherAI/pythia-410m` | 410M | B0 · G1 | future work | mid-scale; checks that gate gain is not specific to tiny models |
| GPT-2 Medium | `gpt2-medium` | 355M | B0 · G1 | future work | same family as flagship, 3× params |
| Pythia 1.4B | `EleutherAI/pythia-1.4b` | 1.4B | B0 · G1 | future work | closer to Qwen's 1.7B dense evidence |
| GPT-2 Large | `gpt2-large` | 774M | B0 · G1 | future work | upper end of same-family sweep |

B1/B2 are dropped at scale because the scale claim is specifically
about the gate, not about data-only continual pre-training vs. prefix
metadata. G1 vs B0 is sufficient to test whether the gate margin
ps/ersists.

---

## Referenced but not in scope

Cited only for methodological anchoring to Qwen's gated-attention
study (`02-related-work.tex §Gating in Attention`, `gateattention.pdf`).
No intention to run — we inherit the Qwen team's evidence at this
scale.

| Model class | Size | Source | Role in paper |
|---|---|---|---|
| Qwen dense | 1.7B | `qiu2025gated` | source of the G_1-dominance evidence |
| Qwen MoE | 15B (2.54B active) | `qiu2025gated` | source of the G_1-dominance evidence |

---

## Cross-reference — which `experiments-ideas/v*.md` applies to which model

All variants (`v01`–`v10`) assume **GPT-2 Base** unless noted otherwise.
The scale dimension is orthogonal to the variant catalogue: pick a
variant (e.g. v01 flagship) and a model row above to form a concrete
experimental unit.

| Variant | Model it targets | Notes |
|---|---|---|
| `v01-flagship-sdpa-cefr-l1.md` | GPT-2 Base | the flagship; becomes the worked example in `sections/05-results.tex` |
| `v02-site-value-gate.md` | GPT-2 Base | site ablation, same scale |
| `v03-granularity-headwise.md` | GPT-2 Base | granularity ablation, same scale |
| `v04-mode-additive-silu.md` | GPT-2 Base | mode ablation, same scale |
| `v05-metadata-cefr-only.md` | GPT-2 Base | metadata ablation, same scale |
| `v06-metadata-l1-only.md` | GPT-2 Base | metadata ablation, same scale |
| `v07-metadata-typology.md` | GPT-2 Base | typology feature swap, same scale |
| `v08-task-cloze.md` | GPT-2 Base | task variant, same scale |
| `v09-frozen-backbone.md` | GPT-2 Base | training-regime variant, same scale |
| `v10-efcamdat-plus-c4200m.md` | GPT-2 Base | data-mix variant, deferred |

No variant file currently targets Pythia / GPT-2 Medium / GPT-2 Large /
Pythia 1.4B. When the flagship is verified, add `v11-scale-pythia.md`
and `v12-scale-gpt2-large.md` rather than duplicating v01 four times.

---

## Recommended run order

1. **B0 on GPT-2 Base** — done (smoke test at n=100; full test at full EFCAMDAT-test is trivial)
2. **B1 on GPT-2 Base** — already trained in CMCL companion; reuse
3. **B2 on GPT-2 Base** — one continual-pre-training run
4. **G1 on GPT-2 Base** — one continual-pre-training run; the flagship claim
5. **Variants v02–v09 on GPT-2 Base** — ablations; prioritise v09 (frozen backbone) because it's the cheapest
6. **B0 on Pythia-160M / 410M / GPT-2 Medium** — cheap no-training benchmark extension; one smoke run each
7. **G1 on Pythia-160M / 410M** — the scale-sweep test, gated on the flagship result
