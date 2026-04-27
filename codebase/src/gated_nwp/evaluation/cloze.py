"""Cloze / fill-in-the-gap evaluation.

We form an input of the shape::

    <prefix> <MASK> <suffix>

and rank candidate completions of the masked span by the model's
log-probability of the target word given bilateral context. For a
left-to-right LM, we compute probabilities of the target tokens under
the concatenated ``prefix + target + suffix`` sequence at the target
positions, then marginalise over sub-word tokens of the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from gated_nwp.data.cloze import ClozeExample
from gated_nwp.utils.forward import call_model


@dataclass
class ClozeReport:
    top1: float
    top5: float
    by_cefr_top1: dict[str, float]
    by_l1_top1: dict[str, float]
    n: int


def _score_target_logprob(
    model: Any,
    tokenizer: Any,
    prefix: str,
    target: str,
    suffix: str,
    *,
    cefr_id: int,
    l1_id: int,
    device: torch.device,
) -> float:
    """Log-probability of the target tokens in the context of prefix/suffix.

    For a causal LM, we use the left context only — the suffix is
    reported to the caller for bookkeeping but not used at scoring time
    unless the caller explicitly wants bidirectional-style scoring (see
    v08 variant md).
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(" " + target.strip(), add_special_tokens=False)["input_ids"]

    input_ids = torch.tensor([prefix_ids + target_ids], device=device)
    with torch.no_grad():
        out = call_model(
            model,
            input_ids=input_ids,
            cefr_id=torch.tensor([cefr_id], device=device),
            l1_id=torch.tensor([l1_id], device=device),
        )
    logits = out.logits[0]  # (T, V)
    start = len(prefix_ids) - 1
    logprobs = 0.0
    for i, tid in enumerate(target_ids):
        pos = start + i
        logp = torch.log_softmax(logits[pos], dim=-1)[tid].item()
        logprobs += logp
    return logprobs


def score_cloze(
    model: Any,
    tokenizer: Any,
    examples: list[ClozeExample],
    *,
    encoder: Any,
    topk: tuple[int, ...] = (1, 5),
    device: str | torch.device = "cuda",
) -> ClozeReport:
    """Score cloze accuracy over a list of ClozeExamples.

    For each example, the top-1 (respectively top-5) metric counts it as
    correct if the model's next-token distribution at the mask position
    ranks the first sub-word token of the target in the top-1 (top-5).
    """
    model.eval()
    model = model.to(device)

    top1_correct = 0
    top5_correct = 0
    per_cefr_correct: dict[str, int] = {}
    per_cefr_total: dict[str, int] = {}
    per_l1_correct: dict[str, int] = {}
    per_l1_total: dict[str, int] = {}

    # GPT-2 has no BOS token; we reuse its EOS id as a sentinel prefix so
    # examples whose mask lands at position 0 still produce a
    # well-formed input (empty prefix → [] → GPT-2 can't reshape).
    bos_id = tokenizer.eos_token_id

    for ex in tqdm(examples, desc="cloze"):
        cefr_id = encoder.encode_cefr(ex.cefr)
        l1_id = encoder.encode_l1(ex.l1)

        # Strip trailing whitespace so the BPE lands on the word before
        # the gap. The target carries its leading space separately, and
        # leaving the trailing space in the prefix makes GPT-2 emit it
        # as a standalone token that offsets the prediction.
        prefix_text = ex.prefix.rstrip()
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        if not prefix_ids:
            prefix_ids = [bos_id]
        target_first = tokenizer(" " + ex.target.strip(), add_special_tokens=False)["input_ids"][0]

        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out = call_model(
                model,
                input_ids=input_ids,
                cefr_id=torch.tensor([cefr_id], device=device),
                l1_id=torch.tensor([l1_id], device=device),
            )
        logits = out.logits[0, -1]
        topk_vals = torch.topk(logits, max(topk)).indices.tolist()

        if target_first == topk_vals[0]:
            top1_correct += 1
        if target_first in topk_vals:
            top5_correct += 1

        cefr_key = ex.cefr or "unk"
        l1_key = ex.l1 or "unk"
        per_cefr_total[cefr_key] = per_cefr_total.get(cefr_key, 0) + 1
        per_l1_total[l1_key] = per_l1_total.get(l1_key, 0) + 1
        if target_first == topk_vals[0]:
            per_cefr_correct[cefr_key] = per_cefr_correct.get(cefr_key, 0) + 1
            per_l1_correct[l1_key] = per_l1_correct.get(l1_key, 0) + 1

    n = max(len(examples), 1)
    return ClozeReport(
        top1=top1_correct / n,
        top5=top5_correct / n,
        by_cefr_top1={k: per_cefr_correct.get(k, 0) / v for k, v in per_cefr_total.items()},
        by_l1_top1={k: per_l1_correct.get(k, 0) / v for k, v in per_l1_total.items()},
        n=len(examples),
    )
