"""Stratified token-level perplexity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gated_nwp.utils.forward import call_model


@dataclass
class PPLReport:
    overall: float
    by_cefr: dict[str, float]
    by_l1: dict[str, float]
    # Cell keys are stringified as "CEFR|L1" so the report is JSON-serialisable
    # without custom encoders.
    by_cell: dict[str, float]
    n_tokens_overall: int
    n_tokens_by_cell: dict[str, int]


def _tok_nll(
    logits: torch.Tensor, labels: torch.Tensor, attn: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (summed NLL per example, token count per example)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attn[..., 1:].contiguous().float()

    loss_flat = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss_flat.view(shift_labels.size())
    nll_per_example = (loss * shift_mask).sum(dim=-1)
    n_tokens_per_example = shift_mask.sum(dim=-1)
    return nll_per_example, n_tokens_per_example


def compute_stratified_ppl(
    model: Any,
    dataset: Any,
    *,
    encoder: Any,
    batch_size: int = 16,
    device: str | torch.device = "cuda",
) -> PPLReport:
    """Compute token-level PPL overall and stratified by (CEFR, L1)."""
    model.eval()
    model = model.to(device)

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]).to(device),
            "cefr_id": torch.tensor([b["cefr_id"] for b in batch], dtype=torch.long).to(device),
            "l1_id": torch.tensor([b["l1_id"] for b in batch], dtype=torch.long).to(device),
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    total_nll = 0.0
    total_tokens = 0
    nll_by_cell: dict[tuple[int, int], float] = defaultdict(float)
    tok_by_cell: dict[tuple[int, int], int] = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="ppl"):
            out = call_model(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                cefr_id=batch["cefr_id"],
                l1_id=batch["l1_id"],
            )
            nll, ntok = _tok_nll(out.logits, batch["input_ids"], batch["attention_mask"])
            total_nll += nll.sum().item()
            total_tokens += int(ntok.sum().item())
            for i in range(batch["input_ids"].size(0)):
                key = (int(batch["cefr_id"][i].item()), int(batch["l1_id"][i].item()))
                nll_by_cell[key] += nll[i].item()
                tok_by_cell[key] += int(ntok[i].item())

    def ppl(nll: float, tokens: int) -> float:
        import math

        return math.exp(nll / max(tokens, 1))

    overall = ppl(total_nll, total_tokens)
    by_cefr: dict[str, float] = {}
    by_l1: dict[str, float] = {}
    by_cell: dict[str, float] = {}
    cefr_agg: dict[int, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    l1_agg: dict[int, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for (ci, li), nll in nll_by_cell.items():
        n = tok_by_cell[(ci, li)]
        cell_key = f"{encoder.cefr_classes[ci]}|{encoder.l1_classes[li]}"
        by_cell[cell_key] = ppl(nll, n)
        cefr_agg[ci] = (cefr_agg[ci][0] + nll, cefr_agg[ci][1] + n)
        l1_agg[li] = (l1_agg[li][0] + nll, l1_agg[li][1] + n)
    for ci, (nll, n) in cefr_agg.items():
        by_cefr[encoder.cefr_classes[ci]] = ppl(nll, n)
    for li, (nll, n) in l1_agg.items():
        by_l1[encoder.l1_classes[li]] = ppl(nll, n)

    return PPLReport(
        overall=overall,
        by_cefr=by_cefr,
        by_l1=by_l1,
        by_cell=by_cell,
        n_tokens_overall=total_tokens,
        n_tokens_by_cell={
            f"{encoder.cefr_classes[ci]}|{encoder.l1_classes[li]}": n
            for (ci, li), n in tok_by_cell.items()
        },
    )
