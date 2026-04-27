#!/usr/bin/env python3
"""Train the flagship G1 variant: metadata-aware SDPA-output gated
attention on GPT-2 Base continually pre-trained on EFCAMDAT.

The gate is initialised to pass-through (σ ≈ 1) so step-0 is equivalent
to the ungated baseline.
"""

from __future__ import annotations

from _train_shared import run_training

if __name__ == "__main__":
    run_training(description=__doc__)
