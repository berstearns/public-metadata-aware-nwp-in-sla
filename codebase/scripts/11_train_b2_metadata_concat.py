#!/usr/bin/env python3
"""Train baseline B2: continual pre-training with metadata tokens
prepended to each training sequence.

Implements the prompt-level metadata baseline against which the
architectural gate (G1) is compared.
"""

from __future__ import annotations

from _train_shared import run_training

if __name__ == "__main__":
    run_training(description=__doc__)
