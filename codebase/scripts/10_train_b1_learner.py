#!/usr/bin/env python3
"""Train baseline B1: vanilla GPT-2 continual pre-training on EFCAMDAT.

No metadata is used at training or inference. Produces the "learner
GPT-2" of the CMCL companion.
"""

from __future__ import annotations

from _train_shared import run_training

if __name__ == "__main__":
    run_training(description=__doc__)
