#!/usr/bin/env python3
"""Pre-download base models so training scripts don't race to fetch them.

Idempotent: uses HuggingFace cache if already present.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="gpt2", help="HF model id")
    args = p.parse_args()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[download] base tokenizer: {args.base_model}")
    GPT2Tokenizer.from_pretrained(args.base_model)
    print(f"[download] base model: {args.base_model}")
    GPT2LMHeadModel.from_pretrained(args.base_model)

    print("[done] base model cached")


if __name__ == "__main__":
    main()
