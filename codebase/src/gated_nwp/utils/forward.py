"""Signature-aware model invocation.

Stock ``GPT2LMHeadModel.forward`` does not accept ``cefr_id`` / ``l1_id``
kwargs, whereas our :class:`MetadataAwareGPT2LMHeadModel` does. The
evaluation pipeline is shared across both, so we route all forward /
generate calls through :func:`call_model` / :func:`call_generate`, which
inspect the target's signature and only pass metadata kwargs if they are
accepted.
"""

from __future__ import annotations

import inspect
from typing import Any


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def call_model(model: Any, **kwargs: Any) -> Any:
    """Forward-pass wrapper: drop ``cefr_id`` / ``l1_id`` if the model
    does not declare them."""
    filtered = dict(kwargs)
    for meta_key in ("cefr_id", "l1_id"):
        if meta_key in filtered and not _accepts(model.forward, meta_key):
            filtered.pop(meta_key)
    return model(**filtered)


def call_generate(model: Any, **kwargs: Any) -> Any:
    """``model.generate`` wrapper with the same metadata-dropping
    behaviour."""
    filtered = dict(kwargs)
    for meta_key in ("cefr_id", "l1_id"):
        if meta_key in filtered and not _accepts(model.generate, meta_key):
            filtered.pop(meta_key)
    return model.generate(**filtered)
