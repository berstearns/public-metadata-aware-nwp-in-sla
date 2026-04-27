"""Signature-aware forward shim: passes metadata only to models that
declare it."""

from __future__ import annotations

from gated_nwp.utils.forward import call_generate, call_model


class _ModelNoMeta:
    last_kwargs: dict | None = None

    def forward(self, input_ids, attention_mask=None): ...

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return "ok"

    def generate(self, input_ids, max_new_tokens=10):
        type(self).last_generate_kwargs = {"input_ids": input_ids, "max_new_tokens": max_new_tokens}
        return "gen"


class _ModelWithMeta:
    last_kwargs: dict | None = None

    def forward(self, input_ids, attention_mask=None, cefr_id=None, l1_id=None): ...

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return "ok"

    def generate(self, input_ids, cefr_id=None, l1_id=None, max_new_tokens=10):
        type(self).last_generate_kwargs = dict(
            input_ids=input_ids, cefr_id=cefr_id, l1_id=l1_id, max_new_tokens=max_new_tokens
        )
        return "gen"


def test_call_model_drops_metadata_for_stock_model() -> None:
    m = _ModelNoMeta()
    call_model(m, input_ids=1, attention_mask=2, cefr_id=3, l1_id=4)
    assert m.last_kwargs == {"input_ids": 1, "attention_mask": 2}


def test_call_model_passes_metadata_for_metadata_aware_model() -> None:
    m = _ModelWithMeta()
    call_model(m, input_ids=1, attention_mask=2, cefr_id=3, l1_id=4)
    assert m.last_kwargs == {"input_ids": 1, "attention_mask": 2, "cefr_id": 3, "l1_id": 4}


def test_call_generate_drops_metadata_for_stock_model() -> None:
    m = _ModelNoMeta()
    call_generate(m, input_ids=[[1, 2, 3]], cefr_id=0, l1_id=0, max_new_tokens=5)
    assert "cefr_id" not in m.last_generate_kwargs
    assert "l1_id" not in m.last_generate_kwargs
    assert m.last_generate_kwargs["max_new_tokens"] == 5


def test_call_generate_passes_metadata_for_metadata_aware_model() -> None:
    m = _ModelWithMeta()
    call_generate(m, input_ids=[[1, 2]], cefr_id=7, l1_id=9, max_new_tokens=3)
    assert m.last_generate_kwargs["cefr_id"] == 7
    assert m.last_generate_kwargs["l1_id"] == 9
