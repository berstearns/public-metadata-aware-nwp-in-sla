"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from gated_nwp.data.metadata import MetadataEncoder


@pytest.fixture
def encoder() -> MetadataEncoder:
    return MetadataEncoder.from_config(
        cefr_classes=("A1", "A2", "B1", "B2", "C1", "C2"),
        l1_classes=("Spanish", "German", "Chinese"),
    )
