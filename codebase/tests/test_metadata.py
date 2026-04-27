from gated_nwp.data.metadata import MetadataEncoder


def test_from_config_adds_unk(encoder: MetadataEncoder) -> None:
    assert "unk" in encoder.cefr_classes
    assert "unk" in encoder.l1_classes
    assert encoder.num_cefr == 7
    assert encoder.num_l1 == 4


def test_encode_known_cefr(encoder: MetadataEncoder) -> None:
    assert encoder.encode_cefr("A1") == encoder.cefr_classes.index("A1")
    assert encoder.encode_cefr("a1") == encoder.cefr_classes.index("A1")


def test_encode_unknown_cefr_routes_to_unk(encoder: MetadataEncoder) -> None:
    assert encoder.encode_cefr("Z9") == encoder.unk_cefr_idx
    assert encoder.encode_cefr(None) == encoder.unk_cefr_idx


def test_encode_known_l1(encoder: MetadataEncoder) -> None:
    assert encoder.encode_l1("Spanish") == encoder.l1_classes.index("Spanish")


def test_encode_unknown_l1_routes_to_unk(encoder: MetadataEncoder) -> None:
    assert encoder.encode_l1("Klingon") == encoder.unk_l1_idx
    assert encoder.encode_l1(None) == encoder.unk_l1_idx


def test_encode_batch_length(encoder: MetadataEncoder) -> None:
    cefr, l1 = encoder.encode_batch(["A1", "B2", None], ["German", None, "Spanish"])
    assert len(cefr) == 3
    assert len(l1) == 3
