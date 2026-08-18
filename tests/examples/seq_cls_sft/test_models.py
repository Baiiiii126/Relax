# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest


try:
    import sglang.srt.models.qwen3_5  # noqa: F401
    from sglang.srt.layers.pooler import score_and_pool  # noqa: F401
except Exception as exc:
    pytest.skip(f"requires a Qwen3.5-compatible SGLang >= 0.5.12 image: {exc}", allow_module_level=True)

from examples.seq_cls_sft.models.sglang.model import (  # noqa: E402
    EntryClass,
    Qwen3_5ForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
)


def test_qwen3_5_sequence_classification_external_registry_names():
    assert EntryClass == [Qwen3_5ForSequenceClassification, Qwen3_5MoeForSequenceClassification]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight"),
        ("model.layers.0.mlp.down_proj.weight", "layers.0.mlp.down_proj.weight"),
        ("model.visual.blocks.0.weight", None),
        ("lm_head.weight", None),
        ("mtp.layers.0.weight", None),
    ],
)
def test_qwen3_5_sequence_classification_normalizes_backbone_weight_names(name, expected):
    assert Qwen3_5ForSequenceClassification._backbone_weight_name(name) == expected
