# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import runpy
from pathlib import Path


def test_public_sft_preserves_conversation_and_rejects_invalid_turns():
    path = Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/prepare_chunkflow_sft.py"
    normalize = runpy.run_path(str(path))["conversation"]
    messages = [
        {"role": role, "content": text}
        for role, text in (
            ("system", "instruction"),
            ("user", "first"),
            ("assistant", "answer"),
            ("user", "follow up"),
            ("assistant", "different answer"),
        )
    ]
    assert normalize({"messages": messages}) == messages
    assert normalize({"messages": messages[:-1]}) is None
    assert normalize({"messages": messages[2:]}) is None
    assert normalize({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]}) is None
