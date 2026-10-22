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


def test_chat_token_count_does_not_count_batch_encoding_keys():
    path = Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/prepare_chunkflow_sft.py"
    count = runpy.run_path(str(path))["token_count"]

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=True):
            ids = list(range(messages[0]["length"]))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids

    tokenizer = Tokenizer()
    for length in (100, 5000, 12022, 20000):
        messages = [{"length": length}]
        assert len(tokenizer.apply_chat_template(messages, True, False)) == 2
        assert count(tokenizer, messages) == length
