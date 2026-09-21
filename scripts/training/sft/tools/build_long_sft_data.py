# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a long-sequence SFT dataset by concatenating GSM8K problems.

Why: chunkflow splits per sequence, so a dataset whose samples are shorter than
``--chunkflow-chunk-size`` never produces a second chunk -- chunked attention
degrades to plain attention and an A/B over it proves nothing. GSM8K samples
average ~195 tokens, far below any production chunk size. Concatenating k
problems raises that to a configurable target (2048 by default), which lets the
chunk size be a realistic 512 or 1024 and still cut every sample into several
chunks.

The text is synthetic in the sense that one sample holds several unrelated
problems, but every token is real and the run is fully deterministic, which is
what the loss-curve comparison needs.

Usage:
    python scripts/training/sft/tools/build_long_sft_data.py \
        --input  $DATA_DIR/gsm8k/main/train-00000-of-00001.parquet \
        --output $DATA_DIR/gsm8k-long/train.parquet \
        --tokenizer $MODEL_DIR/Qwen3-4B \
        --target-tokens 2048
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


# GSM8K answers carry calculator annotations like ``<<48/2=24>>24``. They are
# noise for language modelling, so drop the annotation and keep the value.
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def clean_answer(text: str) -> str:
    return _CALC_ANNOTATION.sub("", text).strip()


def sample_length(tokenizer, question: str, answer: str) -> int:
    """Length the SFT loader will see: templated prompt + supervised answer."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer(prompt + answer, add_special_tokens=False)["input_ids"])


def build(input_path: Path, output_path: Path, tokenizer_path: str, target: int, max_tokens: int) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    source = pd.read_parquet(input_path)
    print(f"source: {len(source)} rows from {input_path}")

    questions: list[str] = []
    answers: list[str] = []
    cur_q: list[str] = []
    cur_a: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_q, cur_a, cur_len
        if cur_q:
            questions.append("\n\n".join(cur_q))
            answers.append("\n\n".join(cur_a))
        cur_q, cur_a, cur_len = [], [], 0

    for question, answer in zip(source["question"], source["answer"], strict=True):
        answer = clean_answer(str(answer))
        length = sample_length(tokenizer, str(question), answer)

        # A single problem longer than the cap gets its own sample rather than
        # silently pushing the previous one over.
        if cur_q and cur_len + length > max_tokens:
            flush()

        cur_q.append(str(question))
        cur_a.append(answer)
        cur_len += length

        if cur_len >= target:
            flush()

    flush()

    out = pd.DataFrame({"question": questions, "answer": answers})
    lengths = [sample_length(tokenizer, q, a) for q, a in zip(out["question"], out["answer"], strict=True)]
    out["_tokens"] = lengths

    print(f"built {len(out)} samples (target {target}, cap {max_tokens} tokens)")
    print(
        f"tokens: mean {sum(lengths) / len(lengths):.0f}  "
        f"min {min(lengths)}  max {max(lengths)}  "
        f"p95 {sorted(lengths)[int(0.95 * len(lengths))]}"
    )
    print(f"problems per sample: ~{len(source) / len(out):.1f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["_tokens"]).to_parquet(output_path, index=False)
    print(f"wrote {output_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path, e.g. $MODEL_DIR/Qwen3-4B")
    parser.add_argument("--target-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=3072, help="Hard cap per sample; keep under --max-tokens-per-gpu")
    args = parser.parse_args()

    build(args.input, args.output, args.tokenizer, args.target_tokens, args.max_tokens)


if __name__ == "__main__":
    main()