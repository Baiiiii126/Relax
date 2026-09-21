# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build an SFT dataset with a deliberate MIX of sequence lengths from GSM8K.

Why mixed lengths: chunkflow splits per sequence, so a dataset that is
uniformly short never produces a second chunk (the A/B is vacuous), and a
dataset that is uniformly long never exercises the unchunked path (nor the case
where only some sequences in a packed batch get split). Both regimes, and the
transition between them, need coverage -- so samples are built by concatenating
a *varying* number of GSM8K problems, spanning roughly 100 to 3000 tokens.

The build is deterministic (fixed RNG seed), which is what the loss-curve
comparison needs. Text is real; only the grouping of problems into samples is
synthetic.

Usage:
    python scripts/training/sft/tools/build_mixed_sft_data.py \
        --input  $DATA_DIR/gsm8k/main/train-00000-of-00001.parquet \
        --output $DATA_DIR/gsm8k-mixed/train.parquet \
        --tokenizer $MODEL_DIR/Qwen3-4B \
        --chunk-size 512
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


# GSM8K answers carry calculator annotations like ``<<48/2=24>>24``. They are
# noise for language modelling, so drop the annotation and keep the value.
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")

# (weight, target tokens) per band. The spread is the point: at chunk_size=512
# the 320 band yields single-chunk samples, 1280 yields three chunks, and 2560
# yields five -- so one run covers unchunked, lightly chunked and heavily
# chunked sequences together.
DEFAULT_BANDS = (
    (0.30, 320),
    (0.30, 768),
    (0.25, 1536),
    (0.15, 2560),
)


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


def build(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    bands: tuple[tuple[float, int], ...],
    max_tokens: int,
    seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    source = pd.read_parquet(input_path)
    print(f"source: {len(source)} rows from {input_path}")

    rng = random.Random(seed)
    weights = [w for w, _ in bands]
    targets = [t for _, t in bands]

    questions: list[str] = []
    answers: list[str] = []
    cur_q: list[str] = []
    cur_a: list[str] = []
    cur_len = 0
    target = rng.choices(targets, weights=weights, k=1)[0]

    def flush() -> None:
        nonlocal cur_q, cur_a, cur_len
        if cur_q:
            questions.append("\n\n".join(cur_q))
            answers.append("\n\n".join(cur_a))
        cur_q, cur_a, cur_len = [], [], 0

    for question, answer in zip(source["question"], source["answer"], strict=True):
        answer = clean_answer(str(answer))
        length = sample_length(tokenizer, str(question), answer)

        # A problem that would breach the cap ends the current sample instead of
        # pushing it over; a lone problem longer than the cap becomes its own.
        if cur_q and cur_len + length > max_tokens:
            flush()
            target = rng.choices(targets, weights=weights, k=1)[0]

        cur_q.append(str(question))
        cur_a.append(answer)
        cur_len += length

        if cur_len >= target:
            flush()
            target = rng.choices(targets, weights=weights, k=1)[0]

    flush()

    out = pd.DataFrame({"question": questions, "answer": answers})
    lengths = [sample_length(tokenizer, q, a) for q, a in zip(out["question"], out["answer"], strict=True)]

    print(f"built {len(out)} samples (bands {targets}, cap {max_tokens}, seed {seed})")
    print(
        f"tokens: mean {sum(lengths) / len(lengths):.0f}  "
        f"min {min(lengths)}  max {max(lengths)}  "
        f"p50 {sorted(lengths)[len(lengths) // 2]}  "
        f"p95 {sorted(lengths)[int(0.95 * len(lengths))]}"
    )
    print(f"problems per sample: ~{len(source) / len(out):.1f}")

    # The number that actually matters: how many samples get split at this chunk
    # size, and into how many chunks. If nothing lands above 1 chunk the A/B is
    # a no-op regardless of what the length distribution looks like.
    chunks = [(n + chunk_size - 1) // chunk_size for n in lengths]
    buckets: dict[int, int] = {}
    for c in chunks:
        buckets[c] = buckets.get(c, 0) + 1
    print(f"chunks at chunk_size={chunk_size}:")
    for c in sorted(buckets):
        print(f"  {c} chunk(s): {buckets[c]:>5} samples ({100 * buckets[c] / len(chunks):.0f}%)")
    if max(buckets) < 2:
        raise SystemExit("no sample exceeds one chunk -- pick a smaller --chunk-size or longer bands")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"wrote {output_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path, e.g. $MODEL_DIR/Qwen3-4B")
    parser.add_argument("--max-tokens", type=int, default=3072, help="Hard cap per sample; keep under --max-tokens-per-gpu")
    parser.add_argument("--chunk-size", type=int, default=512, help="Only used for the chunk-count report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build(args.input, args.output, args.tokenizer, DEFAULT_BANDS, args.max_tokens, args.seed, args.chunk_size)


if __name__ == "__main__":
    main()