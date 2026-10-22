# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare deduplicated, length-stratified public SFT conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def conversation(row: dict) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    result = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            return None
        result.append({"role": role, "content": content})
    if result[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in result):
        return None
    turns = result[1:] if result[0]["role"] == "system" else result
    if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(turns)):
        return None
    return result


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], int):
        raise TypeError("Expected a nonempty flat token-ID list from apply_chat_template(return_dict=False)")
    return len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--max-scan", type=int, default=200000)
    parser.add_argument("--max-length", type=int, default=32512)
    args = parser.parse_args()
    if args.rows < 1000 or args.max_length <= 16384 or args.max_scan < args.rows:
        parser.error("require rows >= 1000, max-length > 16384 and max-scan >= rows")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = load_dataset(args.dataset, split="train", revision=revision, streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=10000)
    targets = [int(args.rows * 0.70), int(args.rows * 0.25)]
    targets.append(args.rows - sum(targets))
    ultralong_target = max(1, args.rows // 200)
    buckets: list[list[dict]] = [[], [], []]
    answers, identities = set(), set()
    rejected, observed = Counter(), Counter()
    maximum_length = 0
    scanned = 0
    for scanned, row in enumerate(stream, 1):
        messages = conversation(row)
        if messages is None:
            rejected["invalid_conversation"] += 1
        else:
            answer = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
            answer_hash = hashlib.sha256(" ".join(answer.split()).encode()).hexdigest()
            identity = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            length = token_count(tokenizer, messages)
            maximum_length = max(maximum_length, length)
            bucket = 0 if length <= 4096 else 1 if length <= 8192 else 2
            observed[bucket] += 1
            if scanned <= 3:
                print(f"sample={scanned}; chat_tokens={length}", flush=True)
            if len(tokenizer.encode(answer, add_special_tokens=False)) < 64:
                rejected["short_answer"] += 1
            elif length > args.max_length:
                rejected["oversize"] += 1
            elif answer_hash in answers or identity in identities:
                rejected["duplicate"] += 1
            elif (
                bucket == 2
                and length <= 16384
                and sum(r["token_count"] <= 16384 for r in buckets[2]) >= targets[2] - ultralong_target
            ):
                rejected["reserved_for_ultralong"] += 1
            elif len(buckets[bucket]) < targets[bucket]:
                answers.add(answer_hash)
                identities.add(identity)
                buckets[bucket].append(
                    {
                        "messages": messages,
                        "token_count": length,
                        "id": identity,
                        "source": row.get("source", args.dataset),
                    }
                )
        if all(len(b) == target for b, target in zip(buckets, targets, strict=True)):
            break
        if scanned % 10000 == 0:
            print(
                f"scanned={scanned}; bucket counts={[len(b) for b in buckets]}; targets={targets}; "
                f"observed={[observed[i] for i in range(3)]}; max_tokens={maximum_length}",
                flush=True,
            )
        if scanned >= args.max_scan:
            break
    counts = [len(b) for b in buckets]
    if counts != targets:
        raise SystemExit(
            f"Insufficient natural-length samples: counts={counts}, targets={targets}, scanned={scanned}; "
            f"observed={[observed[i] for i in range(3)]}, max_tokens={maximum_length}, rejected={dict(rejected)}"
        )
    rows = [row for bucket in buckets for row in bucket]
    random.Random(args.seed).shuffle(rows)
    validation = max(1, len(rows) // 20)
    diagnostic = [row for row in rows[validation:] if row["token_count"] > 8448]
    if len(diagnostic) < 24:
        raise SystemExit(f"Need at least 24 real long samples for 3-step GBS8 diagnostics; found {len(diagnostic)}")
    from datasets import Dataset

    args.output.mkdir(parents=True)
    Dataset.from_list(rows[validation:]).to_parquet(str(args.output / "train.parquet"))
    Dataset.from_list(rows[:validation]).to_parquet(str(args.output / "validation.parquet"))
    Dataset.from_list(diagnostic).to_parquet(str(args.output / "diagnostic.parquet"))
    report = {
        "dataset": args.dataset,
        "revision": revision,
        "seed": args.seed,
        "scanned": scanned,
        "observed_bucket_counts": [observed[i] for i in range(3)],
        "maximum_observed_tokens": maximum_length,
        "bucket_bounds": [4096, 8192, args.max_length],
        "bucket_counts_before_split": counts,
        "train_rows": len(rows) - validation,
        "validation_rows": validation,
        "unique_answers": len(answers),
        "diagnostic_rows": len(diagnostic),
        "diagnostic_subset": "training rows >8448 tokens, reserved for cross-chunk correctness checks, not timing claims",
        "over_16384": sum(row["token_count"] > 16384 for row in rows),
        "rejected": dict(rejected),
        "tokenizer": args.tokenizer,
        "token_counts": "chat-template preflight counts; actual training masking/tokenization is authoritative",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
