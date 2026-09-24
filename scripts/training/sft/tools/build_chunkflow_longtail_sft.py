# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a deterministic, length-controlled SFT dataset for ChunkFlow tests.

The source datasets are streamed, so the command does not download their full
snapshots. Output rows use the ``question``/``answer`` schema consumed by the
Qwen3-4B ChunkFlow launch script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


BUCKETS = (
    ("short_512_2k", 512, 2_048),
    ("medium_2k_8k", 2_048, 8_192),
    ("long_8k_16k", 8_192, 16_384),
    # Leave 512 tokens below the 32K training limit as a safety margin.
    ("tail_16k_32k", 16_384, 32_257),
)
ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local Qwen3-4B checkpoint/tokenizer directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--short-dataset", default="BAAI/Infinity-Instruct")
    parser.add_argument("--short-config", default=None)
    parser.add_argument("--short-revision", default="main")
    parser.add_argument("--long-dataset", default="THUDM/LongWriter-6k")
    parser.add_argument("--long-config", default=None)
    parser.add_argument("--long-revision", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--short-count", type=int, default=700)
    parser.add_argument("--medium-count", type=int, default=200)
    parser.add_argument("--long-count", type=int, default=80)
    parser.add_argument("--tail-count", type=int, default=20)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-scan-per-source", type=int, default=200_000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _conversation_pair(messages: Any) -> tuple[str, str] | None:
    if not isinstance(messages, list):
        return None
    normalized: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = ROLE_ALIASES.get(str(message.get("role") or message.get("from") or "").lower())
        content = _text(message.get("content") if "content" in message else message.get("value"))
        if role and content:
            normalized.append((role, content))

    answer_index = next((i for i in range(len(normalized) - 1, -1, -1) if normalized[i][0] == "assistant"), -1)
    if answer_index <= 0:
        return None
    answer = normalized[answer_index][1]
    prompt_parts = [f"<{role}>\n{content}" for role, content in normalized[:answer_index]]
    prompt = "\n\n".join(prompt_parts)
    return (prompt, answer) if prompt and answer else None


def _extract_pair(row: Mapping[str, Any]) -> tuple[str, str] | None:
    for key in ("messages", "conversations"):
        pair = _conversation_pair(row.get(key))
        if pair is not None:
            return pair

    for prompt_key, answer_key in (
        ("instruction", "output"),
        ("question", "answer"),
        ("prompt", "response"),
        ("input", "output"),
    ):
        prompt = _text(row.get(prompt_key))
        answer = _text(row.get(answer_key))
        if prompt and answer:
            extra_input = _text(row.get("input")) if prompt_key == "instruction" else ""
            if extra_input:
                prompt = f"{prompt}\n\n{extra_input}"
            return prompt, answer
    return None


def _token_count(tokenizer, question: str, answer: str) -> int:
    from relax.engine.sft.dataset.chat_template import render_with_loss_mask
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content=question, learn=False),
            CanonicalMessage(role="assistant", content=answer, learn=True),
        ],
        metadata={"source_dataset": "chunkflow_builder", "row_index": 0},
    )
    token_ids, _ = render_with_loss_mask(sample, tokenizer=tokenizer)
    return len(token_ids)


def _bucket_name(token_count: int) -> str | None:
    for name, lower, upper in BUCKETS:
        if lower <= token_count < upper:
            return name
    return None


def _cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in visible)
    return cjk / len(visible)


def _resolve_revision(dataset_name: str, requested: str) -> str | None:
    if Path(dataset_name).exists():
        return None

    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_name, revision=requested).sha


def _resolve_config(dataset_name: str, requested: str | None, revision: str | None) -> str | None:
    if requested:
        return requested

    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(dataset_name, revision=revision)
    if len(configs) <= 1:
        return configs[0] if configs else None
    for preferred in ("7M", "0625", "default"):
        if preferred in configs:
            return preferred
    raise ValueError(
        f"dataset {dataset_name!r} has multiple configs {configs}; select one with the corresponding --*-config"
    )


def _stream(
    dataset_name: str,
    config: str | None,
    revision: str,
    split: str,
    seed: int,
    buffer_size: int,
) -> tuple[Iterable[dict], str]:
    from datasets import load_dataset

    resolved_revision = _resolve_revision(dataset_name, revision)
    resolved_config = _resolve_config(dataset_name, config, resolved_revision)
    load_kwargs = {"split": split, "streaming": True}
    if resolved_revision is not None:
        load_kwargs["revision"] = resolved_revision
    dataset = load_dataset(dataset_name, resolved_config, **load_kwargs)
    return dataset.shuffle(seed=seed, buffer_size=buffer_size), resolved_revision or "local"


def _collect(
    rows_by_bucket: dict[str, list[dict[str, Any]]],
    targets: Mapping[str, int],
    stream: Iterable[dict],
    source_name: str,
    tokenizer,
    max_scan: int,
    allowed_buckets: set[str],
    min_cjk_ratio: float,
) -> int:
    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > max_scan or all(len(rows_by_bucket[name]) >= targets[name] for name in allowed_buckets):
            break
        pair = _extract_pair(row)
        if pair is None:
            continue
        question, answer = pair
        if _cjk_ratio(question + answer) < min_cjk_ratio:
            continue
        token_count = _token_count(tokenizer, question, answer)
        bucket = _bucket_name(token_count)
        if bucket not in allowed_buckets or len(rows_by_bucket[bucket]) >= targets[bucket]:
            continue
        rows_by_bucket[bucket].append(
            {
                "question": question,
                "answer": answer,
                "token_count": token_count,
                "length_bucket": bucket,
                "source": source_name,
            }
        )
    return scanned


def main() -> None:
    args = _parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    targets = {
        "short_512_2k": args.short_count,
        "medium_2k_8k": args.medium_count,
        "long_8k_16k": args.long_count,
        "tail_16k_32k": args.tail_count,
    }
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    sources = (
        (
            args.short_dataset,
            args.short_config,
            args.short_revision,
            args.seed,
            {"short_512_2k", "medium_2k_8k"},
        ),
        (args.long_dataset, args.long_config, args.long_revision, args.seed + 1, set(targets)),
    )
    scanned_by_source = {}
    revisions = {}
    for dataset_name, config, revision, seed, allowed_buckets in sources:
        stream, resolved_revision = _stream(
            dataset_name,
            config,
            revision,
            args.split,
            seed,
            args.shuffle_buffer,
        )
        revisions[dataset_name] = resolved_revision
        scanned_by_source[dataset_name] = _collect(
            rows_by_bucket,
            targets,
            stream,
            dataset_name,
            tokenizer,
            args.max_scan_per_source,
            allowed_buckets,
            args.min_cjk_ratio,
        )

    missing = {name: targets[name] - len(rows) for name, rows in rows_by_bucket.items() if len(rows) < targets[name]}
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"could not fill all length buckets; missing={missing}, scanned={scanned_by_source}. "
            "Increase --max-scan-per-source or pass --allow-partial."
        )

    rows = [row for name, _, _ in BUCKETS for row in rows_by_bucket[name]]
    random.Random(args.seed).shuffle(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    report = {
        "output": str(output),
        "seed": args.seed,
        "min_cjk_ratio": args.min_cjk_ratio,
        "total_rows": len(rows),
        "bucket_counts": {name: len(rows_by_bucket[name]) for name in targets},
        "targets": targets,
        "scanned_by_source": scanned_by_source,
        "source_revisions": revisions,
        "output_sha256": output_sha256,
        "min_tokens": min((row["token_count"] for row in rows), default=0),
        "max_tokens": max((row["token_count"] for row in rows), default=0),
    }
    report_path = output.with_suffix(".stats.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
