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
from itertools import chain
from pathlib import Path
from typing import Any


BUCKETS = (
    ("short_0_2k", 0, 2_048),
    ("medium_2k_8k", 2_048, 8_192),
    ("long_8k_16k", 8_192, 16_384),
    # Leave 512 tokens below the 32K training limit as a safety margin.
    ("tail_16k_32k", 16_384, 32_257),
)
ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
SYNTHETIC_CONTEXTS = (
    "在分布式训练系统中，流水线并行会把模型层划分到不同设备，微批次依次经过各个阶段。",
    "长上下文训练的主要开销来自注意力计算、激活保存、设备通信以及流水线空泡。",
    "为了保证实验可复现，需要固定模型参数、样本顺序、随机种子和每一步的有效令牌数量。",
    "分块计算把一个长序列拆成多个具有依赖关系的片段，并保存后续计算所需的状态。",
    "当计算任务过于细碎时，算子启动和点对点通信的固定成本可能超过减少空泡带来的收益。",
    "性能分析应当分别记录有效计算、填充、重计算、通信和状态存储，而不是只观察总耗时。",
)
SYNTHETIC_ANSWERS = (
    "综合以上信息，应根据序列长度和流水线规模选择合适的分块大小，并验证数值结果。",
    "结论是收益取决于减少的流水线空泡能否覆盖额外算子、通信与重计算成本。",
    "因此需要逐步对比基线与分块方案，并用相同输入检查损失、梯度和执行时间。",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local Qwen3-4B checkpoint/tokenizer directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--local-json", default=None, help="Local JSON/JSONL source for the 0-8K buckets")
    parser.add_argument(
        "--local-long-data",
        default=None,
        help="Local JSON/JSONL/Parquet file or directory containing real samples for the 2K-32K buckets",
    )
    parser.add_argument("--short-dataset", default="BAAI/Infinity-Instruct")
    parser.add_argument("--short-config", default=None)
    parser.add_argument("--short-revision", default="main")
    parser.add_argument("--long-dataset", default="THUDM/LongWriter-6k")
    parser.add_argument("--long-config", default=None)
    parser.add_argument("--long-revision", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--short-count", type=int, default=450)
    parser.add_argument("--medium-count", type=int, default=25)
    parser.add_argument("--long-count", type=int, default=20)
    parser.add_argument("--tail-count", type=int, default=5)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-scan-per-source", type=int, default=200_000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true", help="Generate deterministic Chinese data offline")
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


def _build_synthetic_rows(
    targets: Mapping[str, int],
    tokenizer,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic Chinese samples inside every token bucket."""
    rng = random.Random(seed)
    ranges = {name: (lower, upper) for name, lower, upper in BUCKETS}
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}

    for bucket, count in targets.items():
        lower, upper = ranges[bucket]
        for index in range(count):
            target = rng.randint(lower + (upper - lower) // 4, upper - (upper - lower) // 4 - 1)
            context = SYNTHETIC_CONTEXTS[(index + len(bucket)) % len(SYNTHETIC_CONTEXTS)]
            answer_unit = SYNTHETIC_ANSWERS[index % len(SYNTHETIC_ANSWERS)]
            repeats = max(1, target // 32)

            for _ in range(8):
                question = f"样本编号 {bucket}-{index}。请阅读材料并总结关键结论。\n" + context * repeats
                answer = answer_unit * max(1, repeats // 10)
                token_count = _token_count(tokenizer, question, answer)
                if lower <= token_count < upper:
                    break
                repeats = max(1, int(repeats * target / max(token_count, 1)))
            else:
                raise RuntimeError(f"failed to fit synthetic sample {bucket}-{index} into [{lower}, {upper}) tokens")

            rows_by_bucket[bucket].append(
                {
                    "question": question,
                    "answer": answer,
                    "token_count": token_count,
                    "length_bucket": bucket,
                    "source": "synthetic-chinese-v1",
                }
            )
    return rows_by_bucket


def _resolve_revision(dataset_name: str, requested: str, hf_token: str | None) -> str | None:
    if Path(dataset_name).exists():
        return None

    from huggingface_hub import HfApi

    return HfApi(token=hf_token).dataset_info(dataset_name, revision=requested).sha


def _resolve_config(
    dataset_name: str,
    requested: str | None,
    revision: str | None,
    hf_token: str | None,
) -> str | None:
    if requested:
        return requested

    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(dataset_name, revision=revision, token=hf_token)
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
    hf_token: str | None,
) -> tuple[Iterable[dict], str]:
    from datasets import load_dataset

    resolved_revision = _resolve_revision(dataset_name, revision, hf_token)
    resolved_config = _resolve_config(dataset_name, config, resolved_revision, hf_token)
    load_kwargs = {"split": split, "streaming": True}
    if resolved_revision is not None:
        load_kwargs["revision"] = resolved_revision
    if hf_token is not None:
        load_kwargs["token"] = hf_token
    dataset = load_dataset(dataset_name, resolved_config, **load_kwargs)
    return dataset.shuffle(seed=seed, buffer_size=buffer_size), resolved_revision or "local"


def _local_data_files(path: Path) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(file for file in path.rglob("*") if file.suffix.lower() in {".json", ".jsonl", ".parquet"})
    else:
        raise FileNotFoundError(f"local source does not exist: {path}")
    if not files:
        raise FileNotFoundError(f"no JSON, JSONL, or Parquet files found under: {path}")
    return files


def _local_stream(path: Path, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    streams = []
    files = _local_data_files(path)
    for index, suffix in enumerate((".json", ".jsonl", ".parquet")):
        data_files = [str(file) for file in files if file.suffix.lower() == suffix]
        if not data_files:
            continue
        loader = "parquet" if suffix == ".parquet" else "json"
        streams.append(load_dataset(loader, data_files=data_files, split="train").shuffle(seed=seed + index))
    return chain.from_iterable(streams)


def _local_source_revision(path: Path) -> str:
    digest = hashlib.sha256()
    for file in _local_data_files(path):
        digest.update(str(file.relative_to(path.parent if path.is_file() else path)).encode())
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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
        "short_0_2k": args.short_count,
        "medium_2k_8k": args.medium_count,
        "long_8k_16k": args.long_count,
        "tail_16k_32k": args.tail_count,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.local_long_data and not args.local_json:
        raise ValueError("--local-long-data requires --local-json so both short and long buckets use local real data")
    if args.synthetic_only:
        rows_by_bucket = _build_synthetic_rows(targets, tokenizer, args.seed)
        scanned_by_source = {"synthetic-chinese-v1": sum(targets.values())}
        revisions = {"synthetic-chinese-v1": "v1"}
    elif args.local_json:
        short_path = Path(args.local_json).expanduser().resolve()
        rows_by_bucket = {name: [] for name in targets}
        short_scanned = _collect(
            rows_by_bucket,
            targets,
            _local_stream(short_path, args.seed),
            str(short_path),
            tokenizer,
            args.max_scan_per_source,
            {"short_0_2k", "medium_2k_8k"},
            args.min_cjk_ratio,
        )
        scanned_by_source = {str(short_path): short_scanned}
        revisions = {str(short_path): _local_source_revision(short_path)}
        if args.local_long_data:
            long_path = Path(args.local_long_data).expanduser().resolve()
            long_scanned = _collect(
                rows_by_bucket,
                targets,
                _local_stream(long_path, args.seed + 1),
                str(long_path),
                tokenizer,
                args.max_scan_per_source,
                {"medium_2k_8k", "long_8k_16k", "tail_16k_32k"},
                0.0,
            )
            scanned_by_source[str(long_path)] = long_scanned
            revisions[str(long_path)] = _local_source_revision(long_path)
    else:
        from huggingface_hub import get_token

        rows_by_bucket = {name: [] for name in targets}
        hf_token = get_token()
        sources = (
            (
                args.short_dataset,
                args.short_config,
                args.short_revision,
                args.seed,
                {"short_0_2k", "medium_2k_8k"},
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
                hf_token,
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
