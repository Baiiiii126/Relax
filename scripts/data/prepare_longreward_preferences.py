# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert LongReward-10k rows to Relax's DPO preference schema."""

import argparse
import json
from pathlib import Path
from typing import Any


def _build_row(row: dict[str, Any], *, index: int, separator: str) -> dict[str, Any] | None:
    context = row.get("context") or ""
    query = row.get("query") or ""
    chosen = row.get("win_response") or ""
    rejected = row.get("lose_response") or ""
    if not isinstance(context, str) or not isinstance(query, str):
        return None
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    prompt_text = f"{context}{separator}{query}" if context else query
    if not prompt_text.strip() or not chosen.strip() or not rejected.strip():
        return None
    if chosen == rejected:
        return None
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
        "metadata": {
            "source_dataset": "LongReward-10k",
            "source_index": index,
            "source_idx": row.get("idx"),
            "prompt_chars": len(prompt_text),
            "chosen_chars": len(chosen),
            "rejected_chars": len(rejected),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="source dpo-*.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="destination preference jsonl")
    parser.add_argument("--separator", default="\n\n", help="inserted between context and query")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 keeps every valid pair")
    args = parser.parse_args()

    kept = 0
    skipped = 0
    max_prompt_chars = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8", newline="\n") as sink:
        for index, line in enumerate(source):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            built = _build_row(row, index=index, separator=args.separator)
            if built is None:
                skipped += 1
                continue
            max_prompt_chars = max(max_prompt_chars, built["metadata"]["prompt_chars"])
            sink.write(json.dumps(built, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            kept += 1
            if args.max_pairs and kept >= args.max_pairs:
                break

    manifest = {
        "schema_version": 1,
        "source": {"path": str(args.input), "dataset": "LongReward-10k"},
        "output": {"path": str(args.output), "pairs": kept, "skipped": skipped},
        "max_prompt_chars": max_prompt_chars,
        "separator": args.separator,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
