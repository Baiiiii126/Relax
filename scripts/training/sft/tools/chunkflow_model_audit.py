# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU audit of model direction and its differences from the paper's toy
example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chunkflow_benefit_model import _load_chunkflow, step_metrics


def audit(repo: Path) -> dict[str, Any]:
    cf = _load_chunkflow(repo)
    model = cf.schedule_model
    # Section 3 / Figures 2, 6, 7: sequence lengths [1, 1, 2, 4] * Unit.
    # Scale Unit to 128 tokens to eliminate alignment rounding in this toy test.
    lengths = [128, 128, 256, 512]
    order = tuple(range(4))
    baseline = model.simulate([model.ModelPlan(dict(enumerate(lengths)), order, order, frozenset(), (3, 2, 1, 0))], 4)
    examples = []
    for size, keep, paper_bubble in ((256, 1, 0.541), (256, 2, 0.478), (512, 1, 0.60)):
        chunks = cf.scheduler.build_chunks(lengths, size, repack_tail=False)
        units = cf.units.build_micro_batches(chunks, "staggered")
        prediction = model.simulate([model.make_model_plan(units, chunks, keep, 4)], 4)
        examples.append(
            {
                "chunk_units": size / 128,
                "keep_last_k": keep,
                "model_units": len(units),
                "model_speedup": baseline["makespan"] / prediction["makespan"],
                "model_bubble_fraction": prediction["bubble_fraction"],
                "paper_reported_bubble_fraction": paper_bubble,
                "bubble_matches_to_0_002": abs(prediction["bubble_fraction"] - paper_bubble) <= 0.002,
            }
        )
    controls = []
    for name, samples, chunk, cap in (
        ("single_long_parent", [28160], 2048, 32768),
        ("many_unsplit_parents", [1024] * 8, 2048, 1024),
    ):
        metrics = step_metrics(samples, chunk, cap, 4, "staggered", 0, cf)
        controls.append(
            {
                "name": name,
                "lengths": samples,
                "chunk": chunk,
                "capacity": cap,
                "model_speedup": metrics["schedule_model_speedup"],
            }
        )
    return {
        "source": "https://arxiv.org/html/2503.02356v3",
        "paper_sections": "3, 4.3, 5; Figures 2, 6, 7",
        "note": "Same length example, current unit builder and blocking executor model. Not a paper oracle. "
        "Schedule orders, communication semantics and recomputation placement need separate conformance checks.",
        "baseline_model_bubble": baseline["bubble_fraction"],
        "baseline_paper_bubble": 0.5714,
        "paper_length_examples": examples,
        "runtime_model_controls": controls,
        "conclusion": "Model can predict negative speedup. Toy-example numerical differences prevent claiming "
        "exact paper reproduction; they do not establish an error in the paper.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[4])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
