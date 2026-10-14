# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Theory-side benefit model for ChunkFlow step scheduling.

ChunkFlow's pitch is "remove pipeline bubbles". Its own plan telemetry already
turns that pitch into numbers -- ``equal_unit_fill_drain_{baseline,chunkflow,
delta}_proxy`` -- and also prices what chunkflow pays for the smaller bubble
(fragmentation, recompute, launch count, rectified attention). This tool runs
that telemetry on CPU, on a sequence-length distribution, so the theory side of
an A/B run can be computed before -- or without -- a GPU.

It reuses ``relax/backends/megatron/chunkflow/{scheduler,units,telemetry}.py``
directly (all pure Python, no torch) so a number printed here and the
``train/chunkflow/<key>`` metric a real run logs are the same quantity computed
the same way.

How the model maps onto one optimizer step::

    step samples --first-fit--> micro-batches (M of them, capacity =
    max_tokens_per_gpu) --build_chunks(chunk_size)--> chunks
    --build_micro_batches(policy)--> execution units (U total)

    baseline bubble  = 1 + (pp - 1) / M      # standard 1F1B, M items
    chunkflow bubble = 1 + (pp - 1) / U      # weighted by padded rows
    theory speedup   = baseline / chunkflow

The bubble saving is an upper bound: it assumes a unit's duration is
proportional to its token rows. ``launch_amplification`` (units per
micro-batch), the rectified-attention ratio and the recompute multipliers are
printed next to it precisely so the gap to a *measured* speedup can be
attributed.

Usage::

    python scripts/training/sft/tools/chunkflow_benefit_model.py --synthetic
    python scripts/training/sft/tools/chunkflow_benefit_model.py \
        --lengths lengths.json --pp-size 4 --gbs 1,4,16,64 \
        --chunk-sizes 512,1024,2048,4096 --json benefit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Sequence


# Buckets and default counts mirror scripts/training/sft/tools/build_chunkflow_longtail_sft.py
# so the synthetic distribution matches the dataset the SFT A/B script consumes.
BUCKETS = (
    ("short_0_2k", 0, 2_048, 450),
    ("medium_2k_8k", 2_048, 8_192, 25),
    ("long_8k_16k", 8_192, 16_384, 20),
    ("tail_16k_32k", 16_384, 32_257, 5),
)

# --keep-last-k 0 means "keep everything" upstream (see schedules.py); mirror it.
NO_RECOMPUTE = 1 << 30


# ---------------------------------------------------------------------------
# Loading chunkflow's pure-Python modules without importing the relax package
# ---------------------------------------------------------------------------


def _load_chunkflow(repo_root: Path) -> types.SimpleNamespace:
    """Load scheduler/units/telemetry by file path.

    Importing ``relax.backends.megatron.chunkflow`` normally pulls torch and
    Megatron through the package ``__init__`` chain; these three modules have
    no such dependency, so a synthetic package namespace lets them be loaded
    standalone and keeps this tool runnable anywhere.
    """
    chunkflow_dir = repo_root / "relax" / "backends" / "megatron" / "chunkflow"
    if not chunkflow_dir.is_dir():
        raise SystemExit(f"chunkflow package not found at {chunkflow_dir}; set --repo-root")

    package = "relax.backends.megatron.chunkflow"
    for name in ("relax", "relax.backends", "relax.backends.megatron", package):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module

    def load(leaf: str):
        full = f"{package}.{leaf}"
        spec = importlib.util.spec_from_file_location(full, chunkflow_dir / f"{leaf}.py")
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {chunkflow_dir / f'{leaf}.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    return types.SimpleNamespace(
        scheduler=load("scheduler"),
        units=load("units"),
        telemetry=load("telemetry"),
    )


# ---------------------------------------------------------------------------
# Faithful mirrors of the two non-chunkflow helpers the model needs
# ---------------------------------------------------------------------------


def _first_fit_partitions(total_lengths: Sequence[int], max_tokens_per_gpu: int) -> list[list[int]]:
    """Pack indices in input order; an oversized sample gets its own bin.

    Mirrors ``relax.utils.data.data.get_first_fit_partitions``, which is what
    decides M (the micro-batch count) in a real run. Re-implemented here only
    because that module imports torch; ``_assert_packer_matches`` checks the
    two agree whenever relax happens to be importable.
    """
    batches: list[int] = []
    partitions: list[list[int]] = []
    for index, length in enumerate(total_lengths):
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                partitions[i].append(index)
                break
        else:
            batches.append(length)
            partitions.append([index])
    return partitions


def _assert_packer_matches(total_lengths: Sequence[int], max_tokens_per_gpu: int, local: list[list[int]]) -> None:
    """Cross-check the mirror against the real packer when relax is importable.

    A mismatch means this tool's M would silently diverge from a real run, so
    it is raised rather than warned. Import failures (no torch, missing
    optional deps) are not the mirror's problem and are skipped.
    """
    try:
        from relax.utils.data.data import get_first_fit_partitions
    except Exception:
        return
    if get_first_fit_partitions(list(total_lengths), max_tokens_per_gpu) != local:
        raise RuntimeError(
            "_first_fit_partitions disagrees with relax.utils.data.data.get_first_fit_partitions; "
            "the upstream packer changed -- resync this mirror before trusting M."
        )


def _dependency_delay(units, chunks, keep_last_k: int, units_module) -> int:
    """``C`` from ``pipeline_schedules._pipeline_plan``.

    Extra warmup covering the first dependency group's completion. Only the
    ``warmup_*`` telemetry depends on it -- not the fill/drain proxies -- but
    computing it keeps this model's plan identical to the runtime's.
    """
    standalone = [u.unit_id for u in units if not u.is_dependent]
    dependent = sorted((u for u in units if u.is_dependent), key=lambda u: u.round)
    fwd_seq = standalone + [u.unit_id for u in dependent]
    bwd_seq = units_module.make_unit_bwd_order(units, fwd_seq)

    by_id = {u.unit_id: u for u in units}
    fwd_pos = {u: i for i, u in enumerate(fwd_seq)}
    done_at: dict[int, int] = {}
    for u in units:
        for g in u.groups:
            done_at[g] = max(done_at.get(g, -1), fwd_pos[u.unit_id])

    def ready(unit_id: int) -> int:
        unit = by_id[unit_id]
        if not unit.groups:
            return fwd_pos[unit_id]
        return max(done_at[g] for g in unit.groups)

    return max((ready(u) - i for i, u in enumerate(bwd_seq)), default=0)


# ---------------------------------------------------------------------------
# Length distribution
# ---------------------------------------------------------------------------


def synthetic_lengths(seed: int = 1234) -> list[int]:
    """A deterministic long-tail pool matching the builder's bucket counts.

    Each bucket's samples land in the middle half of its token range (the
    builder's own fitter), then the pool is shuffled so a step prefix is a
    representative mix rather than "all short".
    """
    rng = random.Random(seed)
    lengths: list[int] = []
    for _name, lower, upper, count in BUCKETS:
        lo = lower + (upper - lower) // 4
        hi = upper - (upper - lower) // 4 - 1
        lengths.extend(rng.randint(lo, hi) for _ in range(count))
    rng.shuffle(lengths)
    return lengths


def load_lengths(path: Path) -> list[int]:
    """Read a pool: a JSON int array, or JSONL rows carrying a length."""
    text = path.read_text()
    if path.suffix == ".jsonl":
        lengths = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lengths.append(_row_length(json.loads(line)))
        return lengths
    data = json.loads(text)
    if isinstance(data, list) and all(isinstance(x, int) for x in data):
        return list(data)
    if isinstance(data, list):
        return [_row_length(row) for row in data]
    raise SystemExit(f"{path}: expected a JSON int array or a list of rows")


def _row_length(row: Any) -> int:
    if isinstance(row, int):
        return row
    if isinstance(row, dict):
        for key in ("total_lengths", "token_count", "length", "num_tokens"):
            if key in row:
                return int(row[key])
    raise SystemExit(f"cannot read a token length from row: {row!r}")


# ---------------------------------------------------------------------------
# One step's plan -> telemetry
# ---------------------------------------------------------------------------


def step_metrics(
    samples: Sequence[int],
    chunk_size: int,
    max_tokens_per_gpu: int,
    pp_size: int,
    policy: str,
    keep_last_k: int,
    cf,
) -> dict[str, Any]:
    """Aggregate plan telemetry for one optimizer step at ``chunk_size``.

    ``chunk_size`` 0 means the baseline arm, which does not run chunkflow at
    all: its schedule is plain 1F1B over M micro-batches, so only the
    baseline fill/drain proxy is meaningful and the chunkflow side is left
    equal to it (speedup 1.0 by construction).
    """
    partitions = _first_fit_partitions(samples, max_tokens_per_gpu)
    _assert_packer_matches(samples, max_tokens_per_gpu, partitions)

    if chunk_size <= 0:
        # The baseline arm does not run chunkflow at all: plain 1F1B over M
        # micro-batches, one pipeline pass. Only its fill/drain matters.
        micro_batches = len(partitions)
        bubble = 1.0 + (pp_size - 1) / micro_batches
        return {
            "num_microbatches": micro_batches,
            "num_units": micro_batches,
            "equal_unit_fill_drain_baseline_proxy": bubble,
            "equal_unit_fill_drain_chunkflow_proxy": bubble,
            "equal_unit_fill_drain_delta_proxy": 0.0,
            "fragmentation_amplification": 1.0,
            "token_forward_recompute_multiplier": 1.0,
            "attention_forward_recompute_multiplier": 1.0,
            "launch_amplification": 1.0,
            "logical_attention_pairs": 0,
            "kernel_rect_pairs": 0,
        }

    effective_keep = keep_last_k or NO_RECOMPUTE
    step = cf.telemetry.StepPlanTelemetry()
    for members in partitions:
        lengths = [int(samples[i]) for i in members]
        chunks = cf.scheduler.build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
        units = cf.units.build_micro_batches(chunks, policy)
        delay = _dependency_delay(units, chunks, effective_keep, cf.units)
        warmup = [min(delay + (pp_size - stage - 1), len(units)) for stage in range(pp_size)]
        step.add(
            cf.telemetry.summarize_plan(
                lengths,
                chunks,
                units,
                effective_keep,
                pp_size,
                warmup,
                source_rows=sum(lengths),
            )
        )

    return step.to_metrics(pp_size)


def _oversized_bins(samples: Sequence[int], capacity: int) -> int:
    """Micro-batches holding more tokens than the bin capacity.

    A sample longer than the budget cannot share a bin, so it becomes its own
    oversized micro-batch. Legal (nothing rejects it), but it changes the
    memory profile and is worth knowing before a real run.
    """
    return sum(
        1 for members in _first_fit_partitions(samples, capacity) if sum(samples[i] for i in members) > capacity
    )


def _derived(metrics: dict[str, Any]) -> dict[str, Any]:
    """The few ratios the table reports, derived from the telemetry dict."""
    base = metrics["equal_unit_fill_drain_baseline_proxy"]
    chunk = metrics["equal_unit_fill_drain_chunkflow_proxy"]
    logical = metrics.get("logical_attention_pairs", 0)
    rect = metrics.get("kernel_rect_pairs", 0)
    return {
        "theory_x": base / chunk if chunk else 0.0,
        "rect_attn": rect / logical if logical else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _int_list(value: str) -> list[int]:
    try:
        return [int(part) for part in value.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {value!r}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="Use the built-in long-tail distribution")
    source.add_argument("--lengths", type=Path, help="JSON int array or JSONL rows with a token length")

    parser.add_argument("--pp-size", type=int, default=4, help="Pipeline stages (default: 4)")
    parser.add_argument("--gbs", type=_int_list, default=[64], help="Samples per step to sweep")
    parser.add_argument(
        "--chunk-sizes", type=_int_list, default=[512, 1024, 2048, 4096], help="Chunk sizes; 0 = baseline"
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=_int_list,
        default=[4_096, 8_192, 16_384, 32_768],
        help="Micro-batch bin capacity to sweep. This -- not --gbs -- is the "
        "knob that moves M: a smaller capacity packs fewer samples per micro-batch.",
    )
    parser.add_argument("--policy", default="staggered", choices=["staggered", "lockstep"])
    parser.add_argument("--keep-last-k", type=int, default=0, help="0 = keep every chunk (no recompute)")
    parser.add_argument("--seed", type=int, default=1234, help="Synthetic distribution seed")
    parser.add_argument(
        "--sort",
        choices=["none", "desc", "asc"],
        default="none",
        help="Order the pool before taking step prefixes. 'desc' models a step of the longest samples "
        "(the case chunkflow targets); 'none' keeps the generated (shuffled) order.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Relax checkout root (default: inferred)")
    parser.add_argument("--json", type=Path, default=None, help="Also write the full table as JSON")
    return parser.parse_args()


_COLUMNS = (
    ("gbs", 4, "{:>4}"),
    ("capacity", 8, "{:>8}"),
    ("M", 4, "{:>4}"),
    ("chunk", 6, "{:>6}"),
    ("U", 6, "{:>6}"),
    ("bubble_base", 11, "{:>11.3f}"),
    ("bubble_cf", 9, "{:>9.3f}"),
    ("theory_x", 8, "{:>8.3f}"),
    ("frag", 6, "{:>6.3f}"),
    ("tok_recomp", 10, "{:>10.3f}"),
    ("attn_recomp", 11, "{:>11.3f}"),
    ("rect_attn", 9, "{:>9.3f}"),
    ("launch", 7, "{:>7.1f}"),
)


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = " ".join(f"{name:>{width}}" for name, width, _ in _COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(fmt.format(row[key]) for key, _width, fmt in _COLUMNS))


def main() -> None:
    args = _parse_args()
    if not args.synthetic and args.lengths is None:
        raise SystemExit("pick a distribution: --synthetic or --lengths <file>")

    repo_root = args.repo_root or Path(__file__).resolve().parents[4]
    cf = _load_chunkflow(repo_root)

    pool = synthetic_lengths(args.seed) if args.synthetic else load_lengths(args.lengths)
    if args.sort == "desc":
        pool = sorted(pool, reverse=True)
    elif args.sort == "asc":
        pool = sorted(pool)
    print(f"length pool: {len(pool)} samples, min={min(pool)}, max={max(pool)}, "
          f"mean={sum(pool) / len(pool):.0f}  (pp={args.pp_size}, policy={args.policy})")
    print()

    rows: list[dict[str, Any]] = []
    oversized: list[tuple[int, int, int]] = []
    for gbs in args.gbs:
        if gbs > len(pool):
            print(f"skip gbs={gbs}: pool only has {len(pool)} samples", file=sys.stderr)
            continue
        samples = pool[:gbs]
        for capacity in args.max_tokens_per_gpu:
            n_oversized = _oversized_bins(samples, capacity)
            if n_oversized:
                oversized.append((gbs, capacity, n_oversized))
            for chunk_size in args.chunk_sizes:
                metrics = step_metrics(
                    samples,
                    chunk_size,
                    capacity,
                    args.pp_size,
                    args.policy,
                    args.keep_last_k,
                    cf,
                )
                rows.append(
                    {
                        "gbs": gbs,
                        "capacity": capacity,
                        "M": metrics["num_microbatches"],
                        "chunk": chunk_size,
                        "U": metrics["num_units"],
                        "bubble_base": metrics["equal_unit_fill_drain_baseline_proxy"],
                        "bubble_cf": metrics["equal_unit_fill_drain_chunkflow_proxy"],
                        "frag": metrics["fragmentation_amplification"],
                        "tok_recomp": metrics["token_forward_recompute_multiplier"],
                        "attn_recomp": metrics["attention_forward_recompute_multiplier"],
                        "launch": metrics["launch_amplification"],
                        **_derived(metrics),
                    }
                )

    _print_table(rows)
    print()
    print("M         = micro-batches per step (from --gbs and --max-tokens-per-gpu; see the capacity column)")
    print("theory_x  = baseline_bubble / chunkflow_bubble (upper bound on step speedup)")
    print("rect_attn = kernel_rect_pairs / logical_attention_pairs (chunked attention FLOPs inflation)")
    if oversized:
        print()
        print("WARNING: capacity below the longest sample makes an oversized micro-batch:", file=sys.stderr)
        for gbs, capacity, count in oversized:
            print(f"  gbs={gbs} capacity={capacity}: {count} oversized bin(s)", file=sys.stderr)

    if args.json is not None:
        args.json.write_text(json.dumps({"config": vars(args) | {"lengths": None, "json": None}, "rows": rows},
                                        ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
