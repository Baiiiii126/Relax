#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Compare a saved Megatron checkpoint against the original HF weights.

Checks whether norm weights, norm biases, QK norms, linear biases, and other
parameter groups are actually being updated during training.

Usage (run on the training pod after training completes):

  # Convert latest checkpoint to HF, then compare
  python scripts/tools/compare_ckpt_weights.py \\
      --ckpt-dir /path/to/save/Qwen3-VL-30B-A3B-Thinking-Checkpoint \\
      --original-hf-dir /path/to/Qwen3-VL-30B-A3B-Thinking

  # Skip conversion if already converted
  python scripts/tools/compare_ckpt_weights.py \\
      --ckpt-dir /path/to/save/Qwen3-VL-30B-A3B-Thinking-Checkpoint \\
      --original-hf-dir /path/to/Qwen3-VL-30B-A3B-Thinking \\
      --no-convert --converted-hf-dir /path/to/converted_checkpoint

  # Compare a specific iteration
  python scripts/tools/compare_ckpt_weights.py \\
      --ckpt-dir /path/to/save/Qwen3-VL-30B-A3B-Thinking-Checkpoint \\
      --original-hf-dir /path/to/Qwen3-VL-30B-A3B-Thinking \\
      --iter 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    iter_dirs = sorted(d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    return iter_dirs[-1] if iter_dirs else None


def convert_to_hf(megatron_iter_dir: Path, hf_output_dir: Path, original_hf_dir: Path) -> None:
    """Run convert_torch_dist_to_hf_bridge.py to produce an HF-format checkpoint."""
    script = Path(__file__).parent / "convert_torch_dist_to_hf_bridge.py"
    if not script.exists():
        print(f"ERROR: Conversion script not found at {script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script),
        "--input-dir", str(megatron_iter_dir),
        "--output-dir", str(hf_output_dir),
        "--origin-hf-dir", str(original_hf_dir),
        "--force",
    ]
    print("Converting checkpoint to HF format...")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Saved to: {hf_output_dir}")


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all weights from a safetensors model directory."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    weights: dict[str, torch.Tensor] = {}
    index_file = model_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        loaded_files: set[str] = set()
        for param_name, fname in index["weight_map"].items():
            fpath = model_dir / fname
            if fname not in loaded_files:
                loaded_files.add(fname)
                with safe_open(str(fpath), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        weights[key] = sf.get_tensor(key)
    else:
        for sf_file in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(sf_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weights[key] = sf.get_tensor(key)

    if not weights:
        print(f"ERROR: No safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    return weights


_NORM_PATTERNS = [
    "norm.weight",
    "layernorm.weight",
    "layer_norm.weight",
    "rmsnorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "final_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "ln_q.weight",
    "ln_k.weight",
    "patch_embed.norm.weight",
    "merger.ln_q.weight",
    "vision_model.encoder.layers",
]

_NORM_BIAS_PATTERNS = [
    "norm.bias",
    "layernorm.bias",
    "layer_norm.bias",
    "input_layernorm.bias",
    "post_attention_layernorm.bias",
    "final_layernorm.bias",
    "q_norm.bias",
    "k_norm.bias",
]


def classify_param(name: str) -> str:
    lower = name.lower()
    if any(p in lower for p in _NORM_PATTERNS):
        return "norm_weight"
    if any(p in lower for p in _NORM_BIAS_PATTERNS):
        return "norm_bias"
    if lower.endswith(".bias") and "norm" not in lower:
        return "linear_bias"
    if "embed_tokens" in lower or "lm_head" in lower:
        return "embedding"
    return "weight"


def compare_weights(
    orig: dict[str, torch.Tensor],
    ckpt: dict[str, torch.Tensor],
    threshold: float = 1e-7,
) -> dict[str, dict]:
    categories: dict[str, dict] = {}

    for name, orig_t in orig.items():
        ptype = classify_param(name)
        if ptype not in categories:
            categories[ptype] = {"updated": [], "frozen": [], "missing": []}

        if name not in ckpt:
            categories[ptype]["missing"].append(name)
            continue

        ckpt_t = ckpt[name].float()
        orig_f = orig_t.float()
        diff = (orig_f - ckpt_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        entry = (name, max_diff, mean_diff)
        if max_diff > threshold:
            categories[ptype]["updated"].append(entry)
        else:
            categories[ptype]["frozen"].append(entry)

    return categories


_TYPE_LABELS = {
    "norm_weight": "Norm Weights (RMSNorm / LayerNorm / QK-Norm)",
    "norm_bias":   "Norm Biases",
    "linear_bias": "Linear Biases",
    "embedding":   "Embeddings (embed_tokens / lm_head)",
    "weight":      "Other Weights (attention / MLP projections)",
}

_PRINT_ORDER = ["norm_weight", "norm_bias", "linear_bias", "embedding", "weight"]

# Parameter types we expect to be updated; frozen ones are potential bugs
_SHOULD_UPDATE = {"norm_weight", "norm_bias"}


def print_report(categories: dict, threshold: float) -> None:
    print(f"\n{'='*70}")
    print("CHECKPOINT vs ORIGINAL HF WEIGHT COMPARISON REPORT")
    print(f"{'='*70}")
    print(f"Threshold for 'updated': max_diff > {threshold:.1e}")
    print()

    for ptype in _PRINT_ORDER:
        data = categories.get(ptype)
        if data is None:
            continue
        total = len(data["updated"]) + len(data["frozen"]) + len(data["missing"])
        label = _TYPE_LABELS.get(ptype, ptype)
        print(f"[ {label} ]")
        print(f"  Total: {total}  |  Updated: {len(data['updated'])}  |  "
              f"Frozen: {len(data['frozen'])}  |  Missing in ckpt: {len(data['missing'])}")

        if ptype in _SHOULD_UPDATE and data["frozen"]:
            print(f"  ⚠️  NOT UPDATED (potential issue) — showing up to 10:")
            for name, max_d, mean_d in data["frozen"][:10]:
                print(f"       {name}  (max_diff={max_d:.2e}, mean_diff={mean_d:.2e})")
            if len(data["frozen"]) > 10:
                print(f"       … and {len(data['frozen']) - 10} more")

        if data["updated"]:
            top5 = sorted(data["updated"], key=lambda x: -x[1])[:5]
            print(f"  ✓  Updated (top-5 by max_diff):")
            for name, max_d, mean_d in top5:
                print(f"       {name}  (max_diff={max_d:.2e}, mean_diff={mean_d:.2e})")

        if data["missing"]:
            print(f"  ?  Missing from checkpoint ({len(data['missing'])}):")
            for name in data["missing"][:5]:
                print(f"       {name}")

        print()

    # --- summary ---
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    issues: list[str] = []
    for ptype in _SHOULD_UPDATE:
        data = categories.get(ptype)
        if data and data["frozen"]:
            label = _TYPE_LABELS.get(ptype, ptype)
            issues.append(f"{label}: {len(data['frozen'])} / "
                          f"{len(data['frozen']) + len(data['updated'])} parameters NOT updated")

    if issues:
        print("⚠️  POTENTIAL UPDATE ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("Interpretation:")
        print("  If norm weights show max_diff ≈ 0 after multiple training steps,")
        print("  they may be frozen or excluded from the optimizer update.")
        print("  Possible causes:")
        print("    1. Frozen parameters (check --freeze-* flags or model_provider freeze logic)")
        print("    2. Optimizer not applied to these param groups")
        print("    3. Learning rate / gradient scale so small that diff < threshold")
        print("    4. precision-aware optimizer skipping low-precision norms")
    else:
        print("✓  No update anomalies detected: norm weights appear to be updating normally.")

    print()
    total_params = sum(
        len(d["updated"]) + len(d["frozen"]) + len(d["missing"])
        for d in categories.values()
    )
    total_updated = sum(len(d["updated"]) for d in categories.values())
    total_frozen = sum(len(d["frozen"]) for d in categories.values())
    print(f"Grand total: {total_params} parameters")
    print(f"  Updated: {total_updated}  ({100 * total_updated / max(1, total_params):.1f}%)")
    print(f"  Frozen:  {total_frozen}  ({100 * total_frozen / max(1, total_params):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Megatron checkpoint vs original HF weights (norm/bias focus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ckpt-dir", required=True,
                        help="Parent directory of iter_XXXXXXX/ checkpoint folders")
    parser.add_argument("--original-hf-dir", required=True,
                        help="Original HuggingFace model directory (safetensors)")
    parser.add_argument("--converted-hf-dir",
                        help="Where to write / read the converted HF checkpoint "
                             "(default: <ckpt-dir>/converted_hf_iter_XXXXXXX)")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip conversion and load from --converted-hf-dir directly")
    parser.add_argument("--iter", type=int,
                        help="Specific iteration number to compare (default: latest)")
    parser.add_argument("--threshold", type=float, default=1e-7,
                        help="Max-diff threshold to classify a parameter as 'updated' (default: 1e-7)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"ERROR: ckpt_dir does not exist: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    if args.iter is not None:
        iter_dir = ckpt_dir / f"iter_{args.iter:07d}"
    else:
        iter_dir = find_latest_checkpoint(ckpt_dir)

    if iter_dir is None or not iter_dir.exists():
        print(f"ERROR: No iter_XXXXXXX checkpoint found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint iteration dir : {iter_dir}")
    print(f"Original HF model        : {args.original_hf_dir}")

    converted_dir = Path(
        args.converted_hf_dir
        or str(ckpt_dir / f"converted_hf_{iter_dir.name}")
    )

    if not args.no_convert:
        convert_to_hf(iter_dir, converted_dir, Path(args.original_hf_dir))
    else:
        if not converted_dir.exists():
            print(f"ERROR: --no-convert set but converted_dir does not exist: {converted_dir}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Skipping conversion; using existing: {converted_dir}")

    print(f"\nLoading original HF weights from {args.original_hf_dir} ...")
    orig_weights = load_safetensors(Path(args.original_hf_dir))
    print(f"  {len(orig_weights)} tensors loaded")

    print(f"Loading checkpoint weights from {converted_dir} ...")
    ckpt_weights = load_safetensors(converted_dir)
    print(f"  {len(ckpt_weights)} tensors loaded")

    print(f"\nComparing weights (threshold={args.threshold:.1e}) ...")
    categories = compare_weights(orig_weights, ckpt_weights, threshold=args.threshold)

    print_report(categories, threshold=args.threshold)


if __name__ == "__main__":
    main()
