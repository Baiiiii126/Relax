#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Compare norm/bias parameters between original HF weights and a trained checkpoint.
Reports whether each parameter updated (YES/NO based on max_diff == 0).

Usage:
  python scripts/tools/compare_norm_bias.py \
      --orig /mnt/tidal-nj01/usr/yuzhe1/exps/Qwen3-VL-4B-Instruct \
      --ckpt /mnt/tidal-nj01/usr/yuzhe1/exps/baihaonan-checkpoints-4b/Qwen3-VL-4B-Instruct-Checkpoint/iter_0000789_hf
"""

import argparse
import json
from pathlib import Path

import torch


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    weights: dict[str, torch.Tensor] = {}
    index_file = model_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        loaded: set[str] = set()
        for param_name, fname in index["weight_map"].items():
            fpath = model_dir / fname
            if fname not in loaded:
                loaded.add(fname)
                with safe_open(str(fpath), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        weights[key] = sf.get_tensor(key)
    else:
        for sf_file in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(sf_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weights[key] = sf.get_tensor(key)

    if not weights:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    return weights


_NORM_KW = ("norm", "layernorm", "layer_norm", "rmsnorm", "ln_q", "ln_k")


def classify(name: str):
    low = name.lower()
    if any(k in low for k in _NORM_KW):
        return "norm"
    if low.endswith(".bias"):
        return "bias"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True, help="Original HF model dir")
    parser.add_argument("--ckpt", required=True, help="Trained checkpoint HF dir")
    args = parser.parse_args()

    print(f"Loading original  : {args.orig}")
    orig = load_safetensors(Path(args.orig))
    print(f"Loading checkpoint: {args.ckpt}\n")
    ckpt = load_safetensors(Path(args.ckpt))

    rows = {"norm": [], "bias": []}
    missing = []

    for name in sorted(orig):
        cat = classify(name)
        if cat is None:
            continue
        if name not in ckpt:
            missing.append((cat, name))
            continue
        max_diff = (orig[name].float() - ckpt[name].float()).abs().max().item()
        rows[cat].append((name, max_diff))

    for cat, label in [("norm", "Norm weights"), ("bias", "Bias parameters")]:
        print(f"{'='*65}")
        print(f"  {label}  ({len(rows[cat])} params)")
        print(f"{'='*65}")
        print(f"  {'updated':>7}  {'max_diff':>12}  name")
        print(f"  {'-'*7}  {'-'*12}  ----")
        for name, max_diff in rows[cat]:
            status = "YES" if max_diff > 0 else "NO "
            print(f"  {status:>7}  {max_diff:12.6e}  {name}")
        print()

    not_updated_norm = [n for n, d in rows["norm"] if d == 0]
    not_updated_bias = [n for n, d in rows["bias"] if d == 0]

    print(f"{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Norm not updated: {len(not_updated_norm)} / {len(rows['norm'])}")
    print(f"  Bias not updated: {len(not_updated_bias)} / {len(rows['bias'])}")
    for n in not_updated_norm:
        print(f"    [norm] {n}")
    for n in not_updated_bias:
        print(f"    [bias] {n}")
    if missing:
        print(f"\n  Missing in checkpoint: {len(missing)}")
        for cat, n in missing:
            print(f"    [{cat}] {n}")


if __name__ == "__main__":
    main()
