#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
检查 final_layernorm 梯度是否到达。

用法（在训练 pod 上运行）:
  python3 scripts/tools/check_final_layernorm_grad.py \
      --hf-ckpt /mnt/tidal-nj01/usr/yuzhe1/exps/Qwen3-VL-4B-Instruct

原理:
  1. 用 HF 格式加载模型
  2. 在 model.norm (final_layernorm) 上注册 weight/input/output hook
  3. 跑一次 dummy forward + backward
  4. 打印梯度 norm 和是否为零
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-ckpt", required=True, help="HF 格式模型路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.hf_ckpt} on {args.device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_ckpt,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_ckpt, trust_remote_code=True)
    model.train()

    # ── 找到 final_layernorm ──────────────────────────────────────────────────
    norm_module = None
    norm_name = None
    for name, mod in model.named_modules():
        # HF Qwen3-VL: model.language_model.model.norm  OR  model.model.norm
        if name.endswith(".norm") and hasattr(mod, "weight") and not list(mod.children()):
            # pick the last one (final_layernorm is after all transformer layers)
            norm_module = mod
            norm_name = name
    assert norm_module is not None, "找不到 final_layernorm，检查模型结构"
    print(f"\nfinal_layernorm module : {norm_name}")
    print(f"  weight shape         : {norm_module.weight.shape}")
    print(f"  requires_grad        : {norm_module.weight.requires_grad}")

    # ── 注册 hooks ─────────────────────────────────────────────────────────────
    records = {}

    def fwd_hook(module, inp, out):
        records["fwd_out_norm"] = out.float().norm().item()
        records["fwd_out_requires_grad"] = out.requires_grad

    def bwd_hook(module, grad_in, grad_out):
        records["bwd_grad_out"] = grad_out[0].float().norm().item() if grad_out[0] is not None else None
        records["bwd_grad_in"]  = grad_in[0].float().norm().item()  if grad_in[0]  is not None else None

    def weight_grad_hook(grad):
        records["weight_grad_norm"] = grad.float().norm().item()
        records["weight_grad_max"]  = grad.float().abs().max().item()

    norm_module.register_forward_hook(fwd_hook)
    norm_module.register_full_backward_hook(bwd_hook)
    norm_module.weight.register_hook(weight_grad_hook)

    # ── Dummy forward + backward ───────────────────────────────────────────────
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    # strip multimodal keys if any
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    print("\nRunning forward pass ...")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    print(f"  loss = {loss.item():.6f}")

    print("Running backward pass ...")
    loss.backward()

    # ── 结果报告 ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("final_layernorm GRADIENT REPORT")
    print("="*60)
    print(f"  forward output norm      : {records.get('fwd_out_norm'):.4e}")
    print(f"  forward output requires_grad: {records.get('fwd_out_requires_grad')}")
    print(f"  backward grad_out (→ norm) : {records.get('bwd_grad_out')}")
    print(f"  backward grad_in  (← norm) : {records.get('bwd_grad_in')}")

    wg = records.get("weight_grad_norm")
    wm = records.get("weight_grad_max")
    if wg is None:
        print("\n  ⚠️  weight_grad_hook NEVER FIRED — gradient did NOT reach final_layernorm.weight")
    else:
        print(f"\n  weight grad norm : {wg:.4e}")
        print(f"  weight grad max  : {wm:.4e}")
        if wg == 0.0:
            print("  ⚠️  grad is ZERO — weight will NOT be updated")
        else:
            update_bf16_threshold = norm_module.weight.float().abs().mean().item() * 2**-7
            print(f"  bf16 update threshold (approx): {update_bf16_threshold:.4e}")
            expected_delta_per_step = 1e-6 * wg  # lr=1e-6
            print(f"  expected delta per step (lr=1e-6): {expected_delta_per_step:.4e}")
            if expected_delta_per_step < update_bf16_threshold:
                print("  ⚠️  delta < bf16 precision → checkpoint 显示 0 是精度问题，非真正未更新")
            else:
                print("  ✓  delta >= bf16 precision → 应该能在 checkpoint 中观察到变化")

    print("\ndirect weight.grad check:")
    if norm_module.weight.grad is None:
        print("  norm_module.weight.grad is None  ← 梯度未到达")
    else:
        g = norm_module.weight.grad.float()
        print(f"  norm  = {g.norm():.4e}")
        print(f"  max   = {g.abs().max():.4e}")
        print(f"  zeros = {(g == 0).sum().item()} / {g.numel()}")


if __name__ == "__main__":
    main()
