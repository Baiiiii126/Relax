#!/usr/bin/env python3
import torch
import torch.nn as nn

torch.manual_seed(42)

class MakeViewlessTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, requires_grad):
        out = inp.view(inp.shape)
        out.requires_grad_(requires_grad)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def make_viewless_tensor(inp, requires_grad, keep_graph):
    if inp._base is None:
        return inp
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    return inp.view(inp.shape).requires_grad_(requires_grad)

hidden_size = 2560
layernorm = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)

# Megatron final_layernorm 输入来自 all_gather，通常是 view
batch_seq = torch.randn(16, 1, hidden_size, requires_grad=True)
hidden_view = batch_seq.view(16, 1, hidden_size)

print(f"input is view: {hidden_view._base is not None}")

# CASE 1: 正常流程（Megatron 实际代码路径）
print("\n=== CASE 1: final_layernorm -> make_viewless_tensor(keep_graph=True) -> loss ===")
layernorm.zero_grad()
out1 = layernorm(hidden_view)
out1_vl = make_viewless_tensor(out1, requires_grad=True, keep_graph=True)
loss1 = out1_vl.sum()
loss1.backward()
g = layernorm.weight.grad
print(f"  weight.grad norm : {g.norm():.4e}" if g is not None else "  weight.grad : None")

# CASE 2: make_viewless_tensor 用 requires_grad=False
print("\n=== CASE 2: requires_grad=False in make_viewless_tensor ===")
layernorm.zero_grad()
out2 = layernorm(hidden_view)
out2_vl = make_viewless_tensor(out2, requires_grad=False, keep_graph=True)
loss2 = out2_vl.sum()
loss2.backward()
g2 = layernorm.weight.grad
print(f"  weight.grad norm : {g2.norm():.4e}" if g2 is not None else "  weight.grad : None")

# CASE 3: keep_graph=False
print("\n=== CASE 3: keep_graph=False ===")
layernorm.zero_grad()
out3 = layernorm(hidden_view)
out3_vl = make_viewless_tensor(out3, requires_grad=True, keep_graph=False)
loss3 = out3_vl.sum()
try:
    loss3.backward()
    g3 = layernorm.weight.grad
    print(f"  weight.grad norm : {g3.norm():.4e}" if g3 is not None else "  weight.grad : None")
except Exception as e:
    print(f"  backward failed: {e}")

# CASE 4: loss 计算时 detach
print("\n=== CASE 4: hidden_states detached before loss (GRPO log_prob detach?) ===")
layernorm.zero_grad()
out4 = layernorm(hidden_view)
out4_vl = make_viewless_tensor(out4, requires_grad=True, keep_graph=True)
# 模拟 lm_head 之后做了 detach 再算 loss
loss4 = out4_vl.detach().sum()
try:
    loss4.backward()
    g4 = layernorm.weight.grad
    print(f"  weight.grad norm : {g4.norm():.4e}" if g4 is not None else "  weight.grad : None (detach breaks graph)")
except Exception as e:
    print(f"  backward failed: {e}")

print("\n=== CASE 5: lm_head output_layer 在 post_process=False 的 rank 上 ===")
print("  如果 post_process=False，output_layer 不存在，loss 不经过 final_layernorm")
print("  final_layernorm 仍存在但其输出只传给下一 PP stage，若 PP=1 则没问题")

print("\n=== 结论 ===")
print("Case 1 是正常路径 -> 梯度能到达")
print("关键问题: GRPO loss 的计算里是否对 logits/log_prob 做了 detach?")
