# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest


try:
    import torch
    from megatron.core.pipeline_parallel import p2p_communication, schedules

    from relax.backends.megatron import training_profile as profiling
    from relax.backends.megatron.chunkflow import pipeline_schedules
    from relax.utils import profile_utils
except ImportError as exc:
    pytest.skip(f"Training profiler dependencies unavailable: {exc}", allow_module_level=True)


class GatedDeltaNet(torch.nn.Module):
    def _forward_compute(self, x):
        return x * 2

    def forward(self, x):
        return self._forward_compute(x)


class _CPUOptimizer:
    def __init__(self):
        self._optimizer_step_post_hooks = {0: self.param_copy_back_gpu_hook}

    def param_copy_back_gpu_hook(self, *args):
        pass

    def step(self):
        for hook in self._optimizer_step_post_hooks.values():
            hook(self, (), {})


class _Hybrid:
    def __init__(self):
        self.cpu_optimizers = [_CPUOptimizer()]

    def _set_sub_optimizer_grads(self):
        pass


class _Optimizer:
    def __init__(self):
        self.optimizer = _Hybrid()

    def prepare_grads(self):
        pass

    def step(self):
        self.prepare_grads()
        self.optimizer._set_sub_optimizer_grads()
        self.optimizer.cpu_optimizers[0].step()
        return True


def test_profile_directory_can_use_shared_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAX_TRAIN_PROFILE_DIR", raising=False)
    args = SimpleNamespace(tb_experiment_name="test-run")
    assert profile_utils._get_train_trace_dir(args) == "traces/test-run/train_trace"
    monkeypatch.setenv("RELAX_TRAIN_PROFILE_DIR", str(tmp_path))
    assert profile_utils._get_train_trace_dir(args) == str(tmp_path)


@pytest.mark.parametrize("chunkflow", [False, True])
def test_profile_records_both_arms_and_restores_wrappers(tmp_path, monkeypatch, chunkflow):
    args = SimpleNamespace(
        use_pytorch_profiler=True,
        profile_target=["train_overall"],
        profile_step_start=34,
        profile_step_end=36,
        use_chunkflow=chunkflow,
    )
    monkeypatch.setattr(profile_utils, "_get_train_trace_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(profile_utils, "_get_rank_tag", lambda: "rank0_dp0_tp0_pp0")
    original_backward = lambda output: output.sum().backward()
    monkeypatch.setattr(schedules, "backward_step", original_backward)
    monkeypatch.setattr(pipeline_schedules, "backward_step", original_backward)
    communicator = p2p_communication.P2PCommunicator.__new__(p2p_communication.P2PCommunicator)
    original_send = lambda self, output: output
    monkeypatch.setattr(p2p_communication.P2PCommunicator, "send_forward", original_send)
    module, optimizer = GatedDeltaNet(), _Optimizer()
    hook_map = optimizer.optimizer.cpu_optimizers[0]._optimizer_step_post_hooks
    original_hook = hook_map[0]
    config = SimpleNamespace(finalize_model_grads_func=lambda: None, grad_sync_func=None)
    for rollout in (33, 34):
        x = torch.ones(3, requires_grad=True)
        with (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace,
            profiling.profile_train_step(args, rollout, 0, [module], optimizer, config),
        ):
            with profiling.train_phase("forward_backward"):
                output, loss = profiling.profile_forward_step(lambda value: (module(value), torch.sum))(x)
                assert loss(output).item() == 6
                assert communicator.send_forward(output) is output
                (pipeline_schedules if chunkflow else schedules).backward_step(output)
                config.finalize_model_grads_func()
            assert optimizer.step()
        names = {event.key for event in trace.key_averages()}
        assert ("train.segment.backward" in names) == (rollout == 34)
        assert ("train.segment.offload.params_h2d" in names) == (rollout == 34)
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        assert schedules.backward_step is original_backward
        assert pipeline_schedules.backward_step is original_backward
        assert p2p_communication.P2PCommunicator.send_forward is original_send
        assert "_forward_compute" not in vars(module)
        assert "step" not in vars(optimizer)
        assert hook_map[0] is original_hook
    rows = [json.loads(line) for line in next(tmp_path.glob("segments_*.jsonl")).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["rollout_id"] == 34
    for name in (
        "forward_backward",
        "forward",
        "loss",
        "backward",
        "pipeline.send_forward",
        "grad_finalize",
        "model.gdn_compute",
        "optimizer",
        "offload.grads_d2h",
        "offload.cpu_step",
        "offload.params_h2d",
    ):
        assert rows[0]["host_inclusive"][name]["calls"] == 1
    with pytest.raises(RuntimeError, match="test failure"):
        with profiling.profile_train_step(args, 35, 0, [module], optimizer, config):
            raise RuntimeError("test failure")
    assert profiling._ACTIVE.get() is None
    assert schedules.backward_step is original_backward and hook_map[0] is original_hook
    assert "_forward_compute" not in vars(module) and "step" not in vars(optimizer)
