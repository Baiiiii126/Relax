# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""EP2 dispatch with local-source FC1 work before remote completion.

No polling kernel or network tile-ready protocol is used. CPU split discovery
and the original combine remain. Each invocation owns its metadata and event;
only the communication stream is shared. Activation/FC2 and backward run once
in the original [expert, source-rank] row order.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def _comm_stream(device_index: int) -> Any:
    import torch

    return torch.cuda.Stream(device=device_index)


class Dispatch:
    def __init__(self, group: Any, send: tuple[int, ...], receive: tuple[int, ...], counts: Any) -> None:
        import torch
        import torch.distributed as dist
        import triton

        from . import _expert_tile_kernels as kernels

        self.group = group
        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 2 or len(send) != 2 or len(receive) != 2:
            raise ValueError("Local-first expert dispatch requires EP2")
        if send[self.rank] != receive[self.rank] or min(*send, *receive) < 0:
            raise ValueError("Invalid EP2 split sizes")
        if (
            not counts.is_cuda
            or counts.ndim != 2
            or counts.shape[0] != 2
            or not 0 < counts.shape[1] <= 1024
            or counts.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected device counts with shape [2, local_experts]")
        self.send, self.receive = send, receive
        self.counts = counts.contiguous()
        metadata = torch.empty((3, counts.shape[1] + 1), dtype=torch.int64, device=counts.device)
        kernels.dispatch_offsets[(1,)](self.counts, metadata, counts.shape[1], triton.next_power_of_2(counts.shape[1]))
        self.source_offsets = (metadata[0], metadata[1])
        self.offsets = metadata[2]
        self.ready: Any = None

    def copy(
        self, source: Any, probabilities: Any, canonical: Any, canonical_probs: Any, rank: int, reverse: bool = False
    ) -> None:
        from . import _expert_tile_kernels as kernels

        if source.shape[0]:
            kernels.source_copy[((source.shape[0] + 15) // 16, (source.shape[1] + 255) // 256)](
                source,
                probabilities,
                canonical,
                canonical_probs,
                self.source_offsets[rank],
                self.offsets,
                self.counts,
                source.shape[0],
                source.shape[1],
                self.counts.shape[1],
                rank,
                reverse,
            )

    def exchange(self, x: Any, p: Any, y: Any, q: Any, reverse: bool = False) -> None:
        import torch.distributed as dist

        send, receive = (self.receive, self.send) if reverse else (self.send, self.receive)
        inputs, outputs = list(send), list(receive)
        inputs[self.rank] = outputs[self.rank] = 0
        # Both collectives are issued even when this rank has zero send/receive.
        hidden = dist.all_to_all_single(y, x, outputs, inputs, group=self.group, async_op=True)
        probs = dist.all_to_all_single(q, p, outputs, inputs, group=self.group, async_op=True)
        hidden.wait()
        probs.wait()


@lru_cache(maxsize=1)
def _dispatch_autograd() -> type:
    import torch
    from torch.autograd.function import once_differentiable

    class RemoteDispatch(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, p: Any, state: Dispatch) -> tuple:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Local-first dispatch does not support CUDA Graph capture")
            if x.dtype != torch.bfloat16 or p.dtype != torch.float64:
                raise ValueError("Local-first dispatch requires BF16 hidden states and FP64 router probabilities")
            if x.ndim != 2 or p.shape != (x.shape[0],) or not x.is_contiguous() or not p.is_contiguous():
                raise ValueError("Expected contiguous dispatch inputs [rows, hidden] and [rows]")
            if x.shape[0] != sum(state.send) or x.device != p.device or x.device != state.counts.device:
                raise ValueError("Dispatch input shape/device does not match metadata")
            rank, peer = state.rank, 1 - state.rank
            y = x.new_empty((sum(state.receive), x.shape[1]))
            q = p.new_empty((sum(state.receive),))
            local_start = sum(state.send[:rank])
            remote_start = sum(state.send[:peer])
            local_x = x.narrow(0, local_start, state.send[rank])
            local_p = p.narrow(0, local_start, state.send[rank])
            remote_x = x.narrow(0, remote_start, state.send[peer])
            remote_p = p.narrow(0, remote_start, state.send[peer])
            rx = x.new_empty((state.receive[peer], x.shape[1]))
            rp = p.new_empty((state.receive[peer],))
            current = torch.cuda.current_stream(x.device)
            comm = _comm_stream(x.device.index)
            # Record this dependency before local FC1 can be queued. No cycle.
            prepared = current.record_event()
            state.copy(local_x, local_p, y, q, rank)
            with torch.cuda.stream(comm):
                comm.wait_event(prepared)
                state.exchange(remote_x, remote_p, rx, rp)
                state.copy(rx, rp, y, q, peer)
                state.ready = comm.record_event()
            # Storage allocated on the caller stream must outlive side-stream use.
            for tensor in (x, p, rx, rp, y, q, state.counts, state.offsets, *state.source_offsets):
                tensor.record_stream(comm)
            ctx.state = state
            ctx.hidden = x.shape[1]
            ctx.set_materialize_grads(False)
            return y, q

        @staticmethod
        @once_differentiable
        def backward(ctx: Any, dy: Any, dq: Any) -> tuple:
            state = ctx.state
            rank, peer = state.rank, 1 - state.rank
            device = state.counts.device
            # The complete FC1/activation graph joins remote completion before backward.
            torch.cuda.current_stream(device).wait_event(state.ready)
            if dy is None:
                dy = torch.zeros((sum(state.receive), ctx.hidden), device=device, dtype=torch.bfloat16)
            if dq is None:
                dq = torch.zeros(sum(state.receive), device=device, dtype=torch.float64)
            dy, dq = dy.contiguous(), dq.contiguous()
            dx = dy.new_empty((sum(state.send), ctx.hidden))
            dp = dq.new_empty((sum(state.send),))
            local_start, remote_start = sum(state.send[:rank]), sum(state.send[:peer])
            state.copy(
                dx.narrow(0, local_start, state.send[rank]),
                dp.narrow(0, local_start, state.send[rank]),
                dy,
                dq,
                rank,
                reverse=True,
            )
            tx = dy.new_empty((state.receive[peer], ctx.hidden))
            tp = dq.new_empty((state.receive[peer],))
            state.copy(tx, tp, dy, dq, peer, reverse=True)
            state.exchange(
                tx,
                tp,
                dx.narrow(0, remote_start, state.send[peer]),
                dp.narrow(0, remote_start, state.send[peer]),
                reverse=True,
            )
            return dx if ctx.needs_input_grad[0] else None, dp if ctx.needs_input_grad[1] else None, None

    return RemoteDispatch


def dispatch(x: Any, probabilities: Any, state: Dispatch) -> tuple:
    return _dispatch_autograd().apply(x, probabilities, state)
