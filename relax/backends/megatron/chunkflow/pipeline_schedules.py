# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP>1 chunkflow schedule: chunks are the units that flow through the
pipeline.

The p2p pattern is Megatron's standard non-interleaved 1F1B (bare warmup
forwards, fused steady ``send_forward_recv_backward`` /
``send_backward_recv_forward``, bare cooldown backwards). That pattern is
deadlock-free because the backward a steady step receives lags the forward it
sends by the warmup depth, so the peer has always produced it already.

The op order is what makes it chunkflow. A dependency group's chunk ``m`` can
only backward once every chunk ``j >= m`` has (the dK/dV fold-back must be
complete), and a group's backward can only run once the whole group has
forwarded (the store is only full then). ``_pipeline_schedule`` lays that out
as a forward order and a backward order, and adds ``C`` -- a constant of extra
warmup that covers the first group's completion -- to the standard
``pp - stage - 1`` depth so the steady phase always has a backward ready. Every
stage derives the identical orders from the batch's sequence lengths, so the
p2p send/recv pairs stay aligned without broadcasting the plan.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.utils import get_model_config

from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, attach_runtime
from relax.backends.megatron.chunkflow.unit_batch import slice_parent_batch
from relax.backends.megatron.chunkflow.units import classify_units, make_unit_bwd_order
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _no_loss_step_func(forward_step_func):
    """``forward_step_func`` with the loss replaced by a no-op.

    An Fp unit needs its stage's output (the next stage computes its own
    layers' K/V from it) but no loss -- nothing will backward it, and the
    ``Fr`` pass is what recomputes a graph when one is needed.
    """

    def no_loss(output_tensor):
        return output_tensor.reshape(-1)[0] * 0.0, torch.tensor(0, dtype=torch.int), {}

    def step(data_iterator, model):
        output_tensor, _ = forward_step_func(data_iterator, model)
        return output_tensor, no_loss

    return step


def _pipeline_schedule(units, chunks, keep_last_k, pp, stage):
    """The per-stage plan: forward order, backward order, warmup depth.

    ``fwd_seq`` runs standalones first -- they backward immediately -- then the
    dependent units by depth, so a group always forwards its earlier chunks
    first. ``bwd_seq`` is ``make_unit_bwd_order``'s dependency-valid order. The
    warmup depth is the standard ``pp - stage - 1`` plus ``C``, a constant that
    covers the first group's completion, so the steady phase always has a
    backward ready; ``C`` is identical on every stage, so the offset between
    neighbouring stages stays exactly 1.
    """
    standalone = [u.unit_id for u in units if not u.is_dependent]
    dependent = sorted((u for u in units if u.is_dependent), key=lambda u: u.round)
    fwd_seq = standalone + [u.unit_id for u in dependent]
    bwd_seq = make_unit_bwd_order(units, fwd_seq)
    _, need_fr = classify_units(units, chunks, keep_last_k)

    by_id = {u.unit_id: u for u in units}
    fwd_pos = {u: i for i, u in enumerate(fwd_seq)}
    done_at: dict[int, int] = {}
    for u in units:
        for g in u.groups:
            done_at[g] = max(done_at.get(g, -1), fwd_pos[u.unit_id])

    def ready(unit_id):
        """Forward position at which this backward's store is full."""
        unit = by_id[unit_id]
        if not unit.groups:
            return fwd_pos[unit_id]
        return max(done_at[g] for g in unit.groups)

    C = max((ready(u) - i) for i, u in enumerate(bwd_seq)) if bwd_seq else 0
    W = min(C + (pp - stage - 1), len(fwd_seq))
    return fwd_seq, bwd_seq, need_fr, W


def _trace(enabled: bool, mb_idx: int, msg: str) -> None:
    if not enabled or parallel_state.get_data_parallel_rank() != 0:
        return
    logger.info(f"[chunkflow-trace] pp{parallel_state.get_pipeline_model_parallel_rank()} mb{mb_idx}: {msg}")


def chunkflow_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn=None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection=None,
    force_all_reduce: Optional[bool] = False,
    **kwargs,
):
    """Standard 1F1B structure, chunkflow units and order."""
    if forward_only:
        raise NotImplementedError("chunkflow pipeline schedule is training-only")
    if isinstance(model, list):
        assert len(model) == 1, "chunkflow pipeline schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "chunkflow pipeline schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError("chunkflow pipeline schedule does not support overlap_p2p_comm")

    import os

    from megatron.training.global_vars import get_args

    from relax.backends.megatron.chunkflow.schedules import _BATCH_KEYS, _num_microbatches, _plan
    from relax.backends.megatron.data import get_batch

    args = get_args()
    chunk_size = int(args.chunkflow_chunk_size)
    policy = getattr(args, "chunkflow_pack_policy", "staggered")
    keep_last_k = getattr(args, "chunkflow_keep_last_k", 0) or 1 << 30
    trace = os.environ.get("RELAX_CHUNKFLOW_TRACE") == "1"
    runtime = attach_runtime(model, chunk_size)

    if p2p_communicator is None and pg_collection is None:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        p2p_communicator = P2PCommunicator(pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = parallel_state.get_pipeline_model_parallel_group()
        pg_collection.embd = parallel_state.get_embedding_group(check_initialized=False)
        pg_collection.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    else:
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
        tp_group = getattr(pg_collection, "tp", None)

    is_first = p2p_communicator.is_pp_first_stage
    is_last = p2p_communicator.is_pp_last_stage

    embedding_module = None
    if config.finalize_model_grads_func is not None:
        embedding_module = clear_embedding_activation_buffer(config, model, is_last)

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(recv_tensor_shapes, send_tensor_shapes)

    # All passes run under no_sync; one grad sync at the end of the step.
    no_sync_func = config.no_sync_func or contextlib.nullcontext

    forward_data_store: list = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    discarded: list = []
    fp_step_func = _no_loss_step_func(forward_step_func)

    if trace:
        import faulthandler

        faulthandler.dump_traceback_later(
            int(os.environ.get("RELAX_CHUNKFLOW_WATCHDOG_S", "120")), repeat=True, exit=True
        )

    def next_parent() -> dict:
        get_next = getattr(data_iterator, "get_next", None)
        raw = get_next(_BATCH_KEYS) if get_next is not None else next(data_iterator)[0]
        return get_batch(
            iter([(raw, {})]),
            _BATCH_KEYS,
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )

    with no_sync_func():
        for mb_idx in range(_num_microbatches(num_microbatches)):
            parent = next_parent()
            chunks, units = _plan(parent, chunk_size, policy, runtime.next_group_offset(None))
            used_max = max((c.group for c in chunks if c.group is not None), default=None)
            runtime.next_group_offset(used_max)
            by_id = {u.unit_id: u for u in units}
            fwd_seq, bwd_seq, need_fr, W = _pipeline_schedule(
                units, chunks, keep_last_k, p2p_communicator.total_stages, p2p_communicator.current_stage
            )
            n = len(fwd_seq)
            _trace(trace, mb_idx, f"fwd={fwd_seq} bwd={bwd_seq} W={W} n={n} need_fr={sorted(need_fr)}")

            def do_forward(unit_id, input_tensor, mb_idx):
                """One unit's forward through this stage.

                Fp runs under no_grad (K/V only) and keeps the input for the
                later recompute; F and Fr build a graph and take a loss.
                """
                unit = by_id[unit_id]
                batch = slice_parent_batch(parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args)
                batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                if unit_id in need_fr:
                    batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, False)
                    with torch.no_grad():
                        output_tensor, _ = forward_step(
                            fp_step_func,
                            iter([(batch, {})]),
                            model,
                            1,
                            input_tensor,
                            discarded,
                            config,
                            cp_group_size=cp_size,
                            collect_non_loss_data=False,
                            is_first_microbatch=(mb_idx == 0),
                            current_microbatch=mb_idx,
                            is_last_stage=is_last,
                        )
                    return output_tensor, input_tensor, None
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    iter([(batch, {})]),
                    model,
                    1,
                    input_tensor,
                    forward_data_store,
                    config,
                    cp_group_size=cp_size,
                    collect_non_loss_data=False,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, mb_idx == 0),
                    current_microbatch=mb_idx,
                    is_last_stage=is_last,
                )
                return output_tensor, input_tensor, num_tokens

            # (input, output) of keep units still owing a backward, and the input
            # of recompute units until their Fr rebuilds a graph.
            in_flight: dict[int, tuple] = {}
            fp_inputs: dict[int, torch.Tensor] = {}

            def backward(unit_id, output_tensor_grad):
                """One unit's backward.

                A recompute unit rebuilds its graph first (Fr), reusing the K/V
                leaf the Fp pass stored.
                """
                fr_num_tokens = None
                if unit_id in need_fr:
                    input_tensor = fp_inputs.pop(unit_id)
                    unit = by_id[unit_id]
                    batch = slice_parent_batch(
                        parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args
                    )
                    batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                    batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, True)
                    output_tensor, fr_num_tokens = forward_step(
                        forward_step_func,
                        iter([(batch, {})]),
                        model,
                        1,
                        input_tensor,
                        forward_data_store,
                        config,
                        cp_group_size=cp_size,
                        collect_non_loss_data=False,
                        is_first_microbatch=False,
                        current_microbatch=mb_idx,
                        is_last_stage=is_last,
                    )
                else:
                    input_tensor, output_tensor = in_flight.pop(unit_id)
                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                for group in by_id[unit_id].groups:
                    runtime.release_group(group)
                return input_tensor_grad, fr_num_tokens

            # ── Warmup ────────────────────────────────────────────────────
            for i in range(W):
                input_tensor = None if is_first else p2p_communicator.recv_forward(recv_tensor_shapes, is_first)
                output_tensor, fwd_input, num_tokens = do_forward(fwd_seq[i], input_tensor, mb_idx)
                if num_tokens is not None:
                    total_num_tokens += num_tokens
                    in_flight[fwd_seq[i]] = (fwd_input, output_tensor)
                else:
                    fp_inputs[fwd_seq[i]] = fwd_input
                if not is_last:
                    p2p_communicator.send_forward(output_tensor, is_last)
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # ── Steady 1F1B ────────────────────────────────────────────────
            n_steady = n - W
            input_tensor = None
            if n_steady > 0 and not is_first:
                input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_first)
            for i in range(n_steady):
                last_iter = i == n_steady - 1
                output_tensor, fwd_input, num_tokens = do_forward(fwd_seq[W + i], input_tensor, mb_idx)
                if num_tokens is not None:
                    total_num_tokens += num_tokens
                    in_flight[fwd_seq[W + i]] = (fwd_input, output_tensor)
                else:
                    fp_inputs[fwd_seq[W + i]] = fwd_input
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, is_last
                )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                input_tensor_grad, fr_num_tokens = backward(bwd_seq[i], output_tensor_grad)
                if fr_num_tokens is not None:
                    total_num_tokens += fr_num_tokens
                if last_iter:
                    if not is_first:
                        p2p_communicator.send_backward(input_tensor_grad, is_first)
                else:
                    input_tensor = p2p_communicator.send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, is_first
                    )

            # ── Cooldown ──────────────────────────────────────────────────
            for i in range(n_steady, n):
                output_tensor_grad = None if is_last else p2p_communicator.recv_backward(send_tensor_shapes, is_last)
                input_tensor_grad, fr_num_tokens = backward(bwd_seq[i], output_tensor_grad)
                if fr_num_tokens is not None:
                    total_num_tokens += fr_num_tokens
                if not is_first:
                    p2p_communicator.send_backward(input_tensor_grad, is_first)

            if in_flight or fp_inputs:
                raise RuntimeError(
                    f"chunkflow pipeline: mb{mb_idx} ended with {len(in_flight)} forwarded and "
                    f"{len(fp_inputs)} unrecomputed unit(s) -- schedule bug"
                )

    if config.grad_sync_func is not None:
        config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None:
        finish_embedding_wgrad_compute(config, embedding_module, is_last, tp_group)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return forward_data_store
