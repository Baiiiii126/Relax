# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP>1 chunkflow schedule: chunks are the units that flow through the
pipeline.

The pipeline structure is ``streaming_forward_backward_pipelining_without_interleaving``
(warmup / steady 1F1B / cooldown, ``P2PCommunicator``, ``no_sync`` with one
``finalize_model_grads``). The difference is the unit of work and its order: a
chunk (or a batch of independent chunks) instead of a micro-batch, in the order
``gen_unit_ops`` prescribes for this stage.

Each stage derives the identical chunk plan from the batch's sequence lengths
(see ``units.build_micro_batches`` and ``scheduler.gen_state_aware_1f1b_ops``),
so the p2p send/recv pairs stay aligned without broadcasting the plan. Every
stage issues the same sequence of p2p calls, which is the only invariant the
communicator needs.

The one thing that does *not* follow from "chunk = micro-batch" is ``Fp``: a
unit whose activation is dropped still has to run its layers **on every stage**,
because each stage keeps its own layers' K/V in its own store for the later
chunks of that group to read. So an Fp forward sends its output downstream like
any other, it just keeps no graph and takes no loss.
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
from relax.backends.megatron.chunkflow.units import gen_unit_ops
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


def _trace(enabled: bool, mb_idx: int, ops: list) -> None:
    if not enabled or parallel_state.get_data_parallel_rank() != 0:
        return
    logger.info(f"[chunkflow-trace] pp{parallel_state.get_pipeline_model_parallel_rank()} mb{mb_idx}: {ops}")


def plan_p2p_calls(ops, *, is_first: bool, is_last: bool) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """The p2p calls one stage issues for its op list, as ``(before, after)``.

    ``before`` is issued ahead of the op's compute and ``after`` once its output
    exists, and each tuple is a single atomic communicator call -- everything in
    it is posted together, which is what ``batch_isend_irecv`` gives us.

    Sends and receives in *opposite* directions go in the same call. Two stages
    that each block on a bare send in opposite directions never post the receives
    that would free them, so a forward is fused with the following backward's
    receive (and vice versa) whenever the op order turns. Bare sends are left
    only where the direction is already established and the peer is guaranteed to
    be receiving: a run of forwards, or a run of backwards.

    ``Fr`` is a local recompute and never communicates. A receive on the first
    stage, and a send on the first/last stage, is a no-op and is left out.

    Kept as a pure function of the op list so the deadlock check in
    ``tests/backends/megatron/test_chunkflow_pipeline_schedule.py`` can simulate
    exactly what the executor below will do.
    """
    plan: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    prefetched_input = prefetched_grad = False
    for idx, (kind, _) in enumerate(ops):
        # The next op that pairs with a p2p call: Fr sits inside the sequence
        # without talking to anyone, so it must not be mistaken for a turn.
        next_kind = next((k for k, _ in ops[idx + 1 :] if k != "Fr"), None)

        if kind == "Fr":
            plan.append(((), ()))
            continue

        if kind == "B":
            before = () if (prefetched_grad or is_last) else ("recv_bwd",)
            prefetched_grad = False
            if next_kind in ("F", "Fp"):
                # Both stages are turning at the same time; exchange in one call.
                after = () if is_first else ("send_bwd", "recv_fwd")
                prefetched_input = not is_first
            else:
                after = () if is_first else ("send_bwd",)
        else:
            before = () if (prefetched_input or is_first) else ("recv_fwd",)
            prefetched_input = False
            if next_kind == "B":
                after = () if is_last else ("send_fwd", "recv_bwd")
                prefetched_grad = not is_last
            else:
                after = () if is_last else ("send_fwd",)

        plan.append((before, after))
    return plan


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
    """Streaming 1F1B structure, chunkflow units and order."""
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
    # Fp units are never backwarded, so nothing of theirs reaches the caller --
    # neither the token count nor the loss metrics.
    discarded: list = []

    # A deadlock here shows up as nothing: no traceback, no log, GPUs idle. So
    # when the trace is on, every rank also arms a watchdog that dumps all stacks
    # and exits after a single op has taken this long -- a normal forward is
    # sub-second, so 300s means stuck, not slow. The dump lands in the worker's
    # stderr, which the job log does capture.
    if trace:
        import faulthandler

        faulthandler.dump_traceback_later(
            int(os.environ.get("RELAX_CHUNKFLOW_WATCHDOG_S", "120")), repeat=True, exit=True
        )
    # (input, output) of every forward still owing a backward, keyed by unit: the
    # op order, not a FIFO, decides which one a ``B`` consumes.
    in_flight: dict[int, tuple] = {}
    # Stage input held for an Fp unit until its recompute. One tensor per such
    # chunk, not per layer -- the per-layer activations are still dropped, which
    # is what Fp is for.
    fp_inputs: dict[int, torch.Tensor] = {}
    # Tensor received for the next forward/backward, when the previous op already
    # fetched it as part of a fused send+recv (see ``plan_p2p_calls``).
    pending_input: Optional[torch.Tensor] = None
    pending_grad: Optional[torch.Tensor] = None
    fp_step_func = _no_loss_step_func(forward_step_func)

    def next_parent() -> dict:
        """The processed micro-batch every stage sees, and therefore the same
        plan on every stage."""
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
            ops = gen_unit_ops(
                units,
                chunks,
                keep_last_k,
                pp=p2p_communicator.total_stages,
                stage=p2p_communicator.current_stage,
            )
            _trace(trace, mb_idx, ops)
            # An Fp unit is recomputed by a later Fr, and the recompute needs the
            # activation that entered *this* stage -- only the first stage can
            # rebuild it from the tokens. So the Fp pass keeps the input tensor
            # for exactly the units that have one coming.
            fr_units = {unit_id for kind, unit_id in ops if kind == "Fr"}
            p2p_plan = plan_p2p_calls(ops, is_first=is_first, is_last=is_last)

            for idx, (kind, unit_id) in enumerate(ops):
                unit = units[unit_id]
                before, after = p2p_plan[idx]
                if trace and mb_idx == 0:
                    # One line per op, first micro-batch only: on a hang the last
                    # line each stage printed is the op it never finished, and the
                    # watchdog stack shows the call inside it.
                    logger.info(
                        f"[chunkflow-trace] pp{p2p_communicator.current_stage} mb{mb_idx} "
                        f"op[{idx}/{len(ops)}]=({kind},{unit_id}) before={before} after={after}"
                    )

                if kind == "B":
                    if "recv_bwd" in before:
                        output_tensor_grad = p2p_communicator.recv_backward(send_tensor_shapes, is_last)
                    else:
                        output_tensor_grad, pending_grad = pending_grad, None
                    input_tensor, output_tensor = in_flight.pop(unit_id)
                    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                    if "recv_fwd" in after:
                        pending_input = p2p_communicator.send_backward_recv_forward(
                            input_tensor_grad, recv_tensor_shapes, is_first
                        )
                    elif after:
                        p2p_communicator.send_backward(input_tensor_grad, is_first)
                    # Every forward of this group has run by the time its first
                    # unit backwards (the backward order guarantees it), so no
                    # chunk will look these up again.
                    for group in unit.groups:
                        runtime.release_group(group)
                    continue

                if kind == "Fr":
                    input_tensor = fp_inputs.pop(unit_id)
                elif "recv_fwd" in before:
                    input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_first)
                else:
                    input_tensor, pending_input = pending_input, None
                batch = slice_parent_batch(parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args)
                batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, kind == "Fr")

                if kind == "Fp":
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
                    if unit_id in fr_units:
                        fp_inputs[unit_id] = input_tensor
                else:
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
                    total_num_tokens += num_tokens
                    in_flight[unit_id] = (input_tensor, output_tensor)

                if not after:
                    continue
                if "recv_bwd" in after:
                    pending_grad = p2p_communicator.send_forward_recv_backward(
                        output_tensor, send_tensor_shapes, is_last
                    )
                else:
                    p2p_communicator.send_forward(output_tensor, is_last)
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            if in_flight:
                raise RuntimeError(
                    f"chunkflow pipeline: micro-batch {mb_idx} forwarded {len(in_flight)} unit(s) "
                    f"that its op order never backwarded ({sorted(in_flight)}) -- schedule bug"
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
