# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP>1 chunkflow schedule: chunks are the units that flow through the pipeline.

The pipeline structure is ``streaming_forward_backward_pipelining_without_interleaving``
(warmup / steady 1F1B / cooldown, ``P2PCommunicator``, ``no_sync`` with one
``finalize_model_grads``). The difference is the unit of work and its order:
a chunk (or a batch of independent chunks) instead of a micro-batch, in the
order ``gen_unit_ops`` prescribes for this stage.

Each stage derives the identical chunk plan from the batch's sequence lengths
(see ``units.build_micro_batches`` and ``scheduler.gen_state_aware_1f1b_ops``),
so the p2p send/recv pairs stay aligned without broadcasting the plan.

UNVERIFIED SKELETON -- structure is in place but this has not run end to end.
The pieces most likely to be wrong first are marked TODO(verify).
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    deallocate_output_tensor,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.utils import get_model_config

from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, attach_runtime
from relax.backends.megatron.chunkflow.unit_batch import slice_parent_batch
from relax.backends.megatron.chunkflow.units import gen_unit_ops
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def chunkflow_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,  # ignored -- the iterator drives the count
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
    del collect_non_loss_data, first_val_step
    if isinstance(model, list):
        assert len(model) == 1, "chunkflow pipeline schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "chunkflow pipeline schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError("chunkflow pipeline schedule does not support overlap_p2p_comm")

    from megatron.training.global_vars import get_args

    from relax.backends.megatron.chunkflow.schedules import _BATCH_KEYS, _forward_kwargs, _plan
    from relax.backends.megatron.data import get_batch

    args = get_args()
    chunk_size = int(args.chunkflow_chunk_size)
    policy = getattr(args, "chunkflow_pack_policy", "staggered")
    keep_last_k = getattr(args, "chunkflow_keep_last_k", 0) or 1 << 30
    runtime = attach_runtime(model, chunk_size)

    if p2p_communicator is None and pg_collection is None:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        p2p_communicator = P2PCommunicator(pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    else:
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
        tp_group = getattr(pg_collection, "tp", None)

    # Tensor shapes: with variable_seq_lengths the communicator negotiates real
    # shapes per message, so the returned shapes here are only a fallback.
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

    no_sync_func = config.no_sync_func or contextlib.nullcontext
    forward_data_store: list = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    is_first = p2p_communicator.is_pp_first_stage
    is_last = p2p_communicator.is_pp_last_stage

    # In-flight queues: (input, output) pairs awaiting backward, keyed loosely
    # by unit so the op order -- not a FIFO -- picks which one to pop.
    in_flight: dict[int, tuple] = {}

    def _run_unit_forward(unit, parent, kind, input_tensor, unit_id, mb_idx):
        batch = slice_parent_batch(parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args)
        batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
        if kind == "Fp":
            batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, False)
            with torch.no_grad():
                model(**_forward_kwargs(batch))
            return None, None, torch.zeros((), dtype=torch.int, device="cuda")
        batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, kind == "Fr")
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
            is_first_microbatch=(mb_idx == 0),
            current_microbatch=mb_idx,
            is_last_stage=is_last,
        )
        return input_tensor, output_tensor, num_tokens

    mb_idx = 0
    with no_sync_func():
        for parent_batch in data_iterator:
            parent = get_batch(
                iter([(parent_batch, {})]),
                _BATCH_KEYS,
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
            )
            chunks, units = _plan(parent, chunk_size, policy, runtime.next_group_offset(None))
            used_max = max((c.group for c in chunks if c.group is not None), default=None)
            runtime.next_group_offset(used_max)
            ops = gen_unit_ops(units, chunks, keep_last_k, pp=p2p_communicator.total_stages, stage=p2p_communicator.current_stage)

            for kind, unit_id in ops:
                unit = units[unit_id]
                if kind == "B":
                    if not is_last:
                        # Backward comes from the next stage.
                        output_tensor_grad = p2p_communicator.recv_backward(send_tensor_shapes, is_last)
                    else:
                        # Last stage computes the loss gradient itself.
                        output_tensor_grad = None
                    input_tensor, output_tensor = in_flight.pop(unit_id)
                    # TODO(verify): on the last stage, output_tensor is the loss and
                    # backward_step needs the same grad placeholder Megatron uses.
                    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                    if not is_first:
                        p2p_communicator.send_backward(input_tensor_grad, is_first)
                    for group in unit.groups:
                        runtime.release_group(group)
                    continue

                input_tensor = None if is_first else p2p_communicator.recv_forward(recv_tensor_shapes, is_first)
                input_tensor, output_tensor, num_tokens = _run_unit_forward(unit, parent, kind, input_tensor, unit_id, mb_idx)
                total_num_tokens += num_tokens
                if output_tensor is not None:
                    in_flight[unit_id] = (input_tensor, output_tensor)
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                if not is_last:
                    p2p_communicator.send_forward(output_tensor, is_last)
            mb_idx += 1

    # TODO(verify): anything left in in_flight here means the op sequence did not
    # backward every unit -- that is a schedule bug, not just a leak.
    if in_flight:
        raise RuntimeError(f"chunkflow pipeline: {len(in_flight)} unit(s) forwarded but never backwarded")

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return forward_data_store
