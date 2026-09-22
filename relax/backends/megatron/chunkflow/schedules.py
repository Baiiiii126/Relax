# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The chunkflow schedule: chunks are the units that flow through the model.

``streaming_schedules.py`` is the template -- keyword-only signature, one-shot
iterators handed to ``forward_step``, gradients accumulated under ``no_sync``
with one ``finalize_model_grads`` at the end. The difference is the unit of
work: a chunk (or a batch of independent chunks) instead of a micro-batch, in
the order ``gen_unit_ops`` prescribes.

For PP=1 that order is ascending forward / descending backward. Descending is
not cosmetic: a group's chunk ``m`` accumulates dK/dV from every chunk ``j >= m``,
so it can only backward once the later ones have.

This is the no-recompute path: every unit keeps its activation, so no ``Fp``/``Fr``
is emitted. The recompute policy slots in at the op dispatch below without
changing anything else.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state

from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR, attach_runtime
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.unit_batch import slice_parent_batch
from relax.backends.megatron.chunkflow.units import build_micro_batches, gen_unit_ops
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Keys ``forward_step`` would have asked ``get_batch`` for. Kept in sync with
# model.py's list; a key absent from the rollout data comes back None.
_BATCH_KEYS = (
    "tokens",
    "packed_seq_params",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "log_probs",
    "ref_log_probs",
    "values",
    "advantages",
    "returns",
    "rollout_log_probs",
    "max_seq_lens",
)


def _plan(parent: dict, chunk_size: int, policy: str, group_offset: int):
    """Chunks and execution units for one processed micro-batch.

    ``build_chunks`` numbers a dependency group by the sequence's index in this
    micro-batch, so group ids repeat from one micro-batch to the next -- and the
    runtime's stores live on the model across all of them. Offset by micro-batch
    so a group id is unique for the whole step, not just the batch.
    """
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    chunks = build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
    for chunk in chunks:
        if chunk.group is not None:
            chunk.group += group_offset
    return chunks, build_micro_batches(chunks, policy)


def _forward_kwargs(batch: dict) -> dict:
    return {
        "input_ids": batch["tokens"],
        "position_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seq_params": batch["packed_seq_params"],
        "loss_mask": batch["full_loss_masks"],
    }


def chunkflow_forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,  # ignored -- the iterator drives the count
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    **kwargs,
):
    """PP=1 chunkflow schedule."""
    del seq_length, micro_batch_size, decoder_seq_length, collect_non_loss_data, first_val_step
    if isinstance(model, list):
        assert len(model) == 1, "chunkflow schedule requires a single model chunk"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "chunkflow schedule requires a single iterator"
        data_iterator = data_iterator[0]

    # Unlike the streaming schedules -- which get a StreamingTQIterator and pull
    # until StopIteration -- the standard path hands us a DataIterator, which has
    # no __iter__: micro-batches come from get_next() and the caller owns the count.
    num_mb = num_microbatches
    if isinstance(num_mb, (list, tuple)):
        num_mb = num_mb[0]
    elif torch.is_tensor(num_mb):
        num_mb = int(num_mb.reshape(-1)[0])
    num_mb = int(num_mb)

    from megatron.training.global_vars import get_args

    from relax.backends.megatron.data import get_batch
    from relax.backends.megatron.loss import loss_function

    args = get_args()
    chunk_size = int(args.chunkflow_chunk_size)
    policy = getattr(args, "chunkflow_pack_policy", "staggered")
    # No recompute yet, so every chunk of a group keeps its activation. A group
    # larger than this would need the Fp/Fr path.
    keep_last_k = getattr(args, "chunkflow_keep_last_k", 0) or 1 << 30
    runtime = attach_runtime(model, chunk_size)

    config = model.config if hasattr(model, "config") else None
    no_sync_func = getattr(config, "no_sync_func", None) or contextlib.nullcontext

    forward_data_store: list = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    with no_sync_func():
        for _ in range(num_mb):
            parent_batch = data_iterator.get_next(_BATCH_KEYS)
            # get_batch's non-DataIterator branch unpacks a (batch, meta) pair
            # off the iterator, the shape StreamingTQIterator yields.
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
            ops = gen_unit_ops(units, chunks, keep_last_k, pp=1, stage=0)

            import os as _os

            if _os.environ.get("RELAX_CHUNKFLOW_TRACE") == "1":
                from megatron.core import parallel_state as _mpu

                if _mpu.get_data_parallel_rank() == 0:
                    logger.info(
                        f"[chunkflow-trace] microbatch: {len(chunks)} chunks, "
                        f"groups={sorted({(c.group, c.group_idx) for c in chunks if c.group is not None})}, ops={ops}"
                    )

            pending: dict[int, torch.Tensor] = {}
            for kind, unit_id in ops:
                unit = units[unit_id]
                if kind == "B":
                    loss = pending.pop(unit_id)
                    if not forward_only:
                        loss.backward()
                    for group in unit.groups:
                        runtime.release_group(group)
                    continue

                # No-recompute path: nothing emits Fp/Fr, so every forward builds
                # the graph its own backward will use.
                batch = slice_parent_batch(parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args)
                batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                output = model(**_forward_kwargs(batch))
                loss, num_tokens, log = loss_function(args, batch, 1, output)
                total_num_tokens += num_tokens
                forward_data_store.append(log)
                pending[unit_id] = loss

    if getattr(config, "finalize_model_grads_func", None) is not None and not forward_only:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        pg = ProcessGroupCollection()
        pg.tp = parallel_state.get_tensor_model_parallel_group()
        pg.cp = parallel_state.get_context_parallel_group()
        pg.pp = parallel_state.get_pipeline_model_parallel_group()
        pg.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=False)
        pg.embd = parallel_state.get_embedding_group(check_initialized=False)
        pg.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if getattr(config, "calculate_per_token_loss", False) else None,
            pg_collection=pg,
        )

    return forward_data_store
