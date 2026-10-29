# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
from collections.abc import Sequence

import torch


def execution_optimizations_enabled() -> bool:
    return os.environ.get("RELAX_CHUNKFLOW_EXECUTION_OPT", "0") == "1"


def int32_metadata(values: Sequence[int], device: torch.device) -> torch.Tensor:
    """Copy immutable host metadata on the consumer's current stream."""
    if not execution_optimizations_enabled():
        return torch.tensor(values, dtype=torch.int32, device=device)
    # .tensor(..., device="cuda") otherwise waits for the copy on every chunk.
    # No side stream: subsequent kernels see the copy through stream ordering.
    # This CPU tensor is never mutated or reused while its copy is in flight.
    return torch.tensor(values, dtype=torch.int32, device="cpu").to(device, non_blocking=True)
