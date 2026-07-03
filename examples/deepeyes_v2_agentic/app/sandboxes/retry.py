# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-agnostic retry helper for *session-level* sandbox operations.

This is intentionally separate from :mod:`.backends._retry`, which retries
``backend.create_session`` calls. Here we retry an arbitrary user-supplied
coroutine (typically the ``init_code`` warm-up call after ``wait_ready``)
when it raises :class:`SandboxTransientError`.

Backends decide what counts as transient by raising
:class:`SandboxTransientError` from their session methods; this helper does
not inspect SDK-specific types or messages.
"""

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from relax.utils.logging_utils import get_logger

from .exceptions import SandboxTransientError


logger = get_logger(__name__)

T = TypeVar("T")


async def run_with_transient_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_backoff_s: float = 0.1,
    max_backoff_s: float = 2.0,
    label: str = "sandbox.op",
) -> T:
    """Call ``factory()`` up to ``max_attempts`` times, retrying only on
    :class:`SandboxTransientError`.

    ``factory`` is a zero-arg coroutine factory because awaiting the same
    coroutine twice is a runtime error -- the call site rebuilds the coroutine
    on every attempt.

    Backoff is exponential with full jitter, capped at ``max_backoff_s``. Any
    other exception (including :class:`SandboxError` itself) propagates
    immediately so non-transient failures fail fast.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await factory()
        except SandboxTransientError as exc:
            last_exc = exc
            if attempt == max_attempts:
                logger.warning(
                    "%s: transient failure on final attempt %d/%d (%r); giving up",
                    label,
                    attempt,
                    max_attempts,
                    exc,
                )
                raise
            backoff = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            backoff += random.uniform(0, backoff)
            logger.warning(
                "%s: transient failure attempt %d/%d (%r); retrying in %.3fs",
                label,
                attempt,
                max_attempts,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc
