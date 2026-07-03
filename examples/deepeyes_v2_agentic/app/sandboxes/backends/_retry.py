# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Generic retry helper for ``backend.create_session`` calls.

Lifted unchanged in spirit from
``examples/deepeyes_v2/env_deepeyes_v2.py:228-294`` and made backend-agnostic:
backends pass the names of exception classes that should be treated as
*permanent* (no-retry) so the helper can decide without importing SDK-specific
types.
"""

import asyncio
import random
from typing import Awaitable, Callable, Sequence, TypeVar

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

T = TypeVar("T")

_TRANSIENT_SUBSTRINGS = (
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "Network connectivity",
    # nex gateway returns HTTP 500 with body "context deadline exceeded" when
    # its upstream sandbox lifecycle API times out under load. The failure is
    # gateway-side and resolves once queue depth drains, so treat it as
    # transient even though 500 is normally permanent.
    "context deadline exceeded",
)


def is_transient_create_error(
    exc: BaseException,
    *,
    permanent_exception_names: Sequence[str] = (),
) -> bool:
    """Decide whether a ``create_session`` failure is worth retrying.

    Strategy: retry on gateway / network class errors. Backends that ship their
    own ``*ApiException`` for permanent (HTTP 4xx) failures pass the class name
    via ``permanent_exception_names``.
    """
    msg = str(exc)
    if any(code in msg for code in _TRANSIENT_SUBSTRINGS):
        return True
    if type(exc).__name__ in permanent_exception_names:
        return False
    return True


async def create_with_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    permanent_exception_names: Sequence[str] = (),
    base_backoff: float = 1.0,
    log_label: str = "sandbox.create",
) -> T:
    """Call ``factory()`` until it succeeds or ``max_retries`` is exhausted.

    ``factory`` is a zero-arg coroutine factory so the call site can build the
    coroutine fresh on every attempt — awaiting the same coroutine twice is a
    runtime error.
    """
    max_retries = max(1, max_retries)
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries - 1 or not is_transient_create_error(
                exc, permanent_exception_names=permanent_exception_names
            ):
                raise
            backoff = base_backoff * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                log_label,
                attempt + 1,
                max_retries,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc
