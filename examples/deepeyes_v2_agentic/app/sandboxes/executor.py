# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Singleton executor that mints sandbox sessions with bounded concurrency.

Mirrors :class:`relax.engine.rewards.RewardExecutor` with three deliberate
differences:

* Sessions are long-lived (one per trajectory, many turns); rewards do
  one-shot work.
* The semaphore bounds ``create_session`` only — once a session is
  provisioned the slot is released so per-turn ``run_code`` calls scale with
  active trajectories. Matches the behaviour of the
  ``_get_sandbox_create_semaphore`` helper that previously lived in
  ``examples/deepeyes_v2/env_deepeyes_v2.py``.
* No Ray actor pool — sessions live in the rollout-worker process; remote
  work happens inside the backend (HTTP for nexsandbox, ZMQ for the docker
  jupyter backend).
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator, Optional

from relax.utils.logging_utils import get_logger

from .base import BaseSandbox, BaseSandboxSession, SandboxCapability
from .exceptions import SandboxCapabilityError


logger = get_logger(__name__)


class SandboxExecutor:
    """Process-wide controller for sandbox session creation."""

    _instance: "SandboxExecutor | None" = None

    def __init__(
        self,
        backend: BaseSandbox,
        *,
        max_concurrent_creates: int = 64,
        ensure_session_timeout_s: int = 240,
    ) -> None:
        self._backend = backend
        self._max_concurrent_creates = max_concurrent_creates
        self._ensure_session_timeout_s = ensure_session_timeout_s
        self._create_sem: Optional[asyncio.Semaphore] = None

    @classmethod
    def get_or_create(cls, backend: BaseSandbox, **kwargs) -> "SandboxExecutor":
        if cls._instance is None:
            cls._instance = cls(backend, **kwargs)
            logger.info(
                "SandboxExecutor: backend=%r capabilities=%s max_concurrent_creates=%d ensure_session_timeout_s=%d",
                backend.name,
                sorted(c.value for c in backend.capabilities),
                cls._instance._max_concurrent_creates,
                cls._instance._ensure_session_timeout_s,
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test-only.

        Drops the singleton.
        """
        cls._instance = None

    @property
    def backend(self) -> BaseSandbox:
        return self._backend

    def _ensure_sem(self) -> asyncio.Semaphore:
        if self._create_sem is None:
            self._create_sem = asyncio.Semaphore(self._max_concurrent_creates)
        return self._create_sem

    @asynccontextmanager
    async def acquire_session(
        self,
        *,
        metadata: Optional[dict] = None,
        require: frozenset = frozenset(),
    ) -> AsyncIterator[BaseSandboxSession]:
        """Provision a session, yield it for the duration of the ``with``
        block, then ``close()`` it on exit."""
        missing = set(require) - set(self._backend.capabilities)
        if missing:
            raise SandboxCapabilityError(
                f"Backend {self._backend.name!r} missing required capabilities: "
                f"{sorted(c.value if isinstance(c, SandboxCapability) else c for c in missing)}"
            )
        sem = self._ensure_sem()
        async with sem:
            session = await asyncio.wait_for(
                self._backend.create_session(
                    metadata=metadata,
                    request_timeout=timedelta(seconds=self._ensure_session_timeout_s),
                ),
                timeout=self._ensure_session_timeout_s,
            )
        try:
            yield session
        finally:
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("SandboxExecutor: session.close() raised: %s", exc)
