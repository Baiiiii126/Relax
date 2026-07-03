# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-agnostic sandbox abstractions.

Recipes consume a ``BaseSandboxSession`` returned by
``SandboxExecutor.acquire_session``. A backend advertises what it can do via
``capabilities`` (a frozenset of :class:`SandboxCapability`); the executor
checks the recipe's required capabilities once at acquisition time so mid-
trajectory mismatches are impossible.
"""

import abc
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Optional

from .exceptions import SandboxCapabilityError


RELAX_SANDBOX_DEFAULT_CODE_TIMEOUT_S: float = 200.0
RELAX_SANDBOX_DEFAULT_REQUEST_TIMEOUT_S: int = 60


class SandboxCapability(str, Enum):
    """Capabilities a backend may advertise."""

    STATEFUL_KERNEL = "stateful_kernel"
    SHELL_EXEC = "shell_exec"
    FILE_IO = "file_io"


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: Optional[int] = None
    is_dir: bool = False


@dataclass(frozen=True)
class ExecutionError:
    name: str
    value: str
    traceback: str = ""


@dataclass
class ExecutionResult:
    """Normalised return shape for code or shell execution.

    ``status`` is ``"success" | "error" | "timeout"``. ``raw`` carries the
    backend-specific payload for callers that need it.
    """

    status: str
    stdout: str = ""
    stderr: str = ""
    error: Optional[ExecutionError] = None
    raw: Any = None


class BaseSandboxSession(abc.ABC):
    """A single per-trajectory session against a sandbox backend.

    Subclasses must implement ``wait_ready``, ``run_code``, ``interrupt``, and
    ``close``. Default implementations of ``exec_shell`` / ``list_files`` /
    ``read_bytes`` / ``delete_files`` raise :class:`SandboxCapabilityError`;
    backends that support those features should override them and add the
    corresponding entry to ``capabilities``.
    """

    capabilities: frozenset = field(default_factory=frozenset)  # type: ignore[assignment]

    @abc.abstractmethod
    async def wait_ready(self, timeout: timedelta) -> None:
        """Block until the underlying execution context is ready to serve
        work."""

    @abc.abstractmethod
    async def run_code(
        self,
        code: str,
        *,
        timeout: float,
        language: str = "python",
    ) -> ExecutionResult:
        """Run ``code`` in the session's stateful kernel."""

    async def exec_shell(
        self,
        cmd: list,
        *,
        timeout: float,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        raise SandboxCapabilityError("shell_exec")

    async def list_files(self, path: str) -> list:
        raise SandboxCapabilityError("file_io")

    async def read_bytes(self, path: str) -> bytes:
        raise SandboxCapabilityError("file_io")

    async def delete_files(self, paths: list) -> None:
        raise SandboxCapabilityError("file_io")

    @abc.abstractmethod
    async def interrupt(self) -> None:
        """Interrupt the currently executing call (best effort)."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Tear down the session.

        Must be idempotent.
        """


class BaseSandbox(abc.ABC):
    """A backend able to mint :class:`BaseSandboxSession` instances."""

    name: str = "base"
    capabilities: frozenset = frozenset()

    @abc.abstractmethod
    async def create_session(
        self,
        *,
        metadata: Optional[dict] = None,
        request_timeout: timedelta,
    ) -> BaseSandboxSession:
        """Provision a fresh session.

        Must respect ``request_timeout``.
        """
