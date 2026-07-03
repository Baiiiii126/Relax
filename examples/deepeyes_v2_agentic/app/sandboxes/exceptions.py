# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exceptions raised by the sandbox abstraction layer.

Backends MUST translate their SDK-/runtime-specific exceptions into one of the
classes below before raising. Recipes catch :class:`SandboxError` (or one of
its subclasses) and rely on the translation contract — they should NOT need to
import any backend-side type.
"""


class SandboxError(Exception):
    """Base class for all sandbox-related errors."""


class SandboxCapabilityError(SandboxError):
    """Raised when a recipe asks a backend for a capability it does not
    advertise."""


class SandboxTimeout(SandboxError):
    """Raised when a sandbox operation exceeds its wall-clock budget."""


class SandboxCreateFailed(SandboxError):
    """Raised when ``backend.create_session`` fails after all retry
    attempts."""


class SandboxTransientError(SandboxError):
    """A session-level operation failed with a signal that suggests the next
    attempt may succeed (gateway 5xx, post-create routing flap, ipykernel iopub
    blip, etc.).

    Used by :func:`examples.deepeyes_v2.sandboxes.retry.run_with_transient_retry`
    to decide whether to retry. Backends raise this *only* when they have
    positive evidence of transience; everything else stays a plain
    :class:`SandboxError` so the caller fails fast.
    """
