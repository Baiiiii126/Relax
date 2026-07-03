# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend registry for the sandbox abstraction.

Backends register themselves at module import time via
:func:`register_backend`. Built-in backends are best-effort imported so that
optional dependencies (``nexsandbox``, ``apptainer`` CLI, ``jupyter_client``)
never break the import of :mod:`examples.deepeyes_v2.sandboxes` for users that do
not need them.
"""

import importlib
from typing import Type

from relax.utils.logging_utils import get_logger

from ..base import BaseSandbox


logger = get_logger(__name__)

_REGISTRY: dict[str, Type[BaseSandbox]] = {}


def register_backend(name: str, cls: Type[BaseSandbox]) -> None:
    """Register a sandbox backend class under ``name``.

    Re-registering the same ``(name, cls)`` pair is a no-op; registering a
    different class under an existing name raises ``ValueError`` so accidental
    shadowing is loud.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Sandbox backend {name!r} is already registered as {existing.__module__}.{existing.__qualname__}"
        )
    _REGISTRY[name] = cls


def _autoregister_builtins() -> None:
    for mod_name in ("nexsandbox_backend", "apptainer_jupyter_backend"):
        try:
            importlib.import_module(f".{mod_name}", __name__)
        except Exception as exc:  # noqa: BLE001
            logger.debug("sandbox backend %s skipped (%s): %s", mod_name, type(exc).__name__, exc)


_autoregister_builtins()
