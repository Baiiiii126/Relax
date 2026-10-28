# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply only the opt-in MoE metadata patch to an existing Megatron
installation."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path


TARGET = "megatron/core/transformer/moe/token_dispatcher.py"
PATCH = Path(__file__).resolve().parents[2] / "docker/patch/latest/megatron.patch"


def apply_patch(root: Path, check_only: bool = False) -> str:
    root = root.resolve()
    if not (root / TARGET).is_file():
        raise ValueError(f"Missing {root / TARGET}")
    sections = [
        section for section in PATCH.read_text().split("diff --git ") if section.startswith(f"a/{TARGET} b/{TARGET}\n")
    ]
    if len(sections) != 1:
        raise ValueError("Expected exactly one dispatcher patch section")
    patch_text = "diff --git " + sections[0]
    command = ["git", "apply", "--unsafe-paths", f"--directory={root}"]

    # Run outside any containing worktree's prefix; the include pattern is root-relative.
    def run(*options: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="relax-moe-patch-") as directory:
            return subprocess.run(
                [*command, *options, "-"], input=patch_text, cwd=directory, text=True, capture_output=True, check=False
            )

    if run("--reverse", "--check").returncode == 0:
        return "already applied"
    result = run("--check")
    if result.returncode:
        raise RuntimeError(f"Patch does not match this Megatron source; no files changed:\n{result.stderr}")
    if check_only:
        return "applicable (check only)"
    result = run()
    if result.returncode:
        raise RuntimeError(result.stderr)
    if "_relax_moe_metadata_patch_version = 1" not in (root / TARGET).read_text():
        raise RuntimeError("git apply returned without installing the metadata patch")
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, help="Directory containing megatron/core")
    parser.add_argument("--check", action="store_true", help="Check compatibility without changing files")
    args = parser.parse_args()
    root = args.megatron_root
    if root is None:
        spec = importlib.util.find_spec("megatron.core")
        if spec is None or spec.origin is None:
            parser.error("Cannot locate megatron.core; pass --megatron-root")
        root = Path(spec.origin).resolve().parents[2]
    try:
        print(f"MoE metadata patch: {apply_patch(root, args.check)}; root={root}")
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
