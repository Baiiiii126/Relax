#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""验证脚本 — 端到端测一遍 deepeyes_v2_kernel.sif + ApptainerJupyterBackend。

用法：
    # 默认验证脚本同目录下的 deepeyes_v2_kernel.sif
    python examples/deepeyes_v2/apptainer_env/verify_kernel.py

    # 也可以显式指定 sif 路径
    python examples/deepeyes_v2/apptainer_env/verify_kernel.py /abs/path/foo.sif

依赖：
    pip install jupyter_client      # 后端需要 (host 侧)
    apptainer 在 PATH               # `which apptainer`

退出码：
    0  全部 PASS
    1  任一阶段 FAIL

验证阶段：
    [0] preflight                  apptainer 在 PATH，sif 文件存在
    [1] apptainer exec import      sif 里 python 能 import 所有必备依赖
    [2] backend create + ready     ApptainerJupyterBackend 启动 + jupyter wait_for_ready
    [3] stateful kernel            run_code 之间变量持久化 (deepeyes_v2 强依赖)
    [4] matplotlib + bind mount    plt.savefig 写入 /tmp/_relax_imgs，host 侧能 list/read
    [5] PIL serialize roundtrip    /tmp/_relax_inputs 写 png，host 侧 read_bytes 反序列化一致
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from datetime import timedelta
from io import BytesIO
from pathlib import Path


# 把仓库根加到 sys.path，这样无论从哪 cwd 跑都能 import relax.*
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]  # apptainer_env/ -> deepeyes_v2/ -> examples/ -> root
sys.path.insert(0, str(_REPO_ROOT))


def _print_stage(idx: "int | str", name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{idx}] {mark}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)


def _stage0_preflight(sif_path: Path) -> tuple[bool, str]:
    apptainer = shutil.which("apptainer")
    if apptainer is None:
        return False, "apptainer not on PATH"
    if not sif_path.is_file():
        return False, f"sif missing: {sif_path}"
    return True, f"apptainer={apptainer} sif_size={sif_path.stat().st_size}"


def _stage1_exec_imports(sif_path: Path) -> tuple[bool, str]:
    cmd = [
        "apptainer",
        "exec",
        "--cleanenv",
        str(sif_path),
        "python",
        "-c",
        "import ipykernel, PIL, matplotlib, autopep8, numpy; print('all imports OK')",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "timeout (>120s)"
    ok = proc.returncode == 0 and "all imports OK" in proc.stdout
    if ok:
        return True, proc.stdout.strip()
    err_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    return False, f"rc={proc.returncode} stderr_tail={err_tail!r}"


async def _probe_backend_with_flags(
    sif_path: Path,
    *,
    label: str,
    cleanenv: bool,
    pid_isolation: bool,
    writable_tmpfs: bool,
    home_dir: "str | None",
) -> tuple[bool, str]:
    """Drive the REAL ApptainerJupyterBackend.create_session path with a
    specific flag combo. Same code path as stage 2 — we just isolate one
    variable (apptainer flags) at a time so we can pinpoint which flag causes
    the `Kernel died before replying to kernel_info` handshake failure.

    On failure, reads the backend's stderr+stdout capture files (left in
    host_conn_dir by the production backend) and includes their tails in the
    returned detail, so the user sees what ipykernel actually printed.
    """
    from examples.deepeyes_v2.sandboxes.backends.apptainer_jupyter_backend import (
        ApptainerJupyterBackend,
    )

    backend = ApptainerJupyterBackend(
        image=str(sif_path),
        bind_paths={"/tmp/_relax_imgs": "imgs", "/tmp/_relax_inputs": "inputs"},
        # Short timeout: each failing combo must give up quickly so the
        # bisection completes in a reasonable wall-clock budget.
        kernel_ready_timeout_s=20.0,
        # Cut retries to 1 — we're probing for *the* failure, not papering
        # over flaky retries.
        create_max_retries=1,
        cleanenv=cleanenv,
        pid_isolation=pid_isolation,
        writable_tmpfs=writable_tmpfs,
        home_dir=home_dir,
    )

    session = None
    try:
        try:
            session = await backend.create_session(
                metadata={"probe": label},
                request_timeout=timedelta(seconds=30),
            )
        except Exception as e:
            # The backend's WARNING log line already includes argv + stderr +
            # stdout tails — that's the diagnostic gold. Surface a short
            # one-liner here so the user can scan the probe matrix.
            return False, f"create_session raised {type(e).__name__}: {str(e)[:160]}"

        # Reached `_ready=True` — kernel handshake succeeded. Also do one
        # round-trip run_code to confirm the channel is actually usable
        # (heartbeat-alive but shell-dead would otherwise pass here).
        try:
            await session.wait_ready(timeout=timedelta(seconds=10))
            result = await session.run_code("print('hi from kernel')", timeout=10.0)
        except Exception as e:
            return False, f"post-ready run_code raised {type(e).__name__}: {str(e)[:160]}"

        ok = result.status == "success" and "hi from kernel" in result.stdout
        return ok, (
            f"status={result.status} stdout={result.stdout.strip()[:80]!r} stderr={result.stderr.strip()[:80]!r}"
        )
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass


async def _stage1_5_bisect_apptainer_flags(sif_path: Path) -> tuple[bool, str]:
    """Bisect which apptainer flag breaks the full jupyter_client → ipykernel
    handshake on this host. Same code path as stage 2 (real backend, real
    AsyncKernelClient, real wait_for_ready + run_code round-trip).

    The backend's WARNING log line for each failing combo carries the full argv
    + stderr_tail + stdout_tail — that's where the actual signal is. The
    PASS/BAD matrix below is just a fast index into those WARNING lines.
    """
    cases = [
        # label,             cleanenv, pid,   writable_tmpfs, home_dir
        ("prod-all", True, True, True, "/root"),
        ("no-cleanenv", False, True, True, "/root"),
        ("no-pid", True, False, True, "/root"),
        ("no-writable-tmpfs", True, True, False, "/root"),
        ("no-home", True, True, True, None),
        ("minimal", False, False, False, None),
    ]
    results: list[tuple[str, bool, str]] = []
    for label, cleanenv, pid, writable, home in cases:
        ok, detail = await _probe_backend_with_flags(
            sif_path,
            label=label,
            cleanenv=cleanenv,
            pid_isolation=pid,
            writable_tmpfs=writable,
            home_dir=home,
        )
        print(f"      probe {label:<18} -> {'OK ' if ok else 'BAD'}  {detail}", flush=True)
        results.append((label, ok, detail))
    healthy = [label for label, ok, _ in results if ok]
    failed = [label for label, ok, _ in results if not ok]
    summary = f"healthy=[{','.join(healthy) or 'NONE'}] failed=[{','.join(failed) or 'NONE'}]"
    # Stage 1.5 passes iff ≥1 combo completes the full handshake + run_code —
    # gives the user a known-good fallback to plug into apptainer_config.yaml.
    return bool(healthy), summary


async def _stage2_to_5(sif_path: Path) -> bool:
    try:
        from examples.deepeyes_v2.sandboxes.backends.apptainer_jupyter_backend import ApptainerJupyterBackend
    except ImportError as e:
        _print_stage(2, "import ApptainerJupyterBackend", False, str(e))
        return False

    backend = ApptainerJupyterBackend(
        image=str(sif_path),
        bind_paths={
            "/tmp/_relax_imgs": "imgs",
            "/tmp/_relax_inputs": "inputs",
        },
        kernel_ready_timeout_s=120.0,
    )

    try:
        session = await backend.create_session(
            metadata={"verify": "1"},
            request_timeout=timedelta(seconds=120),
        )
    except Exception as e:
        _print_stage(2, "backend.create_session", False, repr(e))
        return False

    try:
        try:
            await session.wait_ready(timeout=timedelta(seconds=120))
            _print_stage(2, "session create + wait_ready", True)
        except Exception as e:
            _print_stage(2, "session wait_ready", False, repr(e))
            return False

        # ---------- Stage 3: stateful kernel ----------
        r1 = await session.run_code("a = 41 + 1\nprint(a)", timeout=20.0)
        r2 = await session.run_code("print(a + 8)", timeout=20.0)
        ok3 = (
            r1.status == "success"
            and r1.stdout.strip() == "42"
            and r2.status == "success"
            and r2.stdout.strip() == "50"
        )
        _print_stage(
            3,
            "stateful kernel (var persists across run_code)",
            ok3,
            detail=f"r1.stdout={r1.stdout.strip()!r} r2.stdout={r2.stdout.strip()!r} r1.stderr={r1.stderr[:120]!r}",
        )

        # ---------- Stage 4: matplotlib + bind mount ----------
        plot_code = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "fig = plt.figure(figsize=(2, 2))\n"
            "plt.plot([1, 2, 3], [1, 4, 9])\n"
            "fig.savefig('/tmp/_relax_imgs/plot.png')\n"
            "plt.close(fig)\n"
            "print('SAVED')\n"
        )
        r3 = await session.run_code(plot_code, timeout=30.0)
        plot_files = await session.list_files("/tmp/_relax_imgs")
        match = [f for f in plot_files if f.path.endswith("plot.png")]
        ok4 = r3.status == "success" and "SAVED" in r3.stdout and len(match) == 1 and (match[0].size or 0) > 100
        _print_stage(
            4,
            "matplotlib savefig -> /tmp/_relax_imgs (host list_files)",
            ok4,
            detail=f"r3.status={r3.status} files={[(f.path, f.size) for f in plot_files]} stderr={r3.stderr[:120]!r}",
        )

        # ---------- Stage 5: PIL serialize + host roundtrip ----------
        pil_code = (
            "from PIL import Image\n"
            "img = Image.new('RGB', (16, 16), (255, 0, 0))\n"
            "img.save('/tmp/_relax_inputs/red.png')\n"
            "print('PIL_SAVED')\n"
        )
        r4 = await session.run_code(pil_code, timeout=20.0)
        in_files = await session.list_files("/tmp/_relax_inputs")
        red = [f for f in in_files if f.path.endswith("red.png")]
        ok5 = False
        detail5 = f"r4.status={r4.status} files={[(f.path, f.size) for f in in_files]}"
        if r4.status == "success" and red:
            blob = await session.read_bytes("/tmp/_relax_inputs/red.png")
            png_sig_ok = blob[:8] == b"\x89PNG\r\n\x1a\n"
            pil_ok = True
            try:
                from PIL import Image as _PIL  # host-side optional

                im = _PIL.open(BytesIO(blob))
                pil_ok = im.size == (16, 16) and im.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
                detail5 += f" host_pil={im.size} pixel00={im.convert('RGB').getpixel((0, 0))}"
            except ImportError:
                detail5 += " (host PIL not installed; only PNG signature checked)"
            ok5 = png_sig_ok and pil_ok
        _print_stage(
            5,
            "PIL save (kernel) -> read_bytes (host) -> Image.open roundtrip",
            ok5,
            detail=detail5,
        )

        return ok3 and ok4 and ok5
    finally:
        try:
            await session.close()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] session.close raised: {e!r}", flush=True)


async def _amain(sif_path: Path) -> bool:
    print(f"verifying sif: {sif_path}\n", flush=True)

    ok0, d0 = _stage0_preflight(sif_path)
    _print_stage(0, "preflight (apptainer + sif present)", ok0, d0)
    if not ok0:
        return False

    ok1, d1 = _stage1_exec_imports(sif_path)
    _print_stage(1, "apptainer exec sif python -c 'import deps'", ok1, d1)
    if not ok1:
        return False

    # Stage 1.5: bisect apptainer flag combinations to isolate which flag (if
    # any) is breaking the full jupyter_client ↔ ipykernel handshake on this
    # host. Drives the SAME ApptainerJupyterBackend.create_session path as
    # stage 2 (so jupyter_client is in scope), varying only the apptainer
    # flags. The backend's WARNING line includes stdout+stderr tails — that's
    # where the actual Python startup error (if any) lives. Verbose by design.
    print("  [1.5] probing apptainer flag combinations via real backend path:", flush=True)
    ok15, d15 = await _stage1_5_bisect_apptainer_flags(sif_path)
    _print_stage("1.5", "apptainer flag bisection (>=1 combo passes full handshake)", ok15, d15)
    if not ok15:
        # No flag combination works — stage 2 has zero chance. Bail early
        # rather than repeating the same error in stage 2.
        return False

    return await _stage2_to_5(sif_path)


def main() -> None:
    if len(sys.argv) > 1:
        sif = Path(sys.argv[1]).resolve()
    else:
        sif = (_HERE.parent / "deepeyes_v2_kernel.sif").resolve()

    try:
        ok = asyncio.run(_amain(sif))
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(130)
    print("\n" + ("ALL STAGES PASS" if ok else "VERIFY FAILED"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
