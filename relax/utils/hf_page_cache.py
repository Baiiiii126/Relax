# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Node-scoped Linux page-cache warmup for HF checkpoint directories.

Shared by the Megatron bridge loader (main model) and the SGLang engine
launcher (rollout / genrm / teacher). One flag — ``--warm-hf-checkpoint-page-
cache`` — turns warmup on for all of them. Coordination is via a per-node
marker under ``/dev/shm`` keyed by the absolute checkpoint path, so warming the
same path from multiple callers on the same node performs the read once.
"""

import os

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def warm_hf_checkpoint_page_cache(source_path: str) -> None:
    """Pre-fault the HF checkpoint into the Linux page cache, once per node per
    checkpoint path.

    NFS-backed safetensors files are accessed via mmap by the loader, so the
    first per-page touch incurs a synchronous small-read round-trip (we have
    measured ~20 MB/s effective throughput from ``aten::cat``). Reading the
    files end-to-end once promotes them to the page cache, after which the
    loader's lazy tensor reads are memory-fast.

    Implementation is pure-Python (``open(...).read()`` in chunks) — no shell
    invocation, so dynamic paths cannot inject commands.

    Coordination — advisory ``flock`` + tmpfs marker keyed by ``abs_path``:

    - Every caller opens the lock file and acquires an exclusive ``flock``.
    - Inside the lock, if the marker already exists and names this path,
      warmup is a no-op (someone else on the node already did the read).
    - Otherwise the current caller streams the files, writes the marker, and
      releases the lock; subsequent callers on the same node see the marker
      and return immediately.
    - The marker lives under ``/dev/shm`` (tmpfs), so it naturally clears on
      reboot — avoiding stale-marker / cleared-cache mismatches.
    - Errors (missing path, read failure) are logged and swallowed: warmup
      is a best-effort optimization, never a correctness gate.

    Safe to call from any process (megatron rank, SGLang Ray actor). Callers
    with the same checkpoint path serialize on the flock; the first winner
    does the work, everyone else sees the marker and returns.
    """
    import fcntl
    import glob
    import hashlib
    import time

    if not source_path:
        return
    abs_path = os.path.abspath(source_path)
    if not os.path.isdir(abs_path):
        return

    digest = hashlib.sha1(abs_path.encode()).hexdigest()[:16]
    marker_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    lock_path = f"{marker_dir}/relax_hf_warmup_{digest}.lock"
    done_path = f"{marker_dir}/relax_hf_warmup_{digest}.done"

    def _marker_says_warm() -> bool:
        try:
            with open(done_path) as df:
                return df.read().strip() == abs_path
        except OSError:
            return False

    def _stream_files_to_devnull(files: list[str]) -> int:
        """Read each file end-to-end so the kernel pulls every page into the
        page cache.

        Pure-Python — no shell, no command injection surface. Returns total
        bytes read. Shows a per-file tqdm progress bar.
        """
        from tqdm import tqdm

        chunk = 8 * 1024 * 1024  # 8 MiB, large enough that read syscall overhead is negligible
        total = 0
        pbar = tqdm(files, desc="warming HF ckpt", unit="file", dynamic_ncols=True)
        for fp in pbar:
            pbar.set_postfix_str(os.path.basename(fp))
            try:
                with open(fp, "rb") as fh:
                    while True:
                        data = fh.read(chunk)
                        if not data:
                            break
                        total += len(data)
            except OSError as exc:
                logger.warning(f"HF checkpoint warmup: skipping {fp} due to read error: {exc}")
        pbar.close()
        return total

    try:
        lf = open(lock_path, "w")
    except OSError as e:
        logger.warning(f"HF checkpoint warmup: cannot open lock file {lock_path}: {e}")
        return
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            if _marker_says_warm():
                logger.info(f"HF checkpoint page cache already warm on this node: {abs_path}")
                return
            files = sorted(
                glob.glob(os.path.join(abs_path, "*.safetensors")) + glob.glob(os.path.join(abs_path, "*.bin"))
            )
            if not files:
                logger.info(f"HF checkpoint warmup: no *.safetensors / *.bin under {abs_path}, skipping")
                return
            t0 = time.time()
            logger.info(f"Warming HF checkpoint page cache on this node: {abs_path} ({len(files)} files)")
            total_bytes = _stream_files_to_devnull(files)
            elapsed = time.time() - t0
            throughput_mb = total_bytes / max(elapsed, 1e-6) / (1024 * 1024)
            try:
                with open(done_path, "w") as df:
                    df.write(abs_path)
            except OSError as e:
                logger.warning(f"HF checkpoint warmup: cannot write marker {done_path}: {e}")
            logger.info(
                f"HF checkpoint page cache warmed in {elapsed:.1f}s "
                f"({total_bytes / (1024 * 1024):.0f} MiB, {throughput_mb:.0f} MiB/s)"
            )
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    finally:
        lf.close()
