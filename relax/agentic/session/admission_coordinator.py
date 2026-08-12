# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Ray glue for program-aware admission.

Houses the single-writer :class:`AdmissionBudgetCoordinator` actor, the Ray-backed
:class:`RayBudgetClient` the shards depend on, and a background poller that reconciles the
execution-token ledger from each engine's Prometheus ``/metrics``. All budget math lives in
the pure :class:`~relax.agentic.session.admission.BudgetState`; this module only adds Ray,
HTTP, and the wall clock.

Atomicity: the coordinator is an asyncio actor and every ledger mutation
(:meth:`reserve`/:meth:`release`/reconcile) is synchronous with no ``await`` in the middle,
so on the single event loop they cannot interleave — the same guarantee shard 0's semaphore
gives the request permit.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
import ray

from relax.agentic.session.admission import BudgetState, WorkerSnapshot
from relax.utils.http_utils import router_worker_base_urls
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

ADMISSION_BUDGET_COORDINATOR_NAME = "agentic_admission_budget_coordinator"

# KV-pool gauges we read from each engine's Prometheus /metrics. SGLang replicates
# these once per (tp_rank/pp_rank/...) label with an identical per-engine value, so the
# generic ``parse_prometheus_metrics`` (which SUMS same-name lines) inflates absolute
# gauges by the TP degree. These are per-engine pool readings, not additive shards, so we
# aggregate by MAX across label sets instead — which also yields a conservative worst-case
# pressure under dp-attention.
_KV_GAUGE_NAMES = (
    "sglang:max_total_num_tokens",
    "sglang:num_used_tokens",
    "sglang:token_usage",
    "sglang:full_token_usage",
)
_PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(.+)$")


def _parse_engine_kv_gauges(text: str) -> dict[str, float]:
    """Aggregate the KV-pool gauges we need by MAX across label sets.

    See :data:`_KV_GAUGE_NAMES`: summing TP-replicated ``max_total_num_tokens``
    (as the generic parser does) would over-count capacity by the TP degree and
    deflate the derived usage ratio by the same factor, so a saturated engine
    reads as nearly idle. Taking the max per gauge is correct for the
    replicated capacity and conservative for pressure.
    """
    gauges: dict[str, float] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name not in _KV_GAUGE_NAMES:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        prev = gauges.get(name)
        gauges[name] = value if prev is None else max(prev, value)
    return gauges


@ray.remote
class AdmissionBudgetCoordinator:
    """Single logical writer for cluster-wide execution-token reservations.

    Wraps one pure :class:`BudgetState`. A background poller scrapes engine
    ``/metrics`` every reconcile interval to refresh the ceiling and per-worker
    pressure and to expire stale leases. Fails open: when signals are
    stale/absent the ledger reports ``degraded`` and callers bypass to the
    existing request limiter.
    """

    def __init__(self, *, args: Any) -> None:
        self._router_ip = getattr(args, "sglang_router_ip", None)
        self._router_port = getattr(args, "sglang_router_port", None)
        self._reconcile_interval_s = max(0.1, float(getattr(args, "agentic_admission_reconcile_interval_s", 2.0)))
        # A snapshot is "stale" after a few missed reconciles; then reserve() degrades to bypass.
        staleness_s = max(self._reconcile_interval_s * 3.0, 5.0)
        self._state = BudgetState(
            headroom=float(getattr(args, "agentic_admission_headroom", 0.90)),
            pressure_threshold=float(getattr(args, "agentic_admission_pressure_threshold", 0.92)),
            emergency_reserve_frac=float(getattr(args, "agentic_admission_emergency_reserve_frac", 0.05)),
            lease_ttl_s=float(getattr(args, "agentic_admission_lease_ttl_s", 600.0)),
            staleness_s=staleness_s,
        )
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        self._poll_task: asyncio.Task | None = None
        self._last_reconcile_ok = False

    async def start(self) -> None:
        """Do one immediate reconcile (reduce cold warmup) and launch the
        poller."""
        try:
            await self._reconcile_once()
        except Exception as exc:
            logger.warning("Admission coordinator initial reconcile failed: %s", exc)
        self._ensure_poller()

    def _ensure_poller(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def reserve(self, req: dict) -> dict:
        self._ensure_poller()
        now = time.monotonic()
        self._state.expire_ttl(now)
        grant = self._state.reserve(
            tokens=int(req.get("tokens", 0)),
            dispatch_id=str(req.get("dispatch_id", "")),
            admission_decision_id=str(req.get("admission_decision_id", "")),
            aged=bool(req.get("aged", False)),
            now=now,
        )
        return grant.as_dict()

    async def release(self, lease_id: str, actual_tokens: int | None = None) -> None:
        self._state.release(lease_id, actual_tokens=actual_tokens)

    async def capacity_hint(self) -> dict:
        self._ensure_poller()
        return self._state.capacity_hint(time.monotonic())

    async def health(self) -> dict:
        # reset_peak drains the running usage window: health() is the once-per-step metrics read,
        # so the returned peak/mean cover the whole step rather than the drain-instant sample.
        hint = self._state.capacity_hint(time.monotonic(), reset_peak=True)
        hint["last_reconcile_ok"] = self._last_reconcile_ok
        hint["reconcile_interval_s"] = self._reconcile_interval_s
        return hint

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reconcile_interval_s)
            try:
                await self._reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_reconcile_ok = False
                logger.warning("Admission coordinator reconcile failed: %s", exc)

    async def _reconcile_once(self) -> None:
        snapshots = await self._fetch_worker_snapshots()
        now = time.monotonic()
        self._state.reconcile(snapshots, now=now)
        self._state.expire_ttl(now)
        self._last_reconcile_ok = True

    async def _fetch_worker_snapshots(self) -> list[WorkerSnapshot]:
        if not self._router_ip or not self._router_port:
            return []

        base = f"http://{self._router_ip}:{self._router_port}"
        raw_urls: list[str] = []
        try:
            resp = await self._client.get(f"{base}/workers")
            resp.raise_for_status()
            workers = resp.json().get("workers", [])
            raw_urls = [
                worker.get("url")
                for worker in workers
                if isinstance(worker, dict) and worker.get("url") and bool(worker.get("is_healthy", False))
            ]
        except Exception:
            try:
                resp = await self._client.get(f"{base}/list_workers")
                resp.raise_for_status()
                raw_urls = list(resp.json().get("urls", []))
            except Exception as exc:
                logger.debug("Admission coordinator worker discovery failed: %s", exc)
                return []

        engine_urls = router_worker_base_urls(u for u in raw_urls if isinstance(u, str) and u)
        snapshots: list[WorkerSnapshot] = []
        for url in engine_urls:
            try:
                resp = await self._client.get(f"{url}/metrics")
                resp.raise_for_status()
                gauges = _parse_engine_kv_gauges(resp.text)
            except Exception as exc:
                logger.debug("Admission coordinator /metrics scrape failed for %s: %s", url, exc)
                continue
            # Gauges are aggregated by MAX (see _parse_engine_kv_gauges): TP replicates
            # max_total_num_tokens per rank, so summing it would inflate capacity and deflate
            # usage by the TP degree. Prefer the engine's own usage ratio; fall back to the
            # absolute counts only when the ratio gauge is absent.
            max_total = int(gauges.get("sglang:max_total_num_tokens", 0) or 0)
            num_used = int(gauges.get("sglang:num_used_tokens", 0) or 0)
            usage = gauges.get("sglang:token_usage") or gauges.get("sglang:full_token_usage") or 0.0
            if usage <= 0.0 and max_total > 0:
                usage = num_used / max_total
            usage = min(1.0, max(0.0, usage))
            snapshots.append(
                WorkerSnapshot(
                    engine_id=url,
                    max_total_num_tokens=max_total,
                    token_usage=usage,
                    num_used_tokens=num_used,
                    healthy=True,
                )
            )
        return snapshots


class RayBudgetClient:
    """A :class:`~relax.agentic.session.admission.BudgetClient` backed by the
    actor handle.

    Bounds the reserve RPC with a timeout and swallows release errors (the
    lease TTL reclaims anything a failed release would leak), so the shard's
    admission path can always fail open.
    """

    def __init__(self, handle: Any, *, reserve_timeout_s: float) -> None:
        self._handle = handle
        self._reserve_timeout_s = max(0.01, reserve_timeout_s)
        self._hint_timeout_s = max(1.0, reserve_timeout_s * 5.0)

    async def reserve(self, req: dict) -> dict:
        return await asyncio.wait_for(self._handle.reserve.remote(req), timeout=self._reserve_timeout_s)

    async def release(self, lease_id: str, actual_tokens: int | None = None) -> None:
        try:
            await self._handle.release.remote(lease_id, actual_tokens)
        except Exception as exc:
            logger.debug("Admission lease release failed (TTL will reclaim): %s", exc)

    async def capacity_hint(self) -> dict:
        return await asyncio.wait_for(self._handle.capacity_hint.remote(), timeout=self._hint_timeout_s)


def get_or_create_admission_coordinator(args: Any) -> Any:
    """Create (or fetch) the detached, named coordinator actor; returns the
    handle.

    Idempotent via ``get_if_exists`` so re-deploys reuse the running
    coordinator.
    """
    return AdmissionBudgetCoordinator.options(
        name=ADMISSION_BUDGET_COORDINATOR_NAME,
        lifetime="detached",
        num_cpus=0,
        get_if_exists=True,
        max_restarts=-1,
    ).remote(args=args)


def shutdown_admission_coordinator() -> None:
    """Best-effort teardown of the detached coordinator actor."""
    try:
        handle = ray.get_actor(ADMISSION_BUDGET_COORDINATOR_NAME)
    except Exception:
        return
    try:
        ray.kill(handle)
    except Exception as exc:
        logger.debug("Failed to kill admission coordinator: %s", exc)
