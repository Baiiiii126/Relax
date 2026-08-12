# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Pure decision logic and execution-token ledger for program-aware admission.

This module is intentionally free of Ray, GPU, and I/O imports so the admission
policy and budget accounting can be unit-tested deterministically on CPU. All
time-dependent state takes an injected ``now`` (monotonic seconds) — nothing here
reads the wall clock.

Layering (see the RFC):
  * ``decide_admission_prelude`` handles the fail-open / identity / scope / protected
    short-circuits that never need the cluster-wide budget.
  * ``BudgetState`` is the pure ledger a single coordinator actor wraps: it grants,
    releases, and reconciles cluster-wide execution-token reservations.
  * ``interpret_budget_response`` maps a coordinator grant back to an ``AdmissionDecision``.

The admission layer never selects a worker and never interrupts in-flight decode;
final placement stays with the SGLang router, and the full replay payload keeps
generation correct regardless of any admission or signal failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AdmissionAction(str, Enum):
    BYPASS = "bypass"
    ADMIT = "admit"
    DEFER = "defer"


class AdmissionReason(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    MISSING_IDENTITY = "missing_identity"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    DEGRADED = "degraded"
    CAPACITY_AVAILABLE = "capacity_available"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    PRESSURE_GUARD = "pressure_guard"
    FAIRNESS_RESERVE = "fairness_reserve"


@dataclass(frozen=True)
class AdmissionDecision:
    action: AdmissionAction
    reason_code: AdmissionReason
    reservation_tokens: int = 0
    admission_decision_id: str = ""
    owner_epoch: int = -1
    lease_id: str | None = None


@dataclass(frozen=True)
class AdmissionFeatures:
    """Everything the shard gathers synchronously under the session lock."""

    enabled: bool
    session_id: str
    scope_allowed: bool
    is_protected: bool
    marked: bool
    prompt_tokens: int
    expected_decode_tokens: int
    reservation_tokens: int
    dispatch_id: str
    admission_decision_id: str
    serving_weight_version: str | None = None
    aged: bool = False


@dataclass(frozen=True)
class WorkerSnapshot:
    """A capacity/pressure reading for one engine, sourced from its.

    /metrics.
    """

    engine_id: str
    max_total_num_tokens: int
    token_usage: float
    num_used_tokens: int = 0
    healthy: bool = True
    serving_weight_version: str | None = None


@dataclass
class Lease:
    lease_id: str
    tokens: int
    owner_epoch: int
    created_at: float


@dataclass(frozen=True)
class GrantResult:
    granted: bool
    reason: AdmissionReason
    lease_id: str | None
    owner_epoch: int
    reservation_tokens: int

    def as_dict(self) -> dict:
        return {
            "granted": self.granted,
            "reason": self.reason.value,
            "lease_id": self.lease_id,
            "owner_epoch": self.owner_epoch,
            "reservation_tokens": self.reservation_tokens,
        }


def compute_reservation_tokens(
    *,
    prompt_tokens: int,
    sampling_max_new_tokens: int | None,
    expected_decode_cap: int | None,
    rollout_max_response_len: int | None,
) -> int:
    """Two-phase reservation: exact prompt tokens + a bounded decode estimate.

    ``prompt_tokens`` is the exact length of the dispatched ``input_ids``; the
    decode estimate is the tightest of the sampling ``max_new_tokens`` (if
    any), the configured ``expected_decode_cap``, and the hard
    ``rollout_max_response_len`` ceiling. The result is never unbounded.
    """
    bounded_decode: int | None = None
    for cap in (expected_decode_cap, sampling_max_new_tokens, rollout_max_response_len):
        if cap is None or cap <= 0:
            continue
        bounded_decode = cap if bounded_decode is None else min(bounded_decode, cap)
    if bounded_decode is None:
        bounded_decode = 0
    return max(0, int(prompt_tokens)) + max(0, int(bounded_decode))


def decide_admission_prelude(features: AdmissionFeatures) -> AdmissionDecision | None:
    """Return a fail-open/short-circuit decision, or ``None`` to consult the
    budget.

    Order matters: disabled and identity checks come first so non-agentic and
    anonymous traffic is never gated; protected work always bypasses (never
    starved); a pressure ``marked`` session defers at this boundary without
    spending an RPC.
    """
    decision_id = features.admission_decision_id
    reservation = features.reservation_tokens
    if not features.enabled:
        return AdmissionDecision(AdmissionAction.BYPASS, AdmissionReason.FEATURE_DISABLED, reservation, decision_id)
    if not features.session_id:
        return AdmissionDecision(AdmissionAction.BYPASS, AdmissionReason.MISSING_IDENTITY, reservation, decision_id)
    if not features.scope_allowed:
        return AdmissionDecision(
            AdmissionAction.BYPASS, AdmissionReason.CAPABILITY_UNAVAILABLE, reservation, decision_id
        )
    if features.is_protected:
        return AdmissionDecision(AdmissionAction.BYPASS, AdmissionReason.FAIRNESS_RESERVE, reservation, decision_id)
    if features.marked:
        return AdmissionDecision(AdmissionAction.DEFER, AdmissionReason.PRESSURE_GUARD, reservation, decision_id)
    return None


def build_reserve_request(features: AdmissionFeatures) -> dict:
    """Serialize the fields a coordinator ``reserve`` needs from admission
    features."""
    return {
        "tokens": int(features.reservation_tokens),
        "dispatch_id": features.dispatch_id,
        "admission_decision_id": features.admission_decision_id,
        "serving_weight_version": features.serving_weight_version,
        "aged": bool(features.aged),
    }


def interpret_budget_response(grant: dict, features: AdmissionFeatures) -> AdmissionDecision:
    """Map a coordinator grant dict to an ``AdmissionDecision``.

    ``degraded`` (stale/unavailable signals) fails open to bypass;
    ``capacity_exhausted`` and ``pressure_guard`` defer; a grant admits with
    the coordinator-minted lease.
    """
    decision_id = features.admission_decision_id
    reservation = int(grant.get("reservation_tokens", features.reservation_tokens))
    if grant.get("granted"):
        return AdmissionDecision(
            AdmissionAction.ADMIT,
            AdmissionReason.CAPACITY_AVAILABLE,
            reservation,
            decision_id,
            int(grant.get("owner_epoch", -1)),
            grant.get("lease_id"),
        )
    reason = AdmissionReason(grant.get("reason", AdmissionReason.DEGRADED.value))
    if reason == AdmissionReason.DEGRADED:
        return AdmissionDecision(AdmissionAction.BYPASS, AdmissionReason.DEGRADED, reservation, decision_id)
    return AdmissionDecision(
        AdmissionAction.DEFER, reason, reservation, decision_id, int(grant.get("owner_epoch", -1))
    )


class BudgetState:
    """Pure, single-writer ledger of cluster-wide execution-token reservations.

    A single coordinator actor owns one instance and serializes all mutations,
    so the read-modify-write in :meth:`reserve`/:meth:`release` is atomic.
    Every method takes an injected ``now`` (monotonic seconds) to stay
    deterministic under test.
    """

    def __init__(
        self,
        *,
        headroom: float = 0.90,
        pressure_threshold: float = 0.92,
        emergency_reserve_frac: float = 0.05,
        lease_ttl_s: float = 600.0,
        staleness_s: float = 30.0,
    ) -> None:
        self.headroom = headroom
        self.pressure_threshold = pressure_threshold
        self.emergency_reserve_frac = emergency_reserve_frac
        self.lease_ttl_s = lease_ttl_s
        self.staleness_s = staleness_s
        self.epoch = 0
        self.ceiling = 0
        self.reserved = 0
        self.avg_usage = 0.0
        self.max_usage = 0.0
        # Drain-per-step window (see capacity_hint): the once-per-step metric read is sampled at
        # log time, when the rollout has drained and the instantaneous usage undershoots. A running
        # peak/mean over reconciles, drained on read, makes the logged usage reflect the true window.
        self.peak_usage = 0.0
        self._usage_sum = 0.0
        self._usage_count = 0
        self._leases: dict[str, Lease] = {}
        self._worker_sig: tuple[str, ...] = ()
        self._weight_version: str | None = None
        self._last_snapshot_ts = 0.0

    # -- reconciliation -----------------------------------------------------------------

    def reconcile(self, snapshots: list[WorkerSnapshot], *, now: float, weight_version: str | None = None) -> None:
        """Recompute the ceiling from healthy workers; bump epoch on worker-
        set/version change."""
        healthy = [s for s in snapshots if s.healthy and s.max_total_num_tokens > 0]
        if weight_version is not None:
            healthy = [s for s in healthy if s.serving_weight_version in (None, weight_version)]
        if healthy:
            usages = [max(0.0, float(s.token_usage)) for s in healthy]
            self.avg_usage = sum(usages) / len(usages)
            self.max_usage = max(usages)
            self._last_snapshot_ts = now
        else:
            self.avg_usage = 0.0
            self.max_usage = 0.0
        # Accumulate the drain-per-step window so capacity_hint(reset_peak=True) reports the true
        # in-window peak/mean instead of whatever the once-per-step read happens to sample. The
        # instantaneous max_usage above still drives the live pressure guard in reserve().
        self.peak_usage = max(self.peak_usage, self.max_usage)
        self._usage_sum += self.avg_usage
        self._usage_count += 1
        new_sig = tuple(sorted(s.engine_id for s in healthy))
        version_changed = weight_version is not None and weight_version != self._weight_version
        if new_sig != self._worker_sig or version_changed:
            self.epoch += 1
            self._worker_sig = new_sig
            if weight_version is not None:
                self._weight_version = weight_version
            # Drop leases minted under a prior epoch; their tokens leave `reserved`.
            self._leases = {k: v for k, v in self._leases.items() if v.owner_epoch >= self.epoch}
            self.reserved = sum(v.tokens for v in self._leases.values())
        self.ceiling = int(sum(s.max_total_num_tokens for s in healthy) * self.headroom)

    def expire_ttl(self, now: float) -> int:
        """Reclaim leases older than the TTL (covers shards that died mid-
        flight)."""
        expired = [k for k, lease in self._leases.items() if now - lease.created_at > self.lease_ttl_s]
        for key in expired:
            self._release_key(key)
        return len(expired)

    def is_stale(self, now: float) -> bool:
        return self.ceiling <= 0 or (now - self._last_snapshot_ts) > self.staleness_s

    # -- reservations -------------------------------------------------------------------

    def reserve(
        self,
        *,
        tokens: int,
        dispatch_id: str,
        admission_decision_id: str,
        aged: bool,
        now: float,
    ) -> GrantResult:
        tokens = max(0, int(tokens))
        if self.is_stale(now):
            return GrantResult(False, AdmissionReason.DEGRADED, None, self.epoch, tokens)
        lease_id = f"{self.epoch}:{dispatch_id}:{admission_decision_id}"
        existing = self._leases.get(lease_id)
        if existing is not None:
            # Idempotent: a retried reserve for the same dispatch returns the same lease.
            return GrantResult(True, AdmissionReason.CAPACITY_AVAILABLE, lease_id, self.epoch, existing.tokens)
        if not aged and self.max_usage >= self.pressure_threshold:
            return GrantResult(False, AdmissionReason.PRESSURE_GUARD, None, self.epoch, tokens)
        emergency = int(self.ceiling * self.emergency_reserve_frac)
        limit = self.ceiling if aged else max(0, self.ceiling - emergency)
        if self.reserved + tokens > limit:
            return GrantResult(False, AdmissionReason.CAPACITY_EXHAUSTED, None, self.epoch, tokens)
        self._leases[lease_id] = Lease(lease_id, tokens, self.epoch, now)
        self.reserved += tokens
        return GrantResult(True, AdmissionReason.CAPACITY_AVAILABLE, lease_id, self.epoch, tokens)

    def release(self, lease_id: str, *, actual_tokens: int | None = None) -> None:
        # `actual_tokens` is accepted for future calibration; lease lifetime is what frees budget.
        del actual_tokens
        self._release_key(lease_id)

    def _release_key(self, lease_id: str) -> None:
        lease = self._leases.pop(lease_id, None)
        if lease is not None:
            self.reserved = max(0, self.reserved - lease.tokens)

    # -- introspection ------------------------------------------------------------------

    def capacity_hint(self, now: float, *, reset_peak: bool = False) -> dict:
        """Introspection snapshot.

        ``reset_peak`` drains the running peak/mean window and is set only by
        the once-per-step metrics reader (coordinator ``health``); the resume
        pump reads without draining so it never steals another reader's window.
        """
        window_mean = self._usage_sum / self._usage_count if self._usage_count else self.avg_usage
        hint = {
            "epoch": self.epoch,
            "ceiling": self.ceiling,
            "reserved": self.reserved,
            "available": max(0, self.ceiling - self.reserved),
            "avg_usage": self.avg_usage,
            "max_usage": self.max_usage,
            "peak_usage": self.peak_usage,
            "window_mean_usage": window_mean,
            "degraded": self.is_stale(now),
            "num_leases": len(self._leases),
        }
        if reset_peak:
            # Reset to the current level (not zero) so a fresh window still starts from reality.
            self.peak_usage = self.max_usage
            self._usage_sum = 0.0
            self._usage_count = 0
        return hint


class BudgetClient(Protocol):
    """The seam the shard depends on; a Ray-backed client wraps the coordinator
    actor."""

    async def reserve(self, req: dict) -> dict: ...

    async def release(self, lease_id: str, actual_tokens: int | None = None) -> None: ...

    async def capacity_hint(self) -> dict: ...
