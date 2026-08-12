# Agentic KV Scheduling

Two independent, opt-in features that reduce KV-cache pressure during agentic rollout: **session KV lifecycle** releases a finished session's KV immediately, and **program-aware admission** bounds the active working set at request boundaries.

## Overview

In agentic rollout a session is a long-lived multi-turn program, not a single request. Between turns the agent executes tools — which can take seconds to minutes — while the KV prefix of that session stays resident in the engine. With many concurrent sessions the KV pool saturates, the engine falls back to eviction and recompute, and newly arriving requests queue behind work that is not actually decoding.

Relax addresses this from both ends of a session's life:

| Feature | Flag | Acts at | Effect |
|---|---|---|---|
| Session KV lifecycle | `--agentic-session-lifecycle` | End of a session | Releases the session's KV instead of waiting for LRU eviction |
| Program-aware admission | `--agentic-program-admission` | Start of each request | Bounds how much KV the cluster commits to at once |

Both features are **off by default**, are **independent** (either can be enabled alone), and **fail open** — any missing, stale, or failing signal falls back to the existing behaviour. Neither changes generation results: the full replay payload (`input_ids`) is always sent, so a cold cache still serves correctly.

::: tip
These features target *scheduling* overhead, not model quality. Enable them when the rollout is KV-bound — high engine `token_usage`, frequent eviction, requests queueing while the GPU is not saturated.
:::

## Session KV Lifecycle

### What it does

When enabled, the SGLang backend adapter sends the engine session id as a top-level `session_id` field on every `generate` call, which tags the leaf in the server's session-radix cache. When the session reaches a terminal state — either `finalize_and_discard` or `discard_session` — Relax aborts or quiesces the in-flight requests and then issues an idempotent `/close_session`, releasing that session's KV immediately.

### Routing

`/close_session` is **not** proxied by the sgl-router. Relax therefore fans out directly to each engine base URL, mirroring how request aborts are handled. The engine's DP controller broadcasts the release across all DP ranks; ranks that do not hold the session no-op.

```
                    ┌──────────────────┐
   generate ───────►│   sgl-router     │───────► engine (placement decided here)
   (session_id)     └──────────────────┘
                    ┌──────────────────┐
   /close_session ─X│   sgl-router     │   not proxied
                    └──────────────────┘
                             │
   /close_session ───────────┴──────────────► every engine base URL directly
                                              (DP controller broadcasts to all DP ranks)
```

### Requirements

The SGLang server must run with `--sglang-enable-session-radix-cache`. Without it the `session_id` field is simply ignored and the close call is a no-op — enabling the Relax flag alone gains nothing.

::: warning
`--sglang-enable-session-radix-cache` and `--sglang-radix-eviction-policy` are not defined by Relax. They are SGLang `ServerArgs` auto-exposed with a `--sglang-` prefix (see `relax/backends/sglang/arguments.py`), so they exist only if your installed SGLang provides them. Both are present in SGLang 0.5.15.
:::

### Failure behaviour

Close is a KV-release optimisation only. Failures and timeouts are logged and swallowed so they never block the logical terminal state; affected sessions simply fall back to LRU eviction. Look for these warnings:

```
close_session skipped: cannot list SGLang workers: ...
close_session skipped: no SGLang worker urls available.
close_session: 3/8 call(s) failed; affected sessions fall back to LRU eviction.
```

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--agentic-session-lifecycle` | flag | `False` | Enable the feature |
| `--agentic-session-close-timeout-ms` | int | `500` | Per-call timeout for the close fan-out; `0` disables the bound |
| `--agentic-session-close-max-retries` | int | `2` | Retries per call, transient HTTP errors only; must be `>= 1` |

## Program-Aware Admission

### What it does

At each request boundary — after the `InflightRequest` is built and queued, but **before** it takes the SGLang permit or calls `generate()` — admission returns one of three decisions:

| Action | Behaviour |
|---|---|
| **Bypass** | Today's path, unchanged. No lease is held. |
| **Admit** | Holds a cluster-wide execution-token lease, released when the request finishes. |
| **Defer** | Parks the request shard-side. No permit is taken; a resume pump restarts it later. |

Two invariants hold regardless of configuration:

- Admission **never selects a worker**. Final placement stays with the SGLang router.
- Admission **never interrupts in-flight decode**. It only gates work that has not started.

### Architecture

```
┌──────────────────────┐   reserve / release   ┌────────────────────────────────┐
│  AgenticSessionShard │ ────────────────────► │  AdmissionBudgetCoordinator    │
│  (16 Ray actors)     │ ◄──────────────────── │  single Ray actor, one writer  │
│                      │        grant          │  wraps the BudgetState ledger  │
└──────────┬───────────┘                       └───────────────┬────────────────┘
           │                                                   │ poll /metrics
           │ generate() only when not deferred                 │ every reconcile tick
           ▼                                                   ▼
┌──────────────────────┐                       ┌────────────────────────────────┐
│     sgl-router       │ ────────────────────► │        SGLang engines          │
└──────────────────────┘                       └────────────────────────────────┘
```

| Component | Responsibility | Implementation |
|---|---|---|
| **Decision logic** | Pure Bypass/Admit/Defer policy and the token ledger, no Ray or I/O | `relax/agentic/session/admission.py` |
| **Coordinator** | Single-writer Ray actor, `/metrics` poller, lease TTL reclaim | `relax/agentic/session/admission_coordinator.py` |
| **Shard integration** | Feature gathering, defer gate, resume pump, per-shard counters | `relax/agentic/session/service.py` |

The ledger is deliberately separated from Ray so the policy and the budget accounting are unit-testable on CPU (`tests/test_agentic_rollout.py`). All time-dependent state takes an injected monotonic `now`.

### The reservation

A reservation is **exact prompt tokens + a bounded decode estimate**. The decode estimate is the tightest of the sampling `max_new_tokens`, `--agentic-admission-expected-decode-cap`, and `--rollout-max-response-len`, so it is never unbounded.

### Decision order

The prelude runs first and costs no RPC:

1. Feature disabled → **Bypass**
2. No session id → **Bypass** (`missing_identity`)
3. Scope not allowed → **Bypass** (`capability_unavailable`)
4. Protected work → **Bypass** (`fairness_reserve`) — protected work is never starved
5. Session already marked under pressure → **Defer** (`pressure_guard`)

Otherwise the shard consults the coordinator, which grants or refuses:

| Refusal reason | Ledger condition | Caller behaviour |
|---|---|---|
| `degraded` | No snapshot, or the last one is older than 3 reconcile intervals | **Bypass** (fail open) |
| `pressure_guard` | Worst-case engine `token_usage` ≥ the pressure threshold, request not aged | **Defer** |
| `capacity_exhausted` | `reserved + tokens` exceeds the limit; for non-aged requests the limit excludes the emergency reserve | **Defer** |

The ceiling is `sum(max_total_num_tokens of healthy engines) × headroom`.

### Leases

Leases are idempotent per `(epoch, dispatch_id, decision_id)`, so a retried reserve returns the same lease rather than double-charging the budget. A TTL reclaims leases stranded by a dead shard. When the healthy worker set or the serving weight version changes, the coordinator bumps its epoch and drops all prior-epoch leases.

### Anti-starvation

Deferred sessions are resumed oldest-first by a per-shard pump that ticks every `--agentic-admission-reconcile-interval-s`:

- Under the max wait, a session resumes only when the coordinator reports plausible room for its head request.
- If the ledger is degraded, deferred sessions resume immediately — the re-admit will bypass.
- Past `--agentic-admission-max-wait-s` a session is **force-resumed** (aged). Aged requests skip the pressure guard and may draw on the emergency reserve, and an aged request that would still be deferred bypasses instead, so it always makes progress.
- Forced resumes are capped per shard per tick by `--agentic-admission-forced-resume-cap`.

### Requirements

The coordinator discovers engines through the SGLang router (`/workers`, falling back to `/list_workers`) and scrapes each engine's Prometheus `/metrics`. If the router address is unset or unreachable there are no snapshots, the ledger reports `degraded`, and every request bypasses.

::: tip
Because the coordinator reads engine gauges, TP replication matters. SGLang emits `sglang:max_total_num_tokens` and friends once per rank with an identical per-engine value, so Relax aggregates them by **max**, not sum. Summing would inflate capacity and deflate usage by the TP degree, making a saturated engine read as nearly idle.
:::

### Options

| Flag | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `--agentic-program-admission` | flag | `False` | — | Enable the feature |
| `--agentic-admission-headroom` | float | `0.90` | `(0, 1]` | Fraction of aggregate KV capacity usable as the ceiling |
| `--agentic-admission-pressure-threshold` | float | `0.92` | `(0, 1]` | Per-worker `token_usage` at/above which non-aged requests defer |
| `--agentic-admission-expected-decode-cap` | int | `--rollout-max-response-len` | `> 0` | Upper bound on expected decode tokens per reservation |
| `--agentic-admission-max-wait-s` | float | `30.0` | `>= 0` | Max defer time before forced resume |
| `--agentic-admission-reconcile-interval-s` | float | `2.0` | `> 0` | Coordinator reconcile and resume-pump tick |
| `--agentic-admission-lease-ttl-s` | float | `600.0` | `> 0` | Lease TTL; reclaims leases from dead shards |
| `--agentic-admission-reserve-timeout-ms` | int | `100` | `> 0` | Reserve RPC timeout; on timeout the request bypasses |
| `--agentic-admission-emergency-reserve-frac` | float | `0.05` | `[0, 1)` | Ceiling fraction reserved for aged requests |
| `--agentic-admission-forced-resume-cap` | int | `8` | `>= 1` | Max forced resumes per shard per tick |
| `--agentic-admission-scope` | str | `train` | `train` \| `all` | Apply to train only, or train + eval |

The numeric constraints are enforced only when `--agentic-program-admission` is set.

## Quick Start

Add to an existing agentic rollout launch script:

```bash
AGENTIC_ARGS=(
   --use-agentic-rollout
   # ... existing agent flags ...

   # Session KV lifecycle: requires the server-side session radix cache
   --sglang-enable-session-radix-cache
   --sglang-radix-eviction-policy priority
   --agentic-session-lifecycle

   # Program-aware admission: defaults are a reasonable starting point
   --agentic-program-admission
   --agentic-admission-headroom 0.90
   --agentic-admission-pressure-threshold 0.92
)
```

A complete example lives in `examples/mini_swe_agent/run_mini_swe_agent.sh`.

## Metrics

Both features report once per rollout step, alongside the existing `rollout/` and `perf/` metrics. Every series shares the `agentic_kv/` prefix, so trackers that group by the first path segment — ClearML splits a key into `(title, series)` on its first `/` — render them as one panel instead of three.

| Metric | Meaning |
|---|---|
| `agentic_kv/session/lifecycle_enabled` | `1.0` when session lifecycle is on; absent otherwise |
| `agentic_kv/admission/admit` / `defer` / `bypass` | Per-step decision counts (only when at least one decision was made) |
| `agentic_kv/admission/defer_rate` | `defer / (admit + defer + bypass)` |
| `agentic_kv/admission/degraded_rate` | Fraction of decisions that fell open due to degraded signals |
| `agentic_kv/admission/forced_resume` | Aged sessions force-resumed this step |
| `agentic_kv/admission/reserve_error` | Reserve RPCs that raised and fell open to bypass |
| `agentic_kv/admission/defer_wait_ms_mean` | Mean time parked behind the gate |
| `agentic_kv/admission/state_ready` / `state_in_flight` / `state_acting` / `state_deferred` | Live session-state census |
| `agentic_kv/budget/ceiling` / `agentic_kv/budget/reserved` | Cluster execution-token ceiling and current reservations |
| `agentic_kv/budget/reserved_utilization` | `reserved / ceiling` |
| `agentic_kv/budget/kv_token_usage_mean` / `agentic_kv/budget/kv_token_usage_max` | In-window mean and peak engine KV usage |
| `agentic_kv/budget/epoch` | Coordinator epoch; increments on worker-set or weight-version change |
| `agentic_kv/budget/degraded` | `1.0` while the ledger has no fresh snapshot |

::: warning
`agentic_kv/budget/kv_token_usage_*` are sampled over a running window that is drained once per step, because an instantaneous read at log time lands after the rollout has drained and understates the real peak. The engine-side release gains — pool size, forced evictions, freed tokens — are not in this table; read them from the engine's own Prometheus `/metrics`.
:::

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Everything bypasses; `agentic_kv/admission/defer` stays 0 | Ledger degraded | `agentic_kv/budget/degraded == 1.0`. The router address is unset or unreachable, or the `/metrics` scrape fails. Look for `Admission coordinator reconcile failed` in the log. |
| High `agentic_kv/admission/defer_rate` with low `agentic_kv/budget/reserved_utilization` | Pressure guard tripping, not capacity | Engine `token_usage` is at/above the threshold. Raise `--agentic-admission-pressure-threshold`, or lower concurrency. |
| High `defer_rate` and `reserved_utilization` near 1.0 | Genuinely capacity-bound | Raise `--agentic-admission-headroom`, or reduce concurrent sessions. |
| `agentic_kv/admission/forced_resume` climbing every step | Sessions routinely hit the aging deadline | The budget is too tight for the offered load; the gate is degenerating into a delay. Loosen headroom or reduce concurrency. |
| Session lifecycle enabled but KV never drops | Server-side cache not enabled | Confirm the engine was started with `--sglang-enable-session-radix-cache`. |
| `close_session skipped: ...` warnings | Worker discovery failed | The router is unreachable. Sessions fall back to LRU eviction; generation is unaffected. |

## Next Steps

- Read [Agentic Rollout](./agentic-rollout.md) for the session lifecycle these features hook into.
- Read [Performance Tuning](./performance-tuning.md) for the wider rollout throughput checklist.
- Read [OOM Troubleshooting](./oom-troubleshooting.md) when KV pressure turns into out-of-memory failures.
