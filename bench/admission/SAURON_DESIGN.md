# Sauron: one memory manager (implementation design)

Design doc (thesis, evidence, plan): https://claude.ai/code/artifact/199151f6-cce9-4bbc-82c4-5a96e1a27b3a

## Goal
Replace the scattered admission state (SiriusContext admission + query_memory_ledger + budget
watcher + pin registry + shared_scan_cache counters) with one component that owns every GPU byte,
keeps one invariant, measures two pressures and runs one control law with one setpoint.
No new per-workload knobs; existing env knobs stay for baselines only.

## State (one ledger)
- `B(t)`: budget, from the budget watcher (co-tenant) or `usage_limit_fraction`.
- Data grants: resident bytes per pinned table (pin registry, refreshed by the watcher).
- Shared grants: `shared_scan_cache::strong_bytes()`.
- Work grants per query: `min` (plan one-task need `need_task`, or ledger min at T5 history),
  `expected` (plan peak `need_min`, refined at T3 from exact pipeline input sizes),
  `held` and `live` (query_memory_ledger add/sub, add_held/sub_held).
Invariant (admission, safety): `B >= resident + shared + sum_q max(min_q, held_q) + min_new + headroom`.

## Pressures (measured, windowed)
Memory pressure `M` = stalled task-time / total task-time over a sliding window (default 2 s, EWMA):
- reservation waits: time inside `memory::reservation_wait_scope` (pinned_reservation_guard.hpp:135)
  -> add a duration report in its destructor.
- OOM reschedules: `task_reschedule_exception` path (gpu_pipeline_executor.cpp:588) -> count +
  the backoff time (50 ms) and the wasted task time before the throw.
- memory-wait requeues: `_memory_wait_requeues` (gpu_pipeline_executor.cpp:381) x requeue backoff (10 ms).
- task time: gpu_pipeline_task execute span (gpu_pipeline_task.cpp, around the ledger add/sub).
Scheduling pressure `S` = sum over waiting queries of expected remaining work (solo ms from
`solo_ms_` history, plan estimate otherwise) x age factor, plus ready tasks parked on memory.

## Control law (one setpoint `m*`)
Evaluated at events only: arrival, task start/end, query end, budget change, and a 100 ms tick.
- `M > m*` (back pressure), release cheapest-first, multiplicative decrease of the admission window:
  1. stop admitting (window = live - 1, floor 1)
  2. throttle task creation for the lowest-priority query (task_creator hint gate)
  3. shrink shared buffer (set_budget_bytes) when its hit rate is low
  4. demote coldest table not read by any live or queued plan (existing demote_for_deficit, plan-aware)
  5. pause lowest-priority query (quiesce/resume bracket exists: wait_and_drain_query)
- `M <= m*` and `S > 0` (forward pressure), additive increase:
  admit next by smallest remaining work with aging (best fit under the invariant),
  discount a query's expected grant by bytes it will share with live scans (T0 overlap),
  promote hot demoted tables when headroom allows.
- Budget drop: feasibility rule first (B >= resident + largest min); else demote before anything else.
Stability: EWMA on M, hysteresis band [m*/2, m*], at most one decrease per window.

## Steps (each builds, runs the unit suite, and is committed separately)
1. Telemetry only: stall-time counters + M in memlog (no behavior change). Validate M rises with
   thrashing on existing runs (shift_f75 no-limit vs limit 4; pt8 cap4).
2. Ledger consolidation: `sauron` class owning the state above; SiriusContext admission reads it.
3. Control law on admission window only (AIMD on M). Run E3-style sweep of m* (0.02..0.2).
4. Action menu 2-5 (throttle, shared buffer, plan-aware demotion, pause).
5. T3 refinement + size-aware ordering (E10).
6. Ablations (E6) via env switches that disable one rule each.

## Open questions
- Is stall fraction stable enough as a single signal across GPU-resident and host-tier data?
  (host tier: PCIe waits are not memory stalls; exclude scan transfer time.)
- m* per GPU or universal? Test on second GPU (E7).
- Livelock in hash-join task creation under pressure (conc_bench/dbg8/hang_cotenant_grantU.bt):
  must be fixed or bounded before E1/E5 since Sauron will push near the edge.
