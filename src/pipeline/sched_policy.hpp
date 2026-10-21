/*
 * Experimental scheduling/memory policies for the concurrency study (env-gated, SIRIUS_POLICY=<name>).
 *
 *   upstream (default)  no change: cuCascade reservations, FIFO-by-priority dispatch.
 *   rapids              Spark RAPIDS dynamic concurrentGpuTasks (PR #12374) + reactive retry: a task
 *                       holds permits = p80 of the max footprint of completed tasks of its stage
 *                       (padded with a default to >= 4 entries, capped at capacity); oldest task first,
 *                       head-of-line blocking (no backfill); permits are never reconciled with the pool.
 *   multiqx             MultiQx-GPU (PVLDB 2014) admission: a query is delayed while its time-weighted
 *                       device-memory demand exceeds capacity minus the demands of running queries.
 *   sauron3             sauron2 plus a fast-tier working-set gate: a task whose predicted persistent
 *                       growth (predicted output minus consumed input) is positive runs only while that
 *                       growth fits in GPU headroom plus host-tier headroom, so held intermediates never
 *                       have to spill to disk. Tasks that shrink the working set are never held by it.
 *   sauron2             sauron plus (a) a feedback limit K on running tasks, hill-climbed on the input
 *                       bytes completed per second and cut when the host tier nears full (the disk cliff);
 *                       (b) selection over the whole ready set instead of the first fit: under memory
 *                       pressure the task with the largest predicted persistent release (consumed input
 *                       minus predicted output) runs first; otherwise the task of the query with the
 *                       least predicted remaining work (per-shape total task time minus work done) runs.
 *   sauron              operator-level packing: a task runs only if its predicted peak (exact per-class
 *                       peak/input ratio x the task's real input size) fits the real free pool after the
 *                       outstanding predicted peaks of running tasks; first fit in priority order.
 *
 * All non-upstream policies size the cuCascade reservation to the input materialization only, so the
 * policy, not the reservation ledger, gates memory. They must be run with SIRIUS_BORROW set so
 * allocations beyond a reservation are checked against the real pool.
 * Capacity = space limit minus resident bytes (pool use sampled while no query runs), which gives the
 * baselines knowledge of pinned tables that their original systems did not have.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::parallel {
class itask;
}

namespace sirius::sched {

enum class policy_kind { upstream, rapids, multiqx, sauron, sauron2, sauron3 };

policy_kind active_policy() noexcept;

/// True when reservations should cover only input materialization (policy gates memory instead).
bool minimal_reservations() noexcept;

/// Called once by the runtime with the first GPU and first HOST memory spaces.
void bind_gpu_space(const cucascade::memory::memory_space* space) noexcept;
void bind_host_space(const cucascade::memory::memory_space* space) noexcept;

/// sauron2 chooses among all dispatchable tasks: the scheduler calls scan_begin, scan_consider on
/// each candidate that passes device/retry checks, then pops scan_best (nullptr = hold all).
bool policy_selects() noexcept;
void scan_begin() noexcept;
void scan_consider(const parallel::itask& task) noexcept;
const parallel::itask* scan_best() noexcept;

/// Scheduler predicate hook, evaluated per candidate in priority order inside one selection scan.
/// `scan_blocked` is per-scan state for head-of-line blocking; start each scan with false.
bool task_may_dispatch(const parallel::itask& task, bool& scan_blocked) noexcept;

/// The task was popped for dispatch: take its grant.
void on_task_dispatched(parallel::itask& task) noexcept;

/// The task gave up its slot (memory-wait reschedule, completion, destruction): drop its grant.
void on_task_released(parallel::itask& task) noexcept;

/// A task finished successfully. `footprint` = stream peak including input materialization.
void on_task_completed(const parallel::itask& task,
                       std::size_t input_basis,
                       std::size_t footprint,
                       std::size_t output_bytes,
                       bool ratio_eligible) noexcept;

/// Ledger drain (SIRIUS_LEDGER=1, any policy): a task whose reservation has been refused for
/// longer than SIRIUS_LEDGER_DRAIN_MS (default 1000) becomes the drain owner; no other task is
/// dispatched until the owner's reservation is granted, so a large request is served as soon as
/// running work finishes instead of starving behind it. A drain that has not been granted after
/// SIRIUS_LEDGER_DRAIN_MAX_MS (default 10000) is abandoned so it can never become a livelock.
void ledger_note_refusal(parallel::itask& task) noexcept;
void ledger_note_granted(parallel::itask& task) noexcept;
void ledger_task_destroyed(const parallel::itask& task) noexcept;

/// Query lifecycle (all policies record; only multiqx blocks in query_admit_wait).
void query_begin(std::size_t query_id, const std::string& sql) noexcept;
void query_admit_wait(std::size_t query_id, const std::function<bool()>& interrupted);
void query_end(std::size_t query_id) noexcept;

}  // namespace sirius::sched
