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

enum class policy_kind { upstream, rapids, multiqx, sauron };

policy_kind active_policy() noexcept;

/// True when reservations should cover only input materialization (policy gates memory instead).
bool minimal_reservations() noexcept;

/// Called once by the runtime with the first GPU memory space.
void bind_gpu_space(const cucascade::memory::memory_space* space) noexcept;

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
                       bool ratio_eligible) noexcept;

/// Query lifecycle (all policies record; only multiqx blocks in query_admit_wait).
void query_begin(std::size_t query_id, const std::string& sql) noexcept;
void query_admit_wait(std::size_t query_id, const std::function<bool()>& interrupted);
void query_end(std::size_t query_id) noexcept;

}  // namespace sirius::sched
