/*
 * Experimental memory ledger for the concurrency study (env-gated, SIRIUS_LEDGER=1).
 *
 * Resident bytes (pinned tables) are counted exactly, by ownership: every stored pinned column (and
 * compressed payload) is wrapped in a shared_ptr whose deleter subtracts the bytes it added when it
 * was stored. The counter therefore equals the live stored pinned bytes at every instant, including
 * generations shared by a merge, unpins while a query still reads, and re-pins; temporaries made
 * while pinning are never counted.
 *
 * With the ledger on, decisions that judged accounted bytes use evictable bytes instead:
 *   - a reservation request is capped at (space limit - resident), the most that can be granted;
 *   - the downgrade trigger and stop levels apply to consumed - resident over capacity - resident;
 *   - a task refused for longer than SIRIUS_LEDGER_SHRINK_MS asks only for what the accounting can
 *     grant now (bounded reservation) and borrows the rest from the real pool, instead of starving
 *     behind other work until a timeout (the drain alternative is off unless SIRIUS_LEDGER_DRAIN_MS).
 * Run with SIRIUS_BORROW so allocations past a reservation are checked against the real pool.
 */
#pragma once

#include <rmm/resource_ref.hpp>

#include <cudf/column/column.hpp>

#include <memory>

#include <atomic>
#include <cstddef>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::memory {

bool ledger_enabled() noexcept;

/// Ablation switches (each defaults to SIRIUS_LEDGER; "0"/"1" forces it):
///   SIRIUS_CAP_RESIDENT       request cap = limit - resident (the clamp fix)
///   SIRIUS_EVICTABLE_TRIGGER  downgrade levels on evictable bytes
/// Resident bytes are tracked whenever any of them (or the ledger) is on.
bool cap_resident_enabled() noexcept;
bool evictable_trigger_enabled() noexcept;
bool resident_tracking_enabled() noexcept;

/// Allocator for data that stays resident in `space` (pinned tables). With the ledger off this is
/// the space's default allocator.
rmm::device_async_resource_ref resident_allocator(const cucascade::memory::memory_space* space);

/// Wrap a stored pinned column so its alloc_size() is resident until the last holder drops it.
std::shared_ptr<cudf::column> make_resident(std::unique_ptr<cudf::column> col,
                                            const cucascade::memory::memory_space* space);

/// Add / subtract resident bytes directly (compressed pinned payloads).
void add_resident(const cucascade::memory::memory_space* space, std::size_t bytes) noexcept;
void sub_resident(const cucascade::memory::memory_space* space, std::size_t bytes) noexcept;

/// Live resident bytes in `space` (0 with the ledger off or before any pin).
std::size_t resident_bytes(const cucascade::memory::memory_space* space) noexcept;

/// Counter for `space`, for consumers (the cuCascade downgrade trigger) that read it directly.
const std::atomic<std::size_t>* resident_counter(const cucascade::memory::memory_space* space);

}  // namespace sirius::memory
