/*
 * Experimental memory ledger for the concurrency study (env-gated, SIRIUS_LEDGER=1).
 *
 * Resident bytes (pinned tables) are counted exactly, by ownership: the pin path allocates through
 * a counting resource that wraps the space's default allocator. A cuDF column frees through the
 * resource it was allocated from, so the counter is the live pinned bytes at every instant,
 * including unpins and conversions, with no sampling.
 *
 * With the ledger on, decisions that judged accounted bytes use evictable bytes instead:
 *   - a reservation request is capped at (space limit - resident), the most that can be granted;
 *   - the downgrade trigger and stop levels apply to consumed - resident over capacity - resident;
 *   - a task whose reservation keeps failing gets the GPU drained for it (no new dispatch for
 *     other tasks) instead of starving behind other work until a timeout.
 * Run with SIRIUS_BORROW so allocations past a reservation are checked against the real pool.
 */
#pragma once

#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstddef>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::memory {

bool ledger_enabled() noexcept;

/// Allocator for data that stays resident in `space` (pinned tables). With the ledger off this is
/// the space's default allocator.
rmm::device_async_resource_ref resident_allocator(const cucascade::memory::memory_space* space);

/// Live resident bytes in `space` (0 with the ledger off or before any pin).
std::size_t resident_bytes(const cucascade::memory::memory_space* space) noexcept;

/// Counter for `space`, for consumers (the cuCascade downgrade trigger) that read it directly.
const std::atomic<std::size_t>* resident_counter(const cucascade::memory::memory_space* space);

}  // namespace sirius::memory
