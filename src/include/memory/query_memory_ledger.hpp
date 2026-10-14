/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace sirius::memory {

/**
 * @brief Per-query view of the GPU reservations Sirius's executor holds.
 *
 * Every GPU pipeline task runs under a reservation sized by the engine's own
 * estimator. The ledger adds a task's reservation to its query when the task
 * starts and removes it when the task ends, so for each live query it knows the
 * bytes the engine currently commits to it and the peak so far. Admission uses
 * it both to learn per-template needs (peak at query end) and to correct live
 * projections while queries run.
 */
class query_memory_ledger {
 public:
  static query_memory_ledger& instance()
  {
    static query_memory_ledger ledger;
    return ledger;
  }

  void add(std::uint32_t query, std::size_t bytes)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& e = _entries[query];
    e.live += bytes;
    e.peak     = std::max(e.peak, e.live + e.held);
    e.max_task = std::max(e.max_task, bytes);  // largest single task reservation
  }

  void sub(std::uint32_t query, std::size_t bytes)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    if (it == _entries.end()) { return; }
    it->second.live -= std::min(it->second.live, bytes);
  }

  /// Intermediate results a query published (pipeline outputs) and not yet consumed.
  void add_held(std::uint32_t query, std::size_t bytes)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& e = _entries[query];
    e.held += bytes;
    e.peak      = std::max(e.peak, e.live + e.held);
    e.peak_held = std::max(e.peak_held, e.held);
  }

  void sub_held(std::uint32_t query, std::size_t bytes)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    if (it == _entries.end()) { return; }
    it->second.held -= std::min(it->second.held, bytes);
  }

  /// Bytes the engine currently commits to a query: live task reservations plus held results.
  [[nodiscard]] std::size_t live(std::uint32_t query) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    return it == _entries.end() ? 0 : it->second.live + it->second.held;
  }

  /// Intermediate results a live query holds right now.
  [[nodiscard]] std::size_t held(std::uint32_t query) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    return it == _entries.end() ? 0 : it->second.held;
  }

  struct summary {
    std::size_t peak{0};       ///< max(live reservations + held): the most it ever used (max grant)
    std::size_t max_task{0};   ///< largest single task reservation
    std::size_t peak_held{0};  ///< most materialized state held at once
    /// Minimum grant: enough to run its largest task while holding its state.
    [[nodiscard]] std::size_t min_grant() const noexcept { return max_task + peak_held; }
  };

  /// Summary of a finished query; forgets the query.
  summary take(std::uint32_t query)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    if (it == _entries.end()) { return {}; }
    summary s{it->second.peak, it->second.max_task, it->second.peak_held};
    _entries.erase(it);
    return s;
  }

  /// Peak reserved bytes of a finished query; forgets the query.
  std::size_t take_peak(std::uint32_t query) { return take(query).peak; }

 private:
  struct entry {
    std::size_t live{0};
    std::size_t held{0};
    std::size_t peak{0};
    std::size_t max_task{0};
    std::size_t peak_held{0};
  };
  mutable std::mutex _mutex;
  std::unordered_map<std::uint32_t, entry> _entries;
};

}  // namespace sirius::memory
