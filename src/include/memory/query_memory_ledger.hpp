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
    e.peak = std::max(e.peak, e.live);
  }

  void sub(std::uint32_t query, std::size_t bytes)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    if (it == _entries.end()) { return; }
    it->second.live -= std::min(it->second.live, bytes);
  }

  [[nodiscard]] std::size_t live(std::uint32_t query) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    return it == _entries.end() ? 0 : it->second.live;
  }

  /// Peak reserved bytes of a finished query; forgets the query.
  std::size_t take_peak(std::uint32_t query)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(query);
    if (it == _entries.end()) { return 0; }
    auto peak = it->second.peak;
    _entries.erase(it);
    return peak;
  }

 private:
  struct entry {
    std::size_t live{0};
    std::size_t peak{0};
  };
  mutable std::mutex _mutex;
  std::unordered_map<std::uint32_t, entry> _entries;
};

}  // namespace sirius::memory
