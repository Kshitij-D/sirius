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

#include <cudf/column/column.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::scan_manager {

/**
 * @brief Cooperative shared scan for host-tier pinned tables (experimental).
 *
 * Without sharing, every query scanning a host-resident pinned table copies each
 * chunk's columns over PCIe on its own. This cache publishes the GPU copy of each
 * (table identity, chunk, column) so concurrent queries — of any template — reuse
 * the columns another query already brought over and copy only the missing ones.
 * Each query still applies its own filters, projections and aggregations.
 *
 * Lifetime: entries are weak references (alive while any query's batch holds the
 * column) plus a byte-bounded FIFO of strong references to the most recently
 * published columns, which widens the sharing window. The cache also records,
 * per table, the chunk most recently fetched so a new scan can start there and
 * wrap around (cooperative scan order), maximizing overlap with scans in flight.
 *
 * Enabled by SIRIUS_SHARED_SCAN_MB=<strong-buffer budget in MB> (> 0).
 */
class shared_scan_cache {
 public:
  static shared_scan_cache& instance()
  {
    static shared_scan_cache cache;
    return cache;
  }

  [[nodiscard]] bool enabled() const noexcept { return _budget_bytes > 0 && _gpu_space != nullptr; }
  void set_gpu_space(cucascade::memory::memory_space* space) noexcept { _gpu_space = space; }
  /// Programmatic budget (the memory manager sets it; the env var still applies otherwise).
  void set_budget_bytes(std::size_t b) noexcept { _budget_bytes = b; }
  [[nodiscard]] cucascade::memory::memory_space* gpu_space() const noexcept { return _gpu_space; }

  [[nodiscard]] std::shared_ptr<cudf::column> get(std::string const& table, std::size_t chunk, std::size_t column)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _weak.find(key(table, chunk, column));
    if (it == _weak.end()) { return nullptr; }
    auto sp = it->second.lock();
    if (!sp) {
      _weak.erase(it);
    } else {
      _hits.fetch_add(1, std::memory_order_relaxed);
    }
    return sp;
  }

  /// Publish a column; if another query published it first, returns that one instead.
  std::shared_ptr<cudf::column> put(std::string const& table,
                                    std::size_t chunk,
                                    std::size_t column,
                                    std::shared_ptr<cudf::column> col)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto k   = key(table, chunk, column);
    auto& wp = _weak[k];
    if (auto existing = wp.lock()) { return existing; }
    wp = col;
    _misses.fetch_add(1, std::memory_order_relaxed);
    auto const bytes = col->alloc_size();
    _strong.push_back({std::move(k), col, bytes});
    _strong_bytes += bytes;
    while (_strong_bytes > _budget_bytes && !_strong.empty()) {
      _strong_bytes -= _strong.front().bytes;
      _strong.pop_front();
    }
    return col;
  }

  void note_position(std::string const& table, std::size_t chunk)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _position[table] = chunk;
  }

  [[nodiscard]] std::size_t position(std::string const& table)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _position.find(table);
    return it == _position.end() ? 0 : it->second;
  }

  /// Bytes the cache itself keeps alive (the strong buffer): the shared grant.
  [[nodiscard]] std::size_t strong_bytes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _strong_bytes;
  }

  [[nodiscard]] std::size_t hits() const noexcept { return _hits.load(); }
  [[nodiscard]] std::size_t misses() const noexcept { return _misses.load(); }

  // Byte counters for the evaluation: host->GPU bytes copied by scans (shared path and the
  // regular per-query materialization) and bytes served from shared GPU copies instead.
  void add_copied(std::size_t b) noexcept { _copied.fetch_add(b, std::memory_order_relaxed); }
  void add_reused(std::size_t b) noexcept { _reused.fetch_add(b, std::memory_order_relaxed); }
  [[nodiscard]] std::size_t copied_bytes() const noexcept { return _copied.load(); }
  [[nodiscard]] std::size_t reused_bytes() const noexcept { return _reused.load(); }

 private:
  shared_scan_cache()
  {
    if (auto const* v = std::getenv("SIRIUS_SHARED_SCAN_MB")) {
      _budget_bytes = static_cast<std::size_t>(std::atof(v) * (1 << 20));
    }
  }

  static std::string key(std::string const& table, std::size_t chunk, std::size_t column)
  {
    return table + '#' + std::to_string(chunk) + '#' + std::to_string(column);
  }

  struct strong_ref {
    std::string key;
    std::shared_ptr<cudf::column> col;
    std::size_t bytes;
  };

  std::size_t _budget_bytes{0};
  cucascade::memory::memory_space* _gpu_space{nullptr};
  mutable std::mutex _mutex;
  std::unordered_map<std::string, std::weak_ptr<cudf::column>> _weak;
  std::deque<strong_ref> _strong;
  std::size_t _strong_bytes{0};
  std::unordered_map<std::string, std::size_t> _position;
  std::atomic<std::size_t> _hits{0};
  std::atomic<std::size_t> _misses{0};
  std::atomic<std::size_t> _copied{0};
  std::atomic<std::size_t> _reused{0};
};

}  // namespace sirius::scan_manager
