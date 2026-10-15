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

#include <atomic>
#include <chrono>
#include <cstdint>

namespace sirius::memory {

/**
 * @brief Memory pressure telemetry: how much GPU task time is lost to memory (experimental).
 *
 * The GPU analog of a page-fault rate (Denning) or Linux PSI: task time spent stalled on
 * memory instead of computing. Stall time = waits inside reservation_wait_scope, memory-wait
 * re-queue backoffs, OOM-reschedule backoffs, and operator work thrown away by an OOM.
 * Work time = operator execution that completed. Memory pressure over a window is
 * d(stall) / (d(stall) + d(work)); the memory manager's control law holds it at a setpoint.
 * Counters are cumulative and monotonic; readers difference two snapshots.
 */
class memory_pressure {
 public:
  static memory_pressure& instance()
  {
    static memory_pressure p;
    return p;
  }

  void add_stall(std::chrono::nanoseconds d) noexcept
  {
    _stall_ns.fetch_add(static_cast<std::uint64_t>(d.count()), std::memory_order_relaxed);
  }
  void add_work(std::chrono::nanoseconds d) noexcept
  {
    _work_ns.fetch_add(static_cast<std::uint64_t>(d.count()), std::memory_order_relaxed);
  }
  void count_oom_reschedule() noexcept { _oom_reschedules.fetch_add(1, std::memory_order_relaxed); }
  void count_requeue() noexcept { _requeues.fetch_add(1, std::memory_order_relaxed); }

  struct snapshot {
    std::uint64_t stall_ns{0};
    std::uint64_t work_ns{0};
    std::uint64_t oom_reschedules{0};
    std::uint64_t requeues{0};
  };

  [[nodiscard]] snapshot read() const noexcept
  {
    return {_stall_ns.load(std::memory_order_relaxed),
            _work_ns.load(std::memory_order_relaxed),
            _oom_reschedules.load(std::memory_order_relaxed),
            _requeues.load(std::memory_order_relaxed)};
  }

  /// Pressure between two snapshots: stalled share of task time, 0 when nothing ran.
  [[nodiscard]] static double pressure(snapshot const& a, snapshot const& b) noexcept
  {
    auto const stall = static_cast<double>(b.stall_ns - a.stall_ns);
    auto const work  = static_cast<double>(b.work_ns - a.work_ns);
    return stall + work > 0 ? stall / (stall + work) : 0.0;
  }

 private:
  memory_pressure() = default;
  std::atomic<std::uint64_t> _stall_ns{0};
  std::atomic<std::uint64_t> _work_ns{0};
  std::atomic<std::uint64_t> _oom_reschedules{0};
  std::atomic<std::uint64_t> _requeues{0};
};

}  // namespace sirius::memory
