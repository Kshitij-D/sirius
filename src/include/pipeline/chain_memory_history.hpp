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

#include "pipeline/pipeline_memory_history.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sirius::pipeline {

/**
 * @brief Cross-query memory history per pipeline operator chain (experimental).
 *
 * Sirius sizes task reservations from a per-pipeline history that lives only as
 * long as one query, so every query's first task of every pipeline falls back to
 * the generic no-history rule. Pipelines with the same operator chain recur
 * across queries; this keeps one pipeline_memory_history per chain signature so
 * a new query's first task can use what the engine observed before.
 *
 * Mode (SIRIUS_TASK_CHAIN_HISTORY): 0 off, 1 used when the pipeline has no
 * history yet, 2 also a floor under the per-query history estimate.
 */
class chain_memory_history {
 public:
  static int mode()
  {
    static int const m = [] {
      auto const* v = std::getenv("SIRIUS_TASK_CHAIN_HISTORY");
      return v ? std::atoi(v) : 0;
    }();
    return m;
  }

  static pipeline_memory_history& get(std::string const& chain)
  {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<pipeline_memory_history>> histories;
    std::lock_guard<std::mutex> lock(mutex);
    auto& slot = histories[chain];
    if (!slot) { slot = std::make_unique<pipeline_memory_history>(); }
    return *slot;
  }
};

}  // namespace sirius::pipeline
