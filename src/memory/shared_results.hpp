/*
 * Experimental cross-query result sharing for a batch known in advance (env-gated, SIRIUS_SHARE_KEYS).
 *
 * Unit of sharing (v1): the result of a DuckDB materialized CTE. The key is the FNV-1a 64 hash of the
 * CTE body text (whitespace collapsed) plus the keys of the earlier CTEs it references, computed from
 * the statement's SQL text; the batch planner (outside the engine) writes the keys it expects to repeat
 * to the file named by SIRIUS_SHARE_KEYS, one hex key per line. The database is read-only during a batch
 * (setup S2), so equal keys mean equal results.
 *
 * The first query that finishes such a CTE publishes its GPU batches here (if the total stays within
 * SIRIUS_SHARE_BUDGET_BYTES, default 4 GiB); their bytes are charged to the resident ledger so the
 * reservation cap sees them. Later queries that plan the same CTE replay the published batches through
 * a single-shot source instead of recomputing the body. Entries live until the process ends (one batch).
 */
#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cucascade {
class data_batch;
}

namespace sirius::memory {

struct shared_result {
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  duckdb::vector<duckdb::LogicalType> logical_types;
  std::size_t bytes = 0;
};

/// True when SIRIUS_SHARE_KEYS names a readable key file.
bool sharing_enabled() noexcept;

/// Key of CTE `ctename` in the leading WITH clause of `sql`, or nullopt if the CTE is not found there.
std::optional<std::uint64_t> cte_share_key(const std::string& sql, const std::string& ctename);

/// True if the batch planner listed `key` as shared.
bool share_key_allowed(std::uint64_t key) noexcept;

/// The published result for `key`, or nullptr.
std::shared_ptr<const shared_result> find_shared_result(std::uint64_t key);

/// Publish the finished result of `key` (first publisher wins; over budget = not published).
void publish_shared_result(std::uint64_t key,
                           std::vector<std::shared_ptr<::cucascade::data_batch>> batches,
                           duckdb::vector<duckdb::LogicalType> logical_types);

/// Count a consumer that replayed `key` (log only).
void note_shared_hit(std::uint64_t key, std::size_t bytes);

}  // namespace sirius::memory
