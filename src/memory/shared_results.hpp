/*
 * Experimental cross-query result sharing for a batch known in advance (env-gated, SIRIUS_SHARE_KEYS).
 *
 * Unit of sharing (v1): the result of a DuckDB materialized CTE. The batch planner (outside the
 * engine) names candidate CTEs by a text key: FNV-1a 64 of the CTE body text (whitespace collapsed)
 * plus the keys of the earlier CTEs it references, written one hex key per line to the file named by
 * SIRIUS_SHARE_KEYS. Results are keyed by the text key combined with the optimized body plan, so two
 * queries share only when their optimized bodies are identical. The database is read-only during a
 * batch (setup S2), so equal keys mean equal results.
 *
 * The first query that finishes such a CTE publishes its GPU batches here (if the total stays within
 * SIRIUS_SHARE_BUDGET_BYTES, default 4 GiB); their bytes are charged to the resident ledger so the
 * reservation cap sees them. Later queries that plan the same CTE replay the published batches through
 * a single-shot source instead of recomputing the body. With a prepare pass (sirius_share_prepare) a
 * result is published only if a later consumer exists and released after its last consumer finishes;
 * without one, entries live until the process ends.
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
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace duckdb {
class LogicalOperator;
}

namespace sirius::memory {

struct shared_result {
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  duckdb::vector<duckdb::LogicalType> logical_types;
  std::size_t bytes = 0;
  const ::cucascade::memory::memory_space* space = nullptr;
};

/// True when SIRIUS_SHARE_KEYS names a readable key file.
bool sharing_enabled() noexcept;

/// Key of CTE `ctename` in the leading WITH clause of `sql`, or nullopt if the CTE is not found there.
std::optional<std::uint64_t> cte_share_key(const std::string& sql, const std::string& ctename);

/// Registry key: the planner's text key combined with the optimized body signature (DuckDB pushes
/// consumer filters into a materialized CTE and prunes its columns per query, so equal text is not
/// equal result). SIRIUS_SHARE_DEBUG=1 prints each new signature once.
std::uint64_t combine_share_key(std::uint64_t text_key, const std::string& plan_signature);

/// True if the batch planner listed `key` as shared.
bool share_key_allowed(std::uint64_t key) noexcept;

/// The published result for `key`, or nullptr.
std::shared_ptr<const shared_result> find_shared_result(std::uint64_t key);

/// Publish the finished result of `key` (first publisher wins; over budget = not published).
void publish_shared_result(std::uint64_t key,
                           std::vector<std::shared_ptr<::cucascade::data_batch>> batches,
                           duckdb::vector<duckdb::LogicalType> logical_types);

/// Full, untruncated text of an optimized logical subtree: operator types, every parameter string
/// (filters, projections, aggregates, join conditions) and the output types, recursively.
std::string optimized_plan_signature(duckdb::LogicalOperator const& op);

/// Batch prepare pass: count, for the optimized plan of statement `sql`, one planned consumer per
/// allowed materialized CTE key. Once any count exists, a result is published only if a consumer that
/// has not finished will plan it, and released when its last counted consumer finishes.
std::size_t count_expected_shares(const std::string& sql, duckdb::LogicalOperator const& plan);

/// Counting mode for the prepare pass (this thread only): while set, the post-optimizer hook counts
/// the captured plan of `sql` (the same plan create_plan receives) into the batch counts.
void set_counting_sql(const std::string* sql) noexcept;
const std::string* counting_sql() noexcept;
std::size_t& counted_in_thread() noexcept;

/// The prepare pass finished: from now on counts decide publishing and release (entries already
/// published whose counted consumers all finished are released at once).
void share_prepare_done();

/// A query's CTE with this key finished (computed or replayed): count it, release when done.
void note_shared_cte_finished(std::uint64_t key);

/// Count a consumer that replayed `key` (log only).
void note_shared_hit(std::uint64_t key, std::size_t bytes);

}  // namespace sirius::memory
