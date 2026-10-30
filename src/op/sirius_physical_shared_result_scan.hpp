#pragma once

#include "memory/shared_results.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>

namespace sirius::op {

/**
 * @brief Single-shot source that replays a published shared result (memory/shared_results.hpp).
 *
 * Planned in place of a materialized CTE body whose result another query of the batch already
 * computed. Its one task hands the published batches (shared, read-only) to the CTE sink, which fans
 * them out to the CTE scans as usual; nothing is recomputed or copied. Scheduled as a GPU_VALUES
 * source (query.cpp), which is the scheduling contract it follows.
 *
 * The planned body stays as a dormant child: plan post-passes (scan certification, the
 * transparent-execution read-view check) still see its scans, but build_pipelines() makes this
 * operator a source leaf, so the body never gets a pipeline and never runs.
 */
class sirius_physical_shared_result_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_VALUES;

  sirius_physical_shared_result_scan(duckdb::vector<sirius::logical_type> types,
                                     std::size_t estimated_cardinality,
                                     std::shared_ptr<const memory::shared_result> result,
                                     std::uint64_t key,
                                     duckdb::unique_ptr<sirius_physical_operator> dormant_body);

  bool is_source() const override { return true; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_sources() const override;

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// Replaying allocates nothing: a nominal reservation is enough.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  [[nodiscard]] std::optional<std::size_t> total_source_input_bytes() const override
  {
    return _result->bytes;
  }

 private:
  std::shared_ptr<const memory::shared_result> _result;
  std::uint64_t _key;
  std::atomic<bool> _task_scheduled{false};
  std::atomic<bool> _input_handed_out{false};
};

}  // namespace sirius::op
