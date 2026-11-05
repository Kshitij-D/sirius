#include "op/sirius_physical_shared_result_scan.hpp"

#include "op/sirius_physical_gpu_values.hpp"
#include "data/data_batch_utils.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry/nvtx.hpp"

namespace sirius::op {

sirius_physical_shared_result_scan::sirius_physical_shared_result_scan(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<const memory::shared_result> result,
  std::uint64_t key,
  duckdb::unique_ptr<sirius_physical_operator> dormant_body)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_VALUES, std::move(types), estimated_cardinality),
    _result(std::move(result)),
    _key(key)
{
  if (dormant_body) { children.push_back(std::move(dormant_body)); }
}

void sirius_physical_shared_result_scan::build_pipelines(pipeline::sirius_pipeline& current,
                                                         pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Source leaf: the dormant body gets no pipeline.
  meta_pipeline.get_state().add_pipeline_operator(current, *this);
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_shared_result_scan::get_sources() const
{
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> result;
  result.push_back(*this);
  return result;
}

std::optional<task_creation_hint> sirius_physical_shared_result_scan::get_next_task_hint()
{
  bool expected = false;
  if (!_task_scheduled.compare_exchange_strong(expected, true)) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_shared_result_scan::all_ports_empty()
{
  return _input_handed_out.load(std::memory_order_acquire);
}

std::unique_ptr<operator_data> sirius_physical_shared_result_scan::get_next_task_input_data()
{
  if (_input_handed_out.exchange(true, std::memory_order_acq_rel)) { return nullptr; }
  return std::make_unique<gpu_values_input>(64);  // nothing to materialize
}

std::unique_ptr<operator_data> sirius_physical_shared_result_scan::execute(
  const operator_data& /*input_data*/, ::cuda::stream_ref /*stream*/)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_shared_result_scan::execute"};
  memory::note_shared_hit(_key, _result->bytes);
  // Fresh batches over views of the registry's columns (memory/shared_results.hpp): the shared
  // columns stay alive through the views' owners and are never stolen or moved by this query. The
  // views are written on the registry's stream, so this query's readers wait on its event.
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  auto* space = const_cast<::cucascade::memory::memory_space*>(_result->space);
  for (auto const& cols : _result->columns) {
    std::vector<cudf::column_view> views;
    std::size_t alloc_size = 0;
    for (auto const& c : cols) {
      views.push_back(c->view());
      alloc_size += c->alloc_size();
    }
    auto rep = std::make_unique<::cucascade::gpu_table_representation>(
      cudf::table_view(views),
      std::vector<std::shared_ptr<cudf::column>>(cols),
      alloc_size,
      *space,
      ::cuda::stream_ref{_result->writer});
    batches.push_back(::cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(rep)));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_physical_shared_result_scan::no_history_peak_memory_estimate(
  const input_stats& /*stats*/) const
{
  return std::size_t{1} << 20;
}

}  // namespace sirius::op
