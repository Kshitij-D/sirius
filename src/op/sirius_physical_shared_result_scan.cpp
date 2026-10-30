#include "op/sirius_physical_shared_result_scan.hpp"

#include "op/sirius_physical_gpu_values.hpp"
#include "telemetry/nvtx.hpp"

namespace sirius::op {

sirius_physical_shared_result_scan::sirius_physical_shared_result_scan(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<const memory::shared_result> result,
  std::uint64_t key)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_VALUES, std::move(types), estimated_cardinality),
    _result(std::move(result)),
    _key(key)
{
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
  return std::make_unique<pipelineable_operator_data>(_result->batches);
}

std::size_t sirius_physical_shared_result_scan::no_history_peak_memory_estimate(
  const input_stats& /*stats*/) const
{
  return std::size_t{1} << 20;
}

}  // namespace sirius::op
