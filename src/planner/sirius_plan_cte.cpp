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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "helper/type_conversions.hpp"
#include "memory/shared_results.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_shared_result_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalMaterializedCTE& op)
{
  D_ASSERT(op.children.size() == 2);

  // Create the working_table that the PhysicalCTE will use for evaluation.
  auto working_table =
    duckdb::make_shared_ptr<duckdb::ColumnDataCollection>(context, op.children[0]->types);

  // Add the ColumnDataCollection to the context of this PhysicalPlanGenerator
  recursive_cte_tables[op.table_index] = working_table;
  materialized_ctes[op.table_index] =
    duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>>();

  // Cross-query sharing (memory/shared_results.hpp): replay a result another query of the batch
  // already published for this CTE body, or mark this CTE to publish its own.
  std::uint64_t share_key = 0;
  std::shared_ptr<const sirius::memory::shared_result> shared;
  if (sirius::memory::sharing_enabled()) {
    std::string sql;
    try {
      sql = context.GetCurrentQuery();
    } catch (std::exception&) {
      sql.clear();
    }
    auto const key = sirius::memory::cte_share_key(sql, op.ctename.GetIdentifierName());
    if (key && sirius::memory::share_key_allowed(*key)) {
      // The batch planner's text key is not enough: DuckDB pushes consumer filters into a
      // materialized CTE and prunes its columns per query (E66 smoke: q31). The registry key also
      // covers the optimized body.
      share_key = sirius::memory::combine_share_key(
        *key, sirius::memory::optimized_plan_signature(*op.children[0]));
      shared    = sirius::memory::find_shared_result(share_key);
      if (shared && shared->logical_types != op.children[0]->types) { shared = nullptr; }
    }
  }

  // Create the plan for the left side. This is the materialization.
  // The body is planned even when a shared result replaces it, and kept as the replay source's
  // dormant child: plan post-passes (scan certification, the transparent-execution read-view check)
  // still see its scans, but it gets no pipeline. Not shared if the body added CTE scans of an
  // enclosing CTE (that CTE would wait on scans that never run).
  std::size_t cte_scan_refs = 0;
  for (auto const& [idx, scans] : materialized_ctes) {
    cte_scan_refs += scans.size();
  }
  auto left = create_plan(*op.children[0]);
  if (shared) {
    std::size_t after = 0;
    for (auto const& [idx, scans] : materialized_ctes) {
      after += scans.size();
    }
    if (after == cte_scan_refs) {
      left = duckdb::make_uniq<sirius::op::sirius_physical_shared_result_scan>(
        sirius::from_duckdb_vec(op.children[0]->types),
        op.children[0]->estimated_cardinality,
        shared,
        share_key,
        std::move(left));
    } else {
      shared = nullptr;
    }
  }
  // Initialize an empty vector to collect the scan operators.
  auto right = create_plan(*op.children[1]);

  // V2 uses materialized CTEs for correlated joins: parents need the consumer schema,
  // while the producer-forwarding stage opts out of the runtime output-schema check.
  auto result_types = right->types;
  duckdb::unique_ptr<sirius::op::sirius_physical_cte> cte;
  cte = duckdb::make_uniq<sirius::op::sirius_physical_cte>(op.ctename.GetIdentifierName(),
                                                           op.table_index.index,
                                                           std::move(result_types),
                                                           std::move(left),
                                                           std::move(right),
                                                           op.estimated_cardinality);
  cte->working_table = working_table;
  // cte->working_table_gpu = working_table_gpu;
  cte->cte_scans = materialized_ctes[op.table_index];
  if (share_key != 0) {
    cte->share_key           = share_key;
    cte->share_consumer      = shared != nullptr;
    cte->share_logical_types = op.children[0]->types;
  }

  return std::move(cte);
}

}  // namespace sirius::planner
