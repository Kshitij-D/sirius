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

#include "transparent/physical_sirius_execution.hpp"

#include "log/logging.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"
#include "sirius_interface.hpp"
#include "sirius_sql_rewrite.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/execution/executor.hpp>
#include <duckdb/execution/operator/helper/physical_result_collector.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/pending_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/query_result.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <mutex>
#include <cmath>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

namespace sirius::transparent {

// ---------------------------------------------------------------------------
// Global source state — owns the sirius_interface and the materialized result.
// ---------------------------------------------------------------------------
struct SiriusGlobalSourceState : public duckdb::GlobalSourceState {
  duckdb::unique_ptr<sirius::sirius_interface> iface;
  duckdb::unique_ptr<duckdb::QueryResult> result;
  duckdb::unique_ptr<duckdb::DataChunk> current_chunk;
  duckdb::SiriusContext* sirius_context = nullptr;
  bool finished                         = false;
  // Private executor for the CPU fallback plan (see run_cpu_fallback_plan). Kept
  // alive here so the materialized result's backing pipelines outlive GetData.
  duckdb::unique_ptr<duckdb::Executor> cpu_executor;

  duckdb::idx_t MaxThreads() override { return 1; }
};

// Run a stored DuckDB CPU plan on a private Executor. Reusing the outer query's
// ClientContext (not a fresh Connection) keeps it on the same transaction and MVCC
// snapshot, including this transaction's uncommitted writes. Returns a materialized
// result; throws if the CPU plan itself fails.
duckdb::unique_ptr<duckdb::QueryResult> run_cpu_fallback_plan(
  duckdb::ClientContext& client,
  duckdb::PreparedStatementData& cpu_prepared,
  duckdb::unique_ptr<duckdb::Executor>& out_executor)
{
  // cpu_prepared is SHARED across executions of one prepared statement, so this
  // function must not write it (register E4): output_type/memory_type are fixed
  // to FORCE_MATERIALIZED/IN_MEMORY where the stash is built
  // (SiriusContext::OnFinalizePrepare), so the whole plan runs before any row
  // is streamed out of the operator.
  auto collector = duckdb::PhysicalResultCollector::GetResultCollector(client, cpu_prepared);
  D_ASSERT(collector->type == duckdb::PhysicalOperatorType::RESULT_COLLECTOR);

  // Suppress profiling for the nested run: it shares the context's single
  // QueryProfiler with the outer query, so letting it re-initialize would corrupt
  // the outer profile. No-op when profiling is already off. Restored on every path.
  auto& client_config              = duckdb::ClientConfig::GetConfig(client);
  const bool saved_enable_profiler = client_config.enable_profiler;
  client_config.enable_profiler    = false;

  out_executor   = duckdb::make_uniq<duckdb::Executor>(client);
  auto& executor = *out_executor;
  try {
    executor.Initialize(std::move(collector));
    duckdb::PendingExecutionResult exec_result;
    while (!duckdb::PendingQueryResult::IsResultReady(exec_result = executor.ExecuteTask())) {
      if (exec_result == duckdb::PendingExecutionResult::BLOCKED) { executor.WaitForTask(); }
    }
    if (executor.HasError()) { executor.ThrowException(); }
    auto result                   = executor.GetResult();
    client_config.enable_profiler = saved_enable_profiler;
    return result;
  } catch (...) {
    // Drain any still-registered tasks before the executor is destroyed
    // (~Executor asserts executor_tasks == 0), then restore profiling.
    executor.CancelTasks();
    client_config.enable_profiler = saved_enable_profiler;
    throw;
  }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
PhysicalSiriusExecution::PhysicalSiriusExecution(
  duckdb::PhysicalPlan& physical_plan,
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan,
  std::string query_sql,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<std::string> names,
  duckdb::shared_ptr<duckdb::PreparedStatementData> cpu_fallback_prepared,
  bool cpu_plan_reads_s3,
  duckdb::idx_t estimated_cardinality)
  : duckdb::PhysicalOperator(
      physical_plan, PhysicalSiriusExecution::TYPE, std::move(types), estimated_cardinality),
    logical_plan_(std::move(logical_plan)),
    query_sql_(std::move(query_sql)),
    result_names_(std::move(names)),
    cpu_fallback_prepared_(std::move(cpu_fallback_prepared)),
    cpu_plan_reads_s3_(cpu_plan_reads_s3)
{
}

namespace {

// Bytes per row of an operator's output, cuDF-style: fixed-width types at their
// physical size, strings as offsets plus an assumed average payload.
double row_width(duckdb::vector<duckdb::LogicalType> const& types)
{
  double w = 0;
  for (auto const& t : types) {
    auto pt = t.InternalType();
    if (pt == duckdb::PhysicalType::VARCHAR) {
      w += 4 + 20;
    } else if (duckdb::TypeIsConstantSize(pt)) {
      w += static_cast<double>(duckdb::GetTypeIdSize(pt));
    } else {
      w += 16;
    }
  }
  return w;
}

double card(duckdb::LogicalOperator& op, duckdb::ClientContext& ctx)
{
  return static_cast<double>(op.has_estimated_cardinality ? op.estimated_cardinality
                                                          : op.EstimateCardinality(ctx));
}

// Walk the optimized logical plan and accumulate footprint features (see
// plan_footprint_features). Pure estimate: no execution, no GPU state.
void accumulate_features(duckdb::LogicalOperator& op,
                         duckdb::ClientContext& ctx,
                         duckdb::plan_footprint_features& out)
{
  using T                       = duckdb::LogicalOperatorType;
  constexpr double k_batch_bytes = 2.0 * (1ULL << 30);
  switch (op.type) {
    case T::LOGICAL_GET: {
      double b = card(op, ctx) * row_width(op.types);
      out.f[0] += b;
      out.f[1] += std::min(b, k_batch_bytes);
      // Unfiltered bytes of every column the scan reads (filter columns
      // included): scans materialize whole batches before filtering.
      auto& get = op.Cast<duckdb::LogicalGet>();
      if (get.function.cardinality) {
        auto stats = get.function.cardinality(ctx, get.bind_data.get());
        if (stats && stats->has_estimated_cardinality) {
          duckdb::vector<duckdb::LogicalType> read_types;
          for (auto const& col : get.GetColumnIds()) {
            if (col.IsVirtualColumn()) { continue; }
            auto idx = col.GetPrimaryIndex();
            if (idx < get.returned_types.size()) { read_types.push_back(get.returned_types[idx]); }
          }
          double base = static_cast<double>(stats->estimated_cardinality) * row_width(read_types);
          out.f[6] += base;
          out.f[7] += std::min(base, k_batch_bytes);
        }
      }
      break;
    }
    case T::LOGICAL_COMPARISON_JOIN:
    case T::LOGICAL_DELIM_JOIN:
    case T::LOGICAL_ANY_JOIN:
    case T::LOGICAL_ASOF_JOIN:
    case T::LOGICAL_CROSS_PRODUCT:
      if (op.children.size() == 2) {
        auto& build = *op.children[1];
        out.f[2] += card(build, ctx) * row_width(build.types);
      }
      out.f[3] += card(op, ctx) * row_width(op.types);
      break;
    case T::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case T::LOGICAL_WINDOW: out.f[4] += card(op, ctx) * row_width(op.types); break;
    case T::LOGICAL_ORDER_BY:
    case T::LOGICAL_TOP_N:
      if (!op.children.empty()) {
        auto& in = *op.children[0];
        out.f[5] += card(in, ctx) * row_width(in.types);
      }
      break;
    default: break;
  }
  for (auto& child : op.children) {
    accumulate_features(*child, ctx, out);
  }
}

// Structural peak-memory walk over the logical plan (v2). Sizes are DuckDB's
// estimates; shape and units come from how Sirius executes:
//  - a scan task covers one stored chunk: chunks hold `batch_bytes` of the
//    table's full row width, so a task reads chunk_rows x (width of the columns
//    this query reads), and costs 2x that (Sirius's no-history operator rule);
//    up to `threads` tasks are in flight;
//  - a join's build side (2x its bytes) stays live while its probe side runs;
//  - aggregate and sort state are materialized for the parent; aggregate state
//    is bounded by Sirius's hash partitioning (at most `threads` partitions of
//    one batch in flight), because DuckDB's GROUP BY row estimate is unreliable.
struct walk_result {
  double peak{0};   // need_fast: working tasks + live state
  double work{0};   // need_min: working tasks only
  double held{0};   // bytes materialized for the parent
  double task{0};   // bytes of one task of the pipeline feeding the parent
  double base{0};   // base-table bytes read by the subtree (caps join build sizes)
};

walk_result walk_peak(duckdb::LogicalOperator& op,
                      duckdb::ClientContext& ctx,
                      double batch_bytes,
                      double threads)
{
  using T = duckdb::LogicalOperatorType;
  auto bytes_of = [&](duckdb::LogicalOperator& o) { return card(o, ctx) * row_width(o.types); };
  switch (op.type) {
    case T::LOGICAL_GET: {
      auto& get        = op.Cast<duckdb::LogicalGet>();
      double rows      = card(op, ctx);
      if (get.function.cardinality) {
        auto stats = get.function.cardinality(ctx, get.bind_data.get());
        if (stats && stats->has_estimated_cardinality) {
          rows = static_cast<double>(stats->estimated_cardinality);
        }
      }
      duckdb::vector<duckdb::LogicalType> read_types;
      for (auto const& col : get.GetColumnIds()) {
        if (col.IsVirtualColumn()) { continue; }
        auto idx = col.GetPrimaryIndex();
        if (idx < get.returned_types.size()) { read_types.push_back(get.returned_types[idx]); }
      }
      double full_w     = std::max(1.0, row_width(get.returned_types));
      double chunk_rows = std::min(rows, batch_bytes / full_w);
      double task       = chunk_rows * row_width(read_types);
      double tasks      = std::max(1.0, std::ceil(rows / std::max(1.0, chunk_rows)));
      double w          = 2 * task * std::min(threads, tasks);
      return {w, w, 0, task, rows * row_width(read_types)};
    }
    case T::LOGICAL_COMPARISON_JOIN:
    case T::LOGICAL_DELIM_JOIN:
    case T::LOGICAL_ANY_JOIN:
    case T::LOGICAL_ASOF_JOIN:
    case T::LOGICAL_CROSS_PRODUCT: {
      if (op.children.size() != 2) { break; }
      auto build = walk_peak(*op.children[1], ctx, batch_bytes, threads);
      auto probe = walk_peak(*op.children[0], ctx, batch_bytes, threads);
      // DuckDB's join estimates inflate; a build side cannot hold more than its
      // subtree read from base tables (FK joins), nor more than its materialized input.
      double bb = bytes_of(*op.children[1]);
      if (build.held > 0) { bb = std::min(bb, build.held); }
      if (build.base > 0) { bb = std::min(bb, build.base); }
      double ht = 2 * bb;
      return {std::max(build.peak, ht + probe.peak), std::max(build.work, probe.work), probe.held,
              probe.task, build.base + probe.base};
    }
    case T::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case T::LOGICAL_WINDOW: {
      if (op.children.empty()) { break; }
      auto in      = walk_peak(*op.children[0], ctx, batch_bytes, threads);
      double state = std::min(bytes_of(op), threads * batch_bytes);
      if (in.base > 0) { state = std::min(state, in.base); }
      return {in.peak + 2 * state, in.work, state, in.task, in.base};
    }
    case T::LOGICAL_ORDER_BY:
    case T::LOGICAL_TOP_N: {
      if (op.children.empty()) { break; }
      auto in     = walk_peak(*op.children[0], ctx, batch_bytes, threads);
      double data = in.held > 0 ? in.held : bytes_of(*op.children[0]);
      if (op.type == T::LOGICAL_TOP_N) { data = std::min(data, threads * in.task); }
      if (in.base > 0) { data = std::min(data, in.base); }
      return {in.peak + 2 * data, in.work, data, in.task, in.base};
    }
    default: break;
  }
  // Streaming operators: children run one after another; held results add up.
  walk_result r;
  double held_before = 0;
  for (auto& child : op.children) {
    auto c = walk_peak(*child, ctx, batch_bytes, threads);
    r.peak = std::max(r.peak, held_before + c.peak);
    r.work = std::max(r.work, c.work);
    r.task = std::max(r.task, c.task);
    r.base += c.base;
    held_before += c.held;
  }
  r.held = held_before;
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// Source interface
// ---------------------------------------------------------------------------
duckdb::unique_ptr<duckdb::GlobalSourceState> PhysicalSiriusExecution::GetGlobalSourceState(
  duckdb::ClientContext& context) const
{
  auto state      = duckdb::make_uniq<SiriusGlobalSourceState>();
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  auto conn_state = duckdb::get_sirius_connection_state(context);
  auto query_label =
    conn_state ? conn_state->take_pending_query_label() : std::optional<std::string>{};
  state->iface = duckdb::make_uniq<sirius::sirius_interface>(context, std::move(query_label));
  state->sirius_context = sirius_ctx.get();
  return std::move(state);
}

duckdb::unique_ptr<duckdb::LocalSourceState> PhysicalSiriusExecution::GetLocalSourceState(
  duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const
{
  return duckdb::make_uniq<duckdb::LocalSourceState>();
}

duckdb::SourceResultType PhysicalSiriusExecution::GetDataInternal(
  duckdb::ExecutionContext& context,
  duckdb::DataChunk& chunk,
  duckdb::OperatorSourceInput& input) const
{
  auto& state = input.global_state.Cast<SiriusGlobalSourceState>();
  if (state.finished) { return duckdb::SourceResultType::FINISHED; }

  // Lazy execution: run the GPU query on first GetData call.
  if (!state.result) {
    SIRIUS_LOG_INFO("Transparent GPU execution: executing query");
    if (state.sirius_context) { state.sirius_context->record_transparent_execution(); }

    // Attempt GPU execution. Any failure — a thrown exception or an error-carrying
    // result — is captured in gpu_error and routed to the CPU fallback below. The
    // result is fully materialized before the first Fetch, so falling back here
    // cannot duplicate rows.
    duckdb::ErrorData gpu_error;
    bool gpu_failed                = false;
    bool runtime_unavailable_error = false;
    // The execution window: begin mutations and slot acquire in one scope on
    // this thread; finished (mandatory cleanup and release) below, before the
    // first Fetch exposes the result, so an abandoned result holds nothing.
    std::optional<duckdb::SiriusContext::StandaloneQueryScope> window;
    try {
      if (state.sirius_context) {
        if (logical_plan_) {
          // Query-aware pre-admission: estimate this query's GPU footprint from
          // the optimized plan before it competes for an execution window.
          try {
            duckdb::plan_footprint_features features;
            {
              // The template may be shared by concurrent EXECUTEs of one prepared
              // statement; resolving types mutates it, so serialize.
              static std::mutex resolve_mutex;
              std::lock_guard<std::mutex> lock(resolve_mutex);
              logical_plan_->ResolveOperatorTypes();
              accumulate_features(*logical_plan_, context.client, features);
              auto const& cfg = state.sirius_context->get_config();
              auto const w    = walk_peak(
                *logical_plan_,
                context.client,
                static_cast<double>(cfg.get_operator_params().scan_task_batch_size),
                static_cast<double>(std::max(1, cfg.get_gpu_pipeline_executor_config().num_threads)));
              features.need_fast = w.peak;
              features.need_min  = w.work;
              features.need_task = walk_peak(*logical_plan_,
                                             context.client,
                                             static_cast<double>(cfg.get_operator_params().scan_task_batch_size),
                                             1.0)
                                     .work;
            }
            features.valid = true;
            duckdb::SiriusContext::set_pending_plan_features(features);
          } catch (...) {
          }
        }
        window.emplace(*state.sirius_context, context.client, "transparent_execution");
      }
      if (!logical_plan_ && query_sql_.empty()) {
        throw duckdb::ExecutorException(
          "Transparent GPU execution is missing the logical plan template");
      }

      // Build a minimal PreparedStatementData with the output schema.
      auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      prepared->types = types;
      prepared->names = result_names_;

      // Rebuild a fresh Sirius physical plan for this execution. DuckDB may reuse
      // the same prepared physical operator across multiple EXECUTE calls — and
      // concurrently, so all per-execution state lives in the source state and
      // the shared operator is never mutated (register E4).
      //
      // Prefer LogicalOperator::Copy when the plan supports it (cheap deep clone
      // via serialization). When the plan contains a non-serializable LogicalGet,
      // fall back to re-parsing + re-binding the unbound SQL statement, which
      // exercises the same bind path the very first run did.
      duckdb::unique_ptr<duckdb::LogicalOperator> fresh_plan;
      if (logical_plan_ && !plan_copy_unsupported_.load(std::memory_order_acquire)) {
        try {
          // Use the dynamic-filter-aware copy so LogicalComparisonJoin::filter_pushdown and
          // LogicalGet::dynamic_filters survive the serialize/deserialize round-trip — they are
          // not in DuckDB's serialization schema and would otherwise be null on the fresh plan,
          // making downstream Sirius wiring silently miss runtime-computed dynamic filters.
          fresh_plan = sirius::transparent::copy_logical_plan(*logical_plan_, context.client);
        } catch (duckdb::NotImplementedException&) {
          // The template can't be copied. Do NOT reset logical_plan_ — a
          // concurrent execution may be inside its own copy_logical_plan()
          // of the same tree. Latch the monotonic flag instead so future
          // executes skip straight to the replan path.
          plan_copy_unsupported_.store(true, std::memory_order_release);
        }
      }
      if (!fresh_plan) {
        // Suppress the optimizer hooks for this nested replan (the guard is a
        // no-op when Sirius has no per-connection state registered).
        duckdb::SiriusContext::InternalQueryGuard guard(context.client);
        duckdb::Parser parser(context.client.GetParserOptions());
        parser.ParseQuery(query_sql_);
        if (parser.statements.size() != 1) {
          throw duckdb::ExecutorException(
            "Transparent GPU execution: replan expected exactly one statement");
        }
        duckdb::Planner duckdb_planner(context.client);
        duckdb_planner.CreatePlan(std::move(parser.statements[0]));
        duckdb::Optimizer optimizer(*duckdb_planner.binder, context.client);
        fresh_plan = optimizer.Optimize(std::move(duckdb_planner.plan));
      }
      sirius::planner::sirius_physical_plan_generator planner(context.client);
      auto sirius_plan = planner.create_plan(std::move(fresh_plan));

      auto gpu_prepared = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
        std::move(prepared), std::move(sirius_plan));

      // TEST-ONLY fault injection: plan generation has already succeeded, so failing
      // here exercises the runtime CPU fallback path (not the plan-time path).
      {
        duckdb::Value inject;
        if (context.client.TryGetCurrentSetting("sirius_test_inject_transparent_gpu_error",
                                                inject) &&
            !inject.IsNull() && !inject.ToString().empty()) {
          throw duckdb::ExecutorException("injected transparent GPU failure: " + inject.ToString());
        }
      }

      // Execute via the standard sirius_interface path.
      duckdb::PendingQueryParameters parameters;
      state.result = state.iface->sirius_execute_query(
        context.client, "transparent_execution", gpu_prepared, parameters, window->query_id());

      if (state.result->HasError()) {
        gpu_error = state.result->GetErrorObject();
        state.result.reset();
        gpu_failed = true;
      }
    } catch (duckdb::SiriusBeginWindowFailureException&) {
      // The BEGIN mutations failed after possibly part-registering this
      // query's runtime entries (unwound by the backstop; contained to this
      // query): typed, never a fallback candidate.
      throw;
    } catch (duckdb::SiriusRuntimeUnavailableException& e) {
      // Pre-existing unavailability (this query never touched the runtime):
      // a local query may fall back to CPU, but the S3 branch below must not
      // rewrite the stable error.
      gpu_error                 = duckdb::ErrorData(e);
      gpu_failed                = true;
      runtime_unavailable_error = true;
    } catch (std::exception& e) {
      gpu_error  = duckdb::ErrorData(e);
      gpu_failed = true;
    }

    // Mandatory per-query cleanup + slot release, BEFORE any of the throwing
    // exits below and before the result is exposed. This covers every
    // gpu_failed exit (interrupt rethrow, unavailable-s3 rethrow, s3
    // no-fallback, sanitized INTERNAL/FATAL, generic no-fallback Throw) —
    // none may skip cleanup — and moves the CPU fallback outside the held
    // slot. finish() may itself throw (mandatory-cleanup failure ⇒ THIS query
    // errors; the runtime is latched unavailable only for shared corruption);
    // a non-std exception escaping the try above is handled by the window's
    // destructor backstop instead.
    if (window) {
      window->finish();
      window.reset();
    }

    if (gpu_failed) {
      const std::string gpu_msg = gpu_error.RawMessage();
      SIRIUS_LOG_ERROR("Transparent GPU execution error: {}", gpu_msg);

      // A user interrupt is never a fallback candidate — propagate it as-is.
      if (gpu_error.Type() == duckdb::ExceptionType::INTERRUPT) { gpu_error.Throw(); }

      // A pre-existing-unavailable error on an S3 query keeps its stable typed
      // message — the S3 branch below must not rewrite it (S3 has no CPU
      // fallback either way, so propagate as-is).
      if (runtime_unavailable_error &&
          (cpu_plan_reads_s3_ || sirius::references_sirius_owned_s3_parquet(query_sql_))) {
        gpu_error.Throw();
      }

      // S3 is GPU-only: DuckDB's CPU read_parquet cannot serve Sirius-owned s3://,
      // so surface a clear error instead of a fallback that would fail anyway.
      if (cpu_plan_reads_s3_ || sirius::references_sirius_owned_s3_parquet(query_sql_)) {
        throw duckdb::ExecutorException(
          "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, "
          "and Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
          gpu_msg);
      }

      // Fallback disabled, or no CPU plan stashed: surface the GPU error. Sanitize
      // INTERNAL/FATAL types (which would invalidate the whole database/session)
      // down to a plain ExecutorException; preserve other types.
      if (!cpu_fallback_prepared_ || !duckdb::duckdb_fallback_enabled(context.client)) {
        if (gpu_error.Type() == duckdb::ExceptionType::INTERNAL ||
            gpu_error.Type() == duckdb::ExceptionType::FATAL) {
          throw duckdb::ExecutorException("Sirius GPU execution failed: " + gpu_msg);
        }
        gpu_error.Throw("Sirius GPU execution failed: ");
      }

      // Fall back: run the stored CPU plan on a private executor bound to the same
      // ClientContext (same transaction / MVCC snapshot).
      duckdb::print_cpu_fallback_banner();
      SIRIUS_LOG_WARN(
        "Transparent execution: GPU execution failed at runtime; falling back to DuckDB CPU "
        "within the same transaction. GPU error: {}",
        gpu_msg);
      if (state.sirius_context) { state.sirius_context->record_transparent_runtime_fallback(); }

      // CpuFallbackGuard marks the replay so sirius_httpfs refuses to serve s3://
      // reached indirectly (e.g. through a view) to the CPU plan. Binds to the
      // TARGET executing connection's state.
      duckdb::SiriusContext::CpuFallbackGuard fallback_guard(context.client);
      state.result =
        run_cpu_fallback_plan(context.client, *cpu_fallback_prepared_, state.cpu_executor);
    }

    SIRIUS_LOG_INFO("Transparent GPU execution: query completed");
  }

  // Fetch the next chunk from the materialized result.
  state.current_chunk = state.result->Fetch();
  if (!state.current_chunk || state.current_chunk->size() == 0) {
    state.current_chunk.reset();
    state.finished = true;
    return duckdb::SourceResultType::FINISHED;
  }

  chunk.Reference(*state.current_chunk);
  return duckdb::SourceResultType::HAVE_MORE_OUTPUT;
}

}  // namespace sirius::transparent
