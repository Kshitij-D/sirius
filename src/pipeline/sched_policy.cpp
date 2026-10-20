#include "pipeline/sched_policy.hpp"

#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::sched {
namespace {

using clock = std::chrono::steady_clock;

policy_kind parse_policy() noexcept
{
  auto const* p = std::getenv("SIRIUS_POLICY");
  if (p == nullptr) { return policy_kind::upstream; }
  std::string const s(p);
  if (s == "rapids") { return policy_kind::rapids; }
  if (s == "multiqx") { return policy_kind::multiqx; }
  if (s == "sauron") { return policy_kind::sauron; }
  return policy_kind::upstream;
}

char const* policy_name(policy_kind k) noexcept
{
  switch (k) {
    case policy_kind::rapids: return "rapids";
    case policy_kind::multiqx: return "multiqx";
    case policy_kind::sauron: return "sauron";
    default: return "upstream";
  }
}

std::size_t env_size(char const* name, std::size_t dflt) noexcept
{
  auto const* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? static_cast<std::size_t>(std::strtoull(v, nullptr, 10)) : dflt;
}

// SQL text with literals replaced, so parameter variants of one query shape share a fingerprint.
std::size_t sql_fingerprint(std::string const& sql)
{
  std::string out;
  out.reserve(sql.size());
  for (std::size_t i = 0; i < sql.size(); ++i) {
    char const c = sql[i];
    if (c == '\'') {
      std::size_t j = i + 1;
      while (j < sql.size() && sql[j] != '\'') { ++j; }
      out += '?';
      i = j;
    } else if (std::isdigit(static_cast<unsigned char>(c)) &&
               (out.empty() || !(std::isalnum(static_cast<unsigned char>(out.back())) || out.back() == '_'))) {
      while (i + 1 < sql.size() &&
             (std::isdigit(static_cast<unsigned char>(sql[i + 1])) || sql[i + 1] == '.')) {
        ++i;
      }
      out += '?';
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      if (!out.empty() && out.back() != ' ') { out += ' '; }
    } else {
      out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return std::hash<std::string>{}(out);
}

struct task_ids {
  std::size_t query_id    = 0;
  std::size_t pipeline_id = 0;
  std::size_t basis       = 0;
};

bool ids_of(parallel::itask const& t, task_ids& out) noexcept
{
  auto const* g = dynamic_cast<pipeline::gpu_pipeline_task const*>(&t);
  if (g == nullptr) { return false; }
  auto const* pipe = g->get_pipeline();
  if (pipe == nullptr) { return false; }
  out.query_id    = sirius::value_of(pipe->get_query_id());
  out.pipeline_id = pipe->get_pipeline_id();
  out.basis       = g->get_input_size();
  return true;
}

struct running_query {
  std::size_t fp       = 0;
  std::size_t demand   = 0;  // multiqx: admitted weighted demand
  double weighted_sum  = 0;  // sum(duration * footprint)
  double duration_sum  = 0;
  bool admitted        = false;
};

struct state {
  policy_kind kind = parse_policy();
  cucascade::memory::memory_space const* gpu = nullptr;

  std::mutex mu;
  std::condition_variable cv;

  std::size_t resident = 0;  // pool bytes in use while no query ran (pinned tables)
  std::size_t granted  = 0;  // sum of grants of dispatched tasks
  std::size_t running  = 0;  // dispatched tasks holding a grant

  std::unordered_map<std::size_t, running_query> queries;          // by query id
  std::unordered_map<std::size_t, std::size_t> fp_of;              // query id -> fingerprint
  std::unordered_map<std::size_t, std::size_t> qx_demand;          // multiqx: fingerprint -> demand
  std::unordered_map<std::string, std::deque<std::size_t>> stage;  // rapids: "qid:pid" -> footprints
  std::unordered_map<std::string, std::vector<std::size_t>> stage_active;
  std::unordered_map<std::string, std::deque<double>> cls;         // sauron: "fp:pid" -> footprint/basis

  std::atomic<std::uint64_t> holds{0}, dispatches{0}, qholds{0};
  bool announced = false;
};

state& S()
{
  static state s;
  return s;
}

std::size_t limit_bytes(state const& s) noexcept { return s.gpu ? s.gpu->get_max_memory() : 0; }

std::size_t pool_used(state const& s) noexcept
{
  if (s.gpu == nullptr) { return 0; }
  auto const* mr = s.gpu->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (mr == nullptr) { return 0; }
  auto upstream = mr->get_upstream_resource();
  if (auto* amr = ::cuda::mr::resource_cast<rmm::mr::cuda_async_memory_resource>(&upstream)) {
    std::uint64_t u = 0;
    cudaMemPoolGetAttribute(amr->pool_handle(), cudaMemPoolAttrUsedMemCurrent, &u);
    return static_cast<std::size_t>(u);
  }
  return mr->get_total_allocated_bytes();
}

std::size_t capacity(state const& s) noexcept
{
  auto const lim = limit_bytes(s);
  return lim > s.resident ? lim - s.resident : 0;
}

std::string stage_key(task_ids const& id)
{
  return std::to_string(id.query_id) + ":" + std::to_string(id.pipeline_id);
}

std::string class_key(state const& s, task_ids const& id)
{
  auto const it = s.fp_of.find(id.query_id);
  return std::to_string(it == s.fp_of.end() ? id.query_id : it->second) + ":" +
         std::to_string(id.pipeline_id);
}

// Spark RAPIDS GpuStageMemoryEstimator: p80 of completed footprints and active estimates of the
// stage, padded with the default to at least 4 entries, capped at capacity.
std::size_t rapids_estimate(state& s, task_ids const& id)
{
  auto const cap   = capacity(s);
  auto const conc  = std::max<std::size_t>(1, env_size("SIRIUS_POL_RAPIDS_CONC", 4));
  auto const dflt  = cap / conc;
  auto const key   = stage_key(id);
  std::vector<std::size_t> v;
  if (auto it = s.stage.find(key); it != s.stage.end()) { v.assign(it->second.begin(), it->second.end()); }
  if (auto it = s.stage_active.find(key); it != s.stage_active.end()) {
    v.insert(v.end(), it->second.begin(), it->second.end());
  }
  while (v.size() < 4) { v.push_back(dflt); }
  auto const k = static_cast<std::size_t>(0.8 * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
  return std::min(v[k], cap);
}

// Sauron: the per-class peak/input ratio is exact across instances of one query shape (exp30
// replay: p90/p10 = 1.00 over 214 classes), so the prediction is the largest recent ratio times
// this task's real input size. Unknown class: 2x input, upstream's no-history default.
std::size_t sauron_predict(state& s, task_ids const& id)
{
  auto const it = s.cls.find(class_key(s, id));
  if (it == s.cls.end() || it->second.empty()) { return 2 * id.basis; }
  double const r = *std::max_element(it->second.begin(), it->second.end());
  return static_cast<std::size_t>(r * static_cast<double>(std::max<std::size_t>(id.basis, 1)));
}

void announce(state& s)
{
  if (s.announced) { return; }
  s.announced = true;
  std::fprintf(stderr, "[sched] policy=%s limit=%zu resident=%zu capacity=%zu\n", policy_name(s.kind),
               limit_bytes(s), s.resident, capacity(s));
}

}  // namespace

policy_kind active_policy() noexcept { return S().kind; }

bool minimal_reservations() noexcept { return S().kind != policy_kind::upstream; }

void bind_gpu_space(cucascade::memory::memory_space const* space) noexcept
{
  if (S().gpu == nullptr) { S().gpu = space; }
}

bool task_may_dispatch(parallel::itask const& task, bool& scan_blocked) noexcept
{
  auto& s = S();
  if (s.kind == policy_kind::upstream || s.kind == policy_kind::multiqx) { return true; }
  if (scan_blocked) { return false; }
  task_ids id;
  if (!ids_of(task, id)) { return true; }
  std::lock_guard lk(s.mu);
  announce(s);
  if (s.running == 0) { return true; }  // progress: something always runs
  bool fits = false;
  if (s.kind == policy_kind::rapids) {
    auto const est = rapids_estimate(s, id);
    fits           = s.granted + est <= capacity(s);
    if (!fits) { scan_blocked = true; }  // head-of-line blocking, no backfill
    task.sched_pred = est;
  } else {
    auto const pred = sauron_predict(s, id);
    fits            = pool_used(s) + s.granted + pred <= limit_bytes(s);
    task.sched_pred = pred;
  }
  if (!fits) { s.holds.fetch_add(1, std::memory_order_relaxed); }
  return fits;
}

void on_task_dispatched(parallel::itask& task) noexcept
{
  auto& s = S();
  if (s.kind == policy_kind::upstream) { return; }
  std::lock_guard lk(s.mu);
  task.sched_start = clock::now();
  if (task.sched_grant != 0 || s.kind == policy_kind::multiqx) { return; }
  task_ids id;
  if (!ids_of(task, id)) { return; }
  std::size_t const g = std::max<std::size_t>(task.sched_pred, 1);
  task.sched_grant    = g;
  s.granted += g;
  ++s.running;
  if (s.kind == policy_kind::rapids) { s.stage_active[stage_key(id)].push_back(g); }
  s.dispatches.fetch_add(1, std::memory_order_relaxed);
}

void on_task_released(parallel::itask& task) noexcept
{
  auto& s = S();
  if (task.sched_grant == 0) { return; }
  std::lock_guard lk(s.mu);
  s.granted -= std::min(s.granted, task.sched_grant);
  if (s.running > 0) { --s.running; }
  if (s.kind == policy_kind::rapids) {
    task_ids id;
    if (ids_of(task, id)) {
      auto& v = s.stage_active[stage_key(id)];
      if (auto it = std::find(v.begin(), v.end(), task.sched_grant); it != v.end()) { v.erase(it); }
    }
  }
  task.sched_grant = 0;
}

void on_task_completed(parallel::itask const& task,
                       std::size_t input_basis,
                       std::size_t footprint,
                       bool ratio_eligible) noexcept
{
  auto& s = S();
  if (s.kind == policy_kind::upstream) { return; }
  task_ids id;
  if (!ids_of(task, id)) { return; }
  std::lock_guard lk(s.mu);
  if (s.kind == policy_kind::rapids) {
    auto& h = s.stage[stage_key(id)];
    h.push_back(footprint);
    if (h.size() > 200) { h.pop_front(); }
  } else if (s.kind == policy_kind::sauron) {
    if (ratio_eligible && input_basis > 0) {
      auto& h = s.cls[class_key(s, id)];
      h.push_back(static_cast<double>(footprint) / static_cast<double>(input_basis));
      if (h.size() > 8) { h.pop_front(); }
    }
  } else if (s.kind == policy_kind::multiqx) {
    if (auto it = s.queries.find(id.query_id); it != s.queries.end()) {
      double const d =
        std::chrono::duration<double>(clock::now() - task.sched_start).count();
      it->second.weighted_sum += d * static_cast<double>(footprint);
      it->second.duration_sum += d;
    }
  }
}

void query_begin(std::size_t query_id, std::string const& sql) noexcept
{
  auto& s = S();
  if (s.kind == policy_kind::upstream) { return; }
  std::lock_guard lk(s.mu);
  if (s.queries.empty() && s.running == 0) { s.resident = pool_used(s); }
  auto const fp      = sql_fingerprint(sql);
  s.fp_of[query_id]  = fp;
  s.queries[query_id] = running_query{.fp = fp};
}

void query_admit_wait(std::size_t query_id, std::function<bool()> const& interrupted)
{
  auto& s = S();
  if (s.kind != policy_kind::multiqx) { return; }
  std::unique_lock lk(s.mu);
  announce(s);
  auto it = s.queries.find(query_id);
  if (it == s.queries.end()) { return; }
  auto const cap = capacity(s);
  auto const d   = s.qx_demand.find(it->second.fp);
  std::size_t const demand =
    std::min(cap, d != s.qx_demand.end() ? d->second : cap / std::max<std::size_t>(1, env_size("SIRIUS_POL_QX_DEFAULT_DIV", 4)));
  bool held = false;
  while (true) {
    std::size_t admitted = 0, n = 0;
    for (auto const& [q, r] : s.queries) {
      if (r.admitted) { admitted += r.demand; ++n; }
    }
    if (n == 0 || admitted + demand <= cap) { break; }
    if (!held) { s.qholds.fetch_add(1, std::memory_order_relaxed); held = true; }
    s.cv.wait_for(lk, std::chrono::milliseconds(5));
    if (interrupted && interrupted()) { return; }
    it = s.queries.find(query_id);
    if (it == s.queries.end()) { return; }
  }
  it->second.admitted = true;
  it->second.demand   = demand;
}

void query_end(std::size_t query_id) noexcept
{
  auto& s = S();
  if (s.kind == policy_kind::upstream) { return; }
  {
    std::lock_guard lk(s.mu);
    auto it = s.queries.find(query_id);
    if (it != s.queries.end()) {
      if (s.kind == policy_kind::multiqx && it->second.duration_sum > 0) {
        s.qx_demand[it->second.fp] =
          static_cast<std::size_t>(it->second.weighted_sum / it->second.duration_sum);
      }
      s.queries.erase(it);
    }
    s.fp_of.erase(query_id);
    for (auto sit = s.stage.begin(); sit != s.stage.end();) {
      sit = sit->first.rfind(std::to_string(query_id) + ":", 0) == 0 ? s.stage.erase(sit) : std::next(sit);
    }
    static std::uint64_t ended = 0;
    if (++ended % 50 == 0) {
      std::fprintf(stderr, "[sched] ended=%llu dispatches=%llu task_holds=%llu query_holds=%llu granted=%zu running=%zu\n",
                   static_cast<unsigned long long>(ended),
                   static_cast<unsigned long long>(s.dispatches.load()),
                   static_cast<unsigned long long>(s.holds.load()),
                   static_cast<unsigned long long>(s.qholds.load()), s.granted, s.running);
    }
  }
  s.cv.notify_all();
}

}  // namespace sirius::sched
