#include "pipeline/sched_policy.hpp"

#include "memory/resident_ledger.hpp"

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
  if (s == "sauron2") { return policy_kind::sauron2; }
  if (s == "sauron3") { return policy_kind::sauron3; }
  return policy_kind::upstream;
}

char const* policy_name(policy_kind k) noexcept
{
  switch (k) {
    case policy_kind::rapids: return "rapids";
    case policy_kind::multiqx: return "multiqx";
    case policy_kind::sauron: return "sauron";
    case policy_kind::sauron2: return "sauron2";
    case policy_kind::sauron3: return "sauron3";
    default: return "upstream";
  }
}

double env_double(char const* name, double dflt) noexcept
{
  auto const* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::strtod(v, nullptr) : dflt;
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

// Whether a ready task's source is a scan of a resident table (its input is not freed when it completes).
struct task_inputs {
  std::size_t upload = 0;
  bool resident_scan = false;
};

task_inputs inputs_of(parallel::itask const& t) noexcept
{
  task_inputs r;
  auto const* g = dynamic_cast<pipeline::gpu_pipeline_task const*>(&t);
  if (g == nullptr || g->get_pipeline() == nullptr) { return r; }
  r.resident_scan = g->get_pipeline()->get_source_operator().second ==
                    op::SiriusPhysicalOperatorType::TABLE_SCAN;
  // Upload bytes are not read here: that takes batch locks while the scheduler holds the queue
  // lock (lock-order risk with the downgrade executor). The class footprint ratio already covers it.
  return r;
}

// Per (query shape, pipeline) class: recent peak/input ratios (exact across instances), and
// moving averages of output/input and seconds per input byte.
struct class_stats {
  std::deque<double> peak_ratio;
  double out_ratio    = -1;
  double sec_per_byte = -1;
};

struct running_query {
  std::size_t fp       = 0;
  double done_sec      = 0;  // sauron2: task seconds completed so far
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
  std::unordered_map<std::string, class_stats> cls;                // sauron*: "fp:pid" -> stats
  std::unordered_map<std::size_t, double> shape_work;              // sauron2: fp -> task seconds per query
  cucascade::memory::memory_space const* host = nullptr;

  // sauron2 feedback limit on running tasks.
  std::size_t k = 0, k_max = 0, max_running_win = 0;
  int dir = +1;
  clock::time_point win_start{};
  double win_bytes = 0, prev_rate = -1;
  std::FILE* log = nullptr;

  // sauron2 selection scan.
  std::size_t scan_pool_used = 0;
  parallel::itask const* best = nullptr;
  double best_score = 0;

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
  if (it == s.cls.end() || it->second.peak_ratio.empty()) { return 2 * id.basis; }
  double const r = *std::max_element(it->second.peak_ratio.begin(), it->second.peak_ratio.end());
  return static_cast<std::size_t>(r * static_cast<double>(std::max<std::size_t>(id.basis, 1)));
}

bool drain_holds(parallel::itask const& task) noexcept;  // ledger drain, defined below

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

void bind_host_space(cucascade::memory::memory_space const* space) noexcept
{
  if (S().host == nullptr) { S().host = space; }
}

bool policy_selects() noexcept
{
  return S().kind == policy_kind::sauron2 || S().kind == policy_kind::sauron3;
}

namespace {

// Hill-climb K on completed input bytes per second over ~1 s windows. K moves only when it was
// binding (some instant had K tasks running); a drop in rate reverses direction. The host tier
// nearing full means spill is about to reach disk, so K is cut regardless of rate.
void update_k(state& s, clock::time_point now)
{
  if (s.k == 0) {
    s.k_max     = std::max<std::size_t>(1, env_size("SIRIUS_POL_KMAX", 16));
    s.k         = std::min(s.k_max, std::max<std::size_t>(1, env_size("SIRIUS_POL_K0", 4)));
    s.win_start = now;
    if (auto const* p = std::getenv("SIRIUS_SCHEDLOG"); p != nullptr && *p != '\0') {
      s.log = std::fopen(p, "w");
      if (s.log) { std::fprintf(s.log, "t,k,rate,prev_rate,max_running,host_frac,pool_used,granted\n"); }
    }
  }
  double const dt = std::chrono::duration<double>(now - s.win_start).count();
  if (dt < 1.0) { return; }
  double const rate = s.win_bytes / dt;
  double host_frac  = 0;
  if (s.host != nullptr && s.host->get_max_memory() > 0) {
    host_frac = static_cast<double>(s.host->get_total_reserved_memory()) /
                static_cast<double>(s.host->get_max_memory());
  }
  double const eps = env_double("SIRIUS_POL_EPS", 0.03);
  if (host_frac >= env_double("SIRIUS_POL_HOST_BRAKE", 0.9)) {
    s.dir = -1;
    if (s.k > 1) { --s.k; }
  } else if (s.max_running_win >= s.k && s.win_bytes > 0) {
    if (s.prev_rate > 0 && rate < s.prev_rate * (1 - eps)) { s.dir = -s.dir; }
    if (s.dir > 0 && s.k < s.k_max) { ++s.k; }
    if (s.dir < 0 && s.k > 1) { --s.k; }
  }
  if (s.log) {
    std::fprintf(s.log, "%.3f,%zu,%.0f,%.0f,%zu,%.3f,%zu,%zu\n",
                 std::chrono::duration<double>(now.time_since_epoch()).count(), s.k, rate, s.prev_rate,
                 s.max_running_win, host_frac, s.scan_pool_used, s.granted);
    std::fflush(s.log);
  }
  if (s.win_bytes > 0) { s.prev_rate = rate; }
  s.win_bytes       = 0;
  s.max_running_win = s.running;
  s.win_start       = now;
}

}  // namespace

void scan_begin() noexcept
{
  auto& s = S();
  std::lock_guard lk(s.mu);
  announce(s);
  update_k(s, clock::now());
  s.max_running_win = std::max(s.max_running_win, s.running);
  s.scan_pool_used  = pool_used(s);
  s.best            = nullptr;
}

void scan_consider(parallel::itask const& task) noexcept
{
  if (drain_holds(task)) { return; }
  auto& s = S();
  task_ids id;
  if (!ids_of(task, id)) { return; }
  auto const in = inputs_of(task);
  std::lock_guard lk(s.mu);
  if (s.running >= s.k) { return; }
  auto const pred = sauron_predict(s, id);
  auto const lim  = limit_bytes(s);
  if (s.running > 0 && s.scan_pool_used + s.granted + pred > lim) {
    s.holds.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  auto const c_it    = s.cls.find(class_key(s, id));
  double const out_b = (c_it != s.cls.end() && c_it->second.out_ratio >= 0)
                         ? c_it->second.out_ratio * static_cast<double>(id.basis)
                         : static_cast<double>(id.basis);
  double const freed_b = in.resident_scan ? 0.0 : static_cast<double>(id.basis);
  if (s.kind == policy_kind::sauron3 && s.running > 0 && out_b - freed_b > 0) {
    // Persistent growth must fit the fast tiers: GPU headroom (limit minus real use and outstanding
    // peaks) plus host-tier headroom, keeping a 10% host margin before the disk tier.
    double gpu_head  = static_cast<double>(lim) - static_cast<double>(s.scan_pool_used + s.granted);
    double host_head = 0;
    if (s.host != nullptr) {
      host_head = 0.9 * static_cast<double>(s.host->get_max_memory()) -
                  static_cast<double>(s.host->get_total_reserved_memory());
    }
    if (out_b - freed_b > std::max(0.0, gpu_head) + std::max(0.0, host_head)) {
      s.holds.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  task.sched_pred = pred;
  double const pressure =
    lim > 0 ? static_cast<double>(s.scan_pool_used + s.granted) / static_cast<double>(lim) : 0;
  double score = 0;
  if (pressure >= env_double("SIRIUS_POL_PRESSURE", 0.9)) {
    // Largest persistent release first: consumed input (not for resident scans) minus output.
    score = -(freed_b - out_b) + static_cast<double>(in.upload);
  } else {
    // Least predicted remaining work first (shortest-remaining-processing-time on queries).
    double rem = 0;
    if (auto q = s.queries.find(id.query_id); q != s.queries.end()) {
      double total = 0;
      if (auto w = s.shape_work.find(q->second.fp); w != s.shape_work.end()) {
        total = w->second;
      } else if (!s.shape_work.empty()) {
        for (auto const& [f, v] : s.shape_work) { total += v; }
        total /= static_cast<double>(s.shape_work.size());
      }
      rem = std::max(0.0, total - q->second.done_sec);
    }
    score = rem;
  }
  if (s.best == nullptr || score < s.best_score) {  // strict: ties keep priority order
    s.best       = &task;
    s.best_score = score;
  }
}

parallel::itask const* scan_best() noexcept
{
  auto& s = S();
  std::lock_guard lk(s.mu);
  return s.best;
}

namespace {

struct drain_state {
  std::mutex mu;
  parallel::itask const* owner = nullptr;
  clock::time_point since{};
  std::uint64_t drains = 0, abandoned = 0;
};

drain_state& D()
{
  static drain_state d;
  return d;
}

// True when a drain is active and `task` is not its owner. Abandons a drain past its maximum.
bool drain_holds(parallel::itask const& task) noexcept
{
  if (!memory::ledger_enabled()) { return false; }
  auto& d = D();
  std::lock_guard lk(d.mu);
  if (d.owner == nullptr || d.owner == &task) { return false; }
  if (clock::now() - d.since > std::chrono::milliseconds(env_size("SIRIUS_LEDGER_DRAIN_MAX_MS", 10000))) {
    d.owner = nullptr;
    ++d.abandoned;
    std::fprintf(stderr, "[ledger] drain abandoned (drains=%llu abandoned=%llu)\n",
                 static_cast<unsigned long long>(d.drains), static_cast<unsigned long long>(d.abandoned));
    return false;
  }
  return true;
}

}  // namespace

void ledger_note_refusal(parallel::itask& task) noexcept
{
  if (!memory::ledger_enabled()) { return; }
  auto const now = clock::now();
  if (task.sched_wait_since == clock::time_point{}) { task.sched_wait_since = now; }
  auto& d = D();
  std::lock_guard lk(d.mu);
  auto const drain_ms = env_size("SIRIUS_LEDGER_DRAIN_MS", 0);
  if (drain_ms > 0 && d.owner == nullptr &&
      now - task.sched_wait_since > std::chrono::milliseconds(drain_ms)) {
    d.owner = &task;
    d.since = now;
    ++d.drains;
  }
}

bool ledger_should_shrink(parallel::itask const& task) noexcept
{
  if (!memory::ledger_enabled() || task.sched_wait_since == clock::time_point{}) { return false; }
  static auto const ms = std::chrono::milliseconds(env_size("SIRIUS_LEDGER_SHRINK_MS", 1000));
  return clock::now() - task.sched_wait_since > ms;
}

void ledger_note_granted(parallel::itask& task) noexcept
{
  task.sched_wait_since = clock::time_point{};
  if (!memory::ledger_enabled()) { return; }
  auto& d = D();
  std::lock_guard lk(d.mu);
  if (d.owner == &task) { d.owner = nullptr; }
}

void ledger_task_destroyed(parallel::itask const& task) noexcept
{
  if (!memory::ledger_enabled()) { return; }
  auto& d = D();
  std::lock_guard lk(d.mu);
  if (d.owner == &task) { d.owner = nullptr; }
}

bool task_may_dispatch(parallel::itask const& task, bool& scan_blocked) noexcept
{
  if (drain_holds(task)) { return false; }
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
                       std::size_t output_bytes,
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
  } else if (s.kind == policy_kind::sauron || s.kind == policy_kind::sauron2 ||
             s.kind == policy_kind::sauron3) {
    double const sec = std::chrono::duration<double>(clock::now() - task.sched_start).count();
    if (ratio_eligible && input_basis > 0) {
      auto& c = s.cls[class_key(s, id)];
      double const b = static_cast<double>(input_basis);
      c.peak_ratio.push_back(static_cast<double>(footprint) / b);
      if (c.peak_ratio.size() > 8) { c.peak_ratio.pop_front(); }
      double const o = static_cast<double>(output_bytes) / b;
      c.out_ratio    = c.out_ratio < 0 ? o : 0.75 * c.out_ratio + 0.25 * o;
      c.sec_per_byte = c.sec_per_byte < 0 ? sec / b : 0.75 * c.sec_per_byte + 0.25 * sec / b;
    }
    if (auto it = s.queries.find(id.query_id); it != s.queries.end()) { it->second.done_sec += sec; }
    s.win_bytes += static_cast<double>(input_basis);
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
  if (memory::resident_tracking_enabled()) {
    static std::atomic<std::uint64_t> n{0};
    if (n.fetch_add(1) % 50 == 0) {
      auto& d = D();
      std::lock_guard lk(d.mu);
      std::fprintf(stderr, "[ledger] queries_ended=%llu resident=%zu limit=%zu drains=%llu abandoned=%llu\n",
                   static_cast<unsigned long long>(n.load()), memory::resident_bytes(s.gpu),
                   limit_bytes(s), static_cast<unsigned long long>(d.drains),
                   static_cast<unsigned long long>(d.abandoned));
    }
  }
  if (s.kind == policy_kind::upstream) { return; }
  {
    std::lock_guard lk(s.mu);
    auto it = s.queries.find(query_id);
    if (it != s.queries.end()) {
      if ((s.kind == policy_kind::sauron2 || s.kind == policy_kind::sauron3) &&
          it->second.done_sec > 0) {
        auto& w = s.shape_work[it->second.fp];
        w       = w <= 0 ? it->second.done_sec : 0.5 * w + 0.5 * it->second.done_sec;
      }
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
