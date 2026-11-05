#include "memory/shared_uploads.hpp"

#include "data/sirius_converter_registry.hpp"
#include "memory/resident_ledger.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>

#include <cudf/table/table.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <deque>
#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::memory {

namespace {

struct column_key {
  const void* allocation;
  std::uint64_t layout;
  bool operator==(const column_key& o) const
  {
    return allocation == o.allocation && layout == o.layout;
  }
};

struct column_key_hash {
  std::size_t operator()(const column_key& k) const
  {
    return std::hash<const void*>{}(k.allocation) ^ (k.layout * 0x9E3779B97F4A7C15ULL);
  }
};

void mix(std::uint64_t& h, std::uint64_t v)
{
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
}

// A column's identity inside its allocation: type and the offsets/sizes of every buffer, children
// included. Slices of one allocation share these offsets, so equal layout in the same allocation
// is the same stored column.
std::uint64_t layout_of(const ::cucascade::memory::column_metadata& m)
{
  std::uint64_t h = 1469598103934665603ULL;
  mix(h, static_cast<std::uint64_t>(m.type_id));
  mix(h, static_cast<std::uint64_t>(m.scale));
  mix(h, m.has_null_mask ? m.null_mask_offset + 1 : 0);
  mix(h, m.null_mask_size);
  mix(h, m.has_data ? m.data_offset + 1 : 0);
  mix(h, m.data_size);
  for (auto const& c : m.children) {
    mix(h, layout_of(c));
  }
  return h;
}

struct entry {
  std::weak_ptr<const void> allocation;  // guards against address reuse after an unpin
  std::weak_ptr<cudf::column> column;
  std::shared_future<std::shared_ptr<cudf::column>> pending;
};

struct chunk_place {
  const void* entry;
  std::size_t index;
};

// Plan-driven retention (SIRIUS_SHARE_UPLOAD_POLICY=plan): a retained copy carries the number of reads
// the batch still owes it; the copy with the fewest remaining reads is evicted first, and a copy is
// released at its last expected read.
struct planned_slot {
  std::shared_ptr<cudf::column> column;
  long long remaining = 0;
  std::uint64_t seq   = 0;
};

using plan_order = std::tuple<long long, std::uint64_t, const void*, std::uint64_t>;

struct cache {
  std::mutex mu;
  bool plan_policy = false;
  std::unordered_map<std::string, long long> expected_readers;  // column name -> queries reading it
  std::unordered_map<column_key, std::string, column_key_hash> name_of;
  std::unordered_map<column_key, planned_slot, column_key_hash> planned;
  std::set<plan_order> plan_index;  // (remaining, seq, allocation, layout), smallest first
  std::uint64_t plan_seq = 0;
  std::atomic<std::uint64_t> plan_released{0}, plan_evicted{0}, plan_skipped{0};
  std::unordered_map<const void*, chunk_place> chunk_of;        // host allocation -> place
  std::unordered_map<const void*, std::size_t> hot;             // entry -> last uploaded chunk
  std::unordered_map<column_key, entry, column_key_hash> entries;
  std::list<std::pair<column_key, std::shared_ptr<cudf::column>>> lru;  // retained copies
  // SIRIUS_SHARE_UPLOAD_POLICY=lru: a hit moves the copy to the back (true LRU). Default: insertion order (FIFO).
  bool true_lru = false;
  // SIRIUS_SHARE_UPLOAD_POLICY=nextuse (B1b): host-tier scans in flight, for the next read of a copy.
  bool nextuse = false;
  struct active_scan {
    const void* entry;
    std::unordered_set<std::string> names;
    std::unordered_map<std::size_t, std::size_t> position;  // chunk index -> position in the scan order
    std::size_t n;
    const std::atomic<std::size_t>* cursor;
    const std::size_t* rotate;
  };
  std::unordered_map<std::uint64_t, active_scan> scans;
  std::uint64_t scan_seq = 0;
  std::atomic<std::uint64_t> nu_evicted{0}, nu_skipped{0};
  // SIRIUS_SHARE_PREFETCH=1 (B2): next chunks handed over by scans, uploaded by one worker.
  bool prefetch                                  = false;
  ::cucascade::memory::memory_space* space        = nullptr;  // device space of the first shared upload
  std::mutex pf_mu;
  std::condition_variable pf_cv;
  std::deque<std::unique_ptr<::cucascade::host_data_representation>> pf_queue;
  std::once_flag pf_start;
  std::atomic<std::uint64_t> pf_uploads{0}, pf_bytes{0}, pf_present{0}, pf_skipped{0}, pf_dropped{0}, pf_reclaimed{0};
  std::unordered_map<column_key, std::list<std::pair<column_key, std::shared_ptr<cudf::column>>>::iterator,
                     column_key_hash>
    lru_pos;
  std::map<std::pair<const void*, std::size_t>, std::size_t> retained_by_chunk;  // (entry, chunk) -> bytes
  std::atomic<std::uint64_t> rf_picks{0}, rf_resident{0};
  // SIRIUS_SHARE_ELASTIC=1: retained copies give way to working memory (see shared_uploads.hpp).
  bool elastic = false;
  std::size_t margin = std::size_t{1} << 30;
  std::chrono::steady_clock::time_point last_pressure{};
  std::atomic<std::size_t> retained_atomic{0};
  std::atomic<std::uint64_t> reclaim_events{0}, reclaimed_bytes{0}, elastic_skips{0};
  // SIRIUS_SHARE_TRACE=<file>: one line per column read (t_ns event entry chunk layout bytes), written at exit.
  bool trace = false;
  std::string trace_path;
  struct trace_rec {
    std::uint64_t t_ns;
    char ev;  // U upload, H hit, W wait
    const void* entry;
    long long chunk;
    std::uint64_t layout;
    std::uint64_t bytes;
  };
  std::vector<trace_rec> trace_log;
  std::size_t retained_bytes = 0;
  std::size_t budget         = 0;
  std::atomic<std::uint64_t> uploads{0}, hits{0}, waits{0}, bytes_uploaded{0}, bytes_hit{0};
};

// Leaked on purpose: retained device columns must not be freed during static destruction, after
// the CUDA context may be gone. The summary is printed from an atexit handler instead.
cache& the_cache()
{
  static cache* c = [] {
    auto* x = new cache;
    if (auto const* b = std::getenv("SIRIUS_SHARE_UPLOAD_BYTES"); b != nullptr && *b != '\0') {
      x->budget = std::stoull(b);
    }
    if (auto const* pol = std::getenv("SIRIUS_SHARE_UPLOAD_POLICY"); pol != nullptr && std::string(pol) == "plan") {
      x->plan_policy = true;
      if (auto const* f = std::getenv("SIRIUS_SHARE_UPLOAD_READERS"); f != nullptr && *f != '\0') {
        std::ifstream in(f);
        std::string name;
        long long n = 0;
        while (in >> name >> n) { x->expected_readers[name] = n; }
      }
      std::fprintf(stderr, "[share-upload] plan policy: %zu columns with reader counts, budget %zu\n",
                   x->expected_readers.size(), x->budget);
    }
    if (auto const* pol = std::getenv("SIRIUS_SHARE_UPLOAD_POLICY"); pol != nullptr && std::string(pol) == "nextuse") {
      x->nextuse  = true;
      x->true_lru = true;  // the retained list keeps recency order for ties
      std::fprintf(stderr, "[share-upload] next-use policy, budget %zu\n", x->budget);
    }
    if (auto const* pol = std::getenv("SIRIUS_SHARE_UPLOAD_POLICY"); pol != nullptr && std::string(pol) == "lru") {
      x->true_lru = true;
      std::fprintf(stderr, "[share-upload] true LRU policy, budget %zu\n", x->budget);
    }
    if (auto const* e = std::getenv("SIRIUS_SHARE_ELASTIC"); e != nullptr && *e == '1') {
      x->elastic = true;
      // SIRIUS_SHARE_ELASTIC_NOMARGIN=1: HeavyDB-style (working memory evicts cache on demand, no admission margin).
      if (auto const* nm = std::getenv("SIRIUS_SHARE_ELASTIC_NOMARGIN"); nm != nullptr && *nm == '1') { x->margin = 0; }
      std::fprintf(stderr, "[share-upload] elastic retention, cap %zu, margin %zu\n", x->budget, x->margin);
    }
    if (auto const* pf = std::getenv("SIRIUS_SHARE_PREFETCH"); pf != nullptr && *pf == '1') {
      x->prefetch = true;
      std::fprintf(stderr, "[share-upload] prefetch on (next chunk per scan, 2 GiB free-memory margin)\n");
    }
    if (auto const* t = std::getenv("SIRIUS_SHARE_TRACE"); t != nullptr && *t != '\0') {
      x->trace      = true;
      x->trace_path = t;
      x->trace_log.reserve(1 << 20);
    }
    std::atexit([] {
      auto& y = the_cache();
      std::fprintf(stderr,
                   "[share-upload] summary uploads=%llu hits=%llu waits=%llu bytes_uploaded=%llu "
                   "bytes_hit=%llu budget=%zu plan_released=%llu plan_evicted=%llu plan_skipped=%llu\n",
                   static_cast<unsigned long long>(y.uploads.load()),
                   static_cast<unsigned long long>(y.hits.load()),
                   static_cast<unsigned long long>(y.waits.load()),
                   static_cast<unsigned long long>(y.bytes_uploaded.load()),
                   static_cast<unsigned long long>(y.bytes_hit.load()),
                   y.budget,
                   static_cast<unsigned long long>(y.plan_released.load()),
                   static_cast<unsigned long long>(y.plan_evicted.load()),
                   static_cast<unsigned long long>(y.plan_skipped.load()));
      std::fprintf(stderr, "[share-upload] elastic=%d reclaim_events=%llu reclaimed_bytes=%llu skips=%llu margin=%zu\n",
                   y.elastic ? 1 : 0, static_cast<unsigned long long>(y.reclaim_events.load()),
                   static_cast<unsigned long long>(y.reclaimed_bytes.load()),
                   static_cast<unsigned long long>(y.elastic_skips.load()), y.margin);
      std::fprintf(stderr, "[share-upload] resident_first picks=%llu resident=%llu true_lru=%d\n",
                   static_cast<unsigned long long>(y.rf_picks.load()),
                   static_cast<unsigned long long>(y.rf_resident.load()), y.true_lru ? 1 : 0);
      std::fprintf(stderr, "[share-upload] prefetch=%d uploads=%llu bytes=%llu present=%llu skipped=%llu dropped=%llu reclaimed=%llu\n",
                   y.prefetch ? 1 : 0, static_cast<unsigned long long>(y.pf_uploads.load()),
                   static_cast<unsigned long long>(y.pf_bytes.load()),
                   static_cast<unsigned long long>(y.pf_present.load()),
                   static_cast<unsigned long long>(y.pf_skipped.load()),
                   static_cast<unsigned long long>(y.pf_dropped.load()),
                   static_cast<unsigned long long>(y.pf_reclaimed.load()));
      std::fprintf(stderr, "[share-upload] nextuse=%d evicted=%llu skipped=%llu\n", y.nextuse ? 1 : 0,
                   static_cast<unsigned long long>(y.nu_evicted.load()),
                   static_cast<unsigned long long>(y.nu_skipped.load()));
      if (y.trace) {
        std::lock_guard lk(y.mu);
        if (std::FILE* f = std::fopen(y.trace_path.c_str(), "w")) {
          for (auto const& r : y.trace_log) {
            std::fprintf(f, "%llu %c %p %lld %llu %llu\n", static_cast<unsigned long long>(r.t_ns), r.ev, r.entry,
                         r.chunk, static_cast<unsigned long long>(r.layout), static_cast<unsigned long long>(r.bytes));
          }
          std::fclose(f);
        }
      }
    });
    return x;
  }();
  return *c;
}

// Retained bytes of the chunk that host allocation `allocation` belongs to (resident-first order).
void chunk_adjust(cache& c, const void* allocation, std::size_t bytes, bool add)
{
  c.retained_atomic.store(c.retained_bytes, std::memory_order_relaxed);
  auto pl = c.chunk_of.find(allocation);
  if (pl == c.chunk_of.end()) { return; }
  auto& v = c.retained_by_chunk[{pl->second.entry, pl->second.index}];
  v       = add ? v + bytes : (v > bytes ? v - bytes : 0);
}

void trace_read(cache& c, const column_key& key, char ev, std::size_t bytes)
{
  if (!c.trace) { return; }
  auto pl = c.chunk_of.find(key.allocation);
  auto const t = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
  c.trace_log.push_back(cache::trace_rec{t,
                                         ev,
                                         pl == c.chunk_of.end() ? nullptr : pl->second.entry,
                                         pl == c.chunk_of.end() ? -1LL : static_cast<long long>(pl->second.index),
                                         key.layout,
                                         bytes});
}

// Elastic admission: keep `margin` free on the device after retaining `bytes`; the margin halves after 30 s quiet.
bool elastic_admit(cache& c, std::size_t bytes, std::size_t available)
{
  if (!c.elastic || c.margin == 0) { return true; }
  auto const now = std::chrono::steady_clock::now();
  if (c.margin > (std::size_t{1} << 30) && now - c.last_pressure > std::chrono::seconds(30)) {
    c.margin /= 2;
    c.last_pressure = now;
  }
  if (available >= bytes + c.margin) { return true; }
  c.elastic_skips.fetch_add(1, std::memory_order_relaxed);
  return false;
}

// True LRU: a hit makes the retained copy the most recent.
void lru_touch(cache& c, const column_key& key)
{
  if (!c.true_lru) { return; }
  auto it = c.lru_pos.find(key);
  if (it != c.lru_pos.end()) { c.lru.splice(c.lru.end(), c.lru, it->second); }
}

void retain(cache& c, const column_key& key, const std::shared_ptr<cudf::column>& col)
{
  if (c.budget == 0) { return; }
  if (c.lru_pos.count(key) != 0) { return; }
  c.lru.emplace_back(key, col);
  c.lru_pos[key] = std::prev(c.lru.end());
  c.retained_bytes += col->alloc_size();
  chunk_adjust(c, key.allocation, col->alloc_size(), true);
  while (c.retained_bytes > c.budget && !c.lru.empty()) {
    auto const& front = c.lru.front();
    c.retained_bytes -= front.second->alloc_size();
    chunk_adjust(c, front.first.allocation, front.second->alloc_size(), false);
    c.lru_pos.erase(front.first);
    c.lru.pop_front();
  }
}

// Chunks the nearest scan in flight still takes before it reads `key`'s chunk; SIZE_MAX if none will.
std::size_t next_use(cache const& c, const column_key& key)
{
  auto const pl = c.chunk_of.find(key.allocation);
  auto const nm = c.name_of.find(key);
  if (pl == c.chunk_of.end() || nm == c.name_of.end()) { return SIZE_MAX; }
  std::size_t best = SIZE_MAX;
  for (auto const& [id, sc] : c.scans) {
    if (sc.entry != pl->second.entry || sc.n == 0 || sc.names.count(nm->second) == 0) { continue; }
    auto const pos = sc.position.find(pl->second.index);
    if (pos == sc.position.end()) { continue; }
    auto const step = (pos->second + sc.n - *sc.rotate % sc.n) % sc.n;  // when this scan takes it
    auto const cur  = sc.cursor->load(std::memory_order_relaxed);
    if (step >= cur) { best = std::min(best, step - cur); }
  }
  return best;
}

void drop_lru(cache& c, std::list<std::pair<column_key, std::shared_ptr<cudf::column>>>::iterator it)
{
  c.retained_bytes -= it->second->alloc_size();
  chunk_adjust(c, it->first.allocation, it->second->alloc_size(), false);
  c.lru_pos.erase(it->first);
  c.lru.erase(it);
}

// Next-use retention (B1b): evict the copy read farthest in the future; do not retain a new copy that
// would itself be the farthest.
void nextuse_retain(cache& c, const column_key& key, const std::shared_ptr<cudf::column>& col)
{
  auto const bytes = col->alloc_size();
  if (c.budget == 0 || bytes > c.budget || c.lru_pos.count(key) != 0) { return; }
  auto const mine = next_use(c, key);
  while (c.retained_bytes + bytes > c.budget && !c.lru.empty()) {
    auto victim     = c.lru.begin();
    std::size_t far = next_use(c, victim->first);
    for (auto it = std::next(c.lru.begin()); it != c.lru.end(); ++it) {  // oldest first wins ties
      if (auto const d = next_use(c, it->first); d > far) {
        victim = it;
        far    = d;
      }
    }
    if (far < mine) {
      c.nu_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    drop_lru(c, victim);
    c.nu_evicted.fetch_add(1, std::memory_order_relaxed);
  }
  c.lru.emplace_back(key, col);
  c.lru_pos[key] = std::prev(c.lru.end());
  c.retained_bytes += bytes;
  chunk_adjust(c, key.allocation, bytes, true);
}

void plan_drop(cache& c, std::unordered_map<column_key, planned_slot, column_key_hash>::iterator it)
{
  c.plan_index.erase(plan_order{it->second.remaining, it->second.seq, it->first.allocation, it->first.layout});
  c.retained_bytes -= it->second.column->alloc_size();
  chunk_adjust(c, it->first.allocation, it->second.column->alloc_size(), false);
  c.planned.erase(it);
}

// One read of `key` happened (upload or hit): count it; release at the last expected read.
void plan_read(cache& c, const column_key& key)
{
  auto it = c.planned.find(key);
  if (it == c.planned.end()) { return; }
  c.plan_index.erase(plan_order{it->second.remaining, it->second.seq, key.allocation, key.layout});
  if (--it->second.remaining <= 0) {
    c.retained_bytes -= it->second.column->alloc_size();
    chunk_adjust(c, key.allocation, it->second.column->alloc_size(), false);
    c.planned.erase(it);
    c.plan_released.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  c.plan_index.insert(plan_order{it->second.remaining, it->second.seq, key.allocation, key.layout});
}

// Retain a freshly uploaded copy for its remaining expected reads, evicting copies with fewer
// remaining reads while over budget; skip it if it would rank lowest.
void plan_retain(cache& c, const column_key& key, const std::shared_ptr<cudf::column>& col)
{
  auto const nm      = c.name_of.find(key);
  long long remaining = 0;
  if (nm != c.name_of.end()) {
    auto const ex = c.expected_readers.find(nm->second);
    if (ex != c.expected_readers.end()) { remaining = ex->second - 1; }  // this upload is one read
  }
  auto const bytes = col->alloc_size();
  if (remaining <= 0 || bytes > c.budget) { return; }
  while (c.retained_bytes + bytes > c.budget && !c.plan_index.empty()) {
    auto const& lowest = *c.plan_index.begin();
    if (std::get<0>(lowest) >= remaining) {
      c.plan_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    plan_drop(c, c.planned.find(column_key{std::get<2>(lowest), std::get<3>(lowest)}));
    c.plan_evicted.fetch_add(1, std::memory_order_relaxed);
  }
  if (c.retained_bytes + bytes > c.budget) { return; }
  auto const seq = ++c.plan_seq;
  c.planned[key] = planned_slot{col, remaining, seq};
  c.plan_index.insert(plan_order{remaining, seq, key.allocation, key.layout});
  c.retained_bytes += bytes;
  chunk_adjust(c, key.allocation, bytes, true);
}

std::shared_ptr<cudf::column> upload_one(::cucascade::host_data_representation& host,
                                         std::size_t i,
                                         ::cucascade::memory::memory_space& space)
{
  std::size_t const idx[1] = {i};
  auto single              = host.slice(idx);
  auto& registry           = ::sirius::converter_registry::get();
  // The host-to-GPU converter copies on the space's own stream and synchronizes before returning;
  // the column's buffers stay bound to that (untracked) stream, so whichever holder drops the last
  // reference frees outside any task's tracked reservation.
  auto gpu   = registry.convert<::cucascade::gpu_table_representation>(*single, &space);
  auto table = gpu->release_table(::cuda::stream_ref{cudaStream_t{}});
  auto cols  = table->release();
  return make_resident(std::move(cols.at(0)), &space);
}

std::shared_ptr<cudf::column> acquire(cache& c,
                                      ::cucascade::host_data_representation& host,
                                      std::size_t i,
                                      ::cucascade::memory::memory_space& space)
{
  auto const& table = host.get_host_table();
  column_key const key{table->allocation.get(), layout_of(table->columns.at(i))};
  std::promise<std::shared_ptr<cudf::column>> mine;
  {
    std::unique_lock lk(c.mu);
    auto it = c.entries.find(key);
    if (it != c.entries.end()) {
      auto alloc = it->second.allocation.lock();
      if (alloc && alloc.get() == key.allocation) {
        if (auto col = it->second.column.lock()) {
          c.hits.fetch_add(1, std::memory_order_relaxed);
          c.bytes_hit.fetch_add(col->alloc_size(), std::memory_order_relaxed);
          if (c.plan_policy) { plan_read(c, key); }
          lru_touch(c, key);
          trace_read(c, key, 'H', col->alloc_size());
          return col;
        }
        if (it->second.pending.valid()) {
          auto pending = it->second.pending;
          lk.unlock();
          if (auto col = pending.get()) {
            c.waits.fetch_add(1, std::memory_order_relaxed);
            c.hits.fetch_add(1, std::memory_order_relaxed);
            c.bytes_hit.fetch_add(col->alloc_size(), std::memory_order_relaxed);
            {
              std::lock_guard relk(c.mu);
              if (c.plan_policy) { plan_read(c, key); }
              lru_touch(c, key);
              trace_read(c, key, 'W', col->alloc_size());
            }
            return col;
          }
          return upload_one(host, i, space);  // the other upload failed: do our own
        }
      }
    }
    auto& e      = c.entries[key];
    e.allocation = std::shared_ptr<const void>(table->allocation, table->allocation.get());
    e.column.reset();
    e.pending = mine.get_future().share();
  }
  std::shared_ptr<cudf::column> col;
  try {
    col = upload_one(host, i, space);
  } catch (...) {
    {
      std::lock_guard lk(c.mu);
      c.entries[key].pending = {};
    }
    mine.set_value(nullptr);
    throw;
  }
  c.uploads.fetch_add(1, std::memory_order_relaxed);
  c.bytes_uploaded.fetch_add(col->alloc_size(), std::memory_order_relaxed);
  auto const available = c.elastic ? space.get_available_memory() : std::size_t{0};
  {
    std::lock_guard lk(c.mu);
    if (auto pl = c.chunk_of.find(key.allocation); pl != c.chunk_of.end()) {
      c.hot[pl->second.entry] = pl->second.index;
    }
    auto& e   = c.entries[key];
    e.column  = col;
    e.pending = {};
    trace_read(c, key, 'U', col->alloc_size());
    if (!elastic_admit(c, col->alloc_size(), available)) {
      // not retained: the copy lives while its scans hold it
    } else if (c.plan_policy) {
      plan_retain(c, key, col);
    } else if (c.nextuse) {
      nextuse_retain(c, key, col);
    } else {
      retain(c, key, col);
    }
  }
  mine.set_value(col);
  return col;
}

// Drop retained copies no scan holds, lowest priority first under the active policy, until about
// `bytes` are freed; c.mu held. Shared by memory pressure (reclaim_retained) and upload-ahead.
std::size_t reclaim_locked(cache& c, std::size_t bytes)
{
  std::size_t freed = 0;
  if (c.nextuse) {
    // Farthest next read first (ties: least recently used first).
    std::vector<std::pair<std::size_t, std::list<std::pair<column_key, std::shared_ptr<cudf::column>>>::iterator>> order;
    std::size_t rank = 0;
    for (auto it = c.lru.begin(); it != c.lru.end(); ++it, ++rank) { order.emplace_back(next_use(c, it->first), it); }
    std::stable_sort(order.begin(), order.end(), [](auto const& a, auto const& b) { return a.first > b.first; });
    for (auto& [d, it] : order) {
      if (freed >= bytes) { break; }
      if (it->second.use_count() > 1) { continue; }  // a scan holds it
      freed += it->second->alloc_size();
      drop_lru(c, it);
    }
  } else if (c.plan_policy) {
    for (auto it = c.plan_index.begin(); it != c.plan_index.end() && freed < bytes;) {
      auto slot = c.planned.find(column_key{std::get<2>(*it), std::get<3>(*it)});
      ++it;
      if (slot == c.planned.end() || slot->second.column.use_count() > 1) { continue; }  // a scan holds it
      freed += slot->second.column->alloc_size();
      plan_drop(c, slot);
    }
  } else {
    for (auto it = c.lru.begin(); it != c.lru.end() && freed < bytes;) {
      if (it->second.use_count() > 1) {
        ++it;
        continue;
      }
      auto const sz = it->second->alloc_size();
      freed += sz;
      c.retained_bytes -= sz;
      chunk_adjust(c, it->first.allocation, sz, false);
      c.lru_pos.erase(it->first);
      it = c.lru.erase(it);
    }
  }
  return freed;
}

std::size_t host_bytes(const ::cucascade::memory::column_metadata& m)
{
  std::size_t b = (m.has_data ? m.data_size : 0) + (m.has_null_mask ? m.null_mask_size : 0);
  for (auto const& ch : m.children) {
    b += host_bytes(ch);
  }
  return b;
}

// Upload-ahead of column `i` of `host` (B2): only if it is neither on the device nor in flight and the
// device has room for it plus a margin; the copy is retained by the active policy (no read counted).
void prefetch_one(cache& c, ::cucascade::host_data_representation& host, std::size_t i)
{
  auto* space = c.space;
  if (space == nullptr) { return; }
  auto const& table = host.get_host_table();
  column_key const key{table->allocation.get(), layout_of(table->columns.at(i))};
  std::promise<std::shared_ptr<cudf::column>> mine;
  {
    std::lock_guard lk(c.mu);
    auto it = c.entries.find(key);
    if (it != c.entries.end()) {
      auto alloc = it->second.allocation.lock();
      if (alloc && alloc.get() == key.allocation &&
          (!it->second.column.expired() || it->second.pending.valid())) {
        c.pf_present.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    // Make room the way an upload by the scan itself would: drop the lowest-priority retained copies
    // (not a pressure event: the admission margin is unchanged). E94 s28: without this the elastic
    // cache kept the device full and no prefetch ever ran.
    auto const need = host_bytes(table->columns.at(i)) + (std::size_t{2} << 30);
    if (auto const avail = space->get_available_memory(); avail < need) {
      c.pf_reclaimed.fetch_add(reclaim_locked(c, need - avail), std::memory_order_relaxed);
      if (space->get_available_memory() < need) {
        c.pf_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    auto& e      = c.entries[key];
    e.allocation = std::shared_ptr<const void>(table->allocation, table->allocation.get());
    e.column.reset();
    e.pending = mine.get_future().share();
  }
  std::shared_ptr<cudf::column> col;
  try {
    col = upload_one(host, i, *space);
  } catch (...) {
    {
      std::lock_guard lk(c.mu);
      c.entries[key].pending = {};
    }
    mine.set_value(nullptr);
    c.pf_skipped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  c.pf_uploads.fetch_add(1, std::memory_order_relaxed);
  c.pf_bytes.fetch_add(col->alloc_size(), std::memory_order_relaxed);
  c.uploads.fetch_add(1, std::memory_order_relaxed);
  c.bytes_uploaded.fetch_add(col->alloc_size(), std::memory_order_relaxed);
  auto const available = c.elastic ? space->get_available_memory() : std::size_t{0};
  {
    std::lock_guard lk(c.mu);
    auto& e   = c.entries[key];
    e.column  = col;
    e.pending = {};
    trace_read(c, key, 'P', col->alloc_size());
    if (!elastic_admit(c, col->alloc_size(), available)) {
      // not retained: dropped unless a scan picks it up while the worker still holds it
    } else if (c.plan_policy) {
      plan_retain(c, key, col);
    } else if (c.nextuse) {
      nextuse_retain(c, key, col);
    } else {
      retain(c, key, col);
    }
  }
  mine.set_value(col);
}

void prefetch_worker(cache& c)
{
  for (;;) {
    std::unique_ptr<::cucascade::host_data_representation> next;
    {
      std::unique_lock lk(c.pf_mu);
      c.pf_cv.wait(lk, [&] { return !c.pf_queue.empty(); });
      next = std::move(c.pf_queue.front());
      c.pf_queue.pop_front();
    }
    for (std::size_t i = 0; i < next->num_columns(); ++i) {
      prefetch_one(c, *next, i);
    }
  }
}

}  // namespace

bool shared_uploads_enabled() noexcept
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_SHARE_UPLOADS");
    return v != nullptr && *v == '1';
  }();
  return on;
}

bool cooperative_start_enabled() noexcept
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_SHARE_NOROTATE");
    return shared_uploads_enabled() && !(v != nullptr && *v == '1');
  }();
  return on;
}

std::unique_ptr<::cucascade::gpu_table_representation> shared_upload(
  ::cucascade::host_data_representation& host, ::cucascade::memory::memory_space& space)
{
  auto& c = the_cache();
  if (c.prefetch && c.space == nullptr) {
    std::lock_guard lk(c.mu);
    if (c.space == nullptr) { c.space = &space; }
  }
  std::vector<std::shared_ptr<cudf::column>> columns;
  std::vector<cudf::column_view> views;
  std::size_t alloc_size = 0;
  for (std::size_t i = 0; i < host.num_columns(); ++i) {
    columns.push_back(acquire(c, host, i, space));
    views.push_back(columns.back()->view());
    alloc_size += columns.back()->alloc_size();
  }
  cudf::table_view view(views);
  return std::make_unique<::cucascade::gpu_table_representation>(
    view, std::move(columns), alloc_size, space, ::cuda::stream_ref{cudaStream_t{}});
}

bool plan_retention_enabled() noexcept
{
  auto const& c = the_cache();
  return c.plan_policy || c.nextuse;  // both need the column names of host chunks
}

bool nextuse_enabled() noexcept { return the_cache().nextuse; }

bool prefetch_enabled() noexcept { return the_cache().prefetch; }

void prefetch_host_chunk(std::unique_ptr<::cucascade::host_data_representation> next_chunk)
{
  auto& c = the_cache();
  if (!c.prefetch || !next_chunk) { return; }
  std::call_once(c.pf_start, [&c] { std::thread([&c] { prefetch_worker(c); }).detach(); });
  {
    std::lock_guard lk(c.pf_mu);
    if (c.pf_queue.size() >= 32) {  // stale look-ahead: the scans have moved on
      c.pf_queue.pop_front();
      c.pf_dropped.fetch_add(1, std::memory_order_relaxed);
    }
    c.pf_queue.push_back(std::move(next_chunk));
  }
  c.pf_cv.notify_one();
}

std::uint64_t register_scan(const void* entry,
                            std::vector<std::string> const& column_names,
                            std::vector<std::size_t> const& survivor_chunks,
                            const std::atomic<std::size_t>* cursor,
                            const std::size_t* rotate)
{
  auto& c = the_cache();
  if (!c.nextuse) { return 0; }
  cache::active_scan sc{entry, {column_names.begin(), column_names.end()}, {}, survivor_chunks.size(), cursor, rotate};
  for (std::size_t i = 0; i < survivor_chunks.size(); ++i) { sc.position.emplace(survivor_chunks[i], i); }
  std::lock_guard lk(c.mu);
  auto const id = ++c.scan_seq;
  c.scans.emplace(id, std::move(sc));
  return id;
}

void unregister_scan(std::uint64_t id)
{
  if (id == 0) { return; }
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  c.scans.erase(id);
}

void note_host_column(const void* allocation,
                      const ::cucascade::memory::column_metadata& column,
                      const std::string& name)
{
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  c.name_of[column_key{allocation, layout_of(column)}] = name;
}

void note_host_chunk(const void* allocation, const void* entry, std::size_t chunk_index)
{
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  c.chunk_of[allocation] = chunk_place{entry, chunk_index};
}

bool resident_first_enabled() noexcept
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_SHARE_RESIDENT_FIRST");
    return shared_uploads_enabled() && v != nullptr && *v == '1';
  }();
  return on;
}

std::vector<std::size_t> retained_bytes_by_chunk(const void* entry, std::size_t chunk_count)
{
  auto& c = the_cache();
  std::vector<std::size_t> out(chunk_count, 0);
  std::lock_guard lk(c.mu);
  for (auto it = c.retained_by_chunk.lower_bound({entry, 0});
       it != c.retained_by_chunk.end() && it->first.first == entry;
       ++it) {
    if (it->first.second < chunk_count) { out[it->first.second] = it->second; }
  }
  return out;
}

bool elastic_enabled() noexcept
{
  static bool const on = shared_uploads_enabled() && the_cache().elastic;
  return on;
}

std::size_t retained_bytes() noexcept { return the_cache().retained_atomic.load(std::memory_order_relaxed); }

std::size_t reclaim_retained(std::size_t bytes)
{
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  if (c.margin != 0) { c.margin = std::min(c.margin * 2, std::size_t{16} << 30); }
  c.last_pressure = std::chrono::steady_clock::now();
  c.reclaim_events.fetch_add(1, std::memory_order_relaxed);
  auto const freed = reclaim_locked(c, bytes);
  c.reclaimed_bytes.fetch_add(freed, std::memory_order_relaxed);
  return freed;
}

void note_resident_first_pick(bool resident) noexcept
{
  auto& c = the_cache();
  c.rf_picks.fetch_add(1, std::memory_order_relaxed);
  if (resident) { c.rf_resident.fetch_add(1, std::memory_order_relaxed); }
}

std::optional<std::size_t> hot_chunk(const void* entry)
{
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  auto it = c.hot.find(entry);
  if (it == c.hot.end()) { return std::nullopt; }
  return it->second;
}

}  // namespace sirius::memory
