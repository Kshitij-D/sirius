#include "memory/shared_uploads.hpp"

#include "data/sirius_converter_registry.hpp"
#include "memory/resident_ledger.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>

#include <cudf/table/table.hpp>

#include <atomic>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <list>
#include <mutex>
#include <unordered_map>
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
    });
    return x;
  }();
  return *c;
}

void retain(cache& c, const column_key& key, const std::shared_ptr<cudf::column>& col)
{
  if (c.budget == 0) { return; }
  c.lru.emplace_back(key, col);
  c.retained_bytes += col->alloc_size();
  while (c.retained_bytes > c.budget && !c.lru.empty()) {
    c.retained_bytes -= c.lru.front().second->alloc_size();
    c.lru.pop_front();
  }
}

void plan_drop(cache& c, std::unordered_map<column_key, planned_slot, column_key_hash>::iterator it)
{
  c.plan_index.erase(plan_order{it->second.remaining, it->second.seq, it->first.allocation, it->first.layout});
  c.retained_bytes -= it->second.column->alloc_size();
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
          return col;
        }
        if (it->second.pending.valid()) {
          auto pending = it->second.pending;
          lk.unlock();
          if (auto col = pending.get()) {
            c.waits.fetch_add(1, std::memory_order_relaxed);
            c.hits.fetch_add(1, std::memory_order_relaxed);
            c.bytes_hit.fetch_add(col->alloc_size(), std::memory_order_relaxed);
            if (c.plan_policy) {
              std::lock_guard relk(c.mu);
              plan_read(c, key);
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
  {
    std::lock_guard lk(c.mu);
    if (auto pl = c.chunk_of.find(key.allocation); pl != c.chunk_of.end()) {
      c.hot[pl->second.entry] = pl->second.index;
    }
    auto& e   = c.entries[key];
    e.column  = col;
    e.pending = {};
    if (c.plan_policy) {
      plan_retain(c, key, col);
    } else {
      retain(c, key, col);
    }
  }
  mine.set_value(col);
  return col;
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

bool plan_retention_enabled() noexcept { return the_cache().plan_policy; }

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

std::optional<std::size_t> hot_chunk(const void* entry)
{
  auto& c = the_cache();
  std::lock_guard lk(c.mu);
  auto it = c.hot.find(entry);
  if (it == c.hot.end()) { return std::nullopt; }
  return it->second;
}

}  // namespace sirius::memory
