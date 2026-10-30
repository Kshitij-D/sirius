#include "memory/shared_uploads.hpp"

#include "data/sirius_converter_registry.hpp"
#include "memory/resident_ledger.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>

#include <cudf/table/table.hpp>

#include <atomic>
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

struct cache {
  std::mutex mu;
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
    std::atexit([] {
      auto& y = the_cache();
      std::fprintf(stderr,
                   "[share-upload] summary uploads=%llu hits=%llu waits=%llu bytes_uploaded=%llu "
                   "bytes_hit=%llu budget=%zu\n",
                   static_cast<unsigned long long>(y.uploads.load()),
                   static_cast<unsigned long long>(y.hits.load()),
                   static_cast<unsigned long long>(y.waits.load()),
                   static_cast<unsigned long long>(y.bytes_uploaded.load()),
                   static_cast<unsigned long long>(y.bytes_hit.load()),
                   y.budget);
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
          return col;
        }
        if (it->second.pending.valid()) {
          auto pending = it->second.pending;
          lk.unlock();
          if (auto col = pending.get()) {
            c.waits.fetch_add(1, std::memory_order_relaxed);
            c.hits.fetch_add(1, std::memory_order_relaxed);
            c.bytes_hit.fetch_add(col->alloc_size(), std::memory_order_relaxed);
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
    retain(c, key, col);
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
