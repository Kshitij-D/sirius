#include "memory/resident_ledger.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <cuda/memory_resource>

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>

namespace sirius::memory {
namespace {

// Pass-through resource that adds live bytes to a counter. Columns keep a reference to it, so it
// is never destroyed (one per memory space, process lifetime).
class counting_resource {
 public:
  counting_resource(rmm::device_async_resource_ref upstream, std::atomic<std::size_t>* counter)
    : _upstream(upstream), _counter(counter)
  {
  }

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    void* p = _upstream.allocate(stream, bytes, alignment);
    _counter->fetch_add(bytes, std::memory_order_relaxed);
    return p;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    _upstream.deallocate(stream, p, bytes, alignment);
    _counter->fetch_sub(bytes, std::memory_order_relaxed);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    void* p = _upstream.allocate_sync(bytes, alignment);
    _counter->fetch_add(bytes, std::memory_order_relaxed);
    return p;
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    _upstream.deallocate_sync(p, bytes, alignment);
    _counter->fetch_sub(bytes, std::memory_order_relaxed);
  }

  bool operator==(counting_resource const& other) const noexcept { return this == &other; }

  friend void get_property(counting_resource const&, ::cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref _upstream;
  std::atomic<std::size_t>* _counter;
};

struct entry {
  std::atomic<std::size_t> bytes{0};
  std::unique_ptr<counting_resource> mr;
};

std::mutex& mu()
{
  static std::mutex m;
  return m;
}

std::map<const cucascade::memory::memory_space*, std::unique_ptr<entry>>& entries()
{
  static auto* e = new std::map<const cucascade::memory::memory_space*, std::unique_ptr<entry>>();
  return *e;
}

entry& entry_for(const cucascade::memory::memory_space* space)
{
  std::lock_guard lk(mu());
  auto& e = entries()[space];
  if (!e) {
    e     = std::make_unique<entry>();
    e->mr = std::make_unique<counting_resource>(space->get_default_allocator(), &e->bytes);
  }
  return *e;
}

}  // namespace

namespace {
// Tri-state env switch: "1" on, "0" off, unset -> dflt.
bool env_switch(char const* name, bool dflt) noexcept
{
  auto const* v = std::getenv(name);
  if (v == nullptr || *v == '\0') { return dflt; }
  return *v == '1';
}
}  // namespace

bool cap_resident_enabled() noexcept
{
  static bool const on = env_switch("SIRIUS_CAP_RESIDENT", ledger_enabled());
  return on;
}

bool evictable_trigger_enabled() noexcept
{
  static bool const on = env_switch("SIRIUS_EVICTABLE_TRIGGER", ledger_enabled());
  return on;
}

bool resident_tracking_enabled() noexcept
{
  return ledger_enabled() || cap_resident_enabled() || evictable_trigger_enabled();
}

bool ledger_enabled() noexcept
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_LEDGER");
    return v != nullptr && *v == '1';
  }();
  return on;
}

rmm::device_async_resource_ref resident_allocator(const cucascade::memory::memory_space* space)
{
  if (!resident_tracking_enabled()) { return space->get_default_allocator(); }
  return rmm::device_async_resource_ref{*entry_for(space).mr};
}

std::size_t resident_bytes(const cucascade::memory::memory_space* space) noexcept
{
  if (!resident_tracking_enabled() || space == nullptr) { return 0; }
  std::lock_guard lk(mu());
  auto const it = entries().find(space);
  return it == entries().end() ? 0 : it->second->bytes.load(std::memory_order_relaxed);
}

std::shared_ptr<cudf::column> make_resident(std::unique_ptr<cudf::column> col,
                                            const cucascade::memory::memory_space* space)
{
  if (!resident_tracking_enabled() || space == nullptr || !col) { return std::shared_ptr<cudf::column>(std::move(col)); }
  auto const bytes = col->alloc_size();
  auto* counter    = &entry_for(space).bytes;
  counter->fetch_add(bytes, std::memory_order_relaxed);
  return std::shared_ptr<cudf::column>(col.release(), [counter, bytes](cudf::column* c) {
    delete c;
    counter->fetch_sub(bytes, std::memory_order_relaxed);
  });
}

void add_resident(const cucascade::memory::memory_space* space, std::size_t bytes) noexcept
{
  if (!resident_tracking_enabled() || space == nullptr || bytes == 0) { return; }
  entry_for(space).bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void sub_resident(const cucascade::memory::memory_space* space, std::size_t bytes) noexcept
{
  if (!resident_tracking_enabled() || space == nullptr || bytes == 0) { return; }
  entry_for(space).bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

const std::atomic<std::size_t>* resident_counter(const cucascade::memory::memory_space* space)
{
  if (!resident_tracking_enabled() || space == nullptr) { return nullptr; }
  return &entry_for(space).bytes;
}

}  // namespace sirius::memory
