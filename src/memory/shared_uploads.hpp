/*
 * Experimental shared uploads of host-tier pinned data (env-gated, SIRIUS_SHARE_UPLOADS=1).
 *
 * Today every split of every query uploads its own copy of a host-tier chunk. With sharing on, a
 * scan uploads each needed column of a chunk once: the device copy is keyed by the host allocation
 * and the column's buffer offsets, and any other scan that needs the same column while a holder is
 * still alive reads the same device column through a zero-copy view (a concurrent upload of the same
 * column is waited for, not repeated). Device copies are charged to the resident ledger
 * (make_resident) until the last holder drops them. SIRIUS_SHARE_UPLOAD_BYTES (default 0) also keeps
 * up to that many bytes of recently used columns alive after their readers finish (LRU).
 *
 * Cooperative start: concurrent scans of a table only share uploads if they read the same chunk at
 * about the same time. A host-tier scan therefore starts at the chunk that other scans of the same
 * pinned entry uploaded most recently and wraps around (scan order carries no meaning for a scan).
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cucascade {
class gpu_table_representation;
class host_data_representation;
namespace memory {
class memory_space;
struct column_metadata;
}
}  // namespace cucascade

namespace sirius::memory {

bool shared_uploads_enabled() noexcept;

/// Cooperative start is on with shared uploads unless SIRIUS_SHARE_NOROTATE=1 (control arm).
bool cooperative_start_enabled() noexcept;

/// Device copy of every column of `host` in `space`, sharing columns other scans already uploaded.
/// Returns a view-backed representation whose owner keeps the shared columns alive.
std::unique_ptr<::cucascade::gpu_table_representation> shared_upload(
  ::cucascade::host_data_representation& host, ::cucascade::memory::memory_space& space);

/// Plan-driven retention (SIRIUS_SHARE_UPLOAD_POLICY=plan, reader counts per column name from the
/// batch planner in SIRIUS_SHARE_UPLOAD_READERS): copies are kept for their remaining expected reads
/// and the one with the fewest is evicted first. Column names identify columns across tables in
/// TPC-DS (prefixed names); a general engine would key by table and column.
bool plan_retention_enabled() noexcept;
void note_host_column(const void* allocation,
                      const ::cucascade::memory::column_metadata& column,
                      const std::string& name);

/// Remember that host allocation `allocation` is chunk `chunk_index` of pinned entry `entry`.
void note_host_chunk(const void* allocation, const void* entry, std::size_t chunk_index);

/// The chunk of `entry` most recently uploaded by any scan, if one was.
std::optional<std::size_t> hot_chunk(const void* entry);

/// Resident-first chunk order (SIRIUS_SHARE_RESIDENT_FIRST=1, needs shared uploads): a host-tier scan takes next the
/// chunk with the most retained device bytes among the chunks it has not taken, so concurrent scans read what is
/// already on the device before uploading more. Chunk order carries no meaning for a scan (as for cooperative start).
bool resident_first_enabled() noexcept;

/// Retained device bytes per chunk index of `entry` (size `chunk_count`), for resident-first selection.
std::vector<std::size_t> retained_bytes_by_chunk(const void* entry, std::size_t chunk_count);

/// Count one resident-first pick (`resident` = the chosen chunk had retained bytes).
void note_resident_first_pick(bool resident) noexcept;

}  // namespace sirius::memory
