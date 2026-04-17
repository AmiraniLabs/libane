/**
 * Compile cache — thread-safe LRU cache for compiled ANE programs.
 *
 * ANE compilation without cache: ~4,200 ms
 * With delta compilation (Orion technique): ~494 ms
 * With warm cache hit: <1 ms
 *
 * Key: (op, shape tuple, weight hash)
 * Value: compiled program handle + IOSurface pool
 * Eviction: LRU with configurable size limit (default 512 MB)
 * Persistence: optional disk cache for cross-session reuse
 * Thread safety: read-write lock, concurrent readers, exclusive writers
 */
#pragma once

#include "../../include/libane.h"
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string>
#include <functional>
#include <memory>
#include <list>
#include <unordered_map>
#include <shared_mutex>
#include <optional>
#include <chrono>

namespace libane {

/* ── Cache key ───────────────────────────────────────────────────────────── */

struct CacheKey {
    libane_op_t     op;
    libane_shape_t  shape;
    uint64_t        weight_hash;

    bool operator==(const CacheKey& o) const noexcept;
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const noexcept;
};

/* ── Compiled program entry ──────────────────────────────────────────────── */

/**
 * A compiled program entry stored in the cache.
 * Lifetime is managed by the cache; callers hold shared_ptr references.
 */
struct CacheEntry {
    CacheKey  key;

    /**
     * Backend-specific compiled program handle.
     * For ANE: void* pointing to ObjC object (_ANECompiledModel).
     * For CPU: nullptr (fallback needs no compiled artifact).
     */
    void* backend_handle = nullptr;

    /** Destructor registered by the backend to release backend_handle. */
    std::function<void(void*)> release_fn;

    /** Byte size of the compiled model (for eviction accounting). */
    size_t size_bytes = 0;

    /** True if this entry was compiled for ANE; false = CPU fallback. */
    bool is_ane = false;

    /** Insertion timestamp (for LRU tie-breaking). */
    std::chrono::steady_clock::time_point inserted_at;

    ~CacheEntry() {
        if (backend_handle && release_fn) {
            release_fn(backend_handle);
        }
    }

    /* Non-copyable, movable. */
    CacheEntry() = default;
    CacheEntry(const CacheEntry&) = delete;
    CacheEntry& operator=(const CacheEntry&) = delete;
    CacheEntry(CacheEntry&&) = default;
    CacheEntry& operator=(CacheEntry&&) = default;
};

/* ── Cache ───────────────────────────────────────────────────────────────── */

class CompileCache {
public:
    static constexpr size_t kDefaultMaxBytes = 512ULL * 1024 * 1024; // 512 MB

    explicit CompileCache(size_t max_bytes = kDefaultMaxBytes);
    ~CompileCache() = default;

    /** Look up a cached entry. Returns nullptr on miss. */
    std::shared_ptr<CacheEntry> get(const CacheKey& key);

    /**
     * Insert a compiled entry.
     * Evicts LRU entries if the cache would exceed max_bytes.
     * Returns the stored shared_ptr (callers should keep this alive).
     */
    std::shared_ptr<CacheEntry> put(std::unique_ptr<CacheEntry> entry);

    /** Remove all entries and release backend handles. */
    void flush();

    /** Current total size in bytes. */
    size_t size_bytes() const;

    /** Number of entries in the cache. */
    size_t size() const;

    /** Hit/miss statistics. */
    struct Stats {
        uint64_t hits   = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
    };
    Stats stats() const;

    /**
     * Enable a disk-backed persistence layer at the given path.
     * Compiled models are written as files named by their cache key hash.
     * On cache miss, the disk is checked before invoking the compiler.
     */
    void enable_disk_cache(const std::string& directory);
    void disable_disk_cache();

private:
    using LruList = std::list<std::shared_ptr<CacheEntry>>;
    using LruIter = LruList::iterator;

    struct MapValue {
        LruIter  lru_pos;
        std::shared_ptr<CacheEntry> entry;
    };

    void evict_lru_locked();  // must hold write lock

    size_t             max_bytes_;
    size_t             current_bytes_ = 0;
    LruList            lru_;
    std::unordered_map<CacheKey, MapValue, CacheKeyHash> map_;

    mutable std::shared_mutex mutex_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};

    std::optional<std::string> disk_cache_dir_;
};

} // namespace libane

/* ── Program handle (public libane_program_s) ────────────────────────────── */

/**
 * libane_handle_t points to this struct.
 * Callers hold a reference-counted pointer to the cache entry.
 * Defined in the global namespace to match libane.h's forward declaration.
 */
struct libane_program_s {
    std::shared_ptr<libane::CacheEntry> entry;

    /** Compiled matmul dimensions (convenience for execute path). */
    int M = 0, K = 0, N = 0;
    libane_op_t op = LIBANE_OP_MATMUL;
    libane_shape_t shape{};
};
