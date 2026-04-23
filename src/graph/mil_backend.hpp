/**
 * MilBackend — CompilerBackend implementation for Path A.
 *
 * Translates each FusionGroup into a MIL text program via MilBuilder,
 * then compiles and loads it onto the ANE via ane_compile().
 *
 * ── Warm-path reconnect cache ────────────────────────────────────────────
 *
 * On first compile, captures model_url and hexID.  On subsequent compiles
 * of the same MIL + weights, uses ane_reconnect() to bind the existing
 * aned compile slot in ~0.7 ms rather than paying full cold-compile cost
 * (~111 ms).  Keyed by hexID (aned's own equivalence class), so two
 * compiles aned would deduplicate hit the same cache entry by construction.
 *
 * try_warm_reconnect() is a non-side-effecting probe: it computes the
 * hexID, checks the cache, attempts the reconnect, and returns nullptr
 * on any miss.  Other backends (notably HwxBackend) can call this before
 * their own warm paths to pick up same-MIL reconnect hits.
 */
#pragma once

#include "compiler_backend.hpp"
#include "../runtime/ane_runtime.hpp"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace libane {
namespace graph {

/** Snapshot of MilBackend cache state at a point in time. */
struct MilBackendCacheStats {
    size_t   entries        = 0;  ///< number of cached (hex_id → URL entry) pairs
    size_t   capacity       = 0;  ///< max entries before LRU eviction kicks in
    size_t   bytes          = 0;  ///< approximate footprint (mil_text + weights + strings)
    uint64_t hits           = 0;  ///< try_warm_reconnect calls that returned a program
    uint64_t misses         = 0;  ///< try_warm_reconnect calls that returned nullptr
    uint64_t cold_compiles  = 0;  ///< cache_populate calls (successful cold compiles)
    uint64_t evictions      = 0;  ///< stale entries dropped after reconnect failure
    uint64_t lru_evictions  = 0;  ///< entries dropped because cache hit capacity
};

/** Default LRU capacity — bounds memory growth regardless of compile cadence.
 *  Well below aned's ~115 slot ceiling so the cache never drives the ceiling
 *  on its own.  Callers with unusual workloads can override via
 *  MilBackend::set_cache_capacity(). */
inline constexpr size_t kDefaultMilBackendCacheCapacity = 128;

class MilBackend final : public CompilerBackend {
public:
    /** Catch-all: owns every group not claimed by a higher-priority backend. */
    bool owns(const AneGraph& graph, const FusionGroup& group) const override;

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override;

    /**
     * Non-side-effecting warm-path probe.
     *
     * Computes the hexID for (mil_text, weights) and looks it up in the
     * reconnect cache.  On hit, attempts ane_reconnect().  Returns nullptr
     * on any miss (cache miss, reconnect failure, or aned slot purged) —
     * the caller is responsible for falling through to its own cold path.
     *
     * Unlike compile_group(), this never compiles.  Safe to call from
     * other backends that want to pick up same-MIL reconnect hits before
     * their own warm paths kick in.
     */
    runtime::AneProgram* try_warm_reconnect(const std::string& mil_text,
                                            const std::vector<runtime::WeightEntry>& weights,
                                            const std::string& debug_name);

    /** Snapshot current cache state — safe to call from any thread that owns
     *  this MilBackend instance. */
    MilBackendCacheStats cache_stats() const;

    /** Drop every cached entry. Subsequent compiles pay full cold cost until
     *  the cache repopulates. */
    void cache_clear();

    /** Change the LRU capacity. If new capacity is smaller than current
     *  entries count, the oldest entries are evicted immediately. */
    void set_cache_capacity(size_t capacity);

    /** Trim the cache to at most max_entries via LRU eviction.  If
     *  max_entries is 0, resolves to the current configured capacity
     *  (acts as an assertion that the cap is being honoured).
     *  Returns the number of entries evicted. */
    size_t cache_prune(size_t max_entries = 0);

private:
    struct UrlCacheEntry {
        std::string hex_id;    ///< duplicated from map key for O(1) list→map reverse lookup on eviction
        std::string model_url;
        std::string mil_text;
        // Shared with the AneProgram that populated this entry (and any
        // other program for the same (mil_text, weights) pair).  One copy
        // of the weight bytes serves any number of live references; freed
        // only when the last reference drops.
        std::shared_ptr<const std::vector<runtime::WeightEntry>> weights;
    };

    // LRU: front = most-recently-used, back = candidate for eviction.
    // lru_ holds the entries by value; url_cache_ maps hex_id → list iterator
    // for O(1) lookup + O(1) touch (splice to front).
    using LruList     = std::list<UrlCacheEntry>;
    using LruIterator = LruList::iterator;
    LruList                                          lru_;
    std::unordered_map<std::string, LruIterator>     url_cache_;
    size_t                                           cache_capacity_ = kDefaultMilBackendCacheCapacity;

    // Observability counters.  Use atomic to stay trivially correct if a
    // debug inspector reads stats while another thread increments; update
    // paths themselves are single-threaded per instance so no .load/.store
    // contention matters.
    std::atomic<uint64_t> hits_         {0};
    std::atomic<uint64_t> misses_       {0};
    std::atomic<uint64_t> cold_compiles_{0};
    std::atomic<uint64_t> evictions_    {0};
    std::atomic<uint64_t> lru_evictions_{0};

    /** Populate the cache from a fresh cold-compile result. */
    void cache_populate(const runtime::AneProgram* prog);

    /** Drop the LRU-tail entry.  Caller must ensure lru_ is non-empty. */
    void evict_lru_tail();

    static size_t entry_bytes(const UrlCacheEntry& e);
};

} // namespace graph
} // namespace libane
