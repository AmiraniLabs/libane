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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace libane {
namespace graph {

/** Snapshot of MilBackend cache state at a point in time. */
struct MilBackendCacheStats {
    size_t   entries       = 0;  ///< number of cached (hex_id → URL entry) pairs
    size_t   bytes         = 0;  ///< approximate footprint (mil_text + weights + strings)
    uint64_t hits          = 0;  ///< try_warm_reconnect calls that returned a program
    uint64_t misses        = 0;  ///< try_warm_reconnect calls that returned nullptr
    uint64_t cold_compiles = 0;  ///< cache_populate calls (successful cold compiles)
    uint64_t evictions     = 0;  ///< stale entries dropped after reconnect failure
};

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

private:
    struct UrlCacheEntry {
        std::string model_url;
        std::string mil_text;
        // Shared with the AneProgram that populated this entry (and any
        // other program for the same (mil_text, weights) pair).  One copy
        // of the weight bytes serves any number of live references; freed
        // only when the last reference drops.
        std::shared_ptr<const std::vector<runtime::WeightEntry>> weights;
    };
    std::unordered_map<std::string, UrlCacheEntry> url_cache_;

    // Observability counters.  Use atomic to stay trivially correct if a
    // debug inspector reads stats while another thread increments; update
    // paths themselves are single-threaded per instance so no .load/.store
    // contention matters.
    std::atomic<uint64_t> hits_         {0};
    std::atomic<uint64_t> misses_       {0};
    std::atomic<uint64_t> cold_compiles_{0};
    std::atomic<uint64_t> evictions_    {0};

    /** Populate the cache from a fresh cold-compile result. */
    void cache_populate(const runtime::AneProgram* prog);

    static size_t entry_bytes(const std::string& hex_id, const UrlCacheEntry& e);
};

} // namespace graph
} // namespace libane
