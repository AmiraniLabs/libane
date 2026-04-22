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

#include <string>
#include <unordered_map>
#include <vector>

namespace libane {
namespace graph {

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

private:
    struct UrlCacheEntry {
        std::string model_url;
        std::string mil_text;
        std::vector<runtime::WeightEntry> weights;
    };
    std::unordered_map<std::string, UrlCacheEntry> url_cache_;

    /** Populate the cache from a fresh cold-compile result. */
    void cache_populate(const runtime::AneProgram* prog);
};

} // namespace graph
} // namespace libane
