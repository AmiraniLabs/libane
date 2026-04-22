/**
 * HwxBackend — Path C compiler backend.
 *
 * For weight-free single-node activation groups (RELU, TANH, SIGMOID,
 * HARDSWISH, LEAKY_RELU, ELU), HwxBackend emits BEEFFACE HWX binaries
 * and loads them directly via ane_load_hwx(), bypassing MIL→aned
 * compilation after the first warm-up call per shape.
 *
 * ── Performance tiers ────────────────────────────────────────────────────
 *
 * COLD  (first compile for a given (C, S) shape)
 *   Falls back to MilBackend (Path A).  Full ane_compile() cost applies:
 *   ~4200 ms on M-series.  Captures the resulting HWX and caches it.
 *
 * WARM  (same shape, any known op)
 *   HwxEmitter::emit() cross-patches the 5 op-specific words in the cached
 *   HWX template in microseconds — no recompile, no XPC to ANECompilerService.
 *   ane_load_hwx() stubs in the patched binary and calls loadWithQoS: through
 *   aned (~20–40 ms).  The compile cost is zero.  This is the production ceiling
 *   under standard system configuration.
 *
 * ── UNet / transformer use case ──────────────────────────────────────────
 *
 * A UNet or transformer has a fixed set of shapes that repeat across layers.
 * First forward pass pays the compile cost once per unique (C, S) pair.
 * Every subsequent call — regardless of which activation op — pays only the
 * loader cost (~20–40 ms), not the compiler cost (~4200 ms).
 *
 * The cache is append-only within a process lifetime: warm-cache calls can
 * never regress to cold-cache cost for a previously seen shape.
 *
 * ── Delegation ───────────────────────────────────────────────────────────
 *
 * Weight-bearing ops, multi-input ops, and multi-node fusion groups always
 * delegate to MilBackend.  HwxBackend is strictly additive — it never
 * changes the semantics of any compiled graph.
 */
#pragma once

#include "compiler_backend.hpp"
#include "hwx_emitter.hpp"

#include <string>
#include <unordered_map>

namespace libane {
namespace graph {

class HwxBackend final : public CompilerBackend {
public:
    HwxBackend() = default;

    /**
     * Owns single-node, weight-free activation groups.
     * Multi-node groups, weight-bearing ops, and unrecognised ops are
     * routed to the next backend in the priority list.
     */
    bool owns(const AneGraph& graph, const FusionGroup& group) const override;

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override;

private:
    HwxEmitter emitter_;

    static bool is_hwx_eligible(libane_op_t op);

    // ── Path C URL reconnect cache (macOS 26+) ────────────────────────────
    // Keyed by (channels, seq, op).  Populated on first cold compile;
    // used by ane_reconnect() for subsequent calls at the same shape+op
    // without consuming a new aned compile slot (~0.722ms warm path).
    struct ShapeOpKey {
        int channels, seq, op;
        bool operator==(const ShapeOpKey& o) const noexcept {
            return channels == o.channels && seq == o.seq && op == o.op;
        }
    };
    struct ShapeOpKeyHash {
        size_t operator()(const ShapeOpKey& k) const noexcept {
            size_t h = std::hash<int>{}(k.channels);
            h ^= std::hash<int>{}(k.seq) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.op)  + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct UrlCacheEntry {
        std::string model_url;
        std::string mil_text;
    };
    std::unordered_map<ShapeOpKey, UrlCacheEntry, ShapeOpKeyHash> url_cache_;
};

} // namespace graph
} // namespace libane
