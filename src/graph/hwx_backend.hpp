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
#include "mil_backend.hpp"

namespace libane {
namespace graph {

class HwxBackend final : public CompilerBackend {
public:
    HwxBackend() = default;

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override;

private:
    HwxEmitter emitter_;
    MilBackend mil_;

    static bool is_hwx_eligible(libane_op_t op);
};

} // namespace graph
} // namespace libane
