/**
 * HwxBackend — Path C compiler backend.
 *
 * For weight-free single-node activation groups (RELU, TANH, SIGMOID,
 * HARDSWISH, LEAKY_RELU, ELU), HwxBackend emits BEEFFACE HWX binaries
 * and loads them directly via ane_load_hwx().
 *
 * ── Compile-path ordering (ascending cost) ───────────────────────────────
 *
 * Measured on M3 Pro / macOS 26.3.1 via tests/test_warmpath_tiers_bench.cpp
 * (C=64, S=512 for activations; IC=OC=64, SP=128 for matmul).  Times are
 * end-to-end compile_group() latency, not just the ANE-facing call.
 *
 * 1. MilBackend::try_warm_reconnect               ~1.3–1.7 ms
 *    Same op, aned slot alive.  Hits the hexID-keyed URL cache populated
 *    on any prior cold or cross-op compile.
 *
 * 2. HwxEmitter cross-op patch + ane_load_hwx     ~25–40 ms
 *    Different op than any cached, but same shape template exists.
 *    Patches the op-config words in the cached HWX and pre-stages it
 *    at localModelPath for compileWithQoS:'s compileAsNeeded path.
 *
 * 3. MilBackend::compile_group full cold compile  ~15–70 ms
 *    First time for this (mil_text, weights) pair.  Activation ops at
 *    mid-scale ~65 ms; matmul ~15 ms; actual numbers vary with op
 *    complexity and aned's internal disk cache state.
 *
 * Speedups on repeated compiles of the same graph node:
 *   RELU    : 66 ms → 1.7 ms cold → reconnect  (~38×)
 *   TANH    : 29 ms → 1.3 ms cross-op → reconnect  (~22×)
 *   MATMUL  : 17 ms → 1.6 ms cold → reconnect  (~10×)
 *
 * After each successful cold compile, the HWX template + op-config words
 * are captured into HwxEmitter so subsequent cross-op hits at the same
 * shape can skip ANECompilerService entirely.
 *
 * URL reconnect state lives on MilBackend (Phase 1).  HwxBackend does not
 * maintain its own URL cache — by delegating the same-op warm path to
 * MilBackend::try_warm_reconnect, matmul/rmsnorm/softmax and the
 * HwxEmitter-owned activation ops share one cache, keyed by aned's own
 * hexID equivalence class.
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

#include <memory>

namespace libane {
namespace graph {

class HwxBackend final : public CompilerBackend {
public:
    /**
     * Default constructor: HwxBackend owns its own MilBackend.  Suitable
     * for tests and standalone use.  Warm-path URLs cached internally
     * persist across compile_group calls on this instance.
     */
    HwxBackend();

    /**
     * Injecting constructor: HwxBackend borrows the given MilBackend.
     * Used by GraphCompiler so HwxBackend and the router's MilBackend
     * share one URL-reconnect cache — cold compiles from either path
     * populate the same table.
     */
    explicit HwxBackend(MilBackend* shared_mil);

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
    HwxEmitter  emitter_;
    std::unique_ptr<MilBackend> owned_mil_;  ///< non-null iff default-constructed
    MilBackend* mil_;                         ///< always valid; borrowed or owned

    static bool is_hwx_eligible(libane_op_t op);
};

} // namespace graph
} // namespace libane
