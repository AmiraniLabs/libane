/**
 * HwxBackend — Path C compiler backend.
 *
 * For weight-free single-node activation groups (RELU, TANH, SIGMOID,
 * HARDSWISH, LEAKY_RELU, ELU), HwxBackend emits BEEFFACE HWX binaries
 * and loads them directly via ane_load_hwx().
 *
 * ── Compile-path ordering (ascending cost) ───────────────────────────────
 *
 * 1. MilBackend::try_warm_reconnect (~0.7 ms) — same op, aned slot alive
 * 2. HwxEmitter cross-op patch + ane_load_hwx (~20–40 ms) — different op,
 *    shape template cached
 * 3. MilBackend::compile_group full cold compile (~100 ms on macOS 26) —
 *    first time for this shape and op
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
