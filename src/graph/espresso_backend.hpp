/**
 * EspressoBackend — Path B compiler backend.
 *
 * For single-node MATMUL groups, EspressoBackend builds a .mlmodelc bundle
 * via EspressoBuilder and loads it directly via ane_load_mlmodelc(), using
 * _ANEClient for dispatch instead of _ANEInMemoryModel.
 *
 * ── Ownership ────────────────────────────────────────────────────────────
 *
 * EspressoBackend::owns() returns true for:
 *   - Single-node LIBANE_OP_MATMUL groups with seq == 1.
 *
 * All other groups are delegated to MilBackend (catch-all).
 * Single-node weight-free activations are claimed before this backend by
 * HwxBackend in the priority router.
 *
 * ── Performance ──────────────────────────────────────────────────────────
 *
 * dispatch via _ANEClient.doEvaluateDirectWithModel: avoids the
 * _ANEInMemoryModel overhead and is preferred for weight-bearing ops where
 * the HWX warm-cache trick used by HwxBackend is not applicable.
 */
#pragma once

#include "compiler_backend.hpp"
#include <string>

namespace libane {
namespace graph {

class EspressoBackend final : public CompilerBackend {
public:
    EspressoBackend() = default;

    /**
     * Owns single-node MATMUL groups with seq == 1.
     * Returns false if Path B symbols are unavailable (falls through to MilBackend).
     */
    bool owns(const AneGraph& graph, const FusionGroup& group) const override;

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override;
};

} // namespace graph
} // namespace libane
