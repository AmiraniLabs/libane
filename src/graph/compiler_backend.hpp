/**
 * CompilerBackend — abstract interface for per-group compilation.
 *
 * GraphCompiler::compile() calls compile_group() once per FusionGroup.
 * Everything above (build_plan, fusion, tensor sizing) and below
 * (buffer allocation, execution) is backend-agnostic.
 *
 * The only current implementation is MilBackend (Path A: MIL text → ANE).
 */
#pragma once

#include "ane_graph.hpp"
#include "fusion_rules.hpp"
#include "../runtime/ane_runtime.hpp"

#include <string>

namespace libane {
namespace graph {

class CompilerBackend {
public:
    /**
     * Compile one fusion group into a loaded AneProgram ready for dispatch.
     *
     * @param graph       The source graph (read-only — shapes, weights, ops).
     * @param group       The fusion group to compile.
     * @param debug_name  Human-readable label used in error messages / logs.
     * @return            Heap-allocated AneProgram on success, nullptr on failure.
     *                    Caller takes ownership (released via ane_unload).
     */
    virtual runtime::AneProgram* compile_group(const AneGraph&    graph,
                                               const FusionGroup& group,
                                               const std::string& debug_name) = 0;

    virtual ~CompilerBackend() = default;
};

} // namespace graph
} // namespace libane
