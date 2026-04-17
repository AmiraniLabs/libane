/**
 * CompilerBackend — abstract interface for per-group compilation.
 *
 * GraphCompiler uses a RoutingBackend (priority-ordered list of backends)
 * to compile each FusionGroup.  Each backend declares the groups it owns
 * via owns(); the router calls compile_group() only on the first match.
 *
 * Implementations:
 *   MilBackend      (Path A) — MIL text → ANE via _ANEInMemoryModel
 *   HwxBackend      (Path C) — BEEFFACE HWX cache + swap loader
 *   EspressoBackend (Path B) — raw .mlmodelc → aned → IOSurface dispatch
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
     * Return true if this backend owns the given fusion group.
     *
     * Called by the router in priority order; the first backend that returns
     * true receives the group.  MilBackend returns true unconditionally and
     * must therefore be last in any priority list.
     */
    virtual bool owns(const AneGraph&    graph,
                      const FusionGroup& group) const = 0;

    /**
     * Compile one fusion group into a loaded AneProgram ready for dispatch.
     * Only called when owns() returned true for this backend.
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
