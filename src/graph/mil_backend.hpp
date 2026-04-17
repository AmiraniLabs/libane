/**
 * MilBackend — CompilerBackend implementation for Path A.
 *
 * Translates each FusionGroup into a MIL text program via MilBuilder,
 * then compiles and loads it onto the ANE via ane_compile().
 */
#pragma once

#include "compiler_backend.hpp"

namespace libane {
namespace graph {

class MilBackend final : public CompilerBackend {
public:
    /** Catch-all: owns every group not claimed by a higher-priority backend. */
    bool owns(const AneGraph& graph, const FusionGroup& group) const override;

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override;
};

} // namespace graph
} // namespace libane
