/**
 * Graph compiler — turns a validated AneGraph into a CompiledGraph ready for
 * repeated execution.
 *
 * Two-step design:
 *
 *  1. build_plan()  — validate + fuse + pre-compute tensor sizes.
 *                     Pure C++, no ANE I/O.  Always succeeds for valid graphs.
 *                     Throws std::runtime_error on validation failure.
 *                     Useful for unit-testing the planner without hardware.
 *
 *  2. compile()     — calls build_plan(), then compiles each fusion group to an
 *                     ANE program via ane_compile(), and pre-allocates one
 *                     IOSurface-backed AneBuffer per group output tensor.
 *                     Returns nullptr on any failure (ANE unavailable, compile
 *                     error, OOM).
 *
 * CompiledGraph owns:
 *  - All runtime::AneProgram* handles (released in destructor via ane_unload).
 *  - One AneBuffer per group output tensor.  These are kept alive for the
 *    lifetime of the CompiledGraph and reused on every execute() call.
 *    Intermediate tensors (flowing between groups) stay in hardware SRAM
 *    between the two dispatch calls, avoiding DRAM round-trips.
 *
 * Caller is responsible for ensuring the graph outlives calls to compile().
 * The CompiledGraph copies everything it needs; no back-pointer to the graph.
 */
#pragma once

#include "ane_graph.hpp"
#include "fusion_rules.hpp"
#include "../core/mil_builder.hpp"
#include "../core/buffer_manager.hpp"
#include "../runtime/ane_runtime.hpp"
#include "../../include/libane.h"

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace libane {
namespace graph {

/* ── ExecutionPlan ───────────────────────────────────────────────────────── */

/**
 * Pure plan: fusion groups + pre-computed tensor sizes.
 * Produced by GraphCompiler::build_plan(); does not hold ANE resources.
 */
struct ExecutionPlan {
    /** Fusion groups in topological execution order. */
    std::vector<FusionGroup> groups;

    /** Byte size of every tensor in the graph (= shape.bytes()). */
    std::unordered_map<TensorId, size_t> tensor_bytes;

    /**
     * Tensor IDs that need pre-allocated ANE buffers:
     * = the output tensor of every fusion group.
     * Includes both intermediates (flowing to the next group) and final
     * graph outputs (written last, then copied to the caller's buffer).
     */
    std::vector<TensorId> buffer_ids;

    /** Graph input tensor IDs, in the order they were declared. */
    std::vector<TensorId> graph_input_ids;

    /** Graph output tensor IDs, in the order they were marked. */
    std::vector<TensorId> graph_output_ids;
};

/* ── CompiledPlanGroup ───────────────────────────────────────────────────── */

/**
 * One compiled fusion group.
 * program is owned by the enclosing CompiledGraph.
 */
struct CompiledPlanGroup {
    std::vector<uint32_t>  node_ids;
    runtime::AneProgram*   program = nullptr;   // owned by CompiledGraph
    std::vector<TensorId>  inputs;              // external input tensor IDs
    TensorId               output = kInvalidTensorId;
};

/* ── CompiledGraph ───────────────────────────────────────────────────────── */

class CompiledGraph {
public:
    CompiledGraph() = default;
    ~CompiledGraph();

    CompiledGraph(const CompiledGraph&)            = delete;
    CompiledGraph& operator=(const CompiledGraph&) = delete;
    CompiledGraph(CompiledGraph&&)                 = default;
    CompiledGraph& operator=(CompiledGraph&&)      = default;

    /* ── Accessors for the executor ──────────────────────────────────────── */

    const std::vector<CompiledPlanGroup>&       groups()          const { return groups_; }
    const std::unordered_map<TensorId, size_t>& tensor_bytes()    const { return tensor_bytes_; }
    const std::vector<TensorId>&                graph_input_ids() const { return graph_input_ids_; }
    const std::vector<TensorId>&                graph_output_ids()const { return graph_output_ids_; }

    /** Return pre-allocated ANE buffer for tensor id, or nullptr if not found. */
    AneBuffer* ane_buf(TensorId id) const;

    /* ── Accessors for tests ─────────────────────────────────────────────── */

    size_t group_count()              const { return groups_.size(); }
    size_t group_node_count(size_t i) const { return groups_[i].node_ids.size(); }

private:
    friend class GraphCompiler;

    std::vector<CompiledPlanGroup>                           groups_;
    std::unordered_map<TensorId, size_t>                     tensor_bytes_;
    std::unordered_map<TensorId, std::unique_ptr<AneBuffer>> ane_bufs_;
    std::vector<TensorId>                                    graph_input_ids_;
    std::vector<TensorId>                                    graph_output_ids_;
};

/* ── GraphCompiler ───────────────────────────────────────────────────────── */

class GraphCompiler {
public:
    /**
     * Validate + fuse + compute sizes.
     * No ANE hardware access.  Always succeeds for valid graphs.
     * Throws std::runtime_error if validation fails.
     */
    static ExecutionPlan build_plan(const AneGraph& graph);

    /**
     * Full compile: validate, fuse, compile all groups to ANE programs, and
     * pre-allocate ANE buffers for all group outputs.
     *
     * Returns nullptr on failure (validation error, ANE unavailable, or any
     * ane_compile() call fails).  Never throws.
     */
    static std::unique_ptr<CompiledGraph> compile(const AneGraph& graph);
};

} // namespace graph
} // namespace libane
