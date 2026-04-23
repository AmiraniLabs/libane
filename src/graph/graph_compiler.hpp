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
#include "compiler_backend.hpp"
#include "fusion_rules.hpp"
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

/* ── QuantParams ─────────────────────────────────────────────────────────── */

/**
 * Per-tensor int8 quantization parameters.
 * Stored for graph inputs that are quantized to int8 by the caller
 * (e.g. the primary input of a MATMUL_W8A8 group).
 * The executor uses these to dequantize int8 → fp16 before dispatching.
 */
struct QuantParams {
    float   scale      = 1.0f;  ///< input_val_fp16 = (int8_val - zero_point) * scale
    int32_t zero_point = 0;
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
    const std::unordered_map<TensorId, mil::TensorShape>& tensor_shapes() const { return tensor_shapes_; }
    size_t                                      io_alloc_bytes() const { return io_alloc_bytes_; }
    const std::vector<TensorId>&                graph_input_ids() const { return graph_input_ids_; }
    const std::vector<TensorId>&                graph_output_ids()const { return graph_output_ids_; }

    /** Return pre-allocated ANE buffer for tensor id, or nullptr if not found. */
    AneBuffer* ane_buf(TensorId id) const;

    /** Return tensor shape for a tensor id, or nullptr if not found. */
    const mil::TensorShape* tensor_shape(TensorId id) const {
        auto it = tensor_shapes_.find(id);
        return it == tensor_shapes_.end() ? nullptr : &it->second;
    }

    /**
     * Return per-input int8 quantization parameters.
     * Non-empty only when the graph contains MATMUL_W8A8 ops whose
     * primary input is a graph input tensor.
     */
    const std::unordered_map<TensorId, QuantParams>& quant_params() const {
        return quant_params_;
    }

    /* ── Serialization ───────────────────────────────────────────────────── */

    /**
     * Save this compiled graph to a file.
     *
     * Writes MIL text, weight blobs, compiled HWX binaries, and graph
     * metadata in a simple binary format (magic "ANEG", version 1).
     * On restore via GraphCompiler::load(), the expensive compileWithQoS:
     * step is skipped — only loadWithQoS: is called (~8.5× faster).
     *
     * @param path  Destination file path.  Created or overwritten.
     * @return      true on success; false if any write or HWX extraction fails.
     */
    bool save(const std::string& path) const;

    /* ── Accessors for tests ─────────────────────────────────────────────── */

    size_t group_count()              const { return groups_.size(); }
    size_t group_node_count(size_t i) const { return groups_[i].node_ids.size(); }

private:
    friend class GraphCompiler;

    std::vector<CompiledPlanGroup>                           groups_;
    std::unordered_map<TensorId, size_t>                     tensor_bytes_;
    std::unordered_map<TensorId, mil::TensorShape>           tensor_shapes_;
    size_t                                                    io_alloc_bytes_ = 0;
    std::unordered_map<TensorId, std::unique_ptr<AneBuffer>> ane_bufs_;
    std::vector<TensorId>                                    graph_input_ids_;
    std::vector<TensorId>                                    graph_output_ids_;
    std::unordered_map<TensorId, QuantParams>                quant_params_;  ///< int8 input quantization
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
     * Two-argument overload: single backend, no routing.  Used by tests that
     * want to target a specific backend directly.
     *
     * One-argument overload: routes each group through the priority router
     * (HwxBackend → EspressoBackend → MilBackend).
     *
     * Returns nullptr on failure (validation error, ANE unavailable, or any
     * compile_group() call fails).  Never throws.
     */
    static std::unique_ptr<CompiledGraph> compile(const AneGraph& graph,
                                                   CompilerBackend& backend);
    static std::unique_ptr<CompiledGraph> compile(const AneGraph& graph);

    /**
     * Access to the thread-local MilBackend instance the default-routed
     * compile() overload uses.  Exposes cache observability and lifecycle
     * control (see MilBackend::cache_stats() / cache_clear()).
     *
     * Returns the MilBackend the *current* thread would use for compilation.
     * Each thread has its own instance; stats from one thread do not reflect
     * activity on another.
     */
    static class MilBackend& thread_mil_backend();

    /**
     * Load a previously saved CompiledGraph from a file.
     *
     * Restores each group via ane_restore_program(), which calls loadWithQoS:
     * with the saved HWX — skipping compileWithQoS: (~8.5× faster than
     * recompiling from the original graph).
     *
     * Falls back to ane_compile() per group if the HWX load fails (e.g.
     * macOS upgrade changed the binary format).
     *
     * @param path  File produced by CompiledGraph::save().
     * @return      Loaded CompiledGraph, or nullptr on format error or ANE failure.
     */
    static std::unique_ptr<CompiledGraph> load(const std::string& path);
};

} // namespace graph
} // namespace libane
