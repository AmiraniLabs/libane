/**
 * Graph IR — DAG of typed nodes connected by typed tensors.
 *
 * All tensor shapes are known at graph-build time (no dynamic shapes).
 * Each node has exactly one output tensor.
 * The graph is immutable once compile() is called.
 *
 * Usage:
 *   AneGraph g;
 *   TensorId x   = g.add_input("x", {1, 512, 1, 128});
 *   TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, {1, 256, 1, 128}, w, wlen);
 *   g.mark_output(out, "out");
 */
#pragma once

#include "../../include/libane.h"
#include "../core/mil_builder.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace libane {
namespace graph {

using TensorId = uint32_t;

static constexpr TensorId kInvalidTensorId = 0xFFFFFFFFu;
static constexpr uint32_t kInvalidNodeId   = 0xFFFFFFFFu;

/* ── GraphTensor ─────────────────────────────────────────────────────────── */

/**
 * A typed, named edge in the graph.
 *
 * Graph inputs have producer_node_id == kInvalidNodeId.
 */
struct GraphTensor {
    TensorId         id              = kInvalidTensorId;
    std::string      name;
    mil::TensorShape shape;
    uint32_t         producer_node_id = kInvalidNodeId;
};

/* ── GraphNode ───────────────────────────────────────────────────────────── */

/**
 * A single operation in the graph.
 *
 * Weights are stored as raw fp16 bytes (same layout as libane_compile).
 * weight_file is the filename key that will be used in the MIL file() ref;
 * it is generated at node-creation time and is unique within the graph.
 *
 * Layernorm stores gamma in the first half and beta in the second half of
 * the weights vector (each half is C*2 bytes for a [1,C,1,1] tensor).
 * The graph compiler splits them at fusion time.
 */
struct GraphNode {
    uint32_t              id     = kInvalidNodeId;
    libane_op_t           op     = LIBANE_OP_MATMUL;
    std::vector<TensorId> inputs;
    TensorId              output = kInvalidTensorId;
    std::vector<uint8_t>  weights;       // raw fp16; empty for weight-free ops
    std::string           weight_file;   // e.g. "w0.bin"; empty for weight-free ops
};

/* ── AneGraph ────────────────────────────────────────────────────────────── */

/**
 * Mutable graph builder.
 *
 * Tensors and nodes are assigned monotonically-increasing IDs.
 * IDs are stable — they never change after the object is created.
 */
class AneGraph {
public:
    AneGraph() = default;

    /**
     * Declare a graph input tensor.
     * Returns the new tensor's ID.
     */
    TensorId add_input(const std::string& name, mil::TensorShape shape);

    /**
     * Add an op node.
     *
     * @param op           Operation code.
     * @param inputs       Tensor IDs consumed by this op (in op-defined order).
     * @param output_shape Shape of this node's output tensor.
     * @param weights      Raw fp16 weight bytes (nullptr for weight-free ops).
     * @param weights_len  Byte length of weights.
     * @return             TensorId of the new output tensor.
     *
     * Throws std::invalid_argument if any input TensorId is out of range or
     * output_shape fails basic ANE constraints (seq % 8 == 0, batch == 1,
     * height == 1).
     */
    TensorId add_op(libane_op_t op,
                    std::vector<TensorId> inputs,
                    mil::TensorShape output_shape,
                    const void* weights    = nullptr,
                    size_t      weights_len = 0);

    /**
     * Mark a tensor as a graph output.
     * Outputs are returned by libane_graph_execute in the order they are marked.
     * Throws std::invalid_argument if tensor_id is out of range.
     */
    void mark_output(TensorId tensor_id, const std::string& name = "");

    /* ── Read-only accessors ─────────────────────────────────────────────── */

    const std::vector<GraphTensor>& tensors()       const { return tensors_; }
    const std::vector<GraphNode>&   nodes()         const { return nodes_; }
    const std::vector<TensorId>&    graph_inputs()  const { return graph_inputs_; }
    const std::vector<TensorId>&    graph_outputs() const { return graph_outputs_; }

    /** Convenience: look up a tensor by ID (O(1)). */
    const GraphTensor& tensor(TensorId id) const;

    /** Convenience: look up a node by ID (O(1)). */
    const GraphNode& node(uint32_t id) const;

private:
    std::vector<GraphTensor>  tensors_;
    std::vector<GraphNode>    nodes_;
    std::vector<TensorId>     graph_inputs_;
    std::vector<TensorId>     graph_outputs_;

    /** Next tensor ID to assign (zero-based, contiguous). */
    TensorId  next_tensor_id_ = 0;
    /** Next node ID to assign (zero-based, contiguous). */
    uint32_t  next_node_id_   = 0;

    TensorId alloc_tensor(const std::string& name, mil::TensorShape shape,
                           uint32_t producer = kInvalidNodeId);
};

} // namespace graph
} // namespace libane
