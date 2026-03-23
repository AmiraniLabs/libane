/**
 * AneGraph implementation — graph IR builder.
 */
#include "ane_graph.hpp"

#include <stdexcept>
#include <cstring>

namespace libane {
namespace graph {

/* ── Private helpers ─────────────────────────────────────────────────────── */

TensorId AneGraph::alloc_tensor(const std::string& name,
                                 mil::TensorShape shape,
                                 uint32_t producer) {
    GraphTensor t;
    t.id               = next_tensor_id_++;
    t.name             = name;
    t.shape            = shape;
    t.producer_node_id = producer;
    tensors_.push_back(std::move(t));
    return tensors_.back().id;
}

/* ── add_input ───────────────────────────────────────────────────────────── */

TensorId AneGraph::add_input(const std::string& name, mil::TensorShape shape) {
    // Basic ANE constraint check (batch==1, height==1, seq%8==0)
    shape.validate();
    TensorId id = alloc_tensor(name, shape, kInvalidNodeId);
    graph_inputs_.push_back(id);
    return id;
}

/* ── add_op ──────────────────────────────────────────────────────────────── */

TensorId AneGraph::add_op(libane_op_t op,
                           std::vector<TensorId> inputs,
                           mil::TensorShape output_shape,
                           const void* weights,
                           size_t weights_len) {
    // Validate output shape
    output_shape.validate();

    // Validate all input tensor IDs
    for (TensorId tid : inputs) {
        if (tid >= static_cast<TensorId>(tensors_.size())) {
            throw std::invalid_argument(
                "add_op: input tensor id " + std::to_string(tid) +
                " is out of range (have " + std::to_string(tensors_.size()) + " tensors)");
        }
    }

    uint32_t node_id = next_node_id_++;

    GraphNode n;
    n.id     = node_id;
    n.op     = op;
    n.inputs = std::move(inputs);

    if (weights && weights_len > 0) {
        n.weights.resize(weights_len);
        std::memcpy(n.weights.data(), weights, weights_len);
        n.weight_file = "w" + std::to_string(node_id) + ".bin";
    }

    // Allocate output tensor (name derived from weight_file stem or node id)
    std::string out_name = "t" + std::to_string(next_tensor_id_);
    TensorId out_id = alloc_tensor(out_name, output_shape, node_id);

    n.output = out_id;
    nodes_.push_back(std::move(n));

    return out_id;
}

/* ── mark_output ─────────────────────────────────────────────────────────── */

void AneGraph::mark_output(TensorId tensor_id, const std::string& name) {
    if (tensor_id >= static_cast<TensorId>(tensors_.size())) {
        throw std::invalid_argument(
            "mark_output: tensor id " + std::to_string(tensor_id) +
            " is out of range");
    }
    // Allow a custom display name on the output
    if (!name.empty()) {
        tensors_[tensor_id].name = name;
    }
    graph_outputs_.push_back(tensor_id);
}

/* ── tensor / node accessors ─────────────────────────────────────────────── */

const GraphTensor& AneGraph::tensor(TensorId id) const {
    if (id >= static_cast<TensorId>(tensors_.size())) {
        throw std::out_of_range("AneGraph::tensor: id " + std::to_string(id) +
                                " out of range");
    }
    return tensors_[id];
}

const GraphNode& AneGraph::node(uint32_t id) const {
    if (id >= static_cast<uint32_t>(nodes_.size())) {
        throw std::out_of_range("AneGraph::node: id " + std::to_string(id) +
                                " out of range");
    }
    return nodes_[id];
}

} // namespace graph
} // namespace libane
