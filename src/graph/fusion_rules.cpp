/**
 * FusionRules implementation.
 */
#include "fusion_rules.hpp"

namespace libane {
namespace graph {

/* ── op_supported ────────────────────────────────────────────────────────── */

bool FusionRules::op_supported(libane_op_t op) {
    switch (op) {
    case LIBANE_OP_MATMUL:
    case LIBANE_OP_GELU:
    case LIBANE_OP_SOFTMAX:
    case LIBANE_OP_AVG_POOL:
    case LIBANE_OP_MAX_POOL:
    case LIBANE_OP_ADD:
    case LIBANE_OP_MUL:
    case LIBANE_OP_LOGICAL_AND:
    case LIBANE_OP_LOGICAL_OR:
    case LIBANE_OP_LOGICAL_XOR:
    case LIBANE_OP_REDUCE_PROD:
    case LIBANE_OP_REDUCE_SUM:
    case LIBANE_OP_REDUCE_MEAN:
    case LIBANE_OP_REDUCE_MAX:
    case LIBANE_OP_SUB:
    case LIBANE_OP_REAL_DIV:
    case LIBANE_OP_SQRT:
    case LIBANE_OP_LOG:
    case LIBANE_OP_RSQRT:
    case LIBANE_OP_CONCAT:
    case LIBANE_OP_SLICE_BY_INDEX:
    case LIBANE_OP_SLICE:
    case LIBANE_OP_SCATTER:
    case LIBANE_OP_SCATTER_ND:
    case LIBANE_OP_SCATTER_ALONG_AXIS:
    case LIBANE_OP_GATHER:
    case LIBANE_OP_NEG:
    case LIBANE_OP_MOD:
    case LIBANE_OP_SINH:
    case LIBANE_OP_COSH:
    case LIBANE_OP_TAN:
    case LIBANE_OP_ASIN:
    case LIBANE_OP_ACOS:
    case LIBANE_OP_SILU:
    case LIBANE_OP_RMSNORM:
    case LIBANE_OP_LAYERNORM:
    case LIBANE_OP_LAYER_NORM:
    case LIBANE_OP_TRANSPOSE:
    case LIBANE_OP_RESHAPE:
    case LIBANE_OP_SELECT:
    case LIBANE_OP_RELU:
    case LIBANE_OP_TANH:
    case LIBANE_OP_SIGMOID:
    case LIBANE_OP_HARDSWISH:
    case LIBANE_OP_LEAKY_RELU:
    case LIBANE_OP_ELU:
    case LIBANE_OP_PIXEL_SHUFFLE:
    case LIBANE_OP_PWL_ACTIVATION:
    case LIBANE_OP_CLIP:
    case LIBANE_OP_PAD:
        return true;
    case LIBANE_OP_CONV2D:
    case LIBANE_OP_CAST:
    default:
        return false;
    }
}

/* ── build_consumer_count ────────────────────────────────────────────────── */

std::unordered_map<TensorId, size_t>
FusionRules::build_consumer_count(const AneGraph& graph) {
    std::unordered_map<TensorId, size_t> count;

    // Initialise every tensor to 0
    for (const auto& t : graph.tensors())
        count[t.id] = 0;

    // Each input of each node increments the count of the tensor it reads
    for (const auto& n : graph.nodes())
        for (TensorId inp : n.inputs)
            count[inp]++;

    // Graph outputs are "consumed" by the caller — increment so their
    // producers never get mistakenly fused away as dead code.
    for (TensorId oid : graph.graph_outputs())
        count[oid]++;

    return count;
}

/* ── can_extend ──────────────────────────────────────────────────────────── */

bool FusionRules::can_extend(const AneGraph& graph,
                               const FusionGroup& group,
                               uint32_t candidate_id,
                               const std::unordered_map<TensorId, size_t>& consumer_count) {
    if (group.node_ids.empty()) return false;

    const GraphNode& last      = graph.node(group.node_ids.back());
    const GraphNode& candidate = graph.node(candidate_id);

    // Op must be supported
    if (!op_supported(candidate.op)) return false;

    // Rule 1a — chain: candidate's primary input must be last node's output
    if (candidate.inputs.empty()) return false;
    if (candidate.inputs[0] != last.output) return false;

    // Rule 1b — no branching: last node's output is consumed only by candidate
    auto it = consumer_count.find(last.output);
    if (it == consumer_count.end() || it->second != 1) return false;

    // Rule 2 — weight file uniqueness within group
    if (!candidate.weight_file.empty()) {
        for (uint32_t nid : group.node_ids) {
            if (graph.node(nid).weight_file == candidate.weight_file)
                return false;
        }
    }

    // Rule 3 — binary op: inputs[1] must not be produced by any node in group
    if (candidate.inputs.size() >= 2) {
        TensorId side = candidate.inputs[1];
        // Build set of outputs produced within the group
        for (uint32_t nid : group.node_ids) {
            if (graph.node(nid).output == side)
                return false;  // side input is an intra-group tensor — can't fuse
        }
    }

    return true;
}

/* ── compute_groups ──────────────────────────────────────────────────────── */

std::vector<FusionGroup> FusionRules::compute_groups(const AneGraph& graph) {
    const auto consumer_count = build_consumer_count(graph);

    // Build a set of tensor IDs that are graph inputs (no producer node)
    std::unordered_set<TensorId> graph_input_set;
    for (TensorId id : graph.graph_inputs())
        graph_input_set.insert(id);

    // Build a map: tensor_id -> which fusion group index produces it
    // (used for Rule 3 side-input check within compute_groups)
    std::unordered_map<TensorId, size_t> tensor_group;

    std::vector<FusionGroup> groups;

    for (const auto& node : graph.nodes()) {

        // Try to extend the last group
        bool extended = false;
        if (!groups.empty()) {
            FusionGroup& last_group = groups.back();
            if (can_extend(graph, last_group, node.id, consumer_count)) {
                // Add node to group
                last_group.node_ids.push_back(node.id);
                last_group.output = node.output;

                // Add any side inputs (inputs beyond inputs[0]) that come
                // from outside this group
                for (size_t i = 1; i < node.inputs.size(); ++i) {
                    TensorId side = node.inputs[i];
                    // Only add if not already in the inputs list
                    bool already = false;
                    for (TensorId existing : last_group.inputs)
                        if (existing == side) { already = true; break; }
                    if (!already)
                        last_group.inputs.push_back(side);
                }

                tensor_group[node.output] = groups.size() - 1;
                extended = true;
            }
        }

        if (!extended) {
            // Start a new group with this node as its sole member
            FusionGroup g;
            g.node_ids.push_back(node.id);
            g.output = node.output;

            // Chain input = inputs[0]
            if (!node.inputs.empty())
                g.inputs.push_back(node.inputs[0]);

            // Side inputs for binary ops
            for (size_t i = 1; i < node.inputs.size(); ++i)
                g.inputs.push_back(node.inputs[i]);

            tensor_group[node.output] = groups.size();
            groups.push_back(std::move(g));
        }
    }

    return groups;
}

} // namespace graph
} // namespace libane
