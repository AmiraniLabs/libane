/**
 * Fusion rules — decides which graph nodes can be compiled into a single
 * MIL program and dispatched as one ANE invocation.
 *
 * A FusionGroup is a maximal linear chain of nodes satisfying:
 *
 *  1. Linear chain: every consecutive pair (A, B) satisfies
 *       • A's output is B's inputs[0]  (B chains off A's result)
 *       • consumer_count[A.output] == 1  (nothing else reads A's output)
 *     This prevents fusing across branch points.
 *
 *  2. Weight file uniqueness: every weight-bearing node in the group has a
 *     distinct weight_file name (required by the BLOBFILE reference model).
 *
 *  3. Binary op side-input rule: for ADD/MUL nodes, inputs[1] must not be
 *     the output of any node already in the group (it is a "side input"
 *     coming from outside — graph input or a different fusion group).
 *
 *  4. CONV2D rules: CONV2D may start a group and may be extended only by
 *     shape-preserving pointwise activations (RELU, GELU, SILU, TANH,
 *     SIGMOID, HARDSWISH, LEAKY_RELU, ELU, CLIP, NEG).  Non-CONV2D groups
 *     cannot be extended by a CONV2D node.  CAST is always rejected.
 *
 *  5. Supported ops only: CAST is rejected unconditionally.
 *
 * Nodes are processed in topological order (= insertion order, since
 * AneGraph::add_op guarantees inputs pre-exist).
 *
 * FusionGroup.inputs lists ALL external tensor dependencies in the order
 * they are discovered (chain input first, side inputs after).
 * build_fused expects these as FusedInput pairs (var_name, shape).
 */
#pragma once

#include "ane_graph.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace libane {
namespace graph {

/* ── FusionGroup ─────────────────────────────────────────────────────────── */

struct FusionGroup {
    /** Node IDs in execution order. */
    std::vector<uint32_t> node_ids;

    /**
     * External input tensor IDs in discovery order:
     *   [0]   — chain input (first input to node_ids[0])
     *   [1..] — side inputs (additional inputs of binary ops)
     *
     * These are tensors consumed by nodes in this group but NOT produced
     * by any node in this group.
     */
    std::vector<TensorId> inputs;

    /** Output tensor ID (= last node's output). */
    TensorId output = kInvalidTensorId;
};

/* ── FusionRules ─────────────────────────────────────────────────────────── */

class FusionRules {
public:
    /**
     * Compute the fusion partition for a validated graph.
     *
     * Returns one FusionGroup per group, in topological order.
     * Every graph node appears in exactly one group.
     * Single-node groups are valid (and common for branch points).
     *
     * Precondition: GraphValidator::validate(graph).ok() == true.
     */
    static std::vector<FusionGroup> compute_groups(const AneGraph& graph);

    /**
     * Test whether a single candidate node can extend an existing group.
     *
     * @param graph          The containing graph.
     * @param group          Group being extended (non-empty).
     * @param candidate_id   Node ID of the candidate.
     * @param consumer_count Pre-computed map of tensor_id → number of consuming nodes.
     */
    static bool can_extend(const AneGraph& graph,
                            const FusionGroup& group,
                            uint32_t candidate_id,
                            const std::unordered_map<TensorId, size_t>& consumer_count);

    /** Build the consumer_count map for the whole graph. */
    static std::unordered_map<TensorId, size_t>
    build_consumer_count(const AneGraph& graph);

private:
    static bool op_supported(libane_op_t op);
};

} // namespace graph
} // namespace libane
