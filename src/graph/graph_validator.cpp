/**
 * GraphValidator implementation.
 */
#include "graph_validator.hpp"

#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>

namespace libane {
namespace graph {

/* ── Entry point ─────────────────────────────────────────────────────────── */

ValidationResult GraphValidator::validate(const AneGraph& g) {
    ValidationResult r;

    // Run each check; later checks depend on structural sanity so we bail
    // early only if structure is fundamentally broken.
    check_structure(g, r);
    if (!r.ok()) return r;

    check_shapes(g, r);
    check_edges(g, r);
    check_topo(g, r);
    check_reachability(g, r);
    check_weights(g, r);
    check_binary_shapes(g, r);

    return r;
}

/* ── Check 1 — structural sanity ─────────────────────────────────────────── */

void GraphValidator::check_structure(const AneGraph& g, ValidationResult& r) {
    if (g.graph_inputs().empty())
        r.errors.push_back("graph has no inputs");
    if (g.graph_outputs().empty())
        r.errors.push_back("graph has no outputs");
}

/* ── Check 2 — tensor shapes ─────────────────────────────────────────────── */

void GraphValidator::check_shapes(const AneGraph& g, ValidationResult& r) {
    for (const auto& t : g.tensors()) {
        try {
            t.shape.validate();
        } catch (const std::exception& e) {
            r.errors.push_back("tensor " + std::to_string(t.id) +
                               " (\"" + t.name + "\"): " + e.what());
        }
    }
}

/* ── Check 3 — no dangling edges ─────────────────────────────────────────── */

void GraphValidator::check_edges(const AneGraph& g, ValidationResult& r) {
    // Build the set of tensor IDs that are valid "sources":
    // graph inputs + outputs of every node.
    std::unordered_set<TensorId> defined;
    for (TensorId id : g.graph_inputs())
        defined.insert(id);

    // Walk nodes in insertion order — add_op guarantees inputs already exist,
    // so processing in order is safe for the positive case.
    for (const auto& n : g.nodes()) {
        for (TensorId inp : n.inputs) {
            if (defined.find(inp) == defined.end()) {
                r.errors.push_back(
                    "node " + std::to_string(n.id) +
                    " (op " + std::to_string(static_cast<int>(n.op)) +
                    "): input tensor " + std::to_string(inp) +
                    " is not produced by any prior node and is not a graph input");
            }
        }
        defined.insert(n.output);
    }

    // Verify all marked outputs are in the defined set
    for (TensorId oid : g.graph_outputs()) {
        if (oid >= static_cast<TensorId>(g.tensors().size())) {
            r.errors.push_back("graph output tensor id " + std::to_string(oid) +
                               " is out of range");
        } else if (defined.find(oid) == defined.end()) {
            r.errors.push_back("graph output tensor " + std::to_string(oid) +
                               " is not reachable from any node or graph input");
        }
    }
}

/* ── Check 4 — no cycles (Kahn's algorithm) ─────────────────────────────── */

void GraphValidator::check_topo(const AneGraph& g, ValidationResult& r) {
    const auto& nodes = g.nodes();
    if (nodes.empty()) return;

    // Map: tensor_id -> producer node_id (kInvalidNodeId for graph inputs)
    std::unordered_map<TensorId, uint32_t> tensor_producer;
    for (TensorId id : g.graph_inputs())
        tensor_producer[id] = kInvalidNodeId;
    for (const auto& n : nodes)
        tensor_producer[n.output] = n.id;

    // Build in-degree and predecessor map for each node
    std::vector<int> in_degree(nodes.size(), 0);
    // successor_nodes[i] = list of node IDs that consume node i's output
    std::vector<std::vector<uint32_t>> successors(nodes.size());

    for (const auto& n : nodes) {
        std::unordered_set<uint32_t> pred_seen;
        for (TensorId inp : n.inputs) {
            auto it = tensor_producer.find(inp);
            if (it == tensor_producer.end()) continue; // dangling — caught in check_edges
            uint32_t prod = it->second;
            if (prod == kInvalidNodeId) continue; // graph input, not a node
            if (prod >= static_cast<uint32_t>(nodes.size())) continue;
            if (pred_seen.insert(prod).second) {
                in_degree[n.id]++;
                successors[prod].push_back(n.id);
            }
        }
    }

    // Kahn's BFS
    std::queue<uint32_t> q;
    for (const auto& n : nodes) {
        if (in_degree[n.id] == 0)
            q.push(n.id);
    }

    size_t processed = 0;
    while (!q.empty()) {
        uint32_t nid = q.front(); q.pop();
        ++processed;
        for (uint32_t succ : successors[nid]) {
            if (--in_degree[succ] == 0)
                q.push(succ);
        }
    }

    if (processed != nodes.size()) {
        r.errors.push_back(
            "graph contains a cycle: topological sort processed " +
            std::to_string(processed) + " of " + std::to_string(nodes.size()) +
            " nodes");
    }
}

/* ── Check 5 — reachability from graph inputs ────────────────────────────── */

void GraphValidator::check_reachability(const AneGraph& g, ValidationResult& r) {
    // BFS/DFS forward from graph inputs through nodes.
    std::unordered_set<TensorId> reachable;
    for (TensorId id : g.graph_inputs())
        reachable.insert(id);

    // Process nodes in insertion order — since no cycles, one forward pass suffices.
    for (const auto& n : g.nodes()) {
        bool all_reachable = true;
        for (TensorId inp : n.inputs) {
            if (reachable.find(inp) == reachable.end()) {
                all_reachable = false;
                break;
            }
        }
        if (all_reachable)
            reachable.insert(n.output);
    }

    for (TensorId oid : g.graph_outputs()) {
        if (oid >= static_cast<TensorId>(g.tensors().size())) continue; // caught elsewhere
        if (reachable.find(oid) == reachable.end()) {
            r.errors.push_back(
                "graph output tensor " + std::to_string(oid) +
                " (\"" + g.tensor(oid).name + "\") is not reachable from any graph input");
        }
    }
}

/* ── Check 6 — weight sizes ──────────────────────────────────────────────── */

void GraphValidator::check_weights(const AneGraph& g, ValidationResult& r) {
    for (const auto& n : g.nodes()) {
        const GraphTensor& out_t = g.tensor(n.output);
        int OC = out_t.shape.channels;

        // Derive IC from the first input tensor (needed for matmul)
        int IC = 0;
        if (!n.inputs.empty() && n.inputs[0] < static_cast<TensorId>(g.tensors().size()))
            IC = g.tensor(n.inputs[0]).shape.channels;

        auto err = [&](const std::string& msg) {
            r.errors.push_back("node " + std::to_string(n.id) +
                               " (op " + std::to_string(static_cast<int>(n.op)) +
                               "): " + msg);
        };

        switch (n.op) {

        case LIBANE_OP_MATMUL: {
            if (n.inputs.size() != 1) {
                err("matmul requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            size_t expected = static_cast<size_t>(IC) * OC * 2;
            if (n.weights.size() != expected)
                err("weight size mismatch: expected " + std::to_string(expected) +
                    " bytes (IC=" + std::to_string(IC) +
                    " × OC=" + std::to_string(OC) + " × 2), got " +
                    std::to_string(n.weights.size()));
            break;
        }

        case LIBANE_OP_RMSNORM: {
            if (n.inputs.size() != 1) {
                err("rmsnorm requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            size_t expected = static_cast<size_t>(OC) * 2;
            if (n.weights.size() != expected)
                err("scale size mismatch: expected " + std::to_string(expected) +
                    " bytes (C=" + std::to_string(OC) + " × 2), got " +
                    std::to_string(n.weights.size()));
            break;
        }

        case LIBANE_OP_LAYER_NORM:
        case LIBANE_OP_LAYERNORM: {
            if (n.inputs.size() != 1) {
                err("layernorm requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            // gamma (C×2 bytes) packed before beta (C×2 bytes)
            size_t expected = static_cast<size_t>(OC) * 4;
            if (n.weights.size() != expected)
                err("weight size mismatch: expected " + std::to_string(expected) +
                    " bytes (2 × C × 2 for gamma+beta, C=" + std::to_string(OC) +
                    "), got " + std::to_string(n.weights.size()));
            break;
        }

        case LIBANE_OP_GELU:
        case LIBANE_OP_SOFTMAX:
        case LIBANE_OP_AVG_POOL:
        case LIBANE_OP_MAX_POOL:
        case LIBANE_OP_NEG:
        case LIBANE_OP_SINH:
        case LIBANE_OP_COSH:
        case LIBANE_OP_TAN:
        case LIBANE_OP_ASIN:
        case LIBANE_OP_ACOS:
        case LIBANE_OP_SILU:
        case LIBANE_OP_SQRT:
        case LIBANE_OP_LOG:
        case LIBANE_OP_RSQRT:
        case LIBANE_OP_TRANSPOSE:
        case LIBANE_OP_CAST:
            if (!n.weights.empty())
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 1)
                err("requires exactly one input, got " + std::to_string(n.inputs.size()));
            break;

        case LIBANE_OP_REDUCE_PROD: {
            if (!n.weights.empty())
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 1) {
                err("reduce_prod requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& in_t = g.tensor(n.inputs[0]);
            if (out_t.shape.channels != 1 || out_t.shape.seq != in_t.shape.seq) {
                err("reduce_prod output shape must be [1,1,1,S] with S matching input");
            }
            break;
        }

        case LIBANE_OP_SCATTER:
        case LIBANE_OP_SCATTER_ND:
        case LIBANE_OP_SCATTER_ALONG_AXIS: {
            if (n.inputs.size() != 2) {
                err("scatter-like op requires exactly two inputs (base, updates), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            if (n.weights.empty()) {
                err("scatter-like op requires static mask weights [1,C,1,S]");
                break;
            }
            const auto& a = g.tensor(n.inputs[0]).shape;
            const auto& b = g.tensor(n.inputs[1]).shape;
            if (!(a == b && a == out_t.shape)) {
                err("scatter-like op requires input/output shapes to match exactly");
            }
            size_t expected = out_t.shape.bytes();
            if (n.weights.size() != expected) {
                err("scatter-like mask size mismatch: expected " + std::to_string(expected) +
                    " bytes ([1,C,1,S] fp16), got " + std::to_string(n.weights.size()));
            }
            break;
        }

        case LIBANE_OP_GATHER: {
            if (n.inputs.size() == 1) {
                // Static gather mask provided as compile-time weights.
                if (n.weights.empty()) {
                    err("gather(static) requires mask weights [1,C,1,S]");
                    break;
                }
                const auto& in = g.tensor(n.inputs[0]).shape;
                if (!(in == out_t.shape))
                    err("gather(static) requires input/output shapes to match exactly");
                size_t expected = out_t.shape.bytes();
                if (n.weights.size() != expected) {
                    err("gather(static) mask size mismatch: expected " + std::to_string(expected) +
                        " bytes ([1,C,1,S] fp16), got " + std::to_string(n.weights.size()));
                }
                break;
            }

            if (n.inputs.size() == 2) {
                // Dynamic gather mask provided as second runtime input.
                if (!n.weights.empty())
                    err("gather(dynamic) is weight-free; provide mask as second input");
                const auto& x = g.tensor(n.inputs[0]).shape;
                const auto& m = g.tensor(n.inputs[1]).shape;
                if (!(x == m && x == out_t.shape)) {
                    err("gather(dynamic) requires x/mask/output shapes to match exactly");
                }
                break;
            }

            err("gather supports exactly one input (static mask) or two inputs (dynamic mask), got " +
                std::to_string(n.inputs.size()));
            break;
        }

        case LIBANE_OP_ADD:
        case LIBANE_OP_MUL:
        case LIBANE_OP_MOD:
        case LIBANE_OP_SUB:
        case LIBANE_OP_REAL_DIV:
        case LIBANE_OP_LOGICAL_AND:
        case LIBANE_OP_LOGICAL_OR:
        case LIBANE_OP_LOGICAL_XOR:
            if (!n.weights.empty())
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 2)
                err("requires exactly two inputs, got " +
                    std::to_string(n.inputs.size()));
            break;

        case LIBANE_OP_SELECT: {
            if (!n.weights.empty())
                err("select is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 3) {
                err("select requires exactly 3 inputs (condition, x, y), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& cond_s = g.tensor(n.inputs[0]).shape;
            const auto& x_s    = g.tensor(n.inputs[1]).shape;
            const auto& y_s    = g.tensor(n.inputs[2]).shape;
            if (cond_s.channels != x_s.channels || cond_s.seq != x_s.seq)
                err("select: condition and x shapes must match");
            if (x_s.channels != y_s.channels || x_s.seq != y_s.seq)
                err("select: x and y shapes must match");
            if (out_t.shape.channels != x_s.channels || out_t.shape.seq != x_s.seq)
                err("select: output shape must match input shapes");
            break;
        }

        case LIBANE_OP_CONCAT: {
            if (!n.weights.empty()) {
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            }
            if (n.inputs.size() != 2) {
                err("concat requires exactly two inputs, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& a = g.tensor(n.inputs[0]).shape;
            const auto& b = g.tensor(n.inputs[1]).shape;
            if (a.seq != b.seq) {
                err("concat requires matching input seq dimensions, got " +
                    std::to_string(a.seq) + " and " + std::to_string(b.seq));
            }
            if (out_t.shape.seq != a.seq) {
                err("concat output seq must match inputs: expected " +
                    std::to_string(a.seq) + ", got " + std::to_string(out_t.shape.seq));
            }
            if (out_t.shape.channels != a.channels + b.channels) {
                err("concat output channels must equal input0+input1: expected " +
                    std::to_string(a.channels + b.channels) + ", got " +
                    std::to_string(out_t.shape.channels));
            }
            break;
        }

        case LIBANE_OP_RESHAPE: {
            if (!n.weights.empty()) {
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            }
            if (n.inputs.size() != 1) {
                err("reshape requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& in_t = g.tensor(n.inputs[0]).shape;
            if (in_t.numel() != out_t.shape.numel()) {
                err("reshape requires equal input/output element counts: input numel=" +
                    std::to_string(in_t.numel()) + ", output numel=" +
                    std::to_string(out_t.shape.numel()));
            }
            break;
        }

        case LIBANE_OP_SLICE_BY_INDEX: {
            if (!n.weights.empty()) {
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            }
            if (n.inputs.size() != 1) {
                err("slice_by_index requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& in_t = g.tensor(n.inputs[0]).shape;
            if (out_t.shape.channels > in_t.channels || out_t.shape.seq > in_t.seq) {
                err("slice_by_index output dims must be <= input dims: input [1," +
                    std::to_string(in_t.channels) + ",1," + std::to_string(in_t.seq) +
                    "], output [1," + std::to_string(out_t.shape.channels) + ",1," +
                    std::to_string(out_t.shape.seq) + "]");
            }
            break;
        }

        case LIBANE_OP_REDUCE_SUM:
        case LIBANE_OP_REDUCE_MEAN:
        case LIBANE_OP_REDUCE_MAX: {
            if (!n.weights.empty())
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 1)
                err("reduce op requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
            else {
                const auto& in_s = g.tensor(n.inputs[0]).shape;
                if (out_t.shape.channels != 1)
                    err("reduce output channels must be 1, got " +
                        std::to_string(out_t.shape.channels));
                if (out_t.shape.seq != in_s.seq)
                    err("reduce output seq must match input seq: expected " +
                        std::to_string(in_s.seq) + ", got " +
                        std::to_string(out_t.shape.seq));
            }
            break;
        }

        case LIBANE_OP_CONV2D:
            err("CONV2D is not supported in the graph API");
            break;

        default:
            err("unknown op code " + std::to_string(static_cast<int>(n.op)));
            break;
        }
    }
}

/* ── Check 7 — binary arithmetic ops: identical input shapes (ANE #18) ───── */

void GraphValidator::check_binary_shapes(const AneGraph& g, ValidationResult& r) {
    for (const auto& n : g.nodes()) {
        if (n.op != LIBANE_OP_ADD &&
            n.op != LIBANE_OP_MUL &&
            n.op != LIBANE_OP_SUB &&
            n.op != LIBANE_OP_REAL_DIV) continue;
        if (n.inputs.size() < 2) continue;

        const mil::TensorShape& ref =
            g.tensor(n.inputs[0]).shape;

        for (size_t i = 1; i < n.inputs.size(); ++i) {
            TensorId tid = n.inputs[i];
            if (tid >= static_cast<TensorId>(g.tensors().size())) continue;
            const mil::TensorShape& s = g.tensor(tid).shape;
            if (!(s == ref)) {
                r.errors.push_back(
                    "node " + std::to_string(n.id) +
                    " (op " + std::to_string(static_cast<int>(n.op)) +
                    "): input 0 shape [1," + std::to_string(ref.channels) +
                    ",1," + std::to_string(ref.seq) +
                    "] != input " + std::to_string(i) +
                    " shape [1," + std::to_string(s.channels) +
                    ",1," + std::to_string(s.seq) +
                    "] (ANE constraint #18)");
            }
        }
    }
}

} // namespace graph
} // namespace libane
