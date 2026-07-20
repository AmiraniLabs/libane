/**
 * GraphValidator implementation.
 */
#include "graph_validator.hpp"

#include <cstring>
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
            if (t.shape.channels == 1 && t.shape.height > 1)
                t.shape.validate_matrix();
            else if (t.shape.channels > 1 && t.shape.height > 1)
                t.shape.validate_conv_image();
            else
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

        case LIBANE_OP_MATMUL_MULTI: {
            // A: [1,1,K,M]  B: [1,1,N,K]  → C: [1,1,N,M]
            // Matrix-tensor shapes: channels=1, height=K or N, seq=M or K.
            if (!n.weights.empty())
                err("matmul_multi is weight-free, got " +
                    std::to_string(n.weights.size()) + " weight bytes");
            if (n.inputs.size() != 2) {
                err("matmul_multi requires exactly 2 inputs (A, B), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& a_s = g.tensor(n.inputs[0]).shape;
            const auto& b_s = g.tensor(n.inputs[1]).shape;
            if (a_s.channels != 1)
                err("matmul_multi: A must be a matrix tensor with channels=1, got " +
                    std::to_string(a_s.channels));
            if (b_s.channels != 1)
                err("matmul_multi: B must be a matrix tensor with channels=1, got " +
                    std::to_string(b_s.channels));
            // A.height=K, A.seq=M; B.height=N, B.seq=K → B.seq must equal A.height
            if (b_s.seq != a_s.height)
                err("matmul_multi: B.seq (K=" + std::to_string(b_s.seq) +
                    ") must equal A.height (K=" + std::to_string(a_s.height) + ")");
            if (out_t.shape.channels != 1)
                err("matmul_multi: output must be a matrix tensor with channels=1, got " +
                    std::to_string(out_t.shape.channels));
            if (out_t.shape.height != b_s.height)
                err("matmul_multi: output.height (N=" + std::to_string(out_t.shape.height) +
                    ") must equal B.height (N=" + std::to_string(b_s.height) + ")");
            if (out_t.shape.seq != a_s.seq)
                err("matmul_multi: output.seq (M=" + std::to_string(out_t.shape.seq) +
                    ") must equal A.seq (M=" + std::to_string(a_s.seq) + ")");
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
        case LIBANE_OP_RELU:
        case LIBANE_OP_TANH:
        case LIBANE_OP_SIGMOID:
        case LIBANE_OP_HARDSWISH:
        case LIBANE_OP_LEAKY_RELU:
        case LIBANE_OP_ELU:
        case LIBANE_OP_EXP:
        case LIBANE_OP_SIN:
        case LIBANE_OP_COS:
        case LIBANE_OP_ABS:
        case LIBANE_OP_CEIL:
        case LIBANE_OP_FLOOR:
        case LIBANE_OP_ROUND:
        case LIBANE_OP_SIGN:
            if (!n.weights.empty())
                err("op is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 1)
                err("requires exactly one input, got " + std::to_string(n.inputs.size()));
            break;

        case LIBANE_OP_POW:
            if (!n.weights.empty())
                err("pow is weight-free but " + std::to_string(n.weights.size()) +
                    " weight bytes were provided");
            if (n.inputs.size() != 2)
                err("pow requires exactly two inputs (base, exponent), got " +
                    std::to_string(n.inputs.size()));
            break;

        case LIBANE_OP_PIXEL_SHUFFLE: {
            if (n.inputs.size() != 1) {
                err("pixel_shuffle requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            if (n.weights.size() != 4) {
                err("pixel_shuffle requires 4 weight bytes (int32 upscale_factor), got " +
                    std::to_string(n.weights.size()));
                break;
            }
            int32_t r = 0;
            std::memcpy(&r, n.weights.data(), 4);
            if (r <= 0)
                err("pixel_shuffle upscale_factor must be > 0, got " + std::to_string(r));
            const auto& in_t = g.tensor(n.inputs[0]);
            if (in_t.shape.channels % r != 0)
                err("pixel_shuffle input channels (" + std::to_string(in_t.shape.channels) +
                    ") must be divisible by upscale_factor (" + std::to_string(r) + ")");
            break;
        }

        case LIBANE_OP_PWL_ACTIVATION: {
            if (n.inputs.size() != 1) {
                err("pwl_activation requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            // weights: [x_min, x_max, samples...] as float32
            if (n.weights.size() < 12 || (n.weights.size() % 4) != 0) {
                err("pwl_activation weights must be at least 3 float32 values (x_min, x_max, sample[0]), got " +
                    std::to_string(n.weights.size()) + " bytes");
                break;
            }
            int n_floats = static_cast<int>(n.weights.size()) / 4;
            if (n_floats < 4) {
                err("pwl_activation needs at least 2 samples (n_floats >= 4), got " +
                    std::to_string(n_floats));
            }
            break;
        }

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

        case LIBANE_OP_SLICE: {
            if (!n.weights.empty() && n.weights.size() != 8 * sizeof(int32_t)) {
                err("slice weights must be empty or exactly 32 bytes "
                    "(8 × int32: begin[4], stride[4]), got " +
                    std::to_string(n.weights.size()) + " bytes");
            }
            if (n.inputs.size() != 1) {
                err("slice requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            if (!n.weights.empty()) {
                const int32_t* w = reinterpret_cast<const int32_t*>(n.weights.data());
                for (int i = 0; i < 4; ++i) {
                    if (w[4 + i] < 1) {
                        err("slice stride[" + std::to_string(i) +
                            "] must be >= 1, got " + std::to_string(w[4 + i]));
                    }
                }
            }
            break;
        }

        case LIBANE_OP_CLIP: {
            if (n.weights.size() != 2 * sizeof(float))
                err("clip requires exactly 8 weight bytes (float32 lo, hi), got " +
                    std::to_string(n.weights.size()));
            if (n.inputs.size() != 1)
                err("clip requires exactly one input");
            else {
                const auto& in_s = g.tensor(n.inputs[0]).shape;
                if (out_t.shape.channels != in_s.channels ||
                    out_t.shape.seq      != in_s.seq)
                    err("clip output shape must equal input shape");
                if (n.weights.size() == 8) {
                    float lo, hi;
                    std::memcpy(&lo, n.weights.data(),              4);
                    std::memcpy(&hi, n.weights.data() + 4,          4);
                    if (lo > hi)
                        err("clip: lo (" + std::to_string(lo) + ") must be <= hi (" +
                            std::to_string(hi) + ")");
                }
            }
            break;
        }

        case LIBANE_OP_PAD: {
            if (!n.weights.empty() && n.weights.size() != 8 * sizeof(int32_t))
                err("pad weights must be empty or exactly 32 bytes "
                    "(8 × int32), got " + std::to_string(n.weights.size()) + " bytes");
            if (n.inputs.size() != 1)
                err("pad requires exactly one input");
            else if (n.weights.size() == 32) {
                const int32_t* pw =
                    reinterpret_cast<const int32_t*>(n.weights.data());
                const auto& in_s = g.tensor(n.inputs[0]).shape;
                for (int i = 0; i < 8; ++i)
                    if (pw[i] < 0)
                        err("pad amount [" + std::to_string(i) +
                            "] must be >= 0, got " + std::to_string(pw[i]));
                int expected_C  = in_s.channels + pw[1] + pw[5];
                int expected_SP = in_s.seq       + pw[3] + pw[7];
                if (out_t.shape.channels != expected_C)
                    err("pad output C must equal in_C + pad_before_C + pad_after_C = " +
                        std::to_string(expected_C));
                if (out_t.shape.seq != expected_SP)
                    err("pad output S must equal in_S + pad_before_S + pad_after_S = " +
                        std::to_string(expected_SP));
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

        case LIBANE_OP_DYNAMIC_MATMUL: {
            // Matrix format: X=[1,1,K,M]  W=[1,1,N,K]  Y=[1,1,N,M]
            // K/N/M are derived from shapes; no static weights.
            if (!n.weights.empty()) {
                err("dynamic_matmul: no weights expected (K/N/M derived from shapes), got " +
                    std::to_string(n.weights.size()) + " bytes");
                break;
            }
            if (n.inputs.size() != 2) {
                err("dynamic_matmul requires exactly 2 inputs (X, W), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            const auto& x_s = g.tensor(n.inputs[0]).shape;
            const auto& w_s = g.tensor(n.inputs[1]).shape;
            if (x_s.channels != 1 || w_s.channels != 1 || out_t.shape.channels != 1)
                err("dynamic_matmul: all tensors must have channels=1 (matrix format)");
            // X=[1,1,K,M]: height=K (inner), seq=M (output cols)
            // W=[1,1,N,K]: height=N (output rows), seq=K (inner)
            if (x_s.height != w_s.seq)
                err("dynamic_matmul: X.height (K=" + std::to_string(x_s.height) +
                    ") must equal W.seq (K=" + std::to_string(w_s.seq) + ")");
            if (out_t.shape.height != w_s.height)
                err("dynamic_matmul: output.height (N=" + std::to_string(out_t.shape.height) +
                    ") must equal W.height (N=" + std::to_string(w_s.height) + ")");
            if (out_t.shape.seq != x_s.seq)
                err("dynamic_matmul: output.seq (M=" + std::to_string(out_t.shape.seq) +
                    ") must equal X.seq (M=" + std::to_string(x_s.seq) + ")");
            break;
        }

        case LIBANE_OP_SDPA: {
            // Matrix format: Q/K/V=[1,H,S,D]  mask=[1,1,S,S]  out=[1,H,S,D]
            // H/S/D derived from shapes; no static weights.
            if (n.inputs.size() < 3 || n.inputs.size() > 4) {
                err("sdpa requires 3 or 4 inputs (Q, K, V[, mask]), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            if (!n.weights.empty()) {
                err("sdpa: no weights expected (H/S/D derived from shapes), got " +
                    std::to_string(n.weights.size()) + " bytes");
                break;
            }
            const auto& q_s = g.tensor(n.inputs[0]).shape;
            const auto& k_s = g.tensor(n.inputs[1]).shape;
            const auto& v_s = g.tensor(n.inputs[2]).shape;
            int H = q_s.channels;
            int S = q_s.height;
            int D = q_s.seq;
            if (!(k_s == q_s))
                err("sdpa: K shape must match Q shape [1," + std::to_string(H) +
                    "," + std::to_string(S) + "," + std::to_string(D) + "]");
            if (!(v_s == q_s))
                err("sdpa: V shape must match Q shape");
            if (out_t.shape.channels != H || out_t.shape.height != S || out_t.shape.seq != D)
                err("sdpa: output shape must match Q/K/V shape [1," +
                    std::to_string(H) + "," + std::to_string(S) + "," + std::to_string(D) + "]");
            if (n.inputs.size() == 4) {
                const auto& m_s = g.tensor(n.inputs[3]).shape;
                if (m_s.channels != 1 || m_s.height != S || m_s.seq != S)
                    err("sdpa: mask shape must be [1,1,S,S]=[1,1," +
                        std::to_string(S) + "," + std::to_string(S) + "], got [1," +
                        std::to_string(m_s.channels) + "," + std::to_string(m_s.height) +
                        "," + std::to_string(m_s.seq) + "]");
            }
            break;
        }

        case LIBANE_OP_SDPA_GQA: {
            // Matrix format: Q=[1,H_q,S,D]  K/V=[1,H_kv,S,D]  mask=[1,1,S,S]  out=[1,H_q,S,D]
            // H_q/H_kv/S/D derived from input shapes; no static weights.
            if (n.inputs.size() < 3 || n.inputs.size() > 4) {
                err("sdpa_gqa requires 3 or 4 inputs (Q, K, V[, mask]), got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            if (!n.weights.empty()) {
                err("sdpa_gqa: no weights expected (shapes derived from inputs), got " +
                    std::to_string(n.weights.size()) + " bytes");
                break;
            }
            const auto& q_s = g.tensor(n.inputs[0]).shape;
            const auto& k_s = g.tensor(n.inputs[1]).shape;
            const auto& v_s = g.tensor(n.inputs[2]).shape;
            int H_q  = q_s.channels;
            int H_kv = k_s.channels;
            int S    = q_s.height;
            int D    = q_s.seq;
            // K and V must share shape
            if (!(k_s == v_s))
                err("sdpa_gqa: K and V shapes must match; K=[1," +
                    std::to_string(k_s.channels) + "," + std::to_string(k_s.height) +
                    "," + std::to_string(k_s.seq) + "] V=[1," +
                    std::to_string(v_s.channels) + "," + std::to_string(v_s.height) +
                    "," + std::to_string(v_s.seq) + "]");
            // K/V sequence/depth must match Q
            if (k_s.height != S || k_s.seq != D)
                err("sdpa_gqa: K S and D must match Q S=" + std::to_string(S) +
                    " D=" + std::to_string(D));
            // Head divisibility
            if (H_kv <= 0 || H_q % H_kv != 0)
                err("sdpa_gqa: num_q_heads (" + std::to_string(H_q) +
                    ") must be divisible by num_kv_heads (" + std::to_string(H_kv) + ")");
            // Output shape must match Q
            if (out_t.shape.channels != H_q || out_t.shape.height != S || out_t.shape.seq != D)
                err("sdpa_gqa: output shape must be [1," + std::to_string(H_q) +
                    "," + std::to_string(S) + "," + std::to_string(D) +
                    "], got [1," + std::to_string(out_t.shape.channels) + "," +
                    std::to_string(out_t.shape.height) + "," + std::to_string(out_t.shape.seq) + "]");
            // Optional mask: [1,1,S,S]
            if (n.inputs.size() == 4) {
                const auto& m_s = g.tensor(n.inputs[3]).shape;
                if (m_s.channels != 1 || m_s.height != S || m_s.seq != S)
                    err("sdpa_gqa: mask shape must be [1,1,S,S]=[1,1," +
                        std::to_string(S) + "," + std::to_string(S) + "], got [1," +
                        std::to_string(m_s.channels) + "," + std::to_string(m_s.height) +
                        "," + std::to_string(m_s.seq) + "]");
            }
            break;
        }

        case LIBANE_OP_CONV2D: {
            // Weight blob encoding:
            //   Bytes [0..43]  : int32[11] = {kH, kW, stride_h, stride_w,
            //                                 pad_top, pad_left, pad_bottom, pad_right,
            //                                 dilation_h, dilation_w, groups}
            //   Bytes [44..]   : fp16 kernel [OC, IC/groups, kH, kW] row-major
            if (n.inputs.size() != 1) {
                err("conv2d requires exactly 1 input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            constexpr size_t kParamBytes = 11 * sizeof(int32_t);
            if (n.weights.size() <= kParamBytes) {
                err("conv2d weights too small: need > " + std::to_string(kParamBytes) +
                    " bytes (params + kernel), got " + std::to_string(n.weights.size()));
                break;
            }
            const int32_t* wp = reinterpret_cast<const int32_t*>(n.weights.data());
            int kH_v        = wp[0],  kW_v      = wp[1];
            int stride_h_v  = wp[2],  stride_w_v = wp[3];
            int pad_top_v   = wp[4],  pad_left_v = wp[5];
            int pad_bot_v   = wp[6],  pad_right_v = wp[7];
            int dil_h_v     = wp[8],  dil_w_v    = wp[9];
            int groups_v    = wp[10];

            if (kH_v <= 0 || kW_v <= 0)
                err("conv2d kH/kW must be > 0, got kH=" + std::to_string(kH_v) +
                    " kW=" + std::to_string(kW_v));
            if (stride_h_v <= 0 || stride_w_v <= 0)
                err("conv2d strides must be > 0, got " + std::to_string(stride_h_v) +
                    "×" + std::to_string(stride_w_v));
            if (dil_h_v <= 0 || dil_w_v <= 0)
                err("conv2d dilations must be > 0");
            if (groups_v <= 0)
                err("conv2d groups must be > 0, got " + std::to_string(groups_v));
            if (pad_top_v < 0 || pad_left_v < 0 || pad_bot_v < 0 || pad_right_v < 0)
                err("conv2d padding values must be >= 0");
            if (IC % groups_v != 0)
                err("conv2d IC=" + std::to_string(IC) + " not divisible by groups=" +
                    std::to_string(groups_v));

            // Verify kernel byte count
            size_t expected_kernel = static_cast<size_t>(OC) *
                                     (IC / groups_v) * kH_v * kW_v * 2;
            size_t actual_kernel   = n.weights.size() - kParamBytes;
            if (actual_kernel != expected_kernel)
                err("conv2d kernel size mismatch: expected " +
                    std::to_string(expected_kernel) + " bytes (OC=" +
                    std::to_string(OC) + " × IC/groups=" +
                    std::to_string(IC / groups_v) + " × kH=" +
                    std::to_string(kH_v) + " × kW=" + std::to_string(kW_v) +
                    " × 2), got " + std::to_string(actual_kernel));

            // Verify output shape matches computed dimensions
            const auto& in_s_c = g.tensor(n.inputs[0]).shape;
            int H_in_v  = in_s_c.height;
            int W_in_v  = in_s_c.seq;
            int H_out_v = (H_in_v + pad_top_v + pad_bot_v -
                           dil_h_v * (kH_v - 1) - 1) / stride_h_v + 1;
            int W_out_v = (W_in_v + pad_left_v + pad_right_v -
                           dil_w_v * (kW_v - 1) - 1) / stride_w_v + 1;

            if (out_t.shape.channels != OC)
                err("conv2d output channels must equal OC=" + std::to_string(OC) +
                    ", got " + std::to_string(out_t.shape.channels));
            if (out_t.shape.height != H_out_v)
                err("conv2d output H mismatch: expected " + std::to_string(H_out_v) +
                    " got " + std::to_string(out_t.shape.height));
            if (out_t.shape.seq != W_out_v)
                err("conv2d output W mismatch: expected " + std::to_string(W_out_v) +
                    " got " + std::to_string(out_t.shape.seq));
            break;
        }

        case LIBANE_OP_MATMUL_W8A16: {
            // Blob: int32[2]={OC,IC} + int8[IC×OC] + pad-to-4 + float32[OC]
            if (n.inputs.size() != 1) {
                err("matmul_w8a16 requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            constexpr size_t kHdrBytes = 2 * sizeof(int32_t);
            if (n.weights.size() < kHdrBytes + 1) {
                err("matmul_w8a16 weight blob too small (minimum " +
                    std::to_string(kHdrBytes + 1) + " bytes), got " +
                    std::to_string(n.weights.size()));
                break;
            }
            const int32_t* hdr = reinterpret_cast<const int32_t*>(n.weights.data());
            int blob_OC = hdr[0];
            int blob_IC = hdr[1];
            if (blob_OC != OC)
                err("matmul_w8a16 blob OC=" + std::to_string(blob_OC) +
                    " mismatches output channels OC=" + std::to_string(OC));
            if (blob_IC != IC)
                err("matmul_w8a16 blob IC=" + std::to_string(blob_IC) +
                    " mismatches input channels IC=" + std::to_string(IC));
            if (blob_IC <= 0 || blob_OC <= 0) {
                err("matmul_w8a16 IC and OC must be positive");
                break;
            }
            size_t weights_end  = kHdrBytes + static_cast<size_t>(blob_IC) * blob_OC;
            size_t scales_start = (weights_end + 3) & ~size_t(3);
            size_t expected     = scales_start + static_cast<size_t>(blob_OC) * sizeof(float);
            if (n.weights.size() != expected)
                err("matmul_w8a16 weight blob size mismatch: expected " +
                    std::to_string(expected) + " bytes (IC=" + std::to_string(blob_IC) +
                    ", OC=" + std::to_string(blob_OC) + "), got " +
                    std::to_string(n.weights.size()));
            break;
        }

        case LIBANE_OP_MATMUL_W8A8: {
            // Blob: int32[2]={OC,IC} + int8[IC×OC] + pad-to-4
            //       + float32[OC] (weight scales) + float32 (act_scale) + int32 (act_zp)
            if (n.inputs.size() != 1) {
                err("matmul_w8a8 requires exactly one input, got " +
                    std::to_string(n.inputs.size()));
                break;
            }
            constexpr size_t kHdrBytes = 2 * sizeof(int32_t);
            if (n.weights.size() < kHdrBytes + 1) {
                err("matmul_w8a8 weight blob too small, got " +
                    std::to_string(n.weights.size()));
                break;
            }
            const int32_t* hdr = reinterpret_cast<const int32_t*>(n.weights.data());
            int blob_OC = hdr[0];
            int blob_IC = hdr[1];
            if (blob_OC != OC)
                err("matmul_w8a8 blob OC=" + std::to_string(blob_OC) +
                    " mismatches output channels OC=" + std::to_string(OC));
            if (blob_IC != IC)
                err("matmul_w8a8 blob IC=" + std::to_string(blob_IC) +
                    " mismatches input channels IC=" + std::to_string(IC));
            if (blob_IC <= 0 || blob_OC <= 0) {
                err("matmul_w8a8 IC and OC must be positive");
                break;
            }
            size_t weights_end  = kHdrBytes + static_cast<size_t>(blob_IC) * blob_OC;
            size_t scales_start = (weights_end + 3) & ~size_t(3);
            // weight scales + act_scale + act_zero_point
            size_t expected = scales_start + static_cast<size_t>(blob_OC) * sizeof(float)
                              + sizeof(float) + sizeof(int32_t);
            if (n.weights.size() != expected)
                err("matmul_w8a8 weight blob size mismatch: expected " +
                    std::to_string(expected) + " bytes (IC=" + std::to_string(blob_IC) +
                    ", OC=" + std::to_string(blob_OC) + "), got " +
                    std::to_string(n.weights.size()));
            break;
        }

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
