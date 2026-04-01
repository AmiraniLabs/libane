/**
 * GraphCompiler implementation.
 */
#include "graph_compiler.hpp"
#include "graph_validator.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

namespace libane {
namespace graph {

/* ── Local helpers ───────────────────────────────────────────────────────── */

/** Canonical MIL variable name for a tensor ID. */
static std::string tensor_var(TensorId id) {
    return "t" + std::to_string(id);
}

/**
 * Turn a graph node into a MilFragment.
 * in_var  — MIL variable that carries the chain input.
 * out_var — MIL variable to bind the output to.
 */
static mil::MilFragment node_to_fragment(const AneGraph&    graph,
                                          const GraphNode&   node,
                                          const std::string& in_var,
                                          const std::string& out_var) {
    const mil::TensorShape& in_shape  =
        graph.tensor(node.inputs.empty() ? kInvalidTensorId : node.inputs[0]).shape;
    const mil::TensorShape& out_shape = graph.tensor(node.output).shape;

    switch (node.op) {
    case LIBANE_OP_MATMUL:
        return mil::MilBuilder::matmul_fragment(
            in_shape.channels, out_shape.channels, out_shape.seq,
            in_var, out_var, node.weight_file);

    case LIBANE_OP_RMSNORM:
        return mil::MilBuilder::rmsnorm_fragment(
            in_shape.channels, in_shape.seq,
            in_var, out_var, node.weight_file);

    case LIBANE_OP_LAYERNORM:
    case LIBANE_OP_LAYER_NORM: {
        // Gamma file = node.weight_file; beta file = "<stem>b.bin"
        std::string gamma_file = node.weight_file;
        std::string beta_file  = gamma_file;
        if (beta_file.size() > 4)
            beta_file = beta_file.substr(0, beta_file.size() - 4) + "b.bin";
        return mil::MilBuilder::layernorm_fragment(
            in_shape.channels, in_shape.seq,
            in_var, out_var, gamma_file, beta_file);
    }

    case LIBANE_OP_GELU:
        return mil::MilBuilder::gelu_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SILU:
        return mil::MilBuilder::silu_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SOFTMAX:
        return mil::MilBuilder::softmax_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_AVG_POOL:
        return mil::MilBuilder::avg_pool_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_MAX_POOL:
        return mil::MilBuilder::max_pool_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_TRANSPOSE:
        return mil::MilBuilder::transpose_fragment(
            in_shape.channels, in_shape.seq, in_var, out_var);

    case LIBANE_OP_ADD:
        return mil::MilBuilder::add_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_MUL:
        return mil::MilBuilder::mul_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_LOGICAL_AND:
        return mil::MilBuilder::logical_and_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    default:
        throw std::runtime_error(
            "node_to_fragment: unsupported op " + std::to_string(node.op));
    }
}

/* ── CompiledGraph ───────────────────────────────────────────────────────── */

CompiledGraph::~CompiledGraph() {
    for (auto& g : groups_)
        runtime::ane_unload(g.program);
}

AneBuffer* CompiledGraph::ane_buf(TensorId id) const {
    auto it = ane_bufs_.find(id);
    return it == ane_bufs_.end() ? nullptr : it->second.get();
}

/* ── GraphCompiler::build_plan ───────────────────────────────────────────── */

ExecutionPlan GraphCompiler::build_plan(const AneGraph& graph) {
    ValidationResult vr = GraphValidator::validate(graph);
    if (!vr.ok()) {
        std::string msg = "Graph validation failed:";
        for (const auto& e : vr.errors)
            msg += "\n  " + e;
        throw std::runtime_error(msg);
    }

    ExecutionPlan plan;
    plan.groups           = FusionRules::compute_groups(graph);
    plan.graph_input_ids  = graph.graph_inputs();
    plan.graph_output_ids = graph.graph_outputs();

    for (const auto& t : graph.tensors())
        plan.tensor_bytes[t.id] = t.shape.bytes();

    for (const auto& g : plan.groups)
        plan.buffer_ids.push_back(g.output);

    return plan;
}

/* ── GraphCompiler::compile ──────────────────────────────────────────────── */

std::unique_ptr<CompiledGraph> GraphCompiler::compile(const AneGraph& graph) {
    ExecutionPlan plan;
    try {
        plan = build_plan(graph);
    } catch (...) {
        return nullptr;
    }

    if (runtime::initialize() == runtime::AneState::Fallback)
        return nullptr;

    auto cg = std::make_unique<CompiledGraph>();
    cg->tensor_bytes_     = plan.tensor_bytes;
    cg->graph_input_ids_  = plan.graph_input_ids;
    cg->graph_output_ids_ = plan.graph_output_ids;
    for (const auto& t : graph.tensors())
        cg->tensor_shapes_[t.id] = t.shape;

    // Runtime enforces uniform IOSurface allocation size across multi-input
    // dispatches; choose one per-graph allocation target for all I/O tensors.
    for (const auto& kv : plan.tensor_bytes) {
        cg->io_alloc_bytes_ = std::max(cg->io_alloc_bytes_, kv.second);
    }

    auto round_up = [](size_t x, size_t m) {
        return ((x + m - 1) / m) * m;
    };

    // ANE raw channel stride constraints:
    // - Must be at least S*2 bytes for each tensor that shares this alloc size.
    // - Keep 64-byte aligned to match hardware vector length.
    // Choose a single io_alloc_bytes_ that satisfies all graph I/O tensors.
    if (cg->io_alloc_bytes_ > 0) {
        size_t target = std::max(cg->io_alloc_bytes_, static_cast<size_t>(49152));
        target = round_up(target, static_cast<size_t>(64));

        bool ok = false;
        for (int iter = 0; iter < 40960 && !ok; ++iter) {
            ok = true;
            for (TensorId tid : plan.graph_input_ids) {
                const auto& s = graph.tensor(tid).shape;
                size_t stride = target / static_cast<size_t>(s.channels);
                if (stride < static_cast<size_t>(s.seq) * 2 || (stride % 64) != 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                for (TensorId tid : plan.graph_output_ids) {
                    const auto& s = graph.tensor(tid).shape;
                    size_t stride = target / static_cast<size_t>(s.channels);
                    if (stride < static_cast<size_t>(s.seq) * 2 || (stride % 64) != 0) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) target += 64;
        }
        cg->io_alloc_bytes_ = target;
    }

    for (const auto& group : plan.groups) {
        // ── Collect fused inputs (chain + side) ───────────────────────────
        std::vector<mil::FusedInput> fused_inputs;
        {
            TensorId cid = group.inputs[0];
            fused_inputs.push_back({ tensor_var(cid), graph.tensor(cid).shape });
        }
        for (size_t i = 1; i < group.inputs.size(); ++i) {
            TensorId sid = group.inputs[i];
            fused_inputs.push_back({ tensor_var(sid), graph.tensor(sid).shape });
        }

        // ── Build fragments and weight entries ────────────────────────────
        std::vector<mil::MilFragment>     fragments;
        std::vector<runtime::WeightEntry> weight_entries;

        std::string chain_var = tensor_var(group.inputs[0]);

        for (uint32_t nid : group.node_ids) {
            const GraphNode& node    = graph.node(nid);
            std::string      out_var = tensor_var(node.output);

            fragments.push_back(node_to_fragment(graph, node, chain_var, out_var));
            chain_var = out_var;

            if (!node.weight_file.empty() && !node.weights.empty()) {
                if (node.op == LIBANE_OP_LAYERNORM || node.op == LIBANE_OP_LAYER_NORM) {
                    size_t half = node.weights.size() / 2;

                    auto gamma_blob = mil::WeightBlob::from_fp16(node.weights.data(), half);
                    weight_entries.push_back({ node.weight_file,
                                               std::move(gamma_blob.data) });

                    std::string beta_file = node.weight_file;
                    if (beta_file.size() > 4)
                        beta_file = beta_file.substr(0, beta_file.size() - 4) + "b.bin";
                    auto beta_blob = mil::WeightBlob::from_fp16(
                        node.weights.data() + half, half);
                    weight_entries.push_back({ beta_file, std::move(beta_blob.data) });

                } else if (node.op == LIBANE_OP_MATMUL) {
                    // conv1x1 expects [OC, IC]; user provides [IC, OC] → transpose
                    int IC = graph.tensor(node.inputs[0]).shape.channels;
                    int OC = graph.tensor(node.output).shape.channels;
                    auto blob = mil::WeightBlob::from_fp16_transposed(
                        node.weights.data(), IC, OC);
                    weight_entries.push_back({ node.weight_file, std::move(blob.data) });
                } else {
                    auto blob = mil::WeightBlob::from_fp16(
                        node.weights.data(), node.weights.size());
                    weight_entries.push_back({ node.weight_file, std::move(blob.data) });
                }
            }
        }

        // ── Assemble and compile ──────────────────────────────────────────
        mil::MilProgram prog = mil::MilBuilder::build_fused(fused_inputs, fragments);

        std::string debug = "group_" + std::to_string(cg->groups_.size());
        runtime::AneProgram* program =
            runtime::ane_compile(prog.text, weight_entries, debug);
        if (!program) return nullptr;

        CompiledPlanGroup cpg;
        cpg.node_ids = group.node_ids;
        cpg.program  = program;
        cpg.inputs   = group.inputs;
        cpg.output   = group.output;
        cg->groups_.push_back(std::move(cpg));

        // Pre-allocate a persistent ANE buffer for the group output tensor
        const auto& out_shape = graph.tensor(group.output).shape;
        cg->ane_bufs_[group.output] = global_buffer_pool().acquire_tensor_padded(
            out_shape.channels, out_shape.seq, cg->io_alloc_bytes_);
    }

    return cg;
}

} // namespace graph
} // namespace libane
