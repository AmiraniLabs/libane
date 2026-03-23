/**
 * GraphCompiler implementation.
 */
#include "graph_compiler.hpp"
#include "graph_validator.hpp"

#include <stdexcept>
#include <string>

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
        size_t out_bytes = plan.tensor_bytes.at(group.output);
        cg->ane_bufs_[group.output] = global_buffer_pool().acquire(out_bytes);
    }

    return cg;
}

} // namespace graph
} // namespace libane
