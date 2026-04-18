/**
 * MilBackend implementation.
 *
 * Contains everything that is specific to the MIL → ANE compilation path:
 *   - tensor_var()        canonical MIL variable name for a tensor ID
 *   - node_to_fragment()  maps a GraphNode to the appropriate MilFragment
 *   - compile_group()     assembles fragments, builds weights, calls ane_compile
 */
#include "mil_backend.hpp"
#include "../core/mil_builder.hpp"
#include "../../include/libane.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace libane {
namespace graph {

/* ── Local helpers ───────────────────────────────────────────────────────── */

static std::string tensor_var(TensorId id) {
    return "t" + std::to_string(id);
}

static mil::MilFragment node_to_fragment(const AneGraph&    graph,
                                          const GraphNode&   node,
                                          const std::string& in_var,
                                          const std::string& out_var) {
    const mil::TensorShape& in_shape =
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

    case LIBANE_OP_RESHAPE:
        return mil::MilBuilder::reshape_fragment(
            in_shape.channels, in_shape.seq,
            out_shape.channels, out_shape.seq,
            in_var, out_var);

    case LIBANE_OP_ADD:
        return mil::MilBuilder::add_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_MUL:
        return mil::MilBuilder::mul_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_SUB:
        return mil::MilBuilder::sub_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_REAL_DIV:
        return mil::MilBuilder::real_div_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_LOGICAL_AND:
        return mil::MilBuilder::logical_and_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_LOGICAL_OR:
        return mil::MilBuilder::logical_or_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_LOGICAL_XOR:
        return mil::MilBuilder::logical_xor_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_REDUCE_PROD:
        return mil::MilBuilder::reduce_prod_fragment(
            in_shape.channels, in_shape.seq, in_var, out_var);

    case LIBANE_OP_REDUCE_SUM:
        return mil::MilBuilder::reduce_sum_fragment(
            in_shape.channels, in_shape.seq, in_var, out_var);

    case LIBANE_OP_REDUCE_MEAN:
        return mil::MilBuilder::reduce_mean_fragment(
            in_shape.channels, in_shape.seq, in_var, out_var);

    case LIBANE_OP_REDUCE_MAX:
        return mil::MilBuilder::reduce_max_fragment(
            in_shape.channels, in_shape.seq, in_var, out_var);

    case LIBANE_OP_SCATTER:
    case LIBANE_OP_SCATTER_ND:
    case LIBANE_OP_SCATTER_ALONG_AXIS:
        return mil::MilBuilder::scatter_static_mask_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var, node.weight_file);

    case LIBANE_OP_GATHER:
        if (node.inputs.size() == 1) {
            return mil::MilBuilder::gather_static_mask_fragment(
                out_shape.channels, out_shape.seq,
                in_var, out_var, node.weight_file);
        }
        return mil::MilBuilder::gather_dynamic_mask_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_NEG:
        return mil::MilBuilder::neg_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_MOD:
        return mil::MilBuilder::mod_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_SINH:
        return mil::MilBuilder::sinh_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_COSH:
        return mil::MilBuilder::cosh_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_TAN:
        return mil::MilBuilder::tan_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_ASIN:
        return mil::MilBuilder::asin_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_ACOS:
        return mil::MilBuilder::acos_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SELECT:
        return mil::MilBuilder::select_fragment(
            out_shape.channels, out_shape.seq,
            in_var,
            tensor_var(node.inputs[1]),
            tensor_var(node.inputs[2]),
            out_var);

    case LIBANE_OP_CONCAT:
        return mil::MilBuilder::concat_fragment(
            in_shape.channels, out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_SLICE_BY_INDEX:
        return mil::MilBuilder::slice_by_index_fragment(
            in_shape.channels, in_shape.seq,
            out_shape.channels, out_shape.seq,
            in_var, out_var);

    case LIBANE_OP_SLICE: {
        int32_t begin[4]  = {0, 0, 0, 0};
        int32_t stride[4] = {1, 1, 1, 1};
        if (node.weights.size() == 8 * sizeof(int32_t)) {
            const int32_t* w = reinterpret_cast<const int32_t*>(node.weights.data());
            for (int i = 0; i < 4; ++i) begin[i]  = w[i];
            for (int i = 0; i < 4; ++i) stride[i] = w[4 + i];
        }
        return mil::MilBuilder::slice_fragment(
            in_shape.channels, in_shape.seq,
            out_shape.channels, out_shape.seq,
            begin, stride, in_var, out_var);
    }

    case LIBANE_OP_SQRT:
        return mil::MilBuilder::sqrt_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_LOG:
        return mil::MilBuilder::log_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_RSQRT:
        return mil::MilBuilder::rsqrt_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_RELU:
        return mil::MilBuilder::relu_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_TANH:
        return mil::MilBuilder::tanh_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SIGMOID:
        return mil::MilBuilder::sigmoid_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_HARDSWISH:
        return mil::MilBuilder::hardswish_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_LEAKY_RELU:
        return mil::MilBuilder::leaky_relu_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_ELU:
        return mil::MilBuilder::elu_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_PIXEL_SHUFFLE: {
        int32_t r = 0;
        if (node.weights.size() == 4)
            std::memcpy(&r, node.weights.data(), 4);
        int out_C = out_shape.channels;
        int in_SP = in_shape.seq;
        return mil::MilBuilder::pixel_shuffle_fragment(out_C, in_SP, r, in_var, out_var);
    }

    case LIBANE_OP_PWL_ACTIVATION: {
        // weights layout: [x_min, x_max, samples...] as float32
        const auto& w = node.weights;
        int n_floats = static_cast<int>(w.size()) / 4;
        std::vector<float> fv(n_floats);
        std::memcpy(fv.data(), w.data(), w.size());
        float x_min   = fv[0];
        float x_max   = fv[1];
        int n_samples = n_floats - 2;
        return mil::MilBuilder::pwl_activation_fragment(
            out_shape.channels, out_shape.seq,
            x_min, x_max, fv.data() + 2, n_samples,
            in_var, out_var);
    }

    default:
        throw std::runtime_error(
            "MilBackend::node_to_fragment: unsupported op " +
            std::to_string(node.op));
    }
}

/* ── MilBackend::owns ────────────────────────────────────────────────────── */

bool MilBackend::owns(const AneGraph&, const FusionGroup&) const {
    return true;  // catch-all — must be last in any priority list
}

/* ── MilBackend::compile_group ───────────────────────────────────────────── */

runtime::AneProgram* MilBackend::compile_group(const AneGraph&    graph,
                                                const FusionGroup& group,
                                                const std::string& debug_name) {
    // ── Collect fused inputs (chain + side) ───────────────────────────────
    std::vector<mil::FusedInput> fused_inputs;
    {
        TensorId cid = group.inputs[0];
        fused_inputs.push_back({ tensor_var(cid), graph.tensor(cid).shape });
    }
    for (size_t i = 1; i < group.inputs.size(); ++i) {
        TensorId sid = group.inputs[i];
        fused_inputs.push_back({ tensor_var(sid), graph.tensor(sid).shape });
    }

    // ── Build fragments and weight entries ────────────────────────────────
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

        // ── SLICE / SLICE_BY_INDEX: generate conv selection-matrix weights ────
        if (node.op == LIBANE_OP_SLICE || node.op == LIBANE_OP_SLICE_BY_INDEX) {
            int32_t begin[4]  = {0, 0, 0, 0};
            int32_t stride[4] = {1, 1, 1, 1};
            if (node.op == LIBANE_OP_SLICE &&
                node.weights.size() == 8 * sizeof(int32_t)) {
                const int32_t* w =
                    reinterpret_cast<const int32_t*>(node.weights.data());
                for (int i = 0; i < 4; ++i) begin[i]  = w[i];
                for (int i = 0; i < 4; ++i) stride[i] = w[4 + i];
            }

            int in_C   = graph.tensor(node.inputs[0]).shape.channels;
            int in_SP  = graph.tensor(node.inputs[0]).shape.seq;
            int out_C  = graph.tensor(node.output).shape.channels;
            int out_SP = graph.tensor(node.output).shape.seq;
            std::string ovar = tensor_var(node.output);

            // C-selection matrix [out_C, in_C]: W[i, begin_C + i*stride_C] = 1.0
            if (out_C != in_C || begin[1] != 0) {
                std::vector<uint16_t> w(static_cast<size_t>(out_C) * in_C, 0);
                for (int i = 0; i < out_C; ++i)
                    w[static_cast<size_t>(i) * in_C + begin[1] + i * stride[1]]
                        = 0x3C00; // fp16 1.0
                auto blob = mil::WeightBlob::from_fp16(w.data(), w.size() * 2);
                weight_entries.push_back({ ovar + "_csel.bin",
                                           std::move(blob.data) });
            }

            // S-selection matrix [out_SP, in_SP]: W[i, begin_SP + i*stride_SP] = 1.0
            if (out_SP != in_SP || begin[3] != 0) {
                std::vector<uint16_t> w(static_cast<size_t>(out_SP) * in_SP, 0);
                for (int i = 0; i < out_SP; ++i)
                    w[static_cast<size_t>(i) * in_SP + begin[3] + i * stride[3]]
                        = 0x3C00;
                auto blob = mil::WeightBlob::from_fp16(w.data(), w.size() * 2);
                weight_entries.push_back({ ovar + "_ssel.bin",
                                           std::move(blob.data) });
            }
        }
    }

    // ── Assemble MIL program and compile ─────────────────────────────────
    mil::MilProgram prog = mil::MilBuilder::build_fused(fused_inputs, fragments);
    if (getenv("LIBANE_DUMP_MIL")) fprintf(stderr, "=== MIL ===\n%s\n", prog.text.c_str());
    return runtime::ane_compile(prog.text, weight_entries, debug_name);
}

} // namespace graph
} // namespace libane
