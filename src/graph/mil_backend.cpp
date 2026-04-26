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

    case LIBANE_OP_MATMUL_MULTI: {
        // A: [1, K, 1, M]  B: [1, K, 1, N]  → C: [1, N, 1, M]
        // inputs[0] = A (chain var), inputs[1] = B (side input).
        // On macOS 26+ (ios19 NNCompiler) we request the ios19 dialect so the
        // Ios19Backend applies its tiling and fusion passes to the matmul.
        // int8 activation inputs are deferred until the dtype system is extended.
        const auto& b_shape = graph.tensor(node.inputs[1]).shape;
        auto frag = mil::MilBuilder::matmul_multi_fragment(
            in_shape.channels,            // K
            in_shape.seq,                 // M
            b_shape.seq,                  // N
            in_var, tensor_var(node.inputs[1]), out_var,
            /*dtype_int8=*/false);
        if (mil::MilBuilder::os_supports_ios19())
            frag.min_dialect = "ios19";
        return frag;
    }

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

    case LIBANE_OP_EXP:
        return mil::MilBuilder::exp_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SIN:
        return mil::MilBuilder::sin_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_COS:
        return mil::MilBuilder::cos_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_ABS:
        return mil::MilBuilder::abs_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_POW:
        return mil::MilBuilder::pow_fragment(
            out_shape.channels, out_shape.seq,
            in_var, tensor_var(node.inputs[1]), out_var);

    case LIBANE_OP_CEIL:
        return mil::MilBuilder::ceil_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_FLOOR:
        return mil::MilBuilder::floor_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_ROUND:
        return mil::MilBuilder::round_fragment(
            out_shape.channels, out_shape.seq, in_var, out_var);

    case LIBANE_OP_SIGN:
        return mil::MilBuilder::sign_fragment(
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

    case LIBANE_OP_CLIP: {
        float lo = 0.0f, hi = 6.0f;
        if (node.weights.size() == 2 * sizeof(float)) {
            std::memcpy(&lo, node.weights.data(),                   sizeof(float));
            std::memcpy(&hi, node.weights.data() + sizeof(float),   sizeof(float));
        }
        return mil::MilBuilder::clip_fragment(
            out_shape.channels, out_shape.seq, lo, hi, in_var, out_var);
    }

    case LIBANE_OP_PAD: {
        int32_t pad[8] = {0};
        if (node.weights.size() == 8 * sizeof(int32_t)) {
            const int32_t* w = reinterpret_cast<const int32_t*>(node.weights.data());
            for (int i = 0; i < 8; ++i) pad[i] = w[i];
        }
        std::string cfile = tensor_var(node.output) + "_cpad.bin";
        std::string sfile = tensor_var(node.output) + "_spad.bin";
        return mil::MilBuilder::pad_fragment(
            in_shape.channels, in_shape.seq,
            out_shape.channels, out_shape.seq,
            pad[1], pad[3],
            in_var, out_var, cfile, sfile);
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

    case LIBANE_OP_MATMUL_W8A16:
    case LIBANE_OP_MATMUL_W8A8:
        // MIL program is identical to MATMUL — only the weight blob differs.
        // Weight dequantization (int8 × scales → fp16) is done in compile_group().
        // Activation dequantization (for W8A8) is done at execute time.
        return mil::MilBuilder::matmul_fragment(
            in_shape.channels, out_shape.channels, out_shape.seq,
            in_var, out_var, node.weight_file);

    case LIBANE_OP_DYNAMIC_MATMUL: {
        // X=[1,1,K,M]  W=[1,1,N,K]  Y=[1,1,N,M] — K/N/M derived from shapes.
        const mil::TensorShape& x_s = graph.tensor(node.inputs[0]).shape;
        const mil::TensorShape& w_s = graph.tensor(node.inputs[1]).shape;
        int K = x_s.height;
        int M = x_s.seq;
        int N = w_s.height;
        return mil::MilBuilder::dynamic_matmul_fragment(
            K, N, M, in_var, tensor_var(node.inputs[1]), out_var);
    }

    case LIBANE_OP_CONV2D: {
        // Unpack 11 × int32 header from weights blob
        constexpr size_t kParamBytes = 11 * sizeof(int32_t);
        const int32_t* p = reinterpret_cast<const int32_t*>(node.weights.data());
        int kH       = p[0],  kW       = p[1];
        int stride_h = p[2],  stride_w = p[3];
        int pad_top  = p[4],  pad_left = p[5];
        int pad_bot  = p[6],  pad_right= p[7];
        int dil_h    = p[8],  dil_w   = p[9];
        int groups   = p[10];
        (void)kParamBytes;  // used below in compile_group
        return mil::MilBuilder::conv2d_fragment(
            in_shape.channels, out_shape.channels,
            in_shape.height, in_shape.seq,
            kH, kW,
            stride_h, stride_w,
            pad_top, pad_left, pad_bot, pad_right,
            dil_h, dil_w, groups,
            in_var, out_var, node.weight_file);
    }

    case LIBANE_OP_SDPA: {
        // Q/K/V=[1,H,S,D]  mask=[1,1,S,S] — H/S/D derived from shapes.
        int H = in_shape.channels;
        int S = in_shape.height;
        int D = in_shape.seq;
        std::string mask_var = (node.inputs.size() >= 4)
                               ? tensor_var(node.inputs[3]) : "";
        return mil::MilBuilder::sdpa_fragment(
            H, S, D,
            in_var,
            tensor_var(node.inputs[1]),
            tensor_var(node.inputs[2]),
            mask_var,
            out_var);
    }

    case LIBANE_OP_SDPA_GQA: {
        // Q=[1,H_q,S,D]  K/V=[1,H_kv,S,D]  mask=[1,1,S,S] — all derived from shapes.
        int H_q  = in_shape.channels;           // Q channels = num_q_heads
        int S    = in_shape.height;
        int D    = in_shape.seq;
        int H_kv = graph.tensor(node.inputs[1]).shape.channels;  // K channels = num_kv_heads
        std::string mask_var = (node.inputs.size() >= 4)
                               ? tensor_var(node.inputs[3]) : "";
        return mil::MilBuilder::sdpa_gqa_fragment(
            H_q, H_kv, S, D,
            in_var,
            tensor_var(node.inputs[1]),
            tensor_var(node.inputs[2]),
            mask_var,
            out_var);
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
            } else if (node.op == LIBANE_OP_MATMUL_W8A8 ||
                       node.op == LIBANE_OP_MATMUL_W8A16) {
                // Blob layout: int32[2]={OC,IC} + int8[IC×OC] + pad + float32[OC]
                const int32_t* hdr = reinterpret_cast<const int32_t*>(
                    node.weights.data());
                int OC = hdr[0];
                int IC = hdr[1];
                constexpr size_t kHdrBytes = 2 * sizeof(int32_t);
                const int8_t* W_q = reinterpret_cast<const int8_t*>(
                    node.weights.data() + kHdrBytes);
                size_t wbytes      = static_cast<size_t>(IC) * OC;
                size_t scales_off  = (kHdrBytes + wbytes + 3) & ~size_t(3);
                const float* scales = reinterpret_cast<const float*>(
                    node.weights.data() + scales_off);

                // Dequantize int8 → float32: W_fp32[ic, oc] = W_q[ic,oc] × scale[oc]
                // Layout [IC, OC] matches from_fp32(data, IC, OC, true) expectation.
                std::vector<float> W_fp32(static_cast<size_t>(IC) * OC);
                for (int ic = 0; ic < IC; ++ic)
                    for (int oc = 0; oc < OC; ++oc)
                        W_fp32[static_cast<size_t>(ic) * OC + oc] =
                            static_cast<float>(W_q[static_cast<size_t>(ic) * OC + oc])
                            * scales[oc];

                // from_fp32 with transpose=true: [IC,OC] → [OC,IC] for conv1x1
                auto blob = mil::WeightBlob::from_fp32(W_fp32.data(), IC, OC, true);
                weight_entries.push_back({ node.weight_file, std::move(blob.data) });
            } else if (node.op == LIBANE_OP_CONV2D) {
                // Kernel fp16 data starts at byte 44 (after 11 × int32 params).
                // No transpose — user provides [OC, IC/groups, kH, kW] directly.
                constexpr size_t kParamBytes = 11 * sizeof(int32_t);
                auto blob = mil::WeightBlob::from_fp16(
                    node.weights.data() + kParamBytes,
                    node.weights.size() - kParamBytes);
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

        // ── PAD: generate conv projection-matrix weights ──────────────────────
        if (node.op == LIBANE_OP_PAD) {
            int32_t pad[8] = {0};
            if (node.weights.size() == 8 * sizeof(int32_t)) {
                const int32_t* pw =
                    reinterpret_cast<const int32_t*>(node.weights.data());
                for (int i = 0; i < 8; ++i) pad[i] = pw[i];
            }

            int in_C   = graph.tensor(node.inputs[0]).shape.channels;
            int in_SP  = graph.tensor(node.inputs[0]).shape.seq;
            int out_C  = graph.tensor(node.output).shape.channels;
            int out_SP = graph.tensor(node.output).shape.seq;
            std::string ovar = tensor_var(node.output);

            // C-projection matrix [out_C, in_C]:
            //   W[pad_before_C + i, i] = 1.0 for i in [0, in_C); rest = 0.
            if (out_C != in_C) {
                std::vector<uint16_t> w(static_cast<size_t>(out_C) * in_C, 0);
                int before_C = pad[1];
                for (int i = 0; i < in_C; ++i)
                    w[static_cast<size_t>(before_C + i) * in_C + i] = 0x3C00;
                auto blob = mil::WeightBlob::from_fp16(w.data(), w.size() * 2);
                weight_entries.push_back({ ovar + "_cpad.bin",
                                           std::move(blob.data) });
            }

            // S-projection matrix [out_SP, in_SP]:
            //   W[pad_before_S + i, i] = 1.0 for i in [0, in_SP); rest = 0.
            if (out_SP != in_SP) {
                std::vector<uint16_t> w(static_cast<size_t>(out_SP) * in_SP, 0);
                int before_S = pad[3];
                for (int i = 0; i < in_SP; ++i)
                    w[static_cast<size_t>(before_S + i) * in_SP + i] = 0x3C00;
                auto blob = mil::WeightBlob::from_fp16(w.data(), w.size() * 2);
                weight_entries.push_back({ ovar + "_spad.bin",
                                           std::move(blob.data) });
            }
        }
    }

    // ── Assemble MIL program ─────────────────────────────────────────────
    mil::MilProgram prog = mil::MilBuilder::build_fused(fused_inputs, fragments);
    if (getenv("LIBANE_DUMP_MIL")) fprintf(stderr, "=== MIL ===\n%s\n", prog.text.c_str());

    // ── Warm path: try reconnect before paying cold-compile cost ─────────
    if (auto* warm = try_warm_reconnect(prog.text, weight_entries, debug_name))
        return warm;

    // ── Cold path: full compile; cache model_url + hexID for next time ───
    runtime::AneProgram* program = runtime::ane_compile(prog.text, weight_entries, debug_name);
    if (program) cache_populate(program);
    return program;
}

runtime::AneProgram* MilBackend::try_warm_reconnect(
    const std::string&                              mil_text,
    const std::vector<runtime::WeightEntry>&        weights,
    const std::string&                              debug_name) {
    // Compute hexID without compiling — this is aned's own equivalence
    // class for (mil_text, weights), so the lookup is exact.
    std::string hex_id = runtime::ane_compute_hex_id(mil_text, weights);
    if (hex_id.empty()) { misses_.fetch_add(1, std::memory_order_relaxed); return nullptr; }

    auto map_it = url_cache_.find(hex_id);
    if (map_it == url_cache_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    LruIterator lru_it = map_it->second;

    // Cache hit — attempt reconnect.  Returns nullptr if aned's compile
    // slot was purged (compiledModelExists=NO); caller falls through to
    // cold compile.
    const auto& cached_weights = lru_it->weights
        ? *lru_it->weights
        : std::vector<runtime::WeightEntry>{};
    runtime::AneProgram* prog = runtime::ane_reconnect(
        lru_it->mil_text, cached_weights, lru_it->model_url, debug_name);
    if (!prog) {
        // Slot purged — stale entry.  Drop it so the cold compile that
        // follows can refresh the URL on cache_populate().
        lru_.erase(lru_it);
        url_cache_.erase(map_it);
        evictions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // Touch to front — most-recently-used.
    lru_.splice(lru_.begin(), lru_, lru_it);
    hits_.fetch_add(1, std::memory_order_relaxed);
    return prog;
}

void MilBackend::cache_populate(const runtime::AneProgram* prog) {
    if (!prog || prog->hex_id.empty() || prog->model_url.empty()) return;

    auto existing = url_cache_.find(prog->hex_id);
    if (existing != url_cache_.end()) {
        // Refresh model_url (aned may have re-issued the URL after a purge
        // and recompile) and touch to front.
        existing->second->model_url = prog->model_url;
        existing->second->mil_text  = prog->mil_text;
        existing->second->weights   = prog->weights;
        lru_.splice(lru_.begin(), lru_, existing->second);
        cold_compiles_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Insert at front.  prog->weights is a shared_ptr; we store a copy of
    // the shared_ptr (refcount bump, not a buffer copy) so program and
    // cache point at the same underlying vector<WeightEntry>.
    lru_.push_front(UrlCacheEntry{
        prog->hex_id, prog->model_url, prog->mil_text, prog->weights});
    url_cache_[prog->hex_id] = lru_.begin();
    cold_compiles_.fetch_add(1, std::memory_order_relaxed);

    // Enforce capacity — drop LRU tail if we overflowed.
    while (lru_.size() > cache_capacity_) {
        evict_lru_tail();
    }
}

void MilBackend::evict_lru_tail() {
    if (lru_.empty()) return;
    auto& tail = lru_.back();
    url_cache_.erase(tail.hex_id);
    lru_.pop_back();
    lru_evictions_.fetch_add(1, std::memory_order_relaxed);
}

size_t MilBackend::entry_bytes(const UrlCacheEntry& e) {
    size_t b = e.hex_id.size() + e.model_url.size() + e.mil_text.size();
    if (e.weights) {
        for (const auto& w : *e.weights) b += w.filename.size() + w.data.size();
    }
    return b;
}

MilBackendCacheStats MilBackend::cache_stats() const {
    MilBackendCacheStats s;
    s.entries       = lru_.size();
    s.capacity      = cache_capacity_;
    for (const auto& e : lru_) s.bytes += entry_bytes(e);
    s.hits          = hits_.load(std::memory_order_relaxed);
    s.misses        = misses_.load(std::memory_order_relaxed);
    s.cold_compiles = cold_compiles_.load(std::memory_order_relaxed);
    s.evictions     = evictions_.load(std::memory_order_relaxed);
    s.lru_evictions = lru_evictions_.load(std::memory_order_relaxed);
    return s;
}

void MilBackend::cache_clear() {
    lru_.clear();
    url_cache_.clear();
    // Counters are preserved so callers can reason about cumulative
    // activity across clear() cycles.
}

void MilBackend::set_cache_capacity(size_t capacity) {
    cache_capacity_ = capacity;
    while (lru_.size() > cache_capacity_) {
        evict_lru_tail();
    }
}

size_t MilBackend::cache_prune(size_t max_entries) {
    if (max_entries == 0) max_entries = cache_capacity_;
    size_t evicted = 0;
    while (lru_.size() > max_entries) {
        evict_lru_tail();
        ++evicted;
    }
    return evicted;
}

} // namespace graph
} // namespace libane
