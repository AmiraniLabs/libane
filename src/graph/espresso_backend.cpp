#include "espresso_backend.hpp"
#include "espresso_builder.hpp"
#include "../runtime/ane_runtime.hpp"
#include "../../include/libane.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __APPLE__
#  include <unistd.h>
#endif

namespace libane {
namespace graph {

/* ── fp16 → fp32 helpers ─────────────────────────────────────────────────── */

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 0x1u;
    uint32_t exp      = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;

    uint32_t f;
    if (exp == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; exp--; }
            mantissa &= 0x3FFu;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

// Convert fp16 [IC, OC] → fp32 [OC, IC] (transpose)
static std::vector<float> transpose_fp16_to_fp32(const uint8_t* src, int IC, int OC) {
    std::vector<float> out(static_cast<size_t>(OC) * IC);
    for (int r = 0; r < IC; ++r) {
        for (int c = 0; c < OC; ++c) {
            uint16_t h;
            memcpy(&h, src + (static_cast<size_t>(r) * OC + c) * 2, 2);
            out[static_cast<size_t>(c) * IC + r] = fp16_to_fp32(h);
        }
    }
    return out;
}

/* ── EspressoBackend::owns ───────────────────────────────────────────────── */

bool EspressoBackend::owns(const AneGraph& graph, const FusionGroup& group) const {
    if (!runtime::path_b_available()) return false;
    if (group.node_ids.size() != 1) return false;

    const GraphNode& node = graph.node(group.node_ids[0]);
    if (node.op != LIBANE_OP_MATMUL) return false;

    // Limit to seq == 32 (single ANE tile, pure FC); general matmul stays with MilBackend
    const mil::TensorShape& out_shape = graph.tensor(node.output).shape;
    return out_shape.seq == 32;
}

/* ── EspressoBackend::compile_group ─────────────────────────────────────── */

runtime::AneProgram* EspressoBackend::compile_group(const AneGraph&    graph,
                                                     const FusionGroup& group,
                                                     const std::string& debug_name) {
    const GraphNode& node    = graph.node(group.node_ids[0]);
    const int IC = graph.tensor(node.inputs[0]).shape.channels;
    const int OC = graph.tensor(node.output).shape.channels;

    // Weights in node are fp16 [IC, OC]; EspressoBuilder needs fp32 [OC, IC]
    std::vector<float> weights_fp32;
    const float* weights_ptr = nullptr;
    if (!node.weights.empty()) {
        weights_fp32 = transpose_fp16_to_fp32(node.weights.data(), IC, OC);
        weights_ptr  = weights_fp32.data();
    }

    // Espresso inner_product always uses w=1 (seq=1 in Espresso terms).
    // graph_executor handles the stride-extract from the [IC][seq] graph buffer
    // to the [IC][1] compact buffer expected by ane_execute_client.
    EspressoBuilder builder = EspressoBuilder::inner_product(
        IC, OC, weights_ptr, nullptr, false, 1);

    // Write .mlmodelc to a unique temp directory
#ifdef __APPLE__
    char tmpl[] = "/tmp/libane_esp_XXXXXX";
    char* tmp = mkdtemp(tmpl);
    if (!tmp) {
        return nullptr;
    }
    std::string bundle_dir = std::string(tmp) + ".mlmodelc";
#else
    return nullptr;
#endif

    if (!builder.write(bundle_dir)) {
        return nullptr;
    }

    return runtime::ane_load_mlmodelc(bundle_dir, IC, 1, OC, 1, debug_name);
}

} // namespace graph
} // namespace libane
