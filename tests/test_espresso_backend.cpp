/**
 * EspressoBackend hardware dispatch tests.
 *
 * Validates that ane_execute_client() dispatches via _ANEClient and produces
 * numerically correct output.  All tests skip gracefully when Path B symbols
 * are absent (non-Apple Silicon or older firmware).
 *
 * ── Architecture ─────────────────────────────────────────────────────────────
 *
 * EspressoBackend owns single-node MATMUL groups with seq == 16 (the minimum
 * valid ANE seq tile).  The Espresso inner_product model is always compiled with
 * w=1 (single vector).  graph_executor stride-extracts column 0 from the
 * [IC][16] graph buffer → [IC][1] compact buffer → ane_execute_client → result
 * written back to column 0 of the [OC][16] output buffer (columns 1–15 = 0).
 *
 * This makes EspressoBackend correct for single-token decode (seq=16, real data
 * at position 0, positions 1–15 are padding zeros).
 *
 * ── Hardware proof method ─────────────────────────────────────────────────────
 *
 * Known-weight inner product.  If _ANEClient dispatch is broken (wrong
 * IOSurface scatter/gather, stride mismatch, or cacheInference path failure),
 * the output will be all-zero, garbage, or an explicit execution failure.
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"
#include "graph/espresso_backend.hpp"
#include "runtime/ane_runtime.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace libane;
using namespace libane::graph;
using namespace libane::mil;

/* ── fp16 helpers ────────────────────────────────────────────────────────── */

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
using fp16 = __fp16;
static fp16  to_f16(float v) { return static_cast<fp16>(v); }
static float to_f32(fp16 v)  { return static_cast<float>(v); }
#else
using fp16 = uint16_t;
static fp16 to_f16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FF;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00);
    else              h = static_cast<uint16_t>(s | (e << 10) | m);
    return h;
}
static float to_f32(fp16 h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0)  { uint32_t v = s | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    if (e == 31) { uint32_t v = s | 0x7F800000 | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    uint32_t v = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &v, 4); return f;
}
#endif

static TensorShape S(int C, int seq) { return {1, C, 1, seq}; }

/**
 * Build fp16 weight bytes in [IC, OC] row-major layout.
 * W[i, j] = val for all (i, j).
 */
static std::vector<uint8_t> uniform_weight_bytes(int IC, int OC, float val) {
    std::vector<uint8_t> w(static_cast<size_t>(IC) * OC * 2);
    fp16 h = to_f16(val);
    for (size_t k = 0; k < static_cast<size_t>(IC) * OC; ++k)
        std::memcpy(w.data() + k * 2, &h, 2);
    return w;
}

/**
 * Build fp16 weight bytes for the identity projection W[IC, OC] (row-major [IC, OC]).
 * W[i, j] = 1.0 if i == j and i < OC, else 0.0.
 * Effect: output[j, 0] = input[j, 0] for all j < OC.
 */
static std::vector<uint8_t> identity_weight_bytes(int IC, int OC) {
    std::vector<uint8_t> w(static_cast<size_t>(IC) * OC * 2, 0);
    fp16 one = to_f16(1.0f);
    for (int i = 0; i < IC && i < OC; ++i) {
        size_t idx = static_cast<size_t>(i) * OC + i;
        std::memcpy(w.data() + idx * 2, &one, 2);
    }
    return w;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

// SEQ = 16: minimum valid ANE seq tile (required by TensorShape::validate).
// EspressoBackend processes only column 0 (single-token decode semantics).
// Columns 1–15 are padding zeros and should not appear in the output.

TEST_CASE("EspressoBackend dispatch: zero-weight FC produces zero at seq[0]", "[espresso][ane]") {
    if (!libane_available())             SKIP("ANE not available");
    if (!runtime::path_b_available())   SKIP("Path B (_ANEClient) not available");

    const int IC = 64, OC = 32, SEQ = 16;
    const size_t N_in  = static_cast<size_t>(IC)  * SEQ;
    const size_t N_out = static_cast<size_t>(OC) * SEQ;

    auto w = uniform_weight_bytes(IC, OC, 0.0f);

    AneGraph g;
    TensorId x   = g.add_input("x", S(IC, SEQ));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SEQ), w.data(), w.size());
    g.mark_output(out);

    EspressoBackend espresso;
    auto cg = GraphCompiler::compile(g, espresso);
    if (!cg) SKIP("EspressoBackend compilation unavailable in this environment");

    // Data at seq[0] only; seq[1..15] = 0 padding.
    std::vector<fp16> in_data(N_in, to_f16(0.0f));
    for (int c = 0; c < IC; ++c)
        in_data[static_cast<size_t>(c) * SEQ + 0] = to_f16(1.0f);

    std::vector<fp16> out_data(N_out, to_f16(99.0f));

    bool ok = GraphExecutor::execute(*cg,
                                      {in_data.data()}, {N_in  * sizeof(fp16)},
                                      {out_data.data()}, {N_out * sizeof(fp16)});
    REQUIRE(ok);

    // All zeros: zero weights → zero output at all positions
    for (size_t i = 0; i < N_out; ++i)
        CHECK(std::abs(to_f32(out_data[i])) < 0.01f);
}

TEST_CASE("EspressoBackend dispatch: identity-weight FC passes through seq[0]", "[espresso][ane]") {
    if (!libane_available())             SKIP("ANE not available");
    if (!runtime::path_b_available())   SKIP("Path B (_ANEClient) not available");

    // W[IC, OC] with W[i,i] = 1.0 → output[j, 0] = input[j, 0] for j < OC.
    // Proves: weights are loaded, IOSurface scatter/gather is correct,
    // and the seq[0] column is extracted and written back accurately.
    const int IC = 64, OC = 32, SEQ = 16;
    const size_t N_in  = static_cast<size_t>(IC)  * SEQ;
    const size_t N_out = static_cast<size_t>(OC) * SEQ;

    auto w = identity_weight_bytes(IC, OC);

    AneGraph g;
    TensorId x   = g.add_input("x", S(IC, SEQ));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SEQ), w.data(), w.size());
    g.mark_output(out);

    EspressoBackend espresso;
    auto cg = GraphCompiler::compile(g, espresso);
    if (!cg) SKIP("EspressoBackend compilation unavailable in this environment");

    // Fill seq[0] with distinct values; seq[1..15] = 0.
    std::vector<fp16> in_data(N_in, to_f16(0.0f));
    for (int c = 0; c < IC; ++c)
        in_data[static_cast<size_t>(c) * SEQ + 0] =
            to_f16(static_cast<float>(c + 1) * 0.0625f);  // 1/16, 2/16, ...

    std::vector<fp16> out_data(N_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
                                      {in_data.data()}, {N_in  * sizeof(fp16)},
                                      {out_data.data()}, {N_out * sizeof(fp16)});
    REQUIRE(ok);

    // output[j][0] must match input[j][0] (identity projection at seq=0)
    for (int j = 0; j < OC; ++j) {
        float got      = to_f32(out_data[static_cast<size_t>(j) * SEQ + 0]);
        float expected = to_f32(in_data[static_cast<size_t>(j) * SEQ + 0]);
        CHECK(std::abs(got - expected) <= 0.01f + 0.05f * std::abs(expected));
    }

    // Positions 1–15 of output should be zero (stride-insert writes only pos 0)
    for (int j = 0; j < OC; ++j)
        for (int s = 1; s < SEQ; ++s)
            CHECK(std::abs(to_f32(out_data[static_cast<size_t>(j) * SEQ + s])) < 0.01f);
}

TEST_CASE("EspressoBackend warm path: cacheInference:YES re-execute produces correct output", "[espresso][ane]") {
    if (!libane_available())             SKIP("ANE not available");
    if (!runtime::path_b_available())   SKIP("Path B (_ANEClient) not available");

    // Uniform weights (1/IC): output[j, 0] = mean(input[:, 0]) = 1.0 when input all = 1.0
    const int IC = 64, OC = 32, SEQ = 16;
    const size_t N_in  = static_cast<size_t>(IC)  * SEQ;
    const size_t N_out = static_cast<size_t>(OC) * SEQ;
    const float w_val = 1.0f / static_cast<float>(IC);

    auto w = uniform_weight_bytes(IC, OC, w_val);

    AneGraph g;
    TensorId x   = g.add_input("x", S(IC, SEQ));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SEQ), w.data(), w.size());
    g.mark_output(out);

    EspressoBackend espresso;
    auto cg = GraphCompiler::compile(g, espresso);
    if (!cg) SKIP("EspressoBackend compilation unavailable in this environment");

    std::vector<fp16> in_data(N_in, to_f16(0.0f));
    for (int c = 0; c < IC; ++c)
        in_data[static_cast<size_t>(c) * SEQ + 0] = to_f16(1.0f);

    std::vector<fp16> out_data(N_out, to_f16(0.0f));

    // First execute: cold path — creates _ANEIOSurfaceObject wrappers, _ANERequest,
    // calls mapIOSurfaces:cacheInference:YES to register with the ANE kernel.
    bool ok = GraphExecutor::execute(*cg,
                                      {in_data.data()}, {N_in  * sizeof(fp16)},
                                      {out_data.data()}, {N_out * sizeof(fp16)});
    REQUIRE(ok);

    for (int j = 0; j < OC; ++j)
        CHECK(std::abs(to_f32(out_data[static_cast<size_t>(j) * SEQ]) - 1.0f) < 0.1f);

    // Second execute: warm path — client_mapped == true, dispatches via fastConn.
    std::fill(out_data.begin(), out_data.end(), to_f16(0.0f));
    ok = GraphExecutor::execute(*cg,
                                 {in_data.data()}, {N_in  * sizeof(fp16)},
                                 {out_data.data()}, {N_out * sizeof(fp16)});
    REQUIRE(ok);

    for (int j = 0; j < OC; ++j)
        CHECK(std::abs(to_f32(out_data[static_cast<size_t>(j) * SEQ]) - 1.0f) < 0.1f);
}

TEST_CASE("EspressoBackend router: seq==16 matmul routes end-to-end via default compiler", "[espresso][ane]") {
    if (!libane_available())             SKIP("ANE not available");
    if (!runtime::path_b_available())   SKIP("Path B (_ANEClient) not available");

    // Default router (HwxBackend → EspressoBackend → MilBackend):
    // seq==16 MATMUL must be claimed by EspressoBackend with correct output.
    const int IC = 64, OC = 32, SEQ = 16;
    const size_t N_in  = static_cast<size_t>(IC)  * SEQ;
    const size_t N_out = static_cast<size_t>(OC) * SEQ;

    auto w = identity_weight_bytes(IC, OC);

    AneGraph g;
    TensorId x   = g.add_input("x", S(IC, SEQ));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SEQ), w.data(), w.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);  // default router
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    std::vector<fp16> in_data(N_in, to_f16(0.0f));
    for (int c = 0; c < IC; ++c)
        in_data[static_cast<size_t>(c) * SEQ + 0] =
            to_f16(static_cast<float>(c + 1) * 0.0625f);

    std::vector<fp16> out_data(N_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
                                      {in_data.data()}, {N_in  * sizeof(fp16)},
                                      {out_data.data()}, {N_out * sizeof(fp16)});
    REQUIRE(ok);

    for (int j = 0; j < OC; ++j) {
        float got      = to_f32(out_data[static_cast<size_t>(j) * SEQ + 0]);
        float expected = to_f32(in_data[static_cast<size_t>(j) * SEQ + 0]);
        CHECK(std::abs(got - expected) <= 0.01f + 0.05f * std::abs(expected));
    }
}
