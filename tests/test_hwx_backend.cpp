/**
 * HwxBackend integration tests.
 *
 * Verifies that the warm path (HwxEmitter → ane_load_hwx) dispatches to ANE
 * hardware and produces numerically correct output.
 *
 * Hardware-execution proof method: precision fingerprinting.
 * The ANE implements tanh via a fixed-point LUT with linear interpolation.
 * At small inputs the LUT interpolation error produces bit-level different
 * fp16 results from the CPU's minimax polynomial approximation.
 *
 * Verified bit patterns (ANE_5_0_EVIDENCE.md, M5 macOS 26.3.1):
 *   x=0.1 → ANE fp16: 0x2E5D (0.099426)   CPU fp16: 0x2E61 (0.099670)
 *   x=0.3 → ANE fp16: 0x34A5 (0.290283)   CPU fp16: 0x34A9 (0.291260)
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"
#include "graph/hwx_backend.hpp"

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

/* ── Tests ───────────────────────────────────────────────────────────────── */

TEST_CASE("HwxBackend warm path produces ANE LUT fingerprint for tanh", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // C=64, S=512: 64*512*2 = 65536 bytes (> 49 KB IOSurface minimum)
    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId x   = g.add_input("x", S(C, SP));
    TensorId out = g.add_op(LIBANE_OP_TANH, {x}, S(C, SP));
    g.mark_output(out);

    HwxBackend hwx;

    // Cold compile: MilBackend path inside HwxBackend; seeds emitter cache
    auto cg_cold = GraphCompiler::compile(g, hwx);
    if (!cg_cold) SKIP("ANE compiler unavailable in this environment");

    // Warm compile: HwxEmitter emits patched HWX without recompilation
    auto cg_warm = GraphCompiler::compile(g, hwx);
    REQUIRE(cg_warm != nullptr);

    std::vector<fp16> in_data(N, to_f16(0.0f));
    std::vector<fp16> out_data(N, to_f16(0.0f));

    // Probe values at positions [c=0,s=0] and [c=0,s=1]
    // ANE LUT diverges from CPU polynomial at small inputs
    in_data[0] = to_f16(0.1f);   // x=0.1: ANE 0x2E5D, CPU 0x2E61
    in_data[1] = to_f16(0.3f);   // x=0.3: ANE 0x34A5, CPU 0x34A9

    bool ok = GraphExecutor::execute(*cg_warm,
                                      {in_data.data()},  {N * sizeof(fp16)},
                                      {out_data.data()}, {N * sizeof(fp16)});
    REQUIRE(ok);

    uint16_t bits0, bits1;
    std::memcpy(&bits0, &out_data[0], 2);
    std::memcpy(&bits1, &out_data[1], 2);

    // ANE LUT fingerprint: must match hardware values
    CHECK(bits0 == 0x2E5Du);   // tanh(0.1) via ANE LUT
    CHECK(bits1 == 0x34A5u);   // tanh(0.3) via ANE LUT

    // Must not match CPU polynomial values
    CHECK(bits0 != 0x2E61u);
    CHECK(bits1 != 0x34A9u);
}

TEST_CASE("HwxBackend warm path cross-op: tanh bootstrap enables sigmoid warm load", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;

    HwxBackend hwx;

    // Step 1: bootstrap tanh — seeds shape template and tanh op config
    {
        AneGraph g;
        TensorId x = g.add_input("x", S(C, SP));
        g.mark_output(g.add_op(LIBANE_OP_TANH, {x}, S(C, SP)));
        auto cg = GraphCompiler::compile(g, hwx);
        if (!cg) SKIP("ANE compiler unavailable in this environment");
    }

    // Step 2: bootstrap sigmoid — seeds sigmoid op config (shape template already present)
    {
        AneGraph g;
        TensorId x = g.add_input("x", S(C, SP));
        g.mark_output(g.add_op(LIBANE_OP_SIGMOID, {x}, S(C, SP)));
        auto cg = GraphCompiler::compile(g, hwx);
        REQUIRE(cg != nullptr);
    }

    // Step 3: warm sigmoid — HwxEmitter patches tanh template with sigmoid op config
    AneGraph g;
    TensorId x   = g.add_input("x", S(C, SP));
    TensorId out = g.add_op(LIBANE_OP_SIGMOID, {x}, S(C, SP));
    g.mark_output(out);

    auto cg_warm = GraphCompiler::compile(g, hwx);
    REQUIRE(cg_warm != nullptr);

    std::vector<fp16> in_data(N, to_f16(0.0f));
    std::vector<fp16> out_data(N, to_f16(0.0f));

    in_data[0] = to_f16(0.0f);   // sigmoid(0) = 0.5 exactly
    in_data[1] = to_f16(1.0f);   // sigmoid(1) ≈ 0.731

    bool ok = GraphExecutor::execute(*cg_warm,
                                      {in_data.data()},  {N * sizeof(fp16)},
                                      {out_data.data()}, {N * sizeof(fp16)});
    REQUIRE(ok);

    // sigmoid(0) = 0.5 — unambiguous at any precision
    float v0 = to_f32(out_data[0]);
    CHECK(std::abs(v0 - 0.5f) < 0.01f);

    // sigmoid(1) ≈ 0.731
    float v1 = to_f32(out_data[1]);
    CHECK(v1 > 0.70f);
    CHECK(v1 < 0.76f);
}

// ── Helper: bootstrap TANH then warm-compile a target op ─────────────────
//
// All four remaining op tests use the same pattern:
//   1. Bootstrap TANH → seeds shape template + TANH op config.
//   2. Bootstrap target op → seeds target op config (may have different
//      num_words than TANH, exercising the patch_op resize path).
//   3. Warm compile of target op → patches TANH template with target config.
//   4. Execute and verify numerical output.

static std::unique_ptr<CompiledGraph> warm_compile(HwxBackend& hwx,
                                                    libane_op_t target_op,
                                                    int C, int SP) {
    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    g.mark_output(g.add_op(target_op, {x}, S(C, SP)));
    return GraphCompiler::compile(g, hwx);
}

TEST_CASE("HwxBackend warm path: relu", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;
    HwxBackend hwx;

    // Bootstrap TANH (shape template) then RELU (op config)
    { auto cg = warm_compile(hwx, LIBANE_OP_TANH, C, SP); if (!cg) SKIP("ANE compiler unavailable"); }
    { auto cg = warm_compile(hwx, LIBANE_OP_RELU, C, SP); REQUIRE(cg != nullptr); }

    // Warm RELU compile
    auto cg = warm_compile(hwx, LIBANE_OP_RELU, C, SP);
    REQUIRE(cg != nullptr);

    std::vector<fp16> in(N), out(N, to_f16(0.0f));
    in[0] = to_f16(-2.0f);   // relu(-2) = 0
    in[1] = to_f16( 0.0f);   // relu(0)  = 0
    in[2] = to_f16( 1.5f);   // relu(1.5) = 1.5
    in[3] = to_f16( 3.0f);   // relu(3)  = 3
    for (size_t i = 4; i < N; ++i) in[i] = to_f16(0.0f);

    bool ok = GraphExecutor::execute(*cg, {in.data()}, {N*sizeof(fp16)}, {out.data()}, {N*sizeof(fp16)});
    REQUIRE(ok);

    CHECK(std::abs(to_f32(out[0])) < 0.01f);
    CHECK(std::abs(to_f32(out[1])) < 0.01f);
    CHECK(std::abs(to_f32(out[2]) - 1.5f) < 0.05f);
    CHECK(std::abs(to_f32(out[3]) - 3.0f) < 0.05f);
}

TEST_CASE("HwxBackend warm path: leaky_relu (alpha=0.01)", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;
    HwxBackend hwx;

    { auto cg = warm_compile(hwx, LIBANE_OP_TANH,       C, SP); if (!cg) SKIP("ANE compiler unavailable"); }
    { auto cg = warm_compile(hwx, LIBANE_OP_LEAKY_RELU, C, SP); REQUIRE(cg != nullptr); }

    auto cg = warm_compile(hwx, LIBANE_OP_LEAKY_RELU, C, SP);
    REQUIRE(cg != nullptr);

    std::vector<fp16> in(N), out(N, to_f16(0.0f));
    in[0] = to_f16(-2.0f);   // leaky_relu(-2) = -0.02
    in[1] = to_f16( 0.0f);   // leaky_relu(0)  =  0.0
    in[2] = to_f16( 2.0f);   // leaky_relu(2)  =  2.0
    for (size_t i = 3; i < N; ++i) in[i] = to_f16(0.0f);

    bool ok = GraphExecutor::execute(*cg, {in.data()}, {N*sizeof(fp16)}, {out.data()}, {N*sizeof(fp16)});
    REQUIRE(ok);

    CHECK(std::abs(to_f32(out[0]) - (-0.02f)) < 0.005f);
    CHECK(std::abs(to_f32(out[1])) < 0.01f);
    CHECK(std::abs(to_f32(out[2]) - 2.0f) < 0.05f);
}

TEST_CASE("HwxBackend warm path: elu (alpha=1.0)", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;
    HwxBackend hwx;

    { auto cg = warm_compile(hwx, LIBANE_OP_TANH, C, SP); if (!cg) SKIP("ANE compiler unavailable"); }
    { auto cg = warm_compile(hwx, LIBANE_OP_ELU,  C, SP); REQUIRE(cg != nullptr); }

    auto cg = warm_compile(hwx, LIBANE_OP_ELU, C, SP);
    REQUIRE(cg != nullptr);

    std::vector<fp16> in(N), out(N, to_f16(0.0f));
    in[0] = to_f16( 1.0f);   // elu(1)  = 1.0
    in[1] = to_f16( 0.0f);   // elu(0)  = 0.0
    in[2] = to_f16(-1.0f);   // elu(-1) = exp(-1)-1 ≈ -0.632
    for (size_t i = 3; i < N; ++i) in[i] = to_f16(0.0f);

    bool ok = GraphExecutor::execute(*cg, {in.data()}, {N*sizeof(fp16)}, {out.data()}, {N*sizeof(fp16)});
    REQUIRE(ok);

    CHECK(std::abs(to_f32(out[0]) - 1.0f)   < 0.05f);
    CHECK(std::abs(to_f32(out[1]))           < 0.01f);
    CHECK(std::abs(to_f32(out[2]) - (-0.632f)) < 0.05f);
}

TEST_CASE("HwxBackend warm path: hardswish", "[hwx][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 512;
    const size_t N = static_cast<size_t>(C) * SP;
    HwxBackend hwx;

    { auto cg = warm_compile(hwx, LIBANE_OP_TANH,      C, SP); if (!cg) SKIP("ANE compiler unavailable"); }
    { auto cg = warm_compile(hwx, LIBANE_OP_HARDSWISH, C, SP); REQUIRE(cg != nullptr); }

    auto cg = warm_compile(hwx, LIBANE_OP_HARDSWISH, C, SP);
    REQUIRE(cg != nullptr);

    std::vector<fp16> in(N), out(N, to_f16(0.0f));
    in[0] = to_f16(-4.0f);   // hardswish(-4) = 0   (x < -3)
    in[1] = to_f16( 0.0f);   // hardswish(0)  = 0   (0*(0+3)/6)
    in[2] = to_f16( 1.0f);   // hardswish(1)  = 4/6 ≈ 0.667
    in[3] = to_f16( 4.0f);   // hardswish(4)  = 4   (x > 3)
    for (size_t i = 4; i < N; ++i) in[i] = to_f16(0.0f);

    bool ok = GraphExecutor::execute(*cg, {in.data()}, {N*sizeof(fp16)}, {out.data()}, {N*sizeof(fp16)});
    REQUIRE(ok);

    CHECK(std::abs(to_f32(out[0]))           < 0.01f);
    CHECK(std::abs(to_f32(out[1]))           < 0.01f);
    CHECK(std::abs(to_f32(out[2]) - 0.667f)  < 0.05f);
    CHECK(std::abs(to_f32(out[3]) - 4.0f)    < 0.1f);
}
