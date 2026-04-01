/**
 * Graph IR integration tests.
 *
 * Three tiers:
 *
 *  Tier 1 — C++ API: build_plan() on realistic sub-graphs (pure C++, no ANE).
 *  Tier 2 — C API:   libane_graph_* builder + compile path (structure only,
 *                    no execution).
 *  Tier 3 — E2E:     Full compile + execute for simple graphs when ANE is
 *                    available; skipped gracefully on non-ANE machines.
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace libane;
using namespace libane::graph;
using namespace libane::mil;

/* ── fp16 helpers ────────────────────────────────────────────────────────── */

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
using fp16 = __fp16;
static fp16  to_f16(float v)  { return static_cast<fp16>(v); }
static float to_f32(fp16 v)   { return static_cast<float>(v); }
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
    if (e == 0)       { uint32_t v = s | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    if (e == 31)      { uint32_t v = s | 0x7F800000 | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    uint32_t v = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &v, 4); return f;
}
#endif

static TensorShape S(int C, int seq) { return {1, C, 1, seq}; }

/** Build a vector of n fp16 values, all equal to val. */
static std::vector<fp16> fp16_fill(size_t n, float val) {
    return std::vector<fp16>(n, to_f16(val));
}

/** Build a weight matrix [IC × OC] with all values = val (fp16 raw bytes). */
static std::vector<uint8_t> weight_bytes(int IC, int OC, float val = 1.0f) {
    size_t n = static_cast<size_t>(IC) * OC;
    std::vector<uint8_t> w(n * 2);
    fp16 h = to_f16(val);
    for (size_t i = 0; i < n; ++i)
        std::memcpy(w.data() + i * 2, &h, 2);
    return w;
}

static std::vector<uint8_t> scale_bytes(int C, float val = 1.0f) {
    return weight_bytes(C, 1, val);
}

static std::vector<uint8_t> layernorm_bytes(int C, float gamma_val = 1.0f,
                                             float beta_val = 0.0f) {
    auto g = weight_bytes(C, 1, gamma_val);
    auto b = weight_bytes(C, 1, beta_val);
    g.insert(g.end(), b.begin(), b.end());
    return g;
}

static std::vector<uint8_t> scatter_mask_bytes(int C, int SP) {
    // One static write position per channel: mask[c, idx(c)] = 1, else 0.
    const size_t n = static_cast<size_t>(C) * SP;
    std::vector<uint8_t> w(n * 2, 0);
    const fp16 one = to_f16(1.0f);
    for (int c = 0; c < C; ++c) {
        int idx = (7 * c + 3) % SP;
        size_t lin = static_cast<size_t>(c) * SP + static_cast<size_t>(idx);
        std::memcpy(w.data() + lin * 2, &one, 2);
    }
    return w;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tier 1 — C++ build_plan on realistic sub-graphs
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("T1: FFN block — RMSNorm → Matmul → GELU → Matmul", "[integration][tier1]") {
    // Typical transformer FFN: rms-norm input, project up, activate, project down
    const int D = 512, SP = 128;     // model dim, sequence length
    const int FFN = 2048;            // intermediate dim

    AneGraph g;
    auto wrn  = scale_bytes(D);
    auto wup  = weight_bytes(D, FFN);
    auto wdn  = weight_bytes(FFN, D);

    TensorId x   = g.add_input("x", S(D, SP));
    TensorId rn  = g.add_op(LIBANE_OP_RMSNORM, {x},  S(D,   SP), wrn.data(), wrn.size());
    TensorId up  = g.add_op(LIBANE_OP_MATMUL,  {rn}, S(FFN, SP), wup.data(), wup.size());
    TensorId act = g.add_op(LIBANE_OP_GELU,    {up}, S(FFN, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL,  {act},S(D,   SP), wdn.data(), wdn.size());
    g.mark_output(out, "ffn_out");

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // All four ops share no branch → may fuse into ≤2 groups
    // (rmsnorm+matmul blocked by unique weight_file rule for two matmuls in same group)
    CHECK(plan.groups.size() >= 1);
    CHECK(plan.groups.size() <= 4);
    CHECK(plan.graph_output_ids == std::vector<TensorId>{out});
    CHECK(plan.tensor_bytes.at(out) == S(D, SP).bytes());
    CHECK(plan.buffer_ids.size() == plan.groups.size());
}

TEST_CASE("T1: SwiGLU FFN — gate×up merged with down projection", "[integration][tier1]") {
    // SwiGLU: out = (gate_proj(x) * silu) ⊙ up_proj(x)  then down_proj
    const int D = 512, SP = 128, H = 1536;

    AneGraph g;
    auto wgate = weight_bytes(D, H);
    auto wup   = weight_bytes(D, H);
    auto wdown = weight_bytes(H, D);

    TensorId x   = g.add_input("x", S(D, SP));
    TensorId gp  = g.add_op(LIBANE_OP_MATMUL, {x},      S(H, SP), wgate.data(), wgate.size());
    TensorId act = g.add_op(LIBANE_OP_SILU,   {gp},     S(H, SP));
    TensorId up  = g.add_op(LIBANE_OP_MATMUL, {x},      S(H, SP), wup.data(),   wup.size());
    TensorId m   = g.add_op(LIBANE_OP_MUL,    {act, up},S(H, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {m},      S(D, SP), wdown.data(), wdown.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // x is consumed by gp and up (branch) → at least 3 groups
    CHECK(plan.groups.size() >= 3);
    CHECK(plan.tensor_bytes.at(out) == S(D, SP).bytes());
}

TEST_CASE("T1: Attention QK softmax sub-graph", "[integration][tier1]") {
    // x → matmul (Q proj) → transpose → softmax → output
    const int D = 512, SP = 128, HD = 64;

    AneGraph g;
    auto wq = weight_bytes(D, HD);

    TensorId x    = g.add_input("x", S(D, SP));
    TensorId q    = g.add_op(LIBANE_OP_MATMUL,   {x}, S(HD, SP),    wq.data(), wq.size());
    TensorId qt   = g.add_op(LIBANE_OP_TRANSPOSE,{q}, S(SP, HD));
    TensorId smx  = g.add_op(LIBANE_OP_SOFTMAX,  {qt},S(SP, HD));
    g.mark_output(smx);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // The three ops form a linear chain → all should fuse
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 3);
    CHECK(plan.tensor_bytes.at(smx) == S(SP, HD).bytes());
}

TEST_CASE("T1: LayerNorm → Matmul chain", "[integration][tier1]") {
    const int C = 256, SP = 128, OC = 512;

    AneGraph g;
    auto wln = layernorm_bytes(C);
    auto wm  = weight_bytes(C, OC);

    TensorId x  = g.add_input("x", S(C, SP));
    TensorId ln = g.add_op(LIBANE_OP_LAYERNORM, {x},  S(C,  SP), wln.data(), wln.size());
    TensorId mm = g.add_op(LIBANE_OP_MATMUL,    {ln}, S(OC, SP), wm.data(),  wm.size());
    g.mark_output(mm);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // layernorm has two weight files (gamma + beta), matmul has one — no weight_file
    // collision, so they should fuse
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 2);
}

TEST_CASE("T1: Residual add in plan", "[integration][tier1]") {
    // a → matmul → t1
    // a + t1 (residual)
    const int C = 256, SP = 128;

    AneGraph g;
    auto wm = weight_bytes(C, C);

    TensorId a   = g.add_input("a", S(C, SP));
    TensorId t1  = g.add_op(LIBANE_OP_MATMUL, {a},     S(C, SP), wm.data(), wm.size());
    TensorId out = g.add_op(LIBANE_OP_ADD,    {t1, a}, S(C, SP));
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // t1 has a single consumer (the ADD) and the side input `a` is a graph input
    // (not produced within the group) → fusion rules allow them to merge
    CHECK(plan.groups.size() >= 1);
    CHECK(plan.tensor_bytes.at(out) == S(C, SP).bytes());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tier 2 — C API builder
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("T2: libane_graph_create / release round trip", "[integration][tier2]") {
    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_create is idempotent — multiple graphs", "[integration][tier2]") {
    auto* g1 = libane_graph_create();
    auto* g2 = libane_graph_create();
    REQUIRE(g1 != nullptr);
    REQUIRE(g2 != nullptr);
    REQUIRE(g1 != g2);
    libane_graph_release(g1);
    libane_graph_release(g2);
}

TEST_CASE("T2: libane_graph_add_input returns distinct IDs", "[integration][tier2]") {
    auto* g = libane_graph_create();
    libane_shape_t s; s.dims[0]=1; s.dims[1]=256; s.dims[2]=1; s.dims[3]=128; s.ndim=4;

    uint32_t id0 = libane_graph_add_input(g, "x", s);
    uint32_t id1 = libane_graph_add_input(g, "y", s);
    CHECK(id0 != LIBANE_INVALID_TENSOR_ID);
    CHECK(id1 != LIBANE_INVALID_TENSOR_ID);
    CHECK(id0 != id1);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_add_input fails on bad shape", "[integration][tier2]") {
    auto* g = libane_graph_create();
    // S=100 is not a multiple of 8
    libane_shape_t s; s.dims[0]=1; s.dims[1]=256; s.dims[2]=1; s.dims[3]=100; s.ndim=4;
    uint32_t id = libane_graph_add_input(g, "x", s);
    CHECK(id == LIBANE_INVALID_TENSOR_ID);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_add_op returns a tensor ID", "[integration][tier2]") {
    auto* g = libane_graph_create();
    libane_shape_t in_s;  in_s.dims[0]=1;  in_s.dims[1]=512; in_s.dims[2]=1;  in_s.dims[3]=128; in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1; out_s.dims[1]=256; out_s.dims[2]=1; out_s.dims[3]=128; out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    auto w = weight_bytes(512, 256);
    uint32_t out_id = libane_graph_add_op(g, LIBANE_OP_MATMUL, &x, 1,
                                           out_s, w.data(), w.size());
    CHECK(out_id != LIBANE_INVALID_TENSOR_ID);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_mark_output returns LIBANE_OK for valid tensor", "[integration][tier2]") {
    auto* g = libane_graph_create();
    libane_shape_t s; s.dims[0]=1; s.dims[1]=256; s.dims[2]=1; s.dims[3]=128; s.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", s);

    auto w = weight_bytes(256, 256);
    libane_shape_t out_s = s;
    uint32_t out_id = libane_graph_add_op(g, LIBANE_OP_MATMUL, &x, 1,
                                           out_s, w.data(), w.size());
    CHECK(libane_graph_mark_output(g, out_id, "out") == LIBANE_OK);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_mark_output fails for invalid tensor id", "[integration][tier2]") {
    auto* g = libane_graph_create();
    libane_shape_t s; s.dims[0]=1; s.dims[1]=256; s.dims[2]=1; s.dims[3]=128; s.ndim=4;
    libane_graph_add_input(g, "x", s);

    CHECK(libane_graph_mark_output(g, LIBANE_INVALID_TENSOR_ID, nullptr) != LIBANE_OK);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_compile returns null for graph with no outputs", "[integration][tier2]") {
    auto* g = libane_graph_create();
    libane_shape_t s; s.dims[0]=1; s.dims[1]=256; s.dims[2]=1; s.dims[3]=128; s.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", s);
    auto w = weight_bytes(256, 256);
    libane_graph_add_op(g, LIBANE_OP_MATMUL, &x, 1, s, w.data(), w.size());
    // no mark_output

    libane_compiled_graph_t cg = libane_graph_compile(g);
    CHECK(cg == nullptr);

    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_release(NULL) does not crash", "[integration][tier2]") {
    libane_graph_release(nullptr);
}

TEST_CASE("T2: libane_compiled_graph_release(NULL) does not crash", "[integration][tier2]") {
    libane_compiled_graph_release(nullptr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tier 3 — Full E2E (requires ANE; skipped if unavailable)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Clamp-based relative tolerance check for fp16 outputs. */
static bool near(float got, float expected, float rtol = 0.05f, float atol = 0.01f) {
    return std::abs(got - expected) <= atol + rtol * std::abs(expected);
}

TEST_CASE("T3: single GELU node compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    // Build graph
    AneGraph g;
    TensorId x   = g.add_input("x", S(C, SP));
    TensorId out = g.add_op(LIBANE_OP_GELU, {x}, S(C, SP));
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    // All-zeros input → GELU(0) = 0
    auto in_data  = fp16_fill(static_cast<size_t>(C) * SP, 0.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptr   = in_data.data();
    size_t      in_bytes = in_data.size() * sizeof(fp16);
    void*       out_ptr  = out_data.data();
    size_t      ob       = out_data.size() * sizeof(fp16);

    bool ok = GraphExecutor::execute(*cg,
                                      {in_ptr}, {in_bytes},
                                      {out_ptr}, {ob});
    REQUIRE(ok);

    for (size_t i = 0; i < out_data.size(); ++i)
        CHECK(near(to_f32(out_data[i]), 0.0f));
}

TEST_CASE("T3: fused matmul+gelu via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int D = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s;  in_s.dims[0]=1; in_s.dims[1]=D;  in_s.dims[2]=1; in_s.dims[3]=SP; in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1;out_s.dims[1]=D;  out_s.dims[2]=1;out_s.dims[3]=SP;out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    // Identity matmul: weight = I (D×D, all-zeros except diagonal)
    // For simplicity use all-ones weight — just check the graph runs without crash
    auto w = weight_bytes(D, D, 0.0f);  // zero weight → output = 0 → GELU(0) = 0
    uint32_t mm = libane_graph_add_op(g, LIBANE_OP_MATMUL, &x, 1, out_s,
                                       w.data(), w.size());
    REQUIRE(mm != LIBANE_INVALID_TENSOR_ID);

    uint32_t act = libane_graph_add_op(g, LIBANE_OP_GELU, &mm, 1, out_s, nullptr, 0);
    REQUIRE(act != LIBANE_INVALID_TENSOR_ID);

    CHECK(libane_graph_mark_output(g, act, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    // Fill input with 1.0
    auto in_data  = fp16_fill(static_cast<size_t>(D) * SP, 1.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(D) * SP, to_f16(0.0f));

    const void* in_ptr   = in_data.data();
    size_t      in_bytes = in_data.size() * sizeof(fp16);
    void*       out_ptr  = out_data.data();
    size_t      ob       = out_data.size() * sizeof(fp16);

    libane_status_t st = libane_graph_execute(
        cg, &in_ptr, &in_bytes, 1, &out_ptr, &ob, 1);

    CHECK(st == LIBANE_OK);
    // Weight = 0 → matmul output = 0 → GELU(0) = 0
    for (size_t i = 0; i < out_data.size(); ++i)
        CHECK(near(to_f32(out_data[i]), 0.0f));

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: execute returns false for mismatched input count", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    AneGraph g;
    TensorId x   = g.add_input("x", S(C, SP));
    TensorId out = g.add_op(LIBANE_OP_GELU, {x}, S(C, SP));
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    // Pass 0 inputs instead of 1 → executor must return false
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));
    void*  out_ptr   = out_data.data();
    size_t out_bytes = out_data.size() * sizeof(fp16);

    bool ok = GraphExecutor::execute(*cg, {}, {}, {out_ptr}, {out_bytes});
    CHECK(!ok);
}

TEST_CASE("T3: static-mask scatter compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId base = g.add_input("base", S(C, SP));
    TensorId upd  = g.add_input("updates", S(C, SP));
    auto mask = scatter_mask_bytes(C, SP);
    TensorId out = g.add_op(LIBANE_OP_SCATTER, {base, upd}, S(C, SP), mask.data(), mask.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    std::vector<fp16> base_data(n), upd_data(n), out_data(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i) {
        base_data[i] = to_f16(static_cast<float>(i % 97) * 0.125f - 3.0f);
        upd_data[i]  = to_f16(100.0f + static_cast<float>(i % 31));
    }

    const void* in_ptrs[2] = {base_data.data(), upd_data.data()};
    size_t in_sizes[2] = {base_data.size() * sizeof(fp16), upd_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_sizes[1] = {out_data.size() * sizeof(fp16)};

    bool ok = GraphExecutor::execute(*cg,
                                     {in_ptrs[0], in_ptrs[1]},
                                     {in_sizes[0], in_sizes[1]},
                                     {out_ptrs[0]},
                                     {out_sizes[0]});
    REQUIRE(ok);

    for (int c = 0; c < C; ++c) {
        int idx = (7 * c + 3) % SP;
        size_t lin = static_cast<size_t>(c) * SP + static_cast<size_t>(idx);
        CHECK(near(to_f32(out_data[lin]), to_f32(upd_data[lin]), 0.0f, 0.1f));
    }
}

TEST_CASE("T3: static-mask gather compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    auto mask = scatter_mask_bytes(C, SP);
    TensorId out = g.add_op(LIBANE_OP_GATHER, {x}, S(C, SP), mask.data(), mask.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    std::vector<fp16> x_data(n), out_data(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i)
        x_data[i] = to_f16(static_cast<float>(i % 53) * 0.25f - 4.0f);

    const void* in_ptrs[1] = {x_data.data()};
    size_t in_sizes[1] = {x_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_sizes[1] = {out_data.size() * sizeof(fp16)};

    bool ok = GraphExecutor::execute(*cg,
                                     {in_ptrs[0]},
                                     {in_sizes[0]},
                                     {out_ptrs[0]},
                                     {out_sizes[0]});
    REQUIRE(ok);

    for (int c = 0; c < C; ++c) {
        int idx = (7 * c + 3) % SP;
        size_t lin = static_cast<size_t>(c) * SP + static_cast<size_t>(idx);
        CHECK(near(to_f32(out_data[lin]), to_f32(x_data[lin]), 0.0f, 0.1f));
    }
}

TEST_CASE("T3: dynamic-mask gather compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    TensorId m = g.add_input("m", S(C, SP));
    TensorId out = g.add_op(LIBANE_OP_GATHER, {x, m}, S(C, SP));
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    std::vector<fp16> x_data(n), m_data(n), out_data(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i) {
        x_data[i] = to_f16(static_cast<float>(i % 41) * 0.125f - 2.0f);
        m_data[i] = to_f16((i % 5 == 0) ? 1.0f : 0.0f);
    }

    const void* in_ptrs[2] = {x_data.data(), m_data.data()};
    size_t in_sizes[2] = {x_data.size() * sizeof(fp16), m_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_sizes[1] = {out_data.size() * sizeof(fp16)};

    bool ok = GraphExecutor::execute(*cg,
                                     {in_ptrs[0], in_ptrs[1]},
                                     {in_sizes[0], in_sizes[1]},
                                     {out_ptrs[0]},
                                     {out_sizes[0]});
    REQUIRE(ok);

    for (size_t i = 0; i < n; ++i) {
        float expected = to_f32(x_data[i]) * to_f32(m_data[i]);
        CHECK(near(to_f32(out_data[i]), expected, 0.0f, 0.1f));
    }
}
