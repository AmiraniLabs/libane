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

TEST_CASE("T3: concat via C API (axis=1, interleave=false)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C0 = 32, C1 = 16, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s0; s0.dims[0]=1; s0.dims[1]=C0; s0.dims[2]=1; s0.dims[3]=SP; s0.ndim=4;
    libane_shape_t s1; s1.dims[0]=1; s1.dims[1]=C1; s1.dims[2]=1; s1.dims[3]=SP; s1.ndim=4;
    libane_shape_t so; so.dims[0]=1; so.dims[1]=C0+C1; so.dims[2]=1; so.dims[3]=SP; so.ndim=4;

    uint32_t a = libane_graph_add_input(g, "a", s0);
    uint32_t b = libane_graph_add_input(g, "b", s1);
    REQUIRE(a != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(b != LIBANE_INVALID_TENSOR_ID);

    uint32_t in_ids[2] = {a, b};
    uint32_t cat = libane_graph_add_op(g, LIBANE_OP_CONCAT, in_ids, 2, so, nullptr, 0);
    REQUIRE(cat != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, cat, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto a_data = fp16_fill(static_cast<size_t>(C0) * SP, 1.0f);
    auto b_data = fp16_fill(static_cast<size_t>(C1) * SP, 2.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C0 + C1) * SP, to_f16(0.0f));

    const void* in_ptrs[2]  = {a_data.data(), b_data.data()};
    size_t in_bytes[2]      = {a_data.size() * sizeof(fp16), b_data.size() * sizeof(fp16)};
    void* out_ptrs[1]       = {out_data.data()};
    size_t out_bytes[1]     = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 2, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);

    // Output is channel-concat(a, b): first C0*SP entries from a, then C1*SP from b.
    for (size_t i = 0; i < static_cast<size_t>(C0) * SP; ++i)
        CHECK(near(to_f32(out_data[i]), 1.0f));
    for (size_t i = static_cast<size_t>(C0) * SP; i < out_data.size(); ++i)
        CHECK(near(to_f32(out_data[i]), 2.0f));

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: slice_by_index via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const int OC = 32, OSP = 32;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s;  in_s.dims[0]=1; in_s.dims[1]=C;  in_s.dims[2]=1; in_s.dims[3]=SP;  in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1; out_s.dims[1]=OC; out_s.dims[2]=1; out_s.dims[3]=OSP; out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_SLICE_BY_INDEX, &x, 1, out_s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));
    for (size_t i = 0; i < in_data.size(); ++i) {
        in_data[i] = to_f16(static_cast<float>(i % 97) * 0.01f);
    }
    auto out_data = std::vector<fp16>(static_cast<size_t>(OC) * OSP, to_f16(0.0f));

    const void* in_ptrs[1]  = {in_data.data()};
    size_t in_bytes[1]      = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1]       = {out_data.data()};
    size_t out_bytes[1]     = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);

    // Slice is origin-aligned: out[r][c] == in[r][c] for r<OC, c<OSP.
    // in_data is laid out with stride SP; out_data with stride OSP.
    for (int r = 0; r < OC; ++r) {
        for (int c = 0; c < OSP; ++c) {
            CHECK(near(to_f32(out_data[static_cast<size_t>(r) * OSP + c]),
                       to_f32(in_data [static_cast<size_t>(r) * SP  + c])));
        }
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: reduce_sum via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s;  in_s.dims[0]=1; in_s.dims[1]=C; in_s.dims[2]=1; in_s.dims[3]=SP; in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1; out_s.dims[1]=1; out_s.dims[2]=1; out_s.dims[3]=SP; out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_REDUCE_SUM, &x, 1, out_s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    // Fill each channel-slice with 1.0; reduce over channels should produce C.
    auto in_data = fp16_fill(static_cast<size_t>(C) * SP, 1.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(SP), to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), static_cast<float>(C), 0.08f, 0.2f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: reduce_mean via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s;  in_s.dims[0]=1; in_s.dims[1]=C; in_s.dims[2]=1; in_s.dims[3]=SP; in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1; out_s.dims[1]=1; out_s.dims[2]=1; out_s.dims[3]=SP; out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_REDUCE_MEAN, &x, 1, out_s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data = fp16_fill(static_cast<size_t>(C) * SP, 1.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(SP), to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 1.0f, 0.08f, 0.15f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: reduce_max via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 32, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s;  in_s.dims[0]=1; in_s.dims[1]=C; in_s.dims[2]=1; in_s.dims[3]=SP; in_s.ndim=4;
    libane_shape_t out_s; out_s.dims[0]=1; out_s.dims[1]=1; out_s.dims[2]=1; out_s.dims[3]=SP; out_s.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", in_s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_REDUCE_MAX, &x, 1, out_s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));
    for (int c = 0; c < C; ++c) {
        for (int s = 0; s < SP; ++s) {
            in_data[static_cast<size_t>(c) * SP + s] = to_f16(static_cast<float>(c) * 0.5f + static_cast<float>(s) * 0.001f);
        }
    }
    auto out_data = std::vector<fp16>(static_cast<size_t>(SP), to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (int s = 0; s < SP; ++s) {
        float expected = static_cast<float>(C - 1) * 0.5f + static_cast<float>(s) * 0.001f;
        CHECK(near(to_f32(out_data[static_cast<size_t>(s)]), expected, 0.08f, 0.2f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: sub via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;
    uint32_t a = libane_graph_add_input(g, "a", s);
    uint32_t b = libane_graph_add_input(g, "b", s);
    REQUIRE(a != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(b != LIBANE_INVALID_TENSOR_ID);

    uint32_t in_ids[2] = {a, b};
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_SUB, in_ids, 2, s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto a_data = fp16_fill(static_cast<size_t>(C) * SP, 3.0f);
    auto b_data = fp16_fill(static_cast<size_t>(C) * SP, 1.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptrs[2] = {a_data.data(), b_data.data()};
    size_t in_bytes[2] = {a_data.size() * sizeof(fp16), b_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 2, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 2.0f, 0.08f, 0.15f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: real_div via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;
    uint32_t a = libane_graph_add_input(g, "a", s);
    uint32_t b = libane_graph_add_input(g, "b", s);
    REQUIRE(a != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(b != LIBANE_INVALID_TENSOR_ID);

    uint32_t in_ids[2] = {a, b};
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_REAL_DIV, in_ids, 2, s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto a_data = fp16_fill(static_cast<size_t>(C) * SP, 3.0f);
    auto b_data = fp16_fill(static_cast<size_t>(C) * SP, 2.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptrs[2] = {a_data.data(), b_data.data()};
    size_t in_bytes[2] = {a_data.size() * sizeof(fp16), b_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 2, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 1.5f, 0.08f, 0.15f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: sqrt via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_SQRT, &x, 1, s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data  = fp16_fill(static_cast<size_t>(C) * SP, 4.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 2.0f, 0.08f, 0.15f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: log via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_LOG, &x, 1, s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data  = fp16_fill(static_cast<size_t>(C) * SP, 1.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 0.0f, 0.1f, 0.2f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: rsqrt via C API", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", s);
    REQUIRE(x != LIBANE_INVALID_TENSOR_ID);

    uint32_t y = libane_graph_add_op(g, LIBANE_OP_RSQRT, &x, 1, s, nullptr, 0);
    REQUIRE(y != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, y, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    auto in_data  = fp16_fill(static_cast<size_t>(C) * SP, 4.0f);
    auto out_data = std::vector<fp16>(static_cast<size_t>(C) * SP, to_f16(0.0f));

    const void* in_ptrs[1] = {in_data.data()};
    size_t in_bytes[1] = {in_data.size() * sizeof(fp16)};
    void* out_ptrs[1] = {out_data.data()};
    size_t out_bytes[1] = {out_data.size() * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);
    for (size_t i = 0; i < out_data.size(); ++i) {
        CHECK(near(to_f32(out_data[i]), 0.5f, 0.08f, 0.15f));
    }

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: select via graph API (condition, x, y)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 32, SP = 64;
    const size_t N = static_cast<size_t>(C) * SP;

    auto* g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t s; s.dims[0]=1; s.dims[1]=C; s.dims[2]=1; s.dims[3]=SP; s.ndim=4;

    // Input names must be alphabetical: c < x < y (ANE constraint #13)
    uint32_t c_id = libane_graph_add_input(g, "c", s);
    uint32_t x_id = libane_graph_add_input(g, "x", s);
    uint32_t y_id = libane_graph_add_input(g, "y", s);
    REQUIRE(c_id != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(x_id != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(y_id != LIBANE_INVALID_TENSOR_ID);

    uint32_t in_ids[3] = {c_id, x_id, y_id};
    uint32_t out_id = libane_graph_add_op(g, LIBANE_OP_SELECT, in_ids, 3, s, nullptr, 0);
    REQUIRE(out_id != LIBANE_INVALID_TENSOR_ID);
    CHECK(libane_graph_mark_output(g, out_id, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    // Checkerboard condition: even elements = 1.0 (true), odd = 0.0 (false)
    std::vector<fp16> cond_data(N), x_data(N), y_data(N), out_data(N, to_f16(0.0f));
    for (size_t i = 0; i < N; ++i) {
        cond_data[i] = to_f16((i % 2 == 0) ? 1.0f : 0.0f);
        x_data[i]   = to_f16(1.0f);
        y_data[i]   = to_f16(-1.0f);
    }

    const void* in_ptrs[3]  = {cond_data.data(), x_data.data(), y_data.data()};
    size_t      in_bytes[3] = {N * sizeof(fp16), N * sizeof(fp16), N * sizeof(fp16)};
    void*  out_ptrs[1]  = {out_data.data()};
    size_t out_bytes[1] = {N * sizeof(fp16)};

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 3, out_ptrs, out_bytes, 1);
    CHECK(st == LIBANE_OK);

    // Even indices selected from x (1.0), odd from y (-1.0)
    for (size_t i = 0; i < N; ++i) {
        float expected = (i % 2 == 0) ? 1.0f : -1.0f;
        CHECK(near(to_f32(out_data[i]), expected));
    }

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

TEST_CASE("T3: static-mask scatter_nd compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId base = g.add_input("base", S(C, SP));
    TensorId upd  = g.add_input("updates", S(C, SP));
    auto mask = scatter_mask_bytes(C, SP);
    TensorId out = g.add_op(LIBANE_OP_SCATTER_ND, {base, upd}, S(C, SP), mask.data(), mask.size());
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

TEST_CASE("T3: static-mask scatter_along_axis compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId base = g.add_input("base", S(C, SP));
    TensorId upd  = g.add_input("updates", S(C, SP));
    auto mask = scatter_mask_bytes(C, SP);
    TensorId out = g.add_op(LIBANE_OP_SCATTER_ALONG_AXIS, {base, upd}, S(C, SP), mask.data(), mask.size());
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

TEST_CASE("T3: neg lowering compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");
    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;
    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    TensorId y = g.add_op(LIBANE_OP_NEG, {x}, S(C, SP));
    g.mark_output(y);
    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");
    std::vector<fp16> in(n), out(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i) in[i] = to_f16(float(int(i % 41) - 20) * 0.25f);
    bool ok = GraphExecutor::execute(*cg, {in.data()}, {in.size() * sizeof(fp16)}, {out.data()}, {out.size() * sizeof(fp16)});
    REQUIRE(ok);
    for (size_t i = 0; i < n; ++i) CHECK(near(to_f32(out[i]), -to_f32(in[i]), 0.0f, 0.05f));
}

TEST_CASE("T3: mod lowering compile + execute (negative operands)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");
    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;
    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    TensorId d = g.add_input("d", S(C, SP));
    TensorId y = g.add_op(LIBANE_OP_MOD, {x, d}, S(C, SP));
    g.mark_output(y);
    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");
    std::vector<fp16> in(n), den(n), out(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i) {
        float xv = float(int(i % 31) - 15);
        float dv = float((i % 5) + 2); // positive denominator
        if (i % 2) xv = -xv;
        in[i] = to_f16(xv);
        den[i] = to_f16(dv);
    }
    bool ok = GraphExecutor::execute(*cg,
                                     {in.data(), den.data()},
                                     {in.size() * sizeof(fp16), den.size() * sizeof(fp16)},
                                     {out.data()},
                                     {out.size() * sizeof(fp16)});
    REQUIRE(ok);
    for (size_t i = 0; i < n; ++i) {
        float xv = to_f32(in[i]);
        float dv = to_f32(den[i]);
        float q = std::floor(xv / dv);
        float ref = xv - q * dv;
        CHECK(near(to_f32(out[i]), ref, 0.0f, 0.2f));
    }
}

TEST_CASE("T3: sinh/cosh/tan lowering compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");
    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;
    auto run_unary = [&](libane_op_t op, auto ref_fn, float atol) {
        AneGraph g;
        TensorId x = g.add_input("x", S(C, SP));
        TensorId y = g.add_op(op, {x}, S(C, SP));
        g.mark_output(y);
        auto cg = GraphCompiler::compile(g);
        if (!cg) return false;
        std::vector<fp16> in(n), out(n, to_f16(0.0f));
        for (size_t i = 0; i < n; ++i) in[i] = to_f16(float(int(i % 33) - 16) * 0.0625f);
        bool ok = GraphExecutor::execute(*cg, {in.data()}, {in.size() * sizeof(fp16)}, {out.data()}, {out.size() * sizeof(fp16)});
        if (!ok) return false;
        for (size_t i = 0; i < n; ++i) CHECK(near(to_f32(out[i]), ref_fn(to_f32(in[i])), 0.0f, atol));
        return true;
    };
    if (!run_unary(LIBANE_OP_SINH, [](float x){ return std::sinh(x); }, 0.15f)) SKIP("sinh compile/execute unavailable");
    if (!run_unary(LIBANE_OP_COSH, [](float x){ return std::cosh(x); }, 0.15f)) SKIP("cosh compile/execute unavailable");
    if (!run_unary(LIBANE_OP_TAN,  [](float x){ return std::tan(x);  }, 0.20f)) SKIP("tan compile/execute unavailable");
}

TEST_CASE("T3: asin/acos lowering compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");
    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;
    auto run_unary = [&](libane_op_t op, auto ref_fn) {
        AneGraph g;
        TensorId x = g.add_input("x", S(C, SP));
        TensorId y = g.add_op(op, {x}, S(C, SP));
        g.mark_output(y);
        auto cg = GraphCompiler::compile(g);
        if (!cg) return false;
        std::vector<fp16> in(n), out(n, to_f16(0.0f));
        for (size_t i = 0; i < n; ++i) in[i] = to_f16(-0.95f + 1.9f * float(i % 127) / 126.0f);
        bool ok = GraphExecutor::execute(*cg, {in.data()}, {in.size() * sizeof(fp16)}, {out.data()}, {out.size() * sizeof(fp16)});
        if (!ok) return false;
        for (size_t i = 0; i < n; ++i) CHECK(near(to_f32(out[i]), ref_fn(to_f32(in[i])), 0.0f, 0.25f));
        return true;
    };
    if (!run_unary(LIBANE_OP_ASIN, [](float x){ return std::asin(x); })) SKIP("asin compile/execute unavailable");
    if (!run_unary(LIBANE_OP_ACOS, [](float x){ return std::acos(x); })) SKIP("acos compile/execute unavailable");
}

TEST_CASE("T3: clip compile + execute", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const float lo = -2.0f, hi = 5.0f;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    float clip_w[2] = {lo, hi};
    TensorId y = g.add_op(LIBANE_OP_CLIP, {x}, S(C, SP),
                          reinterpret_cast<const uint8_t*>(clip_w), sizeof(clip_w));
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n = static_cast<size_t>(C) * SP;
    std::vector<fp16> in(n), out(n, to_f16(0.0f));
    for (size_t i = 0; i < n; ++i)
        in[i] = to_f16(-6.0f + 12.0f * float(i) / float(n - 1));

    bool ok = GraphExecutor::execute(*cg,
        {in.data()}, {n * sizeof(fp16)},
        {out.data()}, {n * sizeof(fp16)});
    REQUIRE(ok);

    for (size_t i = 0; i < n; ++i) {
        float ref = std::max(lo, std::min(hi, to_f32(in[i])));
        CHECK(near(to_f32(out[i]), ref, 0.0f, 0.1f));
    }
}

TEST_CASE("T3: pad compile + execute (C-dim padding)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const int PAD_C0 = 32;   // pad 32 channels before
    const int OC = C + PAD_C0;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    int32_t pad_w[8] = {0, PAD_C0, 0, 0,  0, 0, 0, 0};
    TensorId y = g.add_op(LIBANE_OP_PAD, {x}, S(OC, SP),
                          reinterpret_cast<const uint8_t*>(pad_w), sizeof(pad_w));
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_in  = static_cast<size_t>(C)  * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<fp16> in(n_in), out(n_out, to_f16(0.0f));
    for (size_t i = 0; i < n_in; ++i)
        in[i] = to_f16(float(i % 97) * 0.01f);

    bool ok = GraphExecutor::execute(*cg,
        {in.data()}, {n_in * sizeof(fp16)},
        {out.data()}, {n_out * sizeof(fp16)});
    REQUIRE(ok);

    // First PAD_C0 channels must be zero
    for (int c = 0; c < PAD_C0; ++c)
        for (int s = 0; s < SP; ++s)
            CHECK(near(to_f32(out[static_cast<size_t>(c) * SP + s]), 0.0f, 0.0f, 0.05f));

    // Remaining channels must equal the input
    for (int c = 0; c < C; ++c)
        for (int s = 0; s < SP; ++s)
            CHECK(near(to_f32(out[static_cast<size_t>(PAD_C0 + c) * SP + s]),
                       to_f32(in [static_cast<size_t>(c) * SP + s])));
}

/* ── DYNAMIC_MATMUL integration ──────────────────────────────────────────── */

TEST_CASE("T3: dynamic_matmul compile + execute (both inputs runtime)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // Matrix format: X=[1,1,K,M]  W=[1,1,N,K]  Y=[1,1,N,M]
    // Using K=32, N=64, M=32: non-square to validate generality.
    // W[n,k] = 0.5 for all n,k; X[k,m] = 1.0 for all k,m.
    // Expected: Y[n,m] = sum_k 1.0 * 0.5 = K * 0.5 = 16.0
    const int K = 32, N = 64, M = 32;

    AneGraph g;
    TensorId x_id = g.add_input("x", {1, 1, K, M});  // [1, 1, K, M]
    TensorId w_id = g.add_input("w", {1, 1, N, K});  // [1, 1, N, K]
    TensorId y_id = g.add_op(LIBANE_OP_DYNAMIC_MATMUL, {x_id, w_id}, {1, 1, N, M},
                              nullptr, 0);
    g.mark_output(y_id);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_x = static_cast<size_t>(K) * M;
    const size_t n_w = static_cast<size_t>(N) * K;
    const size_t n_y = static_cast<size_t>(N) * M;

    std::vector<fp16> x_data(n_x, to_f16(1.0f));
    std::vector<fp16> w_data(n_w, to_f16(0.5f));
    std::vector<fp16> y_data(n_y, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {x_data.data(), w_data.data()},
        {n_x * sizeof(fp16), n_w * sizeof(fp16)},
        {y_data.data()},
        {n_y * sizeof(fp16)});
    REQUIRE(ok);

    // Each output element = sum_k 1.0 * 0.5 = K * 0.5 = 16.0
    const float expected = static_cast<float>(K) * 0.5f;
    int fail_count = 0;
    for (size_t i = 0; i < n_y; ++i) {
        if (!near(to_f32(y_data[i]), expected, 0.05f, 0.5f))
            ++fail_count;
    }
    CHECK(fail_count == 0);
}

/* ── SDPA integration ────────────────────────────────────────────────────── */

TEST_CASE("T3: sdpa compile + execute (unmasked single head)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // Single head (H=1), seq_len=32 tokens, head_dim=32.
    // Q = K = V = ones → uniform attention → output = 1.0 per element.
    // Matrix format: Q/K/V=[1,H,S,D], out=[1,H,S,D]
    const int H = 1, seq_len = 32, head_dim = 32;

    AneGraph g;
    TensorId q_id = g.add_input("q", {1, H, seq_len, head_dim});
    TensorId k_id = g.add_input("k", {1, H, seq_len, head_dim});
    TensorId v_id = g.add_input("v", {1, H, seq_len, head_dim});

    TensorId out_id = g.add_op(LIBANE_OP_SDPA,
                                {q_id, k_id, v_id},
                                {1, H, seq_len, head_dim},
                                nullptr, 0);
    g.mark_output(out_id);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_qkv = static_cast<size_t>(H) * seq_len * head_dim;
    std::vector<fp16> q_data(n_qkv, to_f16(1.0f));
    std::vector<fp16> k_data(n_qkv, to_f16(1.0f));
    std::vector<fp16> v_data(n_qkv, to_f16(1.0f));
    std::vector<fp16> out_data(n_qkv, to_f16(0.0f));

    // Inputs in graph insertion order: q=first, k=second, v=third
    bool ok = GraphExecutor::execute(*cg,
        {q_data.data(), k_data.data(), v_data.data()},
        {n_qkv * sizeof(fp16), n_qkv * sizeof(fp16), n_qkv * sizeof(fp16)},
        {out_data.data()},
        {n_qkv * sizeof(fp16)});
    REQUIRE(ok);

    // With Q=K=V=ones: attn = softmax(ones/sqrt(head_dim)) = uniform, output = 1.0
    int finite_count = 0, near_one_count = 0;
    for (size_t i = 0; i < n_qkv; ++i) {
        float val = to_f32(out_data[i]);
        if (std::isfinite(val)) ++finite_count;
        if (near(val, 1.0f, 0.1f, 0.1f)) ++near_one_count;
    }
    CHECK(finite_count == static_cast<int>(n_qkv));
    CHECK(near_one_count > static_cast<int>(n_qkv) * 9 / 10);
}

/* ── KDA prefill PoC ─────────────────────────────────────────────────────── */

TEST_CASE("T3: KDA prefill — outer product accumulation + readout", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // Ungated KDA prefill PoC.  One chunk of T=32 tokens, one head, D=128.
    //
    // Op 1 — outer product accumulation (state update):
    //   ΔS = V_T @ K    V_T=[D,T], K=[T,D]  →  ΔS=[D,D]
    //   DYNAMIC_MATMUL: inputs[0]=K[1,1,T,D], inputs[1]=V_T[1,1,D,T]
    //
    // Op 2 — readout:
    //   O  = ΔS  @ Q    ΔS=[D,D],  Q=[D,T]  →  O=[D,T]
    //   DYNAMIC_MATMUL: inputs[0]=Q[1,1,D,T], inputs[1]=ΔS[1,1,D,D]
    //
    // Inputs chosen for exact fp16-representable output:
    //   K = ones, V_T = (1/T)*ones, Q = ones
    //   ΔS[d,d'] = Σ_t V_T[d,t]*K[t,d'] = T*(1/T)*1   = 1.0
    //   O [d,t]  = Σ_d' ΔS[d,d']*Q[d',t] = D*1.0*1.0  = 128.0

    const int T = 32, D = 128;

    AneGraph g;
    TensorId k_id  = g.add_input("k",  {1, 1, T, D});  // [1,1, 32,128]
    TensorId vt_id = g.add_input("vt", {1, 1, D, T});  // [1,1,128, 32]  V pre-transposed
    TensorId q_id  = g.add_input("q",  {1, 1, D, T});  // [1,1,128, 32]

    // Op 1: ΔS = V_T @ K  →  [1,1,D,D]
    TensorId ds_id = g.add_op(LIBANE_OP_DYNAMIC_MATMUL,
                               {k_id, vt_id},
                               {1, 1, D, D},
                               nullptr, 0);

    // Op 2: O = ΔS @ Q  →  [1,1,D,T]
    TensorId o_id = g.add_op(LIBANE_OP_DYNAMIC_MATMUL,
                              {q_id, ds_id},
                              {1, 1, D, T},
                              nullptr, 0);

    g.mark_output(o_id);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_k  = static_cast<size_t>(T) * D;   // 4096
    const size_t n_vt = static_cast<size_t>(D) * T;   // 4096
    const size_t n_q  = static_cast<size_t>(D) * T;   // 4096
    const size_t n_o  = static_cast<size_t>(D) * T;   // 4096

    std::vector<fp16> k_data(n_k,  to_f16(1.0f));
    std::vector<fp16> vt_data(n_vt, to_f16(1.0f / T));  // 1/32 — exact in fp16
    std::vector<fp16> q_data(n_q,  to_f16(1.0f));
    std::vector<fp16> o_data(n_o,  to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {k_data.data(), vt_data.data(), q_data.data()},
        {n_k * sizeof(fp16), n_vt * sizeof(fp16), n_q * sizeof(fp16)},
        {o_data.data()},
        {n_o * sizeof(fp16)});
    REQUIRE(ok);

    // Expected: every element = D = 128.0 (exact in fp16)
    const float expected = static_cast<float>(D);
    int fail_count = 0;
    for (size_t i = 0; i < n_o; ++i) {
        if (!near(to_f32(o_data[i]), expected, 0.02f, 0.5f))
            ++fail_count;
    }
    CHECK(fail_count == 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * W8A16 quantized matmul tests
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── W8A16 helpers ───────────────────────────────────────────────────────── */

/**
 * Build a MATMUL_W8A16 weight blob.
 * Weights in [IC, OC] int8 layout, scales in [OC] float32 layout.
 * val_w: uniform int8 value; val_s: uniform float32 scale.
 */
static std::vector<uint8_t> w8a16_blob(int IC, int OC,
                                        int8_t val_w = 1,
                                        float  val_s = 1.0f) {
    const size_t kHdrBytes    = 2 * sizeof(int32_t);
    const size_t wbytes       = static_cast<size_t>(IC) * OC;
    const size_t scales_start = (kHdrBytes + wbytes + 3) & ~size_t(3);
    const size_t total_bytes  = scales_start + static_cast<size_t>(OC) * sizeof(float);

    std::vector<uint8_t> blob(total_bytes, 0);
    int32_t hdr[2] = { OC, IC };
    std::memcpy(blob.data(), hdr, kHdrBytes);
    std::memset(blob.data() + kHdrBytes, static_cast<uint8_t>(val_w), wbytes);
    for (int oc = 0; oc < OC; ++oc) {
        float s = val_s;
        std::memcpy(blob.data() + scales_start + oc * sizeof(float), &s, sizeof(float));
    }
    return blob;
}

/* ── Tier 1 ──────────────────────────────────────────────────────────────── */

TEST_CASE("T1: MATMUL_W8A16 — basic build_plan", "[integration][tier1][w8a16]") {
    // [1,4,1,32] → [1,8,1,32]: IC=4, OC=8, S=32
    const int IC = 4, OC = 8, SP = 32;
    AneGraph g;
    auto blob = w8a16_blob(IC, OC);
    TensorId x   = g.add_input("x", S(IC, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    CHECK(plan.groups.size() == 1);
    CHECK(plan.graph_output_ids == std::vector<TensorId>{out});
    CHECK(plan.tensor_bytes.at(out) == static_cast<size_t>(OC) * SP * 2);
}

TEST_CASE("T1: MATMUL_W8A16 fuses with downstream GELU", "[integration][tier1][w8a16]") {
    // W8A16 must fuse exactly like MATMUL: matmul_w8a16 → gelu → one group
    const int IC = 4, OC = 8, SP = 32;
    AneGraph g;
    auto blob = w8a16_blob(IC, OC);
    TensorId x   = g.add_input("x",  S(IC, SP));
    TensorId mm  = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    TensorId act = g.add_op(LIBANE_OP_GELU, {mm}, S(OC, SP));
    g.mark_output(act);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    // Both ops share the same activation tensor → should fuse into one group
    CHECK(plan.groups.size() == 1);
}

TEST_CASE("T1: MATMUL_W8A16 — validator rejects IC/OC blob mismatch", "[integration][tier1][w8a16]") {
    // Blob declares IC=8 but graph input has IC=4 — validator rejects
    const int IC = 4, OC = 8, SP = 32;
    AneGraph g;
    auto blob = w8a16_blob(8, OC);  // blob IC=8, but graph IC=4
    TensorId x   = g.add_input("x", S(IC, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    CHECK(cg == nullptr);
}

TEST_CASE("T1: MATMUL_W8A16 — validator rejects truncated blob", "[integration][tier1][w8a16]") {
    // Remove the last byte — corrupts the scales region
    const int IC = 4, OC = 8, SP = 32;
    AneGraph g;
    auto blob = w8a16_blob(IC, OC);
    blob.pop_back();
    TensorId x   = g.add_input("x", S(IC, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    CHECK(cg == nullptr);
}

/* ── Tier 2 ──────────────────────────────────────────────────────────────── */

TEST_CASE("T2: libane_graph_add_matmul_w8a16 returns valid tensor ID",
          "[integration][tier2][w8a16]") {
    const int IC = 4, OC = 8, SP = 32;
    std::vector<int8_t> W(IC * OC, int8_t{1});
    std::vector<float>  sc(OC, 1.0f);

    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);

    libane_shape_t in_s  = {.dims = {1, IC, 1, SP}, .ndim = 4};
    libane_shape_t out_s = {.dims = {1, OC, 1, SP}, .ndim = 4};

    uint32_t in_id  = libane_graph_add_input(g, "x", in_s);
    REQUIRE(in_id != LIBANE_INVALID_TENSOR_ID);

    uint32_t out_id = libane_graph_add_matmul_w8a16(
        g, in_id, out_s, W.data(), sc.data(), IC, OC);
    CHECK(out_id != LIBANE_INVALID_TENSOR_ID);

    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_add_matmul_w8a16 rejects bad args",
          "[integration][tier2][w8a16]") {
    const int IC = 4, OC = 8, SP = 32;
    std::vector<int8_t> W(IC * OC, int8_t{1});
    std::vector<float>  sc(OC, 1.0f);

    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);
    libane_shape_t in_s  = {.dims = {1, IC, 1, SP}, .ndim = 4};
    libane_shape_t out_s = {.dims = {1, OC, 1, SP}, .ndim = 4};
    uint32_t in_id = libane_graph_add_input(g, "x", in_s);
    REQUIRE(in_id != LIBANE_INVALID_TENSOR_ID);

    // null weights
    CHECK(libane_graph_add_matmul_w8a16(
            g, in_id, out_s, nullptr, sc.data(), IC, OC) == LIBANE_INVALID_TENSOR_ID);
    // null scales
    CHECK(libane_graph_add_matmul_w8a16(
            g, in_id, out_s, W.data(), nullptr, IC, OC) == LIBANE_INVALID_TENSOR_ID);
    // IC = 0
    CHECK(libane_graph_add_matmul_w8a16(
            g, in_id, out_s, W.data(), sc.data(), 0, OC) == LIBANE_INVALID_TENSOR_ID);
    // OC = 0
    CHECK(libane_graph_add_matmul_w8a16(
            g, in_id, out_s, W.data(), sc.data(), IC, 0) == LIBANE_INVALID_TENSOR_ID);

    libane_graph_release(g);
}

/* ── Tier 3 ──────────────────────────────────────────────────────────────── */

TEST_CASE("T3: MATMUL_W8A16 — identity weights produce correct output",
          "[integration][tier3][ane][w8a16]") {
    // W_int8 = 1, scale = 1.0 → W_dequant = 1.0.
    // Input all 1.0, IC=4, OC=4, S=32.
    // Y[oc, s] = Σ_ic W[ic,oc] × X[ic,s] = IC × 1.0 = 4.0
    const int IC = 4, OC = 4, SP = 32;

    AneGraph g;
    auto blob = w8a16_blob(IC, OC, 1, 1.0f);
    TensorId x   = g.add_input("x", S(IC, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_in  = static_cast<size_t>(IC) * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<fp16> in_data(n_in,  to_f16(1.0f));
    std::vector<fp16> out_data(n_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()},  {n_in  * sizeof(fp16)},
        {out_data.data()}, {n_out * sizeof(fp16)});
    REQUIRE(ok);

    // Expected: every element = IC = 4.0
    const float expected = static_cast<float>(IC);
    int fail = 0;
    for (auto v : out_data)
        if (!near(to_f32(v), expected, 0.02f, 0.1f)) ++fail;
    CHECK(fail == 0);
}

TEST_CASE("T3: MATMUL_W8A16 — scale factor is correctly applied",
          "[integration][tier3][ane][w8a16]") {
    // W_int8 = 2, scale = 0.5 → W_dequant = 1.0.
    // Same arithmetic as the identity test — verifies scale is honoured.
    const int IC = 4, OC = 4, SP = 32;

    AneGraph g;
    auto blob = w8a16_blob(IC, OC, 2, 0.5f);  // 2 × 0.5 = 1.0
    TensorId x   = g.add_input("x", S(IC, SP));
    TensorId out = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n = static_cast<size_t>(IC) * SP;
    std::vector<fp16> in_data(n, to_f16(1.0f));
    std::vector<fp16> out_data(n, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()}, {n * sizeof(fp16)},
        {out_data.data()}, {n * sizeof(fp16)});
    REQUIRE(ok);

    const float expected = static_cast<float>(IC);  // same as identity
    int fail = 0;
    for (auto v : out_data)
        if (!near(to_f32(v), expected, 0.02f, 0.1f)) ++fail;
    CHECK(fail == 0);
}

TEST_CASE("T3: MATMUL_W8A16 fused with GELU executes correctly",
          "[integration][tier3][ane][w8a16]") {
    // W8A16 → GELU fused in one group.
    // W_int8=1, scale=1, IC=4, OC=4: pre-GELU value = IC = 4.0
    // GELU(4.0) ≈ 4.0 (tanh approximation approaches identity for large positive x)
    const int IC = 4, OC = 4, SP = 32;

    AneGraph g;
    auto blob = w8a16_blob(IC, OC, 1, 1.0f);
    TensorId x   = g.add_input("x",  S(IC, SP));
    TensorId mm  = g.add_op(LIBANE_OP_MATMUL_W8A16, {x}, S(OC, SP),
                            blob.data(), blob.size());
    TensorId out = g.add_op(LIBANE_OP_GELU, {mm}, S(OC, SP));
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_in  = static_cast<size_t>(IC) * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<fp16> in_data(n_in,  to_f16(1.0f));
    std::vector<fp16> out_data(n_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()},  {n_in  * sizeof(fp16)},
        {out_data.data()}, {n_out * sizeof(fp16)});
    REQUIRE(ok);

    // GELU(4.0) ≈ 4.0 — verify all outputs are finite, positive, and ≥ 3.9
    int fail = 0;
    for (auto v : out_data) {
        float f = to_f32(v);
        if (!std::isfinite(f) || f < 3.9f) ++fail;
    }
    CHECK(fail == 0);
}

/* ── S=32 minimum alignment probe ───────────────────────────────────────── */

TEST_CASE("T3: S=32 activation tensor compiles and executes (alignment floor)", "[integration][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // Verify that the minimum documented alignment S=32 works end-to-end for
    // a standard [1, C, 1, S] activation tensor.  Uses GELU as the probe op —
    // simple, no weights, output shape equals input shape.
    // Tests multiple channel counts to rule out C-dependent failures.
    for (int C : {1, 32, 64}) {
        AneGraph g;
        TensorId x = g.add_input("x", S(C, 32));
        TensorId y = g.add_op(LIBANE_OP_GELU, {x}, S(C, 32), nullptr, 0);
        g.mark_output(y);

        auto cg = GraphCompiler::compile(g);
        INFO("C=" << C);
        REQUIRE(cg);

        const size_t n = static_cast<size_t>(C) * 32;
        std::vector<fp16> in(n, to_f16(1.0f));
        std::vector<fp16> out(n, to_f16(0.0f));

        bool ok = GraphExecutor::execute(*cg,
            {in.data()}, {n * sizeof(fp16)},
            {out.data()}, {n * sizeof(fp16)});
        REQUIRE(ok);

        // GELU(1.0) ≈ 0.841 — just verify all outputs are finite and non-zero
        int bad = 0;
        for (auto v : out)
            if (!std::isfinite(to_f32(v)) || to_f32(v) == 0.0f) ++bad;
        CHECK(bad == 0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONV2D tests — Tier 1 (build_plan), Tier 2 (C API), Tier 3 (E2E ANE)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── CONV2D helpers ──────────────────────────────────────────────────────── */

/** TensorShape for conv image tensors [1, C, H, W]. */
static TensorShape SC(int C, int H, int W) { return {1, C, H, W}; }

/** libane_shape_t for conv image tensors [1, C, H, W]. */
static libane_shape_t lsc(int C, int H, int W) {
    libane_shape_t s;
    s.dims[0] = 1; s.dims[1] = C; s.dims[2] = H; s.dims[3] = W;
    s.ndim = 4;
    return s;
}

/**
 * Build a flat fp16 conv kernel [OC, IC_per_group, kH, kW], all values = val.
 * Returns raw fp16 bytes (no header — just the payload).
 */
static std::vector<uint8_t> conv_kernel_bytes(int OC, int IC_per_group,
                                              int kH, int kW, float val = 1.0f) {
    const size_t n = static_cast<size_t>(OC) * IC_per_group * kH * kW;
    std::vector<uint8_t> w(n * 2);
    fp16 h = to_f16(val);
    for (size_t i = 0; i < n; ++i)
        std::memcpy(w.data() + i * 2, &h, 2);
    return w;
}

/**
 * Build the full CONV2D weight blob for libane_graph_add_op / AneGraph::add_op:
 *   int32[11] header + fp16 kernel.
 *
 * Header: {kH, kW, stride_h, stride_w, pad_top, pad_left, pad_bot, pad_right,
 *          dil_h, dil_w, groups}
 */
static std::vector<uint8_t> conv2d_blob(
        int OC, int IC, int kH, int kW,
        int stride_h   = 1, int stride_w   = 1,
        int pad_top    = 0, int pad_left   = 0,
        int pad_bot    = 0, int pad_right  = 0,
        int dil_h      = 1, int dil_w      = 1,
        int groups     = 1,
        float val      = 1.0f) {
    const int IC_per_group = IC / groups;
    auto kernel = conv_kernel_bytes(OC, IC_per_group, kH, kW, val);
    constexpr size_t kParamBytes = 11 * sizeof(int32_t);
    std::vector<uint8_t> blob(kParamBytes + kernel.size());
    int32_t params[11] = { kH, kW, stride_h, stride_w,
                           pad_top, pad_left, pad_bot, pad_right,
                           dil_h, dil_w, groups };
    std::memcpy(blob.data(), params, kParamBytes);
    std::memcpy(blob.data() + kParamBytes, kernel.data(), kernel.size());
    return blob;
}

/* ── Tier 1: build_plan (no ANE required) ────────────────────────────────── */

TEST_CASE("T1: CONV2D — conv1x1 pointwise build_plan", "[integration][tier1][conv2d]") {
    // [1,4,4,32] → [1,8,4,32] via 1×1 pointwise conv (IC=4, OC=8)
    const int IC = 4, OC = 8, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 1, 1);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    // CONV2D is always a standalone fusion group (never fused with other ops)
    CHECK(plan.groups.size() == 1);
    CHECK(plan.graph_output_ids == std::vector<TensorId>{out});
    // Output bytes: OC × H × W × 2
    CHECK(plan.tensor_bytes.at(out) == static_cast<size_t>(OC) * H * W * 2);
}

TEST_CASE("T1: CONV2D — 3×3 SAME build_plan", "[integration][tier1][conv2d]") {
    // [1,1,4,32] → [1,1,4,32] with 3×3 kernel and pad=1 all sides (SAME)
    // H_out = (4 + 1+1 - 1*(3-1) - 1)/1 + 1 = (4+2-2-1)/1+1 = 4
    // W_out = (32 + 1+1 - 1*(3-1) - 1)/1 + 1 = (32+2-2-1)/1+1 = 32
    const int IC = 1, OC = 1, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 3, 3, 1, 1, 1, 1, 1, 1);  // pad=1 all sides
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    CHECK(plan.groups.size() == 1);
    CHECK(plan.tensor_bytes.at(out) == static_cast<size_t>(OC) * H * W * 2);
}

TEST_CASE("T1: CONV2D — depthwise 3×3 SAME build_plan", "[integration][tier1][conv2d]") {
    // Depthwise: groups=IC=OC=4, IC/groups=1.  [1,4,4,32] → [1,4,4,32]
    const int IC = 4, OC = 4, H = 4, W = 32;
    const int groups = 4;  // IC == groups → depthwise
    AneGraph g;
    // Kernel: [OC, IC/groups, kH, kW] = [4, 1, 3, 3] = 36 fp16 elements
    auto blob = conv2d_blob(OC, IC, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, groups);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    CHECK(plan.groups.size() == 1);
    // Verify only one node exists in the fusion group
    CHECK(plan.groups.front().node_ids.size() == 1);
    // Verify output size: OC × H × W × 2 = 4×4×32×2 = 1024 bytes
    CHECK(plan.tensor_bytes.at(out) == static_cast<size_t>(OC) * H * W * 2);
}

TEST_CASE("T1: CONV2D — stride-2 downsampling build_plan", "[integration][tier1][conv2d]") {
    // [1,1,4,64] → [1,1,2,32] with 2×2 kernel, stride=2, no padding
    // H_out = (4 - 1*(2-1) - 1)/2 + 1 = 2/2 + 1 = 2
    // W_out = (64 - 1*(2-1) - 1)/2 + 1 = 62/2 + 1 = 32
    const int IC = 1, OC = 1, H_in = 4, W_in = 64, H_out = 2, W_out = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 2, 2, 2, 2);  // stride=2, no padding
    TensorId x   = g.add_input("x", SC(IC, H_in, W_in));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H_out, W_out),
                            blob.data(), blob.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    CHECK(plan.groups.size() == 1);
    CHECK(plan.tensor_bytes.at(out) == static_cast<size_t>(OC) * H_out * W_out * 2);
}

TEST_CASE("T1: CONV2D — validator rejects mismatched output channels", "[integration][tier1][conv2d]") {
    // Kernel encodes OC=8 but output shape claims OC=99 — validator must reject.
    // compile() returns nullptr both from validation failure and from no-ANE;
    // either way the result is nullptr, which is what we assert.
    const int IC = 4, OC_kernel = 8, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC_kernel, IC, 1, 1);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(99, H, W),  // OC=99 is wrong
                            blob.data(), blob.size());
    g.mark_output(out);

    // build_plan throws (validation), compile catches → nullptr
    auto cg = GraphCompiler::compile(g);
    CHECK(cg == nullptr);
}

TEST_CASE("T1: CONV2D — validator rejects truncated kernel blob", "[integration][tier1][conv2d]") {
    // Blob that is 1 byte shorter than the declared kernel.
    const int IC = 4, OC = 8, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 1, 1);
    blob.pop_back();  // corrupt: 1 byte short
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    CHECK(cg == nullptr);
}

/* ── Tier 2: C API with libane_graph_add_conv2d ──────────────────────────── */

TEST_CASE("T2: libane_graph_add_conv2d returns valid tensor ID", "[integration][tier2][conv2d]") {
    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);

    const int IC = 1, OC = 1, H = 4, W = 32;
    auto kernel = conv_kernel_bytes(OC, IC, 1, 1, 1.0f);  // 1×1 identity kernel

    uint32_t in_id = libane_graph_add_input(g, "x", lsc(IC, H, W));
    REQUIRE(in_id != LIBANE_INVALID_TENSOR_ID);

    uint32_t out_id = libane_graph_add_conv2d(
        g, in_id, lsc(OC, H, W),
        1, 1,  // kH, kW
        1, 1,  // stride_h, stride_w
        0, 0, 0, 0,  // padding (none)
        1, 1,  // dilation_h, dilation_w
        1,     // groups
        kernel.data(), kernel.size());

    CHECK(out_id != LIBANE_INVALID_TENSOR_ID);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_add_conv2d rejects null kernel", "[integration][tier2][conv2d]") {
    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);

    uint32_t in_id = libane_graph_add_input(g, "x", lsc(1, 4, 32));
    REQUIRE(in_id != LIBANE_INVALID_TENSOR_ID);

    // Null kernel pointer must return LIBANE_INVALID_TENSOR_ID
    uint32_t out_id = libane_graph_add_conv2d(
        g, in_id, lsc(1, 4, 32),
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
        nullptr, 0);  // null kernel

    CHECK(out_id == LIBANE_INVALID_TENSOR_ID);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_graph_add_conv2d rejects invalid hyperparams", "[integration][tier2][conv2d]") {
    libane_graph_t g = libane_graph_create();
    REQUIRE(g != nullptr);

    auto kernel = conv_kernel_bytes(1, 1, 1, 1, 1.0f);
    uint32_t in_id = libane_graph_add_input(g, "x", lsc(1, 4, 32));
    REQUIRE(in_id != LIBANE_INVALID_TENSOR_ID);

    // kH=0 — illegal
    CHECK(libane_graph_add_conv2d(
            g, in_id, lsc(1, 4, 32),
            0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
            kernel.data(), kernel.size()) == LIBANE_INVALID_TENSOR_ID);

    // stride_w=0 — illegal
    CHECK(libane_graph_add_conv2d(
            g, in_id, lsc(1, 4, 32),
            1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
            kernel.data(), kernel.size()) == LIBANE_INVALID_TENSOR_ID);

    // groups=0 — illegal
    CHECK(libane_graph_add_conv2d(
            g, in_id, lsc(1, 4, 32),
            1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
            kernel.data(), kernel.size()) == LIBANE_INVALID_TENSOR_ID);

    // negative padding — illegal
    CHECK(libane_graph_add_conv2d(
            g, in_id, lsc(1, 4, 32),
            1, 1, 1, 1, -1, 0, 0, 0, 1, 1, 1,
            kernel.data(), kernel.size()) == LIBANE_INVALID_TENSOR_ID);

    libane_graph_release(g);
}

/* ── Tier 3: full compile + execute (requires ANE) ───────────────────────── */

TEST_CASE("T3: CONV2D — 1×1 identity conv produces exact output", "[integration][tier3][ane][conv2d]") {
    // Single-channel 1×1 conv with kernel=1.0: output must equal input exactly.
    // Input all 1.0 → output all 1.0.
    const int IC = 1, OC = 1, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1.0f);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_in  = static_cast<size_t>(IC) * H * W;
    const size_t n_out = static_cast<size_t>(OC) * H * W;
    std::vector<fp16> in_data(n_in,  to_f16(1.0f));
    std::vector<fp16> out_data(n_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()}, {n_in  * sizeof(fp16)},
        {out_data.data()}, {n_out * sizeof(fp16)});
    REQUIRE(ok);

    int fail = 0;
    for (auto v : out_data)
        if (!near(to_f32(v), 1.0f, 0.01f, 0.01f)) ++fail;
    CHECK(fail == 0);
}

TEST_CASE("T3: CONV2D — 1×1 scale conv doubles all values", "[integration][tier3][ane][conv2d]") {
    // Single-channel 1×1 conv with kernel=2.0: output must be 2× the input.
    // Input all 1.0 → output all 2.0.
    const int IC = 1, OC = 1, H = 4, W = 32;
    AneGraph g;
    auto blob = conv2d_blob(OC, IC, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2.0f);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n = static_cast<size_t>(IC) * H * W;
    std::vector<fp16> in_data(n, to_f16(1.0f));
    std::vector<fp16> out_data(n, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()}, {n * sizeof(fp16)},
        {out_data.data()}, {n * sizeof(fp16)});
    REQUIRE(ok);

    int fail = 0;
    for (auto v : out_data)
        if (!near(to_f32(v), 2.0f, 0.01f, 0.01f)) ++fail;
    CHECK(fail == 0);
}

TEST_CASE("T3: CONV2D — multi-channel 1×1 accumulates IC input channels",
          "[integration][tier3][ane][conv2d]") {
    // IC=2, OC=4, kH=kW=1, all-ones kernel, all-ones input.
    // Each output channel = sum over IC = 2 × 1.0 = 2.0.
    const int IC = 2, OC = 4, H = 4, W = 32;
    AneGraph g;
    // Kernel [OC, IC/groups, 1, 1] = [4, 2, 1, 1], all 1.0
    auto blob = conv2d_blob(OC, IC, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1.0f);
    TensorId x   = g.add_input("x", SC(IC, H, W));
    TensorId out = g.add_op(LIBANE_OP_CONV2D, {x}, SC(OC, H, W),
                            blob.data(), blob.size());
    g.mark_output(out);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    const size_t n_in  = static_cast<size_t>(IC) * H * W;
    const size_t n_out = static_cast<size_t>(OC) * H * W;
    std::vector<fp16> in_data(n_in,  to_f16(1.0f));
    std::vector<fp16> out_data(n_out, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {in_data.data()}, {n_in  * sizeof(fp16)},
        {out_data.data()}, {n_out * sizeof(fp16)});
    REQUIRE(ok);

    // Expected: every output element = IC = 2.0
    const float expected = static_cast<float>(IC);
    int fail = 0;
    for (auto v : out_data)
        if (!near(to_f32(v), expected, 0.02f, 0.05f)) ++fail;
    CHECK(fail == 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * delta_reload tests
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Tier 2: null-safety ─────────────────────────────────────────────────── */

TEST_CASE("T2: libane_compiled_graph_delta_reload(NULL) returns LIBANE_ERR_INVALID_ARG",
          "[delta_reload][tier2]") {
    libane_status_t rc = libane_compiled_graph_delta_reload(nullptr);
    CHECK(rc == LIBANE_ERR_INVALID_ARG);
}

/* ── Tier 3: E2E with live ANE ───────────────────────────────────────────── */

TEST_CASE("T3: delta_reload does not crash on a compiled GELU graph",
          "[delta_reload][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // Build a minimal 1-node GELU graph
    AneGraph g;
    TensorId x = g.add_input("x", S(64, 64));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, S(64, 64), nullptr, 0);
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);

    // delta_reload must succeed without error
    bool ok = GraphExecutor::delta_reload(*cg);
    CHECK(ok);
}

TEST_CASE("T3: C API delta_reload returns LIBANE_OK on compiled graph",
          "[delta_reload][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    libane_graph_t g = libane_graph_create();
    REQUIRE(g);

    const int C = 64, SP = 64;
    libane_shape_t sh;
    sh.dims[0]=1; sh.dims[1]=C; sh.dims[2]=1; sh.dims[3]=SP; sh.ndim=4;

    uint32_t x = libane_graph_add_input(g, "x", sh);
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_GELU, &x, 1, sh, nullptr, 0);
    libane_graph_mark_output(g, y, "out");

    libane_compiled_graph_t cg = libane_graph_compile(g);
    REQUIRE(cg);

    libane_status_t rc = libane_compiled_graph_delta_reload(cg);
    CHECK(rc == LIBANE_OK);

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T3: delta_reload then execute produces identical output",
          "[delta_reload][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // A fused GELU→SIGMOID chain (one group), verify output is identical
    // before and after delta_reload.
    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    TensorId y = g.add_op(LIBANE_OP_GELU,    {x}, S(C, SP), nullptr, 0);
    TensorId z = g.add_op(LIBANE_OP_SIGMOID, {y}, S(C, SP), nullptr, 0);
    g.mark_output(z);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);

    // Alternating +1.0 / -1.0 input
    std::vector<fp16> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = to_f16(i % 2 == 0 ? 1.0f : -1.0f);

    // First execute — capture reference output
    std::vector<fp16> out_before(n, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n * sizeof(fp16)},
        {out_before.data()}, {n * sizeof(fp16)}));

    // delta_reload
    REQUIRE(GraphExecutor::delta_reload(*cg));

    // Second execute — must match
    std::vector<fp16> out_after(n, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n * sizeof(fp16)},
        {out_after.data()}, {n * sizeof(fp16)}));

    int mismatch = 0;
    for (size_t i = 0; i < n; ++i)
        if (out_before[i] != out_after[i]) ++mismatch;
    CHECK(mismatch == 0);
}

TEST_CASE("T3: delta_reload is idempotent — multiple calls all return true",
          "[delta_reload][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    AneGraph g;
    TensorId x = g.add_input("x", S(32, 64));
    TensorId y = g.add_op(LIBANE_OP_TANH, {x}, S(32, 64), nullptr, 0);
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);

    for (int i = 0; i < 3; ++i) {
        INFO("iteration " << i);
        CHECK(GraphExecutor::delta_reload(*cg));
    }
}

TEST_CASE("T3: delta_reload on MATMUL graph — execute after reload matches",
          "[delta_reload][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int IC = 64, OC = 32, SP = 64;
    const size_t weight_elems = static_cast<size_t>(IC) * OC;
    std::vector<fp16> W(weight_elems, to_f16(1.0f / IC));

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SP),
                           W.data(), weight_elems * sizeof(fp16));
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);
    REQUIRE(cg->group_count() >= 1);

    // Execute before reload
    const size_t n_in  = static_cast<size_t>(IC) * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<fp16> in(n_in, to_f16(1.0f));
    std::vector<fp16> out_before(n_out, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n_in  * sizeof(fp16)},
        {out_before.data()}, {n_out * sizeof(fp16)}));

    // delta_reload then execute again
    REQUIRE(GraphExecutor::delta_reload(*cg));

    std::vector<fp16> out_after(n_out, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n_in  * sizeof(fp16)},
        {out_after.data()}, {n_out * sizeof(fp16)}));

    // Results must be bit-identical
    int mismatch = 0;
    for (size_t i = 0; i < n_out; ++i)
        if (out_before[i] != out_after[i]) ++mismatch;
    CHECK(mismatch == 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Graph serialization tests
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <cstdio>     // std::remove

static const char* kSavePath = "/tmp/libane_test_save.aneg";

/* ── Tier 2: null-safety ─────────────────────────────────────────────────── */

TEST_CASE("T2: libane_compiled_graph_save with null cg returns LIBANE_ERR_INVALID_ARG",
          "[serialization][tier2]") {
    CHECK(libane_compiled_graph_save(nullptr, kSavePath) == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("T2: libane_compiled_graph_save with null path returns LIBANE_ERR_INVALID_ARG",
          "[serialization][tier2]") {
    // Construct a minimal compiled graph without ANE
    // (just check the null-path guard; skipped if ANE unavailable)
    if (!libane_available()) SKIP("ANE not available");

    libane_graph_t g = libane_graph_create();
    libane_shape_t sh; sh.dims[0]=1; sh.dims[1]=32; sh.dims[2]=1; sh.dims[3]=64; sh.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", sh);
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_GELU, &x, 1, sh, nullptr, 0);
    libane_graph_mark_output(g, y, "out");

    libane_compiled_graph_t cg = libane_graph_compile(g);
    REQUIRE(cg);

    CHECK(libane_compiled_graph_save(cg, nullptr) == LIBANE_ERR_INVALID_ARG);

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
}

TEST_CASE("T2: libane_compiled_graph_load with null path returns null",
          "[serialization][tier2]") {
    CHECK(libane_compiled_graph_load(nullptr) == nullptr);
}

TEST_CASE("T2: libane_compiled_graph_load with bad path returns null",
          "[serialization][tier2]") {
    CHECK(libane_compiled_graph_load("/tmp/does_not_exist_libane_test.aneg") == nullptr);
}

/* ── Tier 3: E2E save + load ─────────────────────────────────────────────── */

TEST_CASE("T3: save and load GELU graph — execute after load matches original",
          "[serialization][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int C = 64, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;

    AneGraph g;
    TensorId x = g.add_input("x", S(C, SP));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, S(C, SP), nullptr, 0);
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);

    // Build input and run original
    std::vector<fp16> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = to_f16(static_cast<float>(i % 16) * 0.1f - 0.8f);

    std::vector<fp16> out_orig(n, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n * sizeof(fp16)},
        {out_orig.data()}, {n * sizeof(fp16)}));

    // Save
    REQUIRE(cg->save(kSavePath));

    // Load
    auto cg2 = GraphCompiler::load(kSavePath);
    REQUIRE(cg2);

    // Execute loaded graph — must match
    std::vector<fp16> out_loaded(n, to_f16(0.0f));
    REQUIRE(GraphExecutor::execute(*cg2,
        {in.data()}, {n * sizeof(fp16)},
        {out_loaded.data()}, {n * sizeof(fp16)}));

    int mismatch = 0;
    for (size_t i = 0; i < n; ++i)
        if (out_orig[i] != out_loaded[i]) ++mismatch;
    CHECK(mismatch == 0);

    std::remove(kSavePath);
}

TEST_CASE("T3: C API save + load roundtrip",
          "[serialization][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    libane_graph_t g = libane_graph_create();
    libane_shape_t sh; sh.dims[0]=1; sh.dims[1]=32; sh.dims[2]=1; sh.dims[3]=64; sh.ndim=4;
    uint32_t x = libane_graph_add_input(g, "x", sh);
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_TANH, &x, 1, sh, nullptr, 0);
    libane_graph_mark_output(g, y, "out");

    libane_compiled_graph_t cg = libane_graph_compile(g);
    REQUIRE(cg);

    CHECK(libane_compiled_graph_save(cg, kSavePath) == LIBANE_OK);

    libane_compiled_graph_t cg2 = libane_compiled_graph_load(kSavePath);
    REQUIRE(cg2);

    // Execute loaded graph with simple input
    const int C = 32, SP = 64;
    const size_t n = static_cast<size_t>(C) * SP;
    std::vector<fp16> in(n, to_f16(0.5f));
    std::vector<fp16> out(n, to_f16(0.0f));
    const void* in_p  = in.data();
    void*       out_p = out.data();
    size_t      in_b  = n * sizeof(fp16);
    size_t      out_b = n * sizeof(fp16);

    CHECK(libane_graph_execute(cg2, &in_p, &in_b, 1, &out_p, &out_b, 1) == LIBANE_OK);

    // TANH(0.5) ≈ 0.462 — verify all outputs finite and non-zero
    int bad = 0;
    for (auto v : out)
        if (!std::isfinite(to_f32(v)) || to_f32(v) == 0.0f) ++bad;
    CHECK(bad == 0);

    libane_compiled_graph_release(cg2);
    libane_compiled_graph_release(cg);
    libane_graph_release(g);
    std::remove(kSavePath);
}

TEST_CASE("T3: loaded graph supports delta_reload",
          "[serialization][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    AneGraph g;
    TensorId x = g.add_input("x", S(32, 64));
    TensorId y = g.add_op(LIBANE_OP_SIGMOID, {x}, S(32, 64), nullptr, 0);
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);
    REQUIRE(cg->save(kSavePath));

    auto cg2 = GraphCompiler::load(kSavePath);
    REQUIRE(cg2);

    // delta_reload must work on a loaded (restored) graph
    CHECK(GraphExecutor::delta_reload(*cg2));

    std::remove(kSavePath);
}

TEST_CASE("T3: save + load MATMUL graph — numerically correct after restore",
          "[serialization][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int IC = 64, OC = 32, SP = 64;
    const size_t w_elems = static_cast<size_t>(IC) * OC;
    // Identity-ish weights: W[oc, ic] = 1/IC → output[oc] = sum(input) / IC
    std::vector<fp16> W(w_elems, to_f16(1.0f / IC));

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {x}, S(OC, SP),
                           W.data(), w_elems * sizeof(fp16));
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);
    REQUIRE(cg->save(kSavePath));

    auto cg2 = GraphCompiler::load(kSavePath);
    REQUIRE(cg2);

    const size_t n_in  = static_cast<size_t>(IC) * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<fp16> in(n_in, to_f16(1.0f));       // all-ones input
    std::vector<fp16> out_orig(n_out, to_f16(0.0f));
    std::vector<fp16> out_load(n_out, to_f16(0.0f));

    REQUIRE(GraphExecutor::execute(*cg,
        {in.data()}, {n_in * sizeof(fp16)},
        {out_orig.data()}, {n_out * sizeof(fp16)}));

    REQUIRE(GraphExecutor::execute(*cg2,
        {in.data()}, {n_in * sizeof(fp16)},
        {out_load.data()}, {n_out * sizeof(fp16)}));

    int mismatch = 0;
    for (size_t i = 0; i < n_out; ++i)
        if (out_orig[i] != out_load[i]) ++mismatch;
    CHECK(mismatch == 0);

    std::remove(kSavePath);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * W8A8 tests — int8 activations + int8 weights
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── W8A8 blob helper ────────────────────────────────────────────────────── */

/**
 * Build a W8A8 weight blob.
 *
 * Layout: int32[2]={OC,IC} + int8[IC×OC] + pad-to-4
 *         + float32[OC] (weight scales) + float32 (act_scale) + int32 (act_zp)
 */
static std::vector<uint8_t> w8a8_blob(int IC, int OC,
                                       int8_t  w_val   = 1,
                                       float   w_scale = 1.0f,
                                       float   a_scale = 1.0f,
                                       int32_t a_zp    = 0) {
    constexpr size_t kHdrBytes = 2 * sizeof(int32_t);
    const size_t wbytes       = static_cast<size_t>(IC) * OC;
    const size_t scales_start = (kHdrBytes + wbytes + 3) & ~size_t(3);
    const size_t total        = scales_start + static_cast<size_t>(OC) * sizeof(float)
                                + sizeof(float) + sizeof(int32_t);
    std::vector<uint8_t> blob(total, 0);
    int32_t hdr[2] = { OC, IC };
    std::memcpy(blob.data(), hdr, kHdrBytes);
    std::fill(blob.data() + kHdrBytes, blob.data() + kHdrBytes + wbytes,
              static_cast<uint8_t>(w_val));
    for (int oc = 0; oc < OC; ++oc)
        std::memcpy(blob.data() + scales_start + oc * sizeof(float), &w_scale, sizeof(float));
    const size_t act_off = scales_start + static_cast<size_t>(OC) * sizeof(float);
    std::memcpy(blob.data() + act_off,     &a_scale, sizeof(float));
    std::memcpy(blob.data() + act_off + 4, &a_zp,    sizeof(int32_t));
    return blob;
}

/* ── Tier 1: build_plan validates W8A8 blob format ─────────────────────────*/

TEST_CASE("T1: MATMUL_W8A8 build_plan succeeds with valid blob", "[w8a8][tier1]") {
    const int IC = 64, OC = 32, SP = 64;
    auto blob = w8a8_blob(IC, OC);

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    TensorId y = g.add_op(LIBANE_OP_MATMUL_W8A8, {x}, S(OC, SP),
                           blob.data(), blob.size());
    g.mark_output(y);

    ExecutionPlan plan;
    REQUIRE_NOTHROW(plan = GraphCompiler::build_plan(g));
    CHECK(plan.groups.size() == 1);
}

TEST_CASE("T1: MATMUL_W8A8 validator rejects IC mismatch in blob", "[w8a8][tier1]") {
    const int IC = 64, OC = 32, SP = 64;
    auto blob = w8a8_blob(IC + 8, OC);  // wrong IC in blob

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    g.add_op(LIBANE_OP_MATMUL_W8A8, {x}, S(OC, SP), blob.data(), blob.size());
    g.mark_output(1);

    REQUIRE_THROWS(GraphCompiler::build_plan(g));
}

TEST_CASE("T1: MATMUL_W8A8 fuses with downstream GELU", "[w8a8][tier1]") {
    const int IC = 64, OC = 32, SP = 64;
    auto blob = w8a8_blob(IC, OC);

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    TensorId y = g.add_op(LIBANE_OP_MATMUL_W8A8, {x}, S(OC, SP),
                           blob.data(), blob.size());
    TensorId z = g.add_op(LIBANE_OP_GELU, {y}, S(OC, SP), nullptr, 0);
    g.mark_output(z);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    CHECK(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 2);
}

/* ── Tier 2: C API null guards ───────────────────────────────────────────── */

TEST_CASE("T2: libane_graph_add_matmul_w8a8 rejects null graph", "[w8a8][tier2]") {
    auto blob = w8a8_blob(64, 32);
    libane_shape_t sh; sh.dims[0]=1; sh.dims[1]=32; sh.dims[2]=1; sh.dims[3]=64; sh.ndim=4;
    const int8_t* w = reinterpret_cast<const int8_t*>(blob.data() + 8);
    const float*  s = nullptr;  // not used by null-cg path
    CHECK(libane_graph_add_matmul_w8a8(nullptr, 0, sh, w, s, 64, 32, 1.0f, 0)
          == LIBANE_INVALID_TENSOR_ID);
}

/* ── Tier 3: E2E W8A8 execution ──────────────────────────────────────────── */

TEST_CASE("T3: W8A8 with identity weights — int8 input dequantized correctly",
          "[w8a8][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // W_int8 = 1, w_scale = 1/IC → W_fp16[ic,oc] = 1/IC → output[oc] = sum(A[ic]) / IC
    // a_scale = 0.5, a_zp = 0 → A_fp16[k] = int8_val * 0.5
    // int8 input = 2 → A_fp16 = 1.0, output = IC * 1.0 * (1/IC) = 1.0
    const int IC = 64, OC = 32, SP = 64;
    const float act_scale = 0.5f;
    const int32_t act_zp  = 0;

    auto blob = w8a8_blob(IC, OC,
                           /*w_val=*/1, /*w_scale=*/1.0f / IC,
                           act_scale, act_zp);

    AneGraph g;
    TensorId x = g.add_input("x", S(IC, SP));
    TensorId y = g.add_op(LIBANE_OP_MATMUL_W8A8, {x}, S(OC, SP),
                           blob.data(), blob.size());
    g.mark_output(y);

    auto cg = GraphCompiler::compile(g);
    REQUIRE(cg);
    // Verify quant params were registered
    CHECK(cg->quant_params().count(x) == 1);

    // int8 input: all = 2 → fp16 = 2*0.5 = 1.0
    const size_t n_in  = static_cast<size_t>(IC) * SP;
    const size_t n_out = static_cast<size_t>(OC) * SP;
    std::vector<int8_t> in_i8(n_in, 2);
    std::vector<fp16>   out(n_out, to_f16(0.0f));

    REQUIRE(GraphExecutor::execute(*cg,
        {in_i8.data()}, {n_in},       // int8: bytes == elements
        {out.data()}, {n_out * sizeof(fp16)}));

    int fail = 0;
    for (auto v : out)
        if (!near(to_f32(v), 1.0f, 0.05f, 0.05f)) ++fail;
    CHECK(fail == 0);
}

TEST_CASE("T3: libane_quantize_i8 round-trips correctly", "[w8a8][tier3]") {
    const size_t N = 256;
    const float scale = 0.01f;
    const int32_t zp  = 0;

    // fp16 values: (i-128)*0.01 for i in [0,255]
    std::vector<fp16> fp16_in(N);
    for (size_t i = 0; i < N; ++i)
        fp16_in[i] = to_f16(static_cast<float>(static_cast<int>(i) - 128) * scale);

    std::vector<int8_t> i8_out(N, 0);
    libane_quantize_i8(fp16_in.data(), i8_out.data(), N, scale, zp);

    // Expected: i8 = i - 128 (clamped to [-128, 127])
    int mismatch = 0;
    for (size_t i = 0; i < N; ++i) {
        auto expected = static_cast<int8_t>(static_cast<int>(i) - 128);
        if (i8_out[i] != expected) ++mismatch;
    }
    CHECK(mismatch == 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SDPA_GQA tests
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("T1: SDPA_GQA build_plan succeeds with valid shapes", "[sdpa_gqa][tier1]") {
    // H_q=4, H_kv=2 (ratio 2:1), S=32, D=32 — minimal valid GQA config.
    const int H_q = 4, H_kv = 2, S = 32, D = 32;

    AneGraph g;
    TensorId q_id = g.add_input("q", {1, H_q,  S, D});
    TensorId k_id = g.add_input("k", {1, H_kv, S, D});
    TensorId v_id = g.add_input("v", {1, H_kv, S, D});
    TensorId out  = g.add_op(LIBANE_OP_SDPA_GQA,
                              {q_id, k_id, v_id},
                              {1, H_q, S, D},
                              nullptr, 0);
    g.mark_output(out);

    // build_plan throws on validation failure; no-throw == success.
    ExecutionPlan plan;
    REQUIRE_NOTHROW(plan = GraphCompiler::build_plan(g));
    REQUIRE(plan.groups.size() >= 1);
}

TEST_CASE("T1: SDPA_GQA validator rejects H_q not divisible by H_kv", "[sdpa_gqa][tier1]") {
    // H_q=4, H_kv=3 — not divisible → build_plan must throw.
    const int H_q = 4, H_kv = 3, S = 32, D = 32;

    AneGraph g;
    TensorId q_id = g.add_input("q", {1, H_q,  S, D});
    TensorId k_id = g.add_input("k", {1, H_kv, S, D});
    TensorId v_id = g.add_input("v", {1, H_kv, S, D});
    TensorId out  = g.add_op(LIBANE_OP_SDPA_GQA,
                              {q_id, k_id, v_id},
                              {1, H_q, S, D},
                              nullptr, 0);
    g.mark_output(out);

    CHECK_THROWS(GraphCompiler::build_plan(g));
}

TEST_CASE("T1: SDPA_GQA validator rejects weight blob", "[sdpa_gqa][tier1]") {
    // SDPA_GQA takes no weights — validator must throw if any are supplied.
    const int H_q = 4, H_kv = 2, S = 32, D = 32;

    AneGraph g;
    TensorId q_id = g.add_input("q", {1, H_q,  S, D});
    TensorId k_id = g.add_input("k", {1, H_kv, S, D});
    TensorId v_id = g.add_input("v", {1, H_kv, S, D});
    std::vector<uint8_t> dummy(8, 0);
    TensorId out  = g.add_op(LIBANE_OP_SDPA_GQA,
                              {q_id, k_id, v_id},
                              {1, H_q, S, D},
                              dummy.data(), dummy.size());
    g.mark_output(out);

    CHECK_THROWS(GraphCompiler::build_plan(g));
}

TEST_CASE("T2: libane_graph_add_sdpa_gqa rejects null graph", "[sdpa_gqa][tier2]") {
    uint32_t r = libane_graph_add_sdpa_gqa(nullptr, 0, 1, 2,
                                            LIBANE_INVALID_TENSOR_ID);
    CHECK(r == LIBANE_INVALID_TENSOR_ID);
}

TEST_CASE("T3: SDPA_GQA compile + execute — Q=K=V=ones gives ~1.0",
          "[sdpa_gqa][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    // H_q=4, H_kv=2 (ratio 2), S=32, D=32
    // Q=K=V=ones → uniform attention → output ≈ 1.0 per element.
    const int H_q = 4, H_kv = 2, S = 32, D = 32;
    const size_t n_q  = static_cast<size_t>(H_q)  * S * D;
    const size_t n_kv = static_cast<size_t>(H_kv) * S * D;

    AneGraph g;
    TensorId q_id = g.add_input("q", {1, H_q,  S, D});
    TensorId k_id = g.add_input("k", {1, H_kv, S, D});
    TensorId v_id = g.add_input("v", {1, H_kv, S, D});
    TensorId out_id = g.add_op(LIBANE_OP_SDPA_GQA,
                                {q_id, k_id, v_id},
                                {1, H_q, S, D},
                                nullptr, 0);
    g.mark_output(out_id);

    auto cg = GraphCompiler::compile(g);
    if (!cg) SKIP("ANE compiler unavailable in this environment");

    std::vector<fp16> q_data(n_q,  to_f16(1.0f));
    std::vector<fp16> k_data(n_kv, to_f16(1.0f));
    std::vector<fp16> v_data(n_kv, to_f16(1.0f));
    std::vector<fp16> out_data(n_q, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {q_data.data(),       k_data.data(),        v_data.data()},
        {n_q  * sizeof(fp16), n_kv * sizeof(fp16),  n_kv * sizeof(fp16)},
        {out_data.data()},
        {n_q  * sizeof(fp16)});
    REQUIRE(ok);

    // Q=K=V=ones → attention weights are uniform → output ≈ 1.0
    int finite_count = 0, near_one_count = 0;
    for (size_t i = 0; i < n_q; ++i) {
        float val = to_f32(out_data[i]);
        if (std::isfinite(val)) ++finite_count;
        if (near(val, 1.0f, 0.1f, 0.1f)) ++near_one_count;
    }
    CHECK(finite_count    == static_cast<int>(n_q));
    CHECK(near_one_count  >  static_cast<int>(n_q) * 9 / 10);
}

TEST_CASE("T3: C API add_sdpa_gqa compile + execute", "[sdpa_gqa][tier3][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    const int H_q = 8, H_kv = 2, S = 32, D = 32;
    const size_t n_q  = static_cast<size_t>(H_q)  * S * D;
    const size_t n_kv = static_cast<size_t>(H_kv) * S * D;

    libane_graph_t gr = libane_graph_create();
    REQUIRE(gr);

    libane_shape_t sq  = {1, H_q,  S, D, 4};
    libane_shape_t skv = {1, H_kv, S, D, 4};
    uint32_t q_id = libane_graph_add_input(gr, "q",  sq);
    uint32_t k_id = libane_graph_add_input(gr, "k",  skv);
    uint32_t v_id = libane_graph_add_input(gr, "v",  skv);
    REQUIRE(q_id != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(k_id != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(v_id != LIBANE_INVALID_TENSOR_ID);

    uint32_t out_id = libane_graph_add_sdpa_gqa(gr, q_id, k_id, v_id,
                                                 LIBANE_INVALID_TENSOR_ID);
    REQUIRE(out_id != LIBANE_INVALID_TENSOR_ID);
    REQUIRE(libane_graph_mark_output(gr, out_id, "out") == LIBANE_OK);

    libane_compiled_graph_t cg = libane_graph_compile(gr);
    libane_graph_release(gr);
    if (!cg) SKIP("ANE compiler unavailable");

    std::vector<fp16> q_data(n_q,  to_f16(1.0f));
    std::vector<fp16> k_data(n_kv, to_f16(1.0f));
    std::vector<fp16> v_data(n_kv, to_f16(1.0f));
    std::vector<fp16> out_data(n_q, to_f16(0.0f));

    const void* in_ptrs[]  = {q_data.data(), k_data.data(), v_data.data()};
    size_t      in_bytes[]  = {n_q*2, n_kv*2, n_kv*2};
    void*       out_ptrs[]  = {out_data.data()};
    size_t      out_bytes[] = {n_q*2};

    libane_status_t exec_ok = libane_graph_execute(cg,
                                                    in_ptrs,  in_bytes,  3,
                                                    out_ptrs, out_bytes, 1);
    libane_compiled_graph_release(cg);
    REQUIRE(exec_ok == LIBANE_OK);

    int near_one = 0;
    for (size_t i = 0; i < n_q; ++i)
        if (near(to_f32(out_data[i]), 1.0f, 0.15f, 0.15f)) ++near_one;
    CHECK(near_one > static_cast<int>(n_q) * 8 / 10);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KV Cache tests
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("T2: libane_kv_cache_create basic sanity", "[kvcache][tier2]") {
    auto* c = libane_kv_cache_create(8, 128, 512);
    REQUIRE(c != nullptr);
    CHECK(libane_kv_cache_position(c) == 0);
    CHECK(libane_kv_cache_k(c) != nullptr);
    CHECK(libane_kv_cache_v(c) != nullptr);
    libane_kv_cache_release(c);
}

TEST_CASE("T2: libane_kv_cache_create rejects invalid dims", "[kvcache][tier2]") {
    CHECK(libane_kv_cache_create(0,  128, 512) == nullptr);
    CHECK(libane_kv_cache_create(8,  0,   512) == nullptr);
    CHECK(libane_kv_cache_create(8,  128, 0)   == nullptr);
    CHECK(libane_kv_cache_create(-1, 128, 512) == nullptr);
}

TEST_CASE("T2: libane_kv_cache_update advances position", "[kvcache][tier2]") {
    const int H = 4, D = 32, S = 64;
    auto* c = libane_kv_cache_create(H, D, S);
    REQUIRE(c);

    std::vector<uint16_t> tok(static_cast<size_t>(H) * D, 0x3C00u); // fp16 1.0
    int p1 = libane_kv_cache_update(c, tok.data(), tok.data());
    CHECK(p1 == 1);
    CHECK(libane_kv_cache_position(c) == 1);

    int p2 = libane_kv_cache_update(c, tok.data(), tok.data());
    CHECK(p2 == 2);
    CHECK(libane_kv_cache_position(c) == 2);

    libane_kv_cache_release(c);
}

TEST_CASE("T2: libane_kv_cache_update writes correct data per head", "[kvcache][tier2]") {
    // H=2, D=4, S=3 — small enough to inspect byte-by-byte.
    const int H = 2, D = 4, S = 3;
    auto* c = libane_kv_cache_create(H, D, S);
    REQUIRE(c);

    // token 0: head0=[1,2,3,4], head1=[5,6,7,8] (fp16 1.0 = 0x3C00)
    // Use distinguishable fp16 bit patterns: h=0 → 0x3C00 (1.0), h=1 → 0x4000 (2.0)
    std::vector<uint16_t> tok(static_cast<size_t>(H) * D);
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
            tok[h * D + d] = static_cast<uint16_t>(0x3C00u + h * 0x0400u);

    libane_kv_cache_update(c, tok.data(), tok.data());

    const auto* k = static_cast<const uint16_t*>(libane_kv_cache_k(c));
    // Buffer layout: [head][max_seq][dim] = k[h * S * D + pos * D + d]
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            uint16_t expected = static_cast<uint16_t>(0x3C00u + h * 0x0400u);
            CHECK(k[static_cast<size_t>(h) * S * D + 0 * D + d] == expected);
        }
        // Position 1 and 2 must still be zero (not yet written)
        for (int p = 1; p < S; ++p)
            for (int d = 0; d < D; ++d)
                CHECK(k[static_cast<size_t>(h) * S * D + p * D + d] == 0);
    }

    libane_kv_cache_release(c);
}

TEST_CASE("T2: libane_kv_cache_update rejects full cache", "[kvcache][tier2]") {
    const int H = 2, D = 4, S = 2;
    auto* c = libane_kv_cache_create(H, D, S);
    REQUIRE(c);

    std::vector<uint16_t> tok(static_cast<size_t>(H) * D, 0);
    CHECK(libane_kv_cache_update(c, tok.data(), tok.data()) == 1);
    CHECK(libane_kv_cache_update(c, tok.data(), tok.data()) == 2);
    // Cache is now full (pos == max_seq == 2)
    CHECK(libane_kv_cache_update(c, tok.data(), tok.data()) == -1);

    libane_kv_cache_release(c);
}

TEST_CASE("T2: libane_kv_cache_reset clears position and buffers", "[kvcache][tier2]") {
    const int H = 2, D = 4, S = 8;
    auto* c = libane_kv_cache_create(H, D, S);
    REQUIRE(c);

    std::vector<uint16_t> tok(static_cast<size_t>(H) * D, 0x3C00u);
    libane_kv_cache_update(c, tok.data(), tok.data());
    libane_kv_cache_update(c, tok.data(), tok.data());
    REQUIRE(libane_kv_cache_position(c) == 2);

    libane_kv_cache_reset(c);
    CHECK(libane_kv_cache_position(c) == 0);

    // All K buffer bytes should be zero after reset
    const auto* k = static_cast<const uint16_t*>(libane_kv_cache_k(c));
    size_t total = static_cast<size_t>(H) * S * D;
    int nonzero = 0;
    for (size_t i = 0; i < total; ++i) if (k[i] != 0) ++nonzero;
    CHECK(nonzero == 0);

    libane_kv_cache_release(c);
}

TEST_CASE("T2: libane_kv_cache_release(NULL) does not crash", "[kvcache][tier2]") {
    libane_kv_cache_release(nullptr);
    libane_kv_cache_reset(nullptr);
    CHECK(libane_kv_cache_position(nullptr) == -1);
}
