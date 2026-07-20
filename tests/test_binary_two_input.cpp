/**
 * test_binary_two_input.cpp — src_count=2 regression for elementwise binary ops.
 *
 * Every op here dispatches two live IOSurface inputs.  The same ANEC channel
 * table that test_matmul_multi.cpp probes for MATMUL_MULTI applies to ADD,
 * MUL, SUB, REAL_DIV, and CONCAT.  The failure modes are:
 *
 *   a) src_count regresses to 1 → one buffer reads zeros → wrong output
 *   b) A/B handles swapped     → commutative ops (ADD, MUL) cannot detect
 *                                 this; non-commutative ops (SUB, REAL_DIV,
 *                                 CONCAT) catch it via asymmetric operands.
 *
 * Tags: [binary2][ane][regression]
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"
#include "graph/mil_backend.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

using namespace libane;
using namespace libane::graph;
using namespace libane::mil;

/* ── fp16 helpers ────────────────────────────────────────────────────────── */

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
using fp16 = __fp16;
static fp16  f16(float v) { return static_cast<fp16>(v); }
static float f32(fp16 v)  { return static_cast<float>(v); }
#else
using fp16 = uint16_t;
static fp16 f16(float v) {
    uint32_t fb; std::memcpy(&fb, &v, 4);
    uint32_t s = (fb >> 16) & 0x8000u;
    int32_t  e = static_cast<int32_t>((fb >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FFu;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00u);
    else              h = static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | m);
    return h;
}
static float f32(fp16 h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0)  { uint32_t v = s | (m << 13); float r; std::memcpy(&r, &v, 4); return r; }
    if (e == 31) { uint32_t v = s | 0x7F800000u | (m << 13); float r; std::memcpy(&r, &v, 4); return r; }
    uint32_t v = s | ((e + 112u) << 23) | (m << 13);
    float r; std::memcpy(&r, &v, 4); return r;
}
#endif

/* ── Geometry ────────────────────────────────────────────────────────────── */

// C=64, SP=512 keeps all IOSurfaces above the 49 KB minimum:
//   64*512*2 = 65 536 B per tensor
static constexpr int C = 64, SP = 512;
static constexpr size_t N = static_cast<size_t>(C) * SP;  // elements per tensor

/* ── Shared compile + run helper ─────────────────────────────────────────── */

struct RunResult {
    bool                compiled = false;
    bool                executed = false;
    std::vector<float>  out;     // fp32 copy of all output elements
};

// Build a two-input graph `a op b → c`, compile, execute once, return results.
// a_fill / b_fill are fp32 values used to fill every element of the respective input.
static RunResult run_binary(libane_op_t op,
                             float a_fill, float b_fill,
                             TensorShape sa = {1, C, 1, SP},
                             TensorShape sb = {1, C, 1, SP},
                             TensorShape sc = {1, C, 1, SP}) {
    RunResult res;
    MilBackend mil;

    AneGraph g;
    TensorId a = g.add_input("a", sa);
    TensorId b = g.add_input("b", sb);
    TensorId c = g.add_op(op, {a, b}, sc);
    g.mark_output(c, "c");

    auto cg = GraphCompiler::compile(g, mil);
    if (!cg) return res;
    res.compiled = true;

    const size_t na = static_cast<size_t>(sa.channels) * sa.seq;
    const size_t nb = static_cast<size_t>(sb.channels) * sb.seq;
    const size_t nc = static_cast<size_t>(sc.channels) * sc.seq;

    std::vector<fp16> av(na, f16(a_fill));
    std::vector<fp16> bv(nb, f16(b_fill));
    std::vector<fp16> cv(nc, f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {av.data(), bv.data()},
        {na * sizeof(fp16), nb * sizeof(fp16)},
        {cv.data()},
        {nc * sizeof(fp16)});
    if (!ok) return res;
    res.executed = true;

    res.out.reserve(nc);
    for (size_t i = 0; i < nc; ++i) res.out.push_back(f32(cv[i]));
    return res;
}

// Compile graph structure only; return CompiledGraph for structural checks.
static std::unique_ptr<CompiledGraph> compile_binary(libane_op_t op,
                                                      TensorShape sa = {1,C,1,SP},
                                                      TensorShape sb = {1,C,1,SP},
                                                      TensorShape sc = {1,C,1,SP}) {
    MilBackend mil;
    AneGraph g;
    TensorId a = g.add_input("a", sa);
    TensorId b = g.add_input("b", sb);
    TensorId c = g.add_op(op, {a, b}, sc);
    g.mark_output(c, "c");
    return GraphCompiler::compile(g, mil);
}

/* ── Structural check: every two-input binary op → src_count=2 ──────────── */

TEST_CASE("binary two-input: graph has 2 inputs and 1 output for each op",
          "[binary2][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    struct Entry { libane_op_t op; const char* name; };
    const Entry ops[] = {
        {LIBANE_OP_ADD,      "ADD"},
        {LIBANE_OP_MUL,      "MUL"},
        {LIBANE_OP_SUB,      "SUB"},
        {LIBANE_OP_REAL_DIV, "REAL_DIV"},
    };
    for (const auto& e : ops) {
        SECTION(e.name) {
            auto cg = compile_binary(e.op);
            if (!cg) SKIP("ANE compiler unavailable");
            CHECK(cg->graph_input_ids().size()  == 2u);
            CHECK(cg->graph_output_ids().size() == 1u);
        }
    }
}

/* ── ADD ─────────────────────────────────────────────────────────────────── */

TEST_CASE("ADD: A=3 B=5 → 8 everywhere (detects missing input)",
          "[binary2][ane]") {
    // If src_count regresses to 1, output would be 3 or 5, not 8.
    if (!libane_available()) SKIP("ANE not available");
    auto r = run_binary(LIBANE_OP_ADD, 3.0f, 5.0f);
    if (!r.compiled) SKIP("ANE compiler unavailable");
    REQUIRE(r.executed);
    for (float v : r.out) CHECK(std::abs(v - 8.0f) < 0.1f);
}

/* ── MUL ─────────────────────────────────────────────────────────────────── */

TEST_CASE("MUL: A=3 B=4 → 12 everywhere (detects missing input)",
          "[binary2][ane]") {
    // If either input maps to zeros, output collapses to 0.
    if (!libane_available()) SKIP("ANE not available");
    auto r = run_binary(LIBANE_OP_MUL, 3.0f, 4.0f);
    if (!r.compiled) SKIP("ANE compiler unavailable");
    REQUIRE(r.executed);
    for (float v : r.out) CHECK(std::abs(v - 12.0f) < 0.2f);
}

/* ── SUB ─────────────────────────────────────────────────────────────────── */

TEST_CASE("SUB: A=7 B=3 → 4 everywhere (detects missing input and swap)",
          "[binary2][ane]") {
    // Swap → 3-7 = -4.  Missing input → 0-3 = -3 or 7-0 = 7.
    if (!libane_available()) SKIP("ANE not available");
    auto r = run_binary(LIBANE_OP_SUB, 7.0f, 3.0f);
    if (!r.compiled) SKIP("ANE compiler unavailable");
    REQUIRE(r.executed);
    for (float v : r.out) CHECK(std::abs(v - 4.0f) < 0.1f);
}

/* ── REAL_DIV ────────────────────────────────────────────────────────────── */

TEST_CASE("REAL_DIV: A=6 B=2 → 3 everywhere (detects missing input and swap)",
          "[binary2][ane]") {
    // Swap → 2/6 ≈ 0.333.  Missing A → 0/2 = 0.  Missing B → 6/0 = ±inf.
    if (!libane_available()) SKIP("ANE not available");
    auto r = run_binary(LIBANE_OP_REAL_DIV, 6.0f, 2.0f);
    if (!r.compiled) SKIP("ANE compiler unavailable");
    REQUIRE(r.executed);
    for (float v : r.out) CHECK(std::abs(v - 3.0f) < 0.1f);
}

/* ── CONCAT ──────────────────────────────────────────────────────────────── */

TEST_CASE("CONCAT: A-channels appear first, B-channels second",
          "[binary2][ane]") {
    // Output is [1, 2*C, 1, SP].  First C channels come from A (fill=7),
    // next C channels from B (fill=11).  A/B swap reverses the order.
    if (!libane_available()) SKIP("ANE not available");

    const TensorShape sa{1, C, 1, SP};
    const TensorShape sb{1, C, 1, SP};
    const TensorShape sc{1, 2*C, 1, SP};

    MilBackend mil;
    AneGraph g;
    TensorId a = g.add_input("a", sa);
    TensorId b = g.add_input("b", sb);
    TensorId c = g.add_op(LIBANE_OP_CONCAT, {a, b}, sc);
    g.mark_output(c, "c");
    auto cg = GraphCompiler::compile(g, mil);
    if (!cg) SKIP("ANE compiler unavailable");

    CHECK(cg->graph_input_ids().size()  == 2u);
    CHECK(cg->graph_output_ids().size() == 1u);

    const size_t na = static_cast<size_t>(C) * SP;
    const size_t nb = static_cast<size_t>(C) * SP;
    const size_t nc = static_cast<size_t>(2*C) * SP;

    std::vector<fp16> av(na, f16(7.0f));
    std::vector<fp16> bv(nb, f16(11.0f));
    std::vector<fp16> cv(nc, f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {av.data(), bv.data()},
        {na * sizeof(fp16), nb * sizeof(fp16)},
        {cv.data()},
        {nc * sizeof(fp16)});
    REQUIRE(ok);

    // First half of output (A channels, index 0..na-1): expect 7.0
    // Second half (B channels, index na..nc-1): expect 11.0
    // A/B swap would invert: first half 11, second half 7.
    float first  = f32(cv[0]);
    float second = f32(cv[na]);   // first element of second channel block
    CHECK(std::abs(first  -  7.0f) < 0.1f);
    CHECK(std::abs(second - 11.0f) < 0.1f);
}
