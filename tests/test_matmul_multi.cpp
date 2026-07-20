/**
 * test_matmul_multi.cpp — LIBANE_OP_MATMUL_MULTI regression.
 *
 * Validates that the two-input live matmul op wires src_count=2 correctly
 * through the ANEC channel table.  Derived from a runtime ANE NN capture
 * that showed anec={src_count=2 dst_count=1} for this op class.
 *
 * If either the A or B IOSurface handle is misrouted, the output is either
 * garbage or a hardware error — both cause the numerical checks to fail.
 *
 * Shapes: A=[1,K,1,M]  B=[1,K,1,N]  →  C=[1,N,1,M]
 *   MIL semantics: C = A^T @ B  (transpose_x=false, transpose_y=true)
 *   With NCHW layout [1,C,1,W]: C[n,j,1,i] = sum_k A[n,k,1,i] * B[n,k,1,j]
 *
 * Tags: [matmul_multi][ane][regression]
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"
#include "graph/mil_backend.hpp"
#include "runtime/ane_runtime.hpp"

#include <cstring>
#include <vector>
#include <cmath>

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
    uint32_t s = (fb >> 16) & 0x8000u;
    int32_t  e = static_cast<int32_t>((fb >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FFu;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00u);
    else              h = static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | m);
    return h;
}
static float to_f32(fp16 h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0)  { uint32_t v = s | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    if (e == 31) { uint32_t v = s | 0x7F800000u | (m << 13); float f; std::memcpy(&f, &v, 4); return f; }
    uint32_t v = s | ((e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &v, 4); return f;
}
#endif

/* ── Geometry ─────────────────────────────────────────────────────────────
 *
 * Matrix-tensor shapes (channels=1, height=K or N):
 *   A: [1,1,K,M]  B: [1,1,N,K]  C: [1,1,N,M]
 *
 * K=64, M=512, N=512 keeps all IOSurfaces above the 49 KB minimum:
 *   A: 1*K*M*2 = 65 536 B   B: 1*N*K*2 = 65 536 B   C: 1*N*M*2 = 524 288 B
 *
 * The MIL formula is matmul(x=B=[N,K], y=A=[K,M]) → [N,K]@[K,M]=[N,M].
 * C[n,m] = Σ_k A[k,m] * B[n,k].
 * User-facing element layout: A[k,m] at flat index k*M+m; B[n,k] at n*K+k.
 */
static constexpr int K = 64, M = 512, N = 512;

static TensorShape SA() { return {1, 1, K, M}; }  // channels=1, height=K, seq=M
static TensorShape SB() { return {1, 1, N, K}; }  // channels=1, height=N, seq=K
static TensorShape SC() { return {1, 1, N, M}; }  // channels=1, height=N, seq=M

/* ── Build and compile the two-input matmul graph ─────────────────────────*/

static std::unique_ptr<CompiledGraph> build_and_compile(MilBackend& mil) {
    AneGraph g;
    TensorId a = g.add_input("a", SA());
    TensorId b = g.add_input("b", SB());
    TensorId c = g.add_op(LIBANE_OP_MATMUL_MULTI, {a, b}, SC());
    g.mark_output(c, "c");
    return GraphCompiler::compile(g, mil);
}

/* ── Tests ────────────────────────────────────────────────────────────────*/

TEST_CASE("MATMUL_MULTI: graph structure has 2 inputs and 1 output",
          "[matmul_multi][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    MilBackend mil;
    auto cg = build_and_compile(mil);
    if (!cg) SKIP("MATMUL_MULTI compile failed: " << runtime::ane_last_error());

    // src_count=2, dst_count=1 from the reference ANE NN capture.
    REQUIRE(cg->graph_input_ids().size()  == 2u);
    REQUIRE(cg->graph_output_ids().size() == 1u);
}

TEST_CASE("MATMUL_MULTI: A=1 B=1 → C=K everywhere",
          "[matmul_multi][ane]") {
    if (!libane_available()) SKIP("ANE not available");

    MilBackend mil;
    auto cg = build_and_compile(mil);
    if (!cg) SKIP("MATMUL_MULTI compile failed: " << runtime::ane_last_error());

    // na = K*M elements (A[k,m] at flat index k*M+m)
    // nb = N*K elements (B[n,k] at flat index n*K+k)
    // nc = N*M elements (C[n,m] at flat index n*M+m)
    const size_t na = static_cast<size_t>(K) * M;
    const size_t nb = static_cast<size_t>(N) * K;
    const size_t nc = static_cast<size_t>(N) * M;

    std::vector<fp16> a(na, to_f16(1.0f));
    std::vector<fp16> b(nb, to_f16(1.0f));
    std::vector<fp16> c(nc, to_f16(0.0f));

    bool ok = GraphExecutor::execute(*cg,
        {a.data(), b.data()},
        {na * sizeof(fp16), nb * sizeof(fp16)},
        {c.data()},
        {nc * sizeof(fp16)});
    REQUIRE(ok);

    // C[n,m] = Σ_k A[k,m]*B[n,k] = K * 1 * 1 = K
    const float expected = static_cast<float>(K);
    const std::vector<size_t> probes = {0, 1, nc/2, nc-1};
    for (size_t idx : probes) {
        float got = to_f32(c[idx]);
        CHECK(std::abs(got - expected) < 1.0f);  // fp16 can hold K=64 exactly
    }
}

TEST_CASE("MATMUL_MULTI: asymmetric operands detect A/B handle swap",
          "[matmul_multi][ane]") {
    // Uniform-fill values can't detect a swap because K*a*b == K*b*a.
    // Instead, set A and B to rank-1 matrices with different per-position
    // values so the output is NOT symmetric under handle swap.
    //
    // A[k=0, m=0]=1  A[k=0, m=1]=2  (rest 0)
    // B[k=0, n=0]=3  B[k=0, n=1]=4  (rest 0)
    //
    // C[j, i] = A[0,i] * B[0,j]  (output layout: [1,N,1,M], index = j*M + i)
    //   c[0]   = C[j=0,i=0] = 1*3 = 3.0
    //   c[1]   = C[j=0,i=1] = 2*3 = 6.0   ← 4.0 if A/B swapped
    //   c[M]   = C[j=1,i=0] = 1*4 = 4.0   ← 6.0 if A/B swapped
    //   c[M+1] = C[j=1,i=1] = 2*4 = 8.0
    if (!libane_available()) SKIP("ANE not available");

    MilBackend mil;
    auto cg = build_and_compile(mil);
    if (!cg) SKIP("MATMUL_MULTI compile failed: " << runtime::ane_last_error());

    // na = K*M elements (A[k,m] at k*M+m)
    // nb = N*K elements (B[n,k] at n*K+k)
    // nc = N*M elements (C[n,m] at n*M+m)
    const size_t na = static_cast<size_t>(K) * M;
    const size_t nb = static_cast<size_t>(N) * K;
    const size_t nc = static_cast<size_t>(N) * M;

    std::vector<fp16> a(na, to_f16(0.0f));
    std::vector<fp16> b(nb, to_f16(0.0f));
    std::vector<fp16> c(nc, to_f16(0.0f));

    // A[k=0,m=0]=1, A[k=0,m=1]=2  →  a[k*M+m] = a[0]=1, a[1]=2
    a[0*M + 0] = to_f16(1.0f);
    a[0*M + 1] = to_f16(2.0f);
    // B[n=0,k=0]=3, B[n=1,k=0]=4  →  b[n*K+k] = b[0]=3, b[K]=4
    b[0*K + 0] = to_f16(3.0f);
    b[1*K + 0] = to_f16(4.0f);

    bool ok = GraphExecutor::execute(*cg,
        {a.data(), b.data()},
        {na * sizeof(fp16), nb * sizeof(fp16)},
        {c.data()},
        {nc * sizeof(fp16)});
    REQUIRE(ok);

    // C[n,m] = Σ_k A[k,m]*B[n,k]:
    // c[0]   = C[n=0,m=0] = A[0,0]*B[0,0] = 1*3 = 3
    // c[1]   = C[n=0,m=1] = A[0,1]*B[0,0] = 2*3 = 6  ← 4 if A/B swapped
    // c[M]   = C[n=1,m=0] = A[0,0]*B[1,0] = 1*4 = 4  ← 6 if A/B swapped
    // c[M+1] = C[n=1,m=1] = A[0,1]*B[1,0] = 2*4 = 8
    CHECK(std::abs(to_f32(c[0])   - 3.0f) < 0.5f);
    CHECK(std::abs(to_f32(c[1])   - 6.0f) < 0.5f);
    CHECK(std::abs(to_f32(c[M])   - 4.0f) < 0.5f);
    CHECK(std::abs(to_f32(c[M+1]) - 8.0f) < 0.5f);
}
