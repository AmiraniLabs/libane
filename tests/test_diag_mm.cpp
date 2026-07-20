#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"
#include "graph/mil_backend.hpp"
#include <cstring>
#include <cstdio>
#include <vector>
using namespace libane; using namespace libane::graph; using namespace libane::mil;
using fp16 = __fp16;

TEST_CASE("diag: matmul_multi sparse probe") {
    if (!libane_available()) SKIP("ANE not available");
    // Matrix-tensor shapes: A=[1,1,K,M], B=[1,1,N,K], C=[1,1,N,M]
    // C[n,m] = Σ_k A[k,m]*B[n,k]
    constexpr int K=64, M=512, N=512;
    AneGraph g;
    auto a = g.add_input("a", {1,1,K,M});
    auto b = g.add_input("b", {1,1,N,K});
    auto c = g.add_op(LIBANE_OP_MATMUL_MULTI, {a,b}, {1,1,N,M});
    g.mark_output(c,"c");
    MilBackend mil;
    auto cg = GraphCompiler::compile(g, mil);
    if (!cg) SKIP("compile failed");

    const size_t na=K*M, nb=N*K, nc=N*M;
    std::vector<fp16> av(na,(__fp16)0.f), bv(nb,(__fp16)0.f), cv(nc,(__fp16)0.f);
    // A[k=0,m=1]=3: a[k*M+m] = a[1]
    // B[n=2,k=0]=5: b[n*K+k] = b[2*64+0] = b[128]
    av[0*M + 1] = (__fp16)3.0f;
    bv[2*K + 0] = (__fp16)5.0f;

    GraphExecutor::execute(*cg,
        {av.data(),bv.data()},{na*2,nb*2},{cv.data()},{nc*2});

    printf("Sparse probe: A[k=0,m=1]=3, B[n=2,k=0]=5\n");
    printf("Expect c[n=2,m=1] = c[%d] = 15\n", 2*M+1);
    int found = 0;
    for (size_t i = 0; i < nc && found < 20; i++) {
        float v = (float)cv[i];
        if (v != 0.0f) {
            int n_idx = (int)(i / M);
            int m_idx = (int)(i % M);
            printf("  c[%zu] = %.1f  (n=%d, m=%d)\n", i, v, n_idx, m_idx);
            found++;
        }
    }
    if (found == 0) printf("  ALL ZERO!\n");
}
