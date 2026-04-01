/**
 * Tests for GraphCompiler::build_plan().
 *
 * build_plan() is pure C++ (no ANE hardware needed), so these tests run
 * everywhere.  compile() is hardware-gated and not tested here.
 */
#include <catch2/catch_test_macros.hpp>
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"

using namespace libane;
using namespace libane::graph;
using namespace libane::mil;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static TensorShape S(int C, int seq) { return {1, C, 1, seq}; }

static std::vector<uint8_t> fp16_weights(size_t n_elements) {
    // n_elements fp16 values, each = 0x3C00 (1.0f in fp16)
    std::vector<uint8_t> w(n_elements * 2, 0);
    for (size_t i = 0; i < n_elements; ++i) {
        w[i * 2]     = 0x00;
        w[i * 2 + 1] = 0x3C;   // 1.0 fp16 big-endian bytes
    }
    return w;
}

// Matmul weight: IC × OC fp16 elements
static std::vector<uint8_t> matmul_weights(int IC, int OC) {
    return fp16_weights(static_cast<size_t>(IC) * OC);
}

// RMSNorm scale: C fp16 elements
static std::vector<uint8_t> rmsnorm_weights(int C) {
    return fp16_weights(static_cast<size_t>(C));
}

// LayerNorm gamma+beta: 2*C fp16 elements (gamma then beta)
static std::vector<uint8_t> layernorm_weights(int C) {
    return fp16_weights(static_cast<size_t>(C) * 2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. build_plan basics
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: single matmul produces one group", "[compiler]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x   = g.add_input("x", S(512, 128));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128),
                             w.data(), w.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == out);
    CHECK(plan.graph_input_ids  == std::vector<TensorId>{x});
    CHECK(plan.graph_output_ids == std::vector<TensorId>{out});
}

TEST_CASE("build_plan: tensor_bytes populated for all tensors", "[compiler]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x   = g.add_input("x", S(512, 128));
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128),
                             w.data(), w.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // x: 512 * 128 * 2 bytes = 131072
    REQUIRE(plan.tensor_bytes.count(x));
    CHECK(plan.tensor_bytes.at(x) == S(512, 128).bytes());

    // out: 256 * 128 * 2 bytes = 65536
    REQUIRE(plan.tensor_bytes.count(out));
    CHECK(plan.tensor_bytes.at(out) == S(256, 128).bytes());
}

TEST_CASE("build_plan: buffer_ids equals group output ids", "[compiler]") {
    AneGraph g;
    auto w1 = matmul_weights(512, 256);
    auto w2 = matmul_weights(256, 128);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL, {x},  S(256, 128),
                             w1.data(), w1.size());
    TensorId t2 = g.add_op(LIBANE_OP_MATMUL, {t1}, S(128, 128),
                             w2.data(), w2.size());
    g.mark_output(t2);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.buffer_ids.size() == plan.groups.size());
    for (size_t i = 0; i < plan.groups.size(); ++i)
        CHECK(plan.buffer_ids[i] == plan.groups[i].output);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Fusing chains
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: matmul+gelu fuse into one group", "[compiler][fusion]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL, {x},  S(256, 128),
                             w.data(), w.size());
    TensorId t2 = g.add_op(LIBANE_OP_GELU,   {t1}, S(256, 128));
    g.mark_output(t2);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 2);
    CHECK(plan.groups[0].output == t2);
}

TEST_CASE("build_plan: matmul+rmsnorm+gelu fuse into one group", "[compiler][fusion]") {
    AneGraph g;
    auto wm  = matmul_weights(512, 256);
    auto wrn = rmsnorm_weights(256);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL,  {x},  S(256, 128),
                             wm.data(),  wm.size());
    TensorId t2 = g.add_op(LIBANE_OP_RMSNORM, {t1}, S(256, 128),
                             wrn.data(), wrn.size());
    TensorId t3 = g.add_op(LIBANE_OP_GELU,    {t2}, S(256, 128));
    g.mark_output(t3);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 3);
}

TEST_CASE("build_plan: branching prevents fusion across split", "[compiler][fusion]") {
    // x -> matmul -> t1 (consumed by both t2 and t3, so t1 is a branch point)
    AneGraph g;
    auto wm1 = matmul_weights(512, 256);
    auto wm2 = matmul_weights(256, 128);
    auto wm3 = matmul_weights(256, 128);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL, {x},  S(256, 128),
                             wm1.data(), wm1.size());
    TensorId t2 = g.add_op(LIBANE_OP_MATMUL, {t1}, S(128, 128),
                             wm2.data(), wm2.size());
    TensorId t3 = g.add_op(LIBANE_OP_MATMUL, {t1}, S(128, 128),
                             wm3.data(), wm3.size());
    g.mark_output(t2);
    g.mark_output(t3);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // t1 is consumed by both t2 and t3 → no fusion across; at minimum 3 groups
    CHECK(plan.groups.size() >= 3);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Binary ops in plan
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: add has two inputs in group", "[compiler][fusion]") {
    AneGraph g;
    TensorId a   = g.add_input("a", S(256, 128));
    TensorId b   = g.add_input("b", S(256, 128));
    TensorId out = g.add_op(LIBANE_OP_ADD, {a, b}, S(256, 128));
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    const auto& grp = plan.groups[0];
    CHECK(grp.inputs.size() == 2);
    CHECK(grp.output == out);
}

TEST_CASE("build_plan: matmul+add fuses with side input tracked", "[compiler][fusion]") {
    AneGraph g;
    auto wm = matmul_weights(512, 256);
    TensorId x   = g.add_input("x",  S(512, 128));
    TensorId res = g.add_input("res", S(256, 128));
    TensorId t1  = g.add_op(LIBANE_OP_MATMUL, {x},       S(256, 128),
                              wm.data(), wm.size());
    TensorId t2  = g.add_op(LIBANE_OP_ADD,    {t1, res}, S(256, 128));
    g.mark_output(t2);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    const auto& grp = plan.groups[0];
    // chain input (x) + side input (res)
    CHECK(grp.inputs.size() == 2);
    CHECK(grp.output == t2);
    CHECK(grp.node_ids.size() == 2);
}

TEST_CASE("build_plan: matmul+avg_pool+max_pool can fuse as unary chain", "[compiler][fusion]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL,   {x},  S(256, 128), w.data(), w.size());
    TensorId t2 = g.add_op(LIBANE_OP_AVG_POOL, {t1}, S(256, 128));
    TensorId t3 = g.add_op(LIBANE_OP_MAX_POOL, {t2}, S(256, 128));
    g.mark_output(t3);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 3);
    CHECK(plan.groups[0].output == t3);
}

TEST_CASE("build_plan: logical_or and logical_xor are supported binary ops", "[compiler][fusion]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(256, 128));
    TensorId b = g.add_input("b", S(256, 128));
    TensorId o = g.add_op(LIBANE_OP_LOGICAL_OR, {a, b}, S(256, 128));
    TensorId x = g.add_op(LIBANE_OP_LOGICAL_XOR, {o, b}, S(256, 128));
    g.mark_output(x);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 2);
    CHECK(plan.groups[0].output == x);
}

TEST_CASE("build_plan: reduce_prod unary reduction node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(256, 128));
    TensorId r = g.add_op(LIBANE_OP_REDUCE_PROD, {x}, S(1, 128));
    g.mark_output(r);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == r);
}

TEST_CASE("build_plan: scatter static-mask node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(64, 128));
    TensorId u = g.add_input("u", S(64, 128));
    auto mask = fp16_weights(static_cast<size_t>(64) * 128);
    TensorId o = g.add_op(LIBANE_OP_SCATTER, {a, u}, S(64, 128), mask.data(), mask.size());
    g.mark_output(o);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == o);
}

TEST_CASE("build_plan: gather static-mask node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(64, 128));
    auto mask = fp16_weights(static_cast<size_t>(64) * 128);
    TensorId o = g.add_op(LIBANE_OP_GATHER, {x}, S(64, 128), mask.data(), mask.size());
    g.mark_output(o);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == o);
}

TEST_CASE("build_plan: gather dynamic-mask node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(64, 128));
    TensorId m = g.add_input("m", S(64, 128));
    TensorId o = g.add_op(LIBANE_OP_GATHER, {x, m}, S(64, 128));
    g.mark_output(o);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == o);
}

TEST_CASE("build_plan: scatter_nd static-mask node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(64, 128));
    TensorId u = g.add_input("u", S(64, 128));
    auto mask = fp16_weights(static_cast<size_t>(64) * 128);
    TensorId o = g.add_op(LIBANE_OP_SCATTER_ND, {a, u}, S(64, 128), mask.data(), mask.size());
    g.mark_output(o);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == o);
}

TEST_CASE("build_plan: scatter_along_axis static-mask node supported", "[compiler][fusion]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(64, 128));
    TensorId u = g.add_input("u", S(64, 128));
    auto mask = fp16_weights(static_cast<size_t>(64) * 128);
    TensorId o = g.add_op(LIBANE_OP_SCATTER_ALONG_AXIS, {a, u}, S(64, 128), mask.data(), mask.size());
    g.mark_output(o);

    ExecutionPlan plan = GraphCompiler::build_plan(g);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    CHECK(plan.groups[0].output == o);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Layernorm weight splitting
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: layernorm node present in a group", "[compiler]") {
    AneGraph g;
    auto wln = layernorm_weights(256);
    TensorId x   = g.add_input("x", S(256, 128));
    TensorId out = g.add_op(LIBANE_OP_LAYERNORM, {x}, S(256, 128),
                             wln.data(), wln.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups[0].node_ids.size() == 1);
    // Verify tensor_bytes are correct
    CHECK(plan.tensor_bytes.at(out) == S(256, 128).bytes());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Invalid graph — build_plan throws
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: throws on empty graph (no inputs)", "[compiler][error]") {
    AneGraph g;
    CHECK_THROWS_AS(GraphCompiler::build_plan(g), std::runtime_error);
}

TEST_CASE("build_plan: throws on graph with no outputs", "[compiler][error]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x = g.add_input("x", S(512, 128));
    g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128), w.data(), w.size());
    // no mark_output
    CHECK_THROWS_AS(GraphCompiler::build_plan(g), std::runtime_error);
}

TEST_CASE("build_plan: throws on bad shape (seq not multiple of 8)", "[compiler][error]") {
    AneGraph g;
    // Shape S(512, 100) — seq=100 is not a multiple of 8
    CHECK_THROWS(g.add_input("x", {1, 512, 1, 100}));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Multi-output graph
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: multi-output graph tracks all output ids", "[compiler]") {
    AneGraph g;
    auto w1 = matmul_weights(512, 256);
    auto w2 = matmul_weights(512, 128);
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId t1 = g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128),
                             w1.data(), w1.size());
    TensorId t2 = g.add_op(LIBANE_OP_MATMUL, {x}, S(128, 128),
                             w2.data(), w2.size());
    g.mark_output(t1, "out1");
    g.mark_output(t2, "out2");

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.graph_output_ids.size() == 2);
    CHECK(plan.graph_output_ids[0] == t1);
    CHECK(plan.graph_output_ids[1] == t2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. FFN sub-graph (matmul→silu, matmul→mul→matmul)
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: SwiGLU FFN groups are sensible", "[compiler][fusion]") {
    // x -> gate_proj(matmul+silu) ──┐
    // x -> up_proj(matmul)   ───────┤ mul -> down_proj(matmul) -> out
    AneGraph g;

    auto wg = matmul_weights(512, 1024);
    auto wu = matmul_weights(512, 1024);
    auto wd = matmul_weights(1024, 512);

    TensorId x  = g.add_input("x", S(512, 128));

    // gate path: matmul + silu
    TensorId gate = g.add_op(LIBANE_OP_MATMUL, {x},    S(1024, 128),
                              wg.data(), wg.size());
    TensorId act  = g.add_op(LIBANE_OP_SILU,   {gate}, S(1024, 128));

    // up path: matmul only
    TensorId up   = g.add_op(LIBANE_OP_MATMUL, {x},    S(1024, 128),
                              wu.data(), wu.size());

    // merge: elementwise mul
    TensorId merged = g.add_op(LIBANE_OP_MUL, {act, up}, S(1024, 128));

    // down proj
    TensorId out = g.add_op(LIBANE_OP_MATMUL, {merged}, S(512, 128),
                             wd.data(), wd.size());
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    // We expect at least 3 groups (gate+silu | up | mul+down or similar).
    // The exact count depends on fusion rules; just verify sanity:
    CHECK(plan.groups.size() >= 2);
    CHECK(plan.graph_output_ids == std::vector<TensorId>{out});
    CHECK(plan.tensor_bytes.at(out) == S(512, 128).bytes());

    // Every group output has an entry in buffer_ids
    CHECK(plan.buffer_ids.size() == plan.groups.size());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. Transpose
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_plan: transpose node in group, output shape flipped", "[compiler]") {
    AneGraph g;
    // [1, 256, 1, 128] -> transpose -> [1, 128, 1, 256]
    TensorId x   = g.add_input("x", S(256, 128));
    TensorId out = g.add_op(LIBANE_OP_TRANSPOSE, {x}, S(128, 256));
    g.mark_output(out);

    ExecutionPlan plan = GraphCompiler::build_plan(g);

    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.tensor_bytes.at(out) == S(128, 256).bytes());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 9. compile() returns nullptr for an invalid graph
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("compile: returns nullptr for graph with no outputs", "[compiler][compile]") {
    AneGraph g;
    auto w = matmul_weights(512, 256);
    TensorId x = g.add_input("x", S(512, 128));
    g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128), w.data(), w.size());
    // no mark_output

    auto cg = GraphCompiler::compile(g);
    CHECK(cg == nullptr);
}
