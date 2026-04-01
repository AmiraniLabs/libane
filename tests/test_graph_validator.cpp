#include <catch2/catch_test_macros.hpp>
#include "graph/ane_graph.hpp"
#include "graph/graph_validator.hpp"

using namespace libane::graph;
using libane::mil::TensorShape;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static TensorShape S(int C, int seq) { return {1, C, 1, seq}; }

// fp16 1.0 repeated
static std::vector<uint16_t> fp16_ones(size_t n) {
    return std::vector<uint16_t>(n, 0x3C00u);
}

// Build a minimal valid single-op graph and return the validation result
static ValidationResult validate_single(libane_op_t op,
                                         TensorShape in_shape,
                                         TensorShape out_shape,
                                         size_t weight_elems = 0) {
    AneGraph g;
    TensorId x = g.add_input("x", in_shape);
    auto w = fp16_ones(weight_elems);
    TensorId y = g.add_op(op, {x}, out_shape,
                           weight_elems ? w.data() : nullptr,
                           weight_elems * 2);
    g.mark_output(y);
    return GraphValidator::validate(g);
}

/* ── Check 1: structure ──────────────────────────────────────────────────── */

TEST_CASE("valid minimal graph passes", "[validator]") {
    auto r = validate_single(LIBANE_OP_GELU, S(512, 128), S(512, 128));
    CHECK(r.ok());
    CHECK(r.errors.empty());
}

TEST_CASE("no inputs fails", "[validator]") {
    AneGraph g;
    // add a node manually is impossible without inputs, so just mark an output
    // on an empty graph — we must add an input first, so the only way to get
    // "no inputs" is to never call add_input.
    // We can't mark_output without a tensor, so just validate immediately.
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
    CHECK(r.errors.size() >= 1);
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("no inputs") != std::string::npos) found = true;
    CHECK(found);
}

TEST_CASE("no outputs fails", "[validator]") {
    AneGraph g;
    g.add_input("x", S(512, 128));
    // never call mark_output
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("no outputs") != std::string::npos) found = true;
    CHECK(found);
}

/* ── Check 2: tensor shapes ──────────────────────────────────────────────── */

// Note: add_input / add_op already validate shapes and throw, so by the time
// we call GraphValidator::validate() the tensors in the graph are always valid.
// This test confirms that the validator's shape pass finds no extra issues for
// a well-formed graph and that the shape error path is exercised via a graph
// whose tensors were inserted before validate() runs.
TEST_CASE("all tensor shapes valid in well-formed graph", "[validator]") {
    auto r = validate_single(LIBANE_OP_GELU, S(512, 128), S(512, 128));
    CHECK(r.ok());
}

/* ── Check 4: topological sort / no cycles ───────────────────────────────── */

TEST_CASE("acyclic graph passes topo check", "[validator]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId h = g.add_op(LIBANE_OP_GELU, {x}, S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_SOFTMAX, {h}, S(512, 128));
    g.mark_output(y);
    auto r = GraphValidator::validate(g);
    CHECK(r.ok());
}

/* ── Check 5: reachability ───────────────────────────────────────────────── */

TEST_CASE("output reachable from input passes", "[validator]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, S(512, 128));
    g.mark_output(y);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("input tensor itself marked as output is reachable", "[validator]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    g.mark_output(x);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("chain of three ops: all reachable", "[validator]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId a = g.add_op(LIBANE_OP_GELU,    {x}, S(512, 128));
    TensorId b = g.add_op(LIBANE_OP_SOFTMAX, {a}, S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_SILU,    {b}, S(512, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

/* ── Check 6: weight sizes ───────────────────────────────────────────────── */

TEST_CASE("matmul: correct weight size passes", "[validator][weights]") {
    int IC = 512, OC = 256, SP = 128;
    // weight bytes = IC × OC × 2
    auto r = validate_single(LIBANE_OP_MATMUL, S(IC, SP), S(OC, SP),
                              static_cast<size_t>(IC) * OC);
    CHECK(r.ok());
}

TEST_CASE("matmul: wrong weight size fails", "[validator][weights]") {
    int IC = 512, OC = 256, SP = 128;
    // provide IC×OC - 1 elements
    auto r = validate_single(LIBANE_OP_MATMUL, S(IC, SP), S(OC, SP),
                              static_cast<size_t>(IC) * OC - 1);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("weight size mismatch") != std::string::npos) found = true;
    CHECK(found);
}

TEST_CASE("matmul: zero weights fails", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_MATMUL, S(512, 128), S(256, 128), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("rmsnorm: correct weight size passes", "[validator][weights]") {
    int C = 512, SP = 128;
    // scale = C elements
    auto r = validate_single(LIBANE_OP_RMSNORM, S(C, SP), S(C, SP),
                              static_cast<size_t>(C));
    CHECK(r.ok());
}

TEST_CASE("rmsnorm: wrong weight size fails", "[validator][weights]") {
    int C = 512, SP = 128;
    auto r = validate_single(LIBANE_OP_RMSNORM, S(C, SP), S(C, SP),
                              static_cast<size_t>(C) + 1);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("layernorm: correct weight size passes", "[validator][weights]") {
    int C = 512, SP = 128;
    // gamma + beta = 2 × C elements
    auto r = validate_single(LIBANE_OP_LAYERNORM, S(C, SP), S(C, SP),
                              static_cast<size_t>(C) * 2);
    CHECK(r.ok());
}

TEST_CASE("layernorm: wrong weight size fails", "[validator][weights]") {
    int C = 512, SP = 128;
    // supply only gamma, not beta
    auto r = validate_single(LIBANE_OP_LAYERNORM, S(C, SP), S(C, SP),
                              static_cast<size_t>(C));
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("LAYER_NORM alias same as LAYERNORM for weight check", "[validator][weights]") {
    int C = 512, SP = 128;
    auto r = validate_single(LIBANE_OP_LAYER_NORM, S(C, SP), S(C, SP),
                              static_cast<size_t>(C) * 2);
    CHECK(r.ok());
}

TEST_CASE("gelu: no weights passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_GELU, S(512, 128), S(512, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("gelu: unexpected weights fails", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_GELU, S(512, 128), S(512, 128), 10);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("weight-free") != std::string::npos) found = true;
    CHECK(found);
}

TEST_CASE("silu: no weights passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_SILU, S(512, 128), S(512, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("softmax: no weights passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_SOFTMAX, S(512, 128), S(512, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("transpose: no weights passes", "[validator][weights]") {
    // transpose [1,C,1,S] -> [1,S,1,C]: output channels = input seq, output seq = input channels
    auto r = validate_single(LIBANE_OP_TRANSPOSE, S(512, 128), S(128, 512), 0);
    CHECK(r.ok());
}

TEST_CASE("reshape: same-numel and no-weights passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_RESHAPE, S(512, 128), S(1024, 64), 0);
    CHECK(r.ok());
}

TEST_CASE("reshape: mismatched numel fails", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_RESHAPE, S(512, 128), S(1024, 128), 0);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("equal input/output element counts") != std::string::npos) found = true;
    CHECK(found);
}

TEST_CASE("reshape: unexpected weights fail", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_RESHAPE, S(512, 128), S(1024, 64), 8);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("concat: valid shape relation passes", "[validator][weights]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(256, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_CONCAT, {a, b}, S(768, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("concat: output channels mismatch fails", "[validator][weights]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(256, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_CONCAT, {a, b}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("concat: mismatched input seq fails", "[validator][weights]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(256, 128));
    TensorId b = g.add_input("b", S(512, 256));
    TensorId c = g.add_op(LIBANE_OP_CONCAT, {a, b}, S(768, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("slice_by_index: output <= input dims passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_SLICE_BY_INDEX, S(512, 128), S(256, 64), 0);
    CHECK(r.ok());
}

TEST_CASE("slice_by_index: output dims larger than input fails", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_SLICE_BY_INDEX, S(256, 64), S(512, 64), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("slice_by_index: unexpected weights fail", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_SLICE_BY_INDEX, S(512, 128), S(256, 64), 8);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("reduce_sum: valid reduced shape passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_SUM, S(512, 128), S(1, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("reduce_sum: output channels must be 1", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_SUM, S(512, 128), S(2, 128), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("reduce_sum: output seq must match input seq", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_SUM, S(512, 128), S(1, 64), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("reduce_mean: valid reduced shape passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_MEAN, S(512, 128), S(1, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("reduce_mean: output channels must be 1", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_MEAN, S(512, 128), S(4, 128), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("reduce_max: valid reduced shape passes", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_MAX, S(512, 128), S(1, 128), 0);
    CHECK(r.ok());
}

TEST_CASE("reduce_max: output seq must match input seq", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_REDUCE_MAX, S(512, 128), S(1, 64), 0);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("conv2d: always fails in graph API", "[validator][weights]") {
    auto r = validate_single(LIBANE_OP_CONV2D, S(512, 128), S(512, 128), 0);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("CONV2D") != std::string::npos) found = true;
    CHECK(found);
}

/* ── Check 7: binary op shape matching ──────────────────────────────────── */

TEST_CASE("add: matching input shapes passes", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_ADD, {a, b}, S(512, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("add: mismatched channels fails", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(256, 128));
    TensorId c = g.add_op(LIBANE_OP_ADD, {a, b}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("#18") != std::string::npos) found = true;
    CHECK(found);
}

TEST_CASE("add: mismatched seq fails", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 256));
    TensorId c = g.add_op(LIBANE_OP_ADD, {a, b}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("mul: matching input shapes passes", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_MUL, {a, b}, S(512, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("sub: matching input shapes passes", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_SUB, {a, b}, S(512, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("sub: mismatched shapes fail", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 64));
    TensorId c = g.add_op(LIBANE_OP_SUB, {a, b}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("real_div: matching input shapes passes", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_REAL_DIV, {a, b}, S(512, 128));
    g.mark_output(c);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("real_div: mismatched shapes fail", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(512, 64));
    TensorId c = g.add_op(LIBANE_OP_REAL_DIV, {a, b}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("sqrt: single-input no-weights passes", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_SQRT, {x}, S(512, 128));
    g.mark_output(y);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("sqrt: unexpected weights fail", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    uint16_t bogus = 0x3C00;
    TensorId y = g.add_op(LIBANE_OP_SQRT, {x}, S(512, 128), &bogus, sizeof(bogus));
    g.mark_output(y);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("log: single-input no-weights passes", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_LOG, {x}, S(512, 128));
    g.mark_output(y);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("log: unexpected weights fail", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    uint16_t bogus = 0x3C00;
    TensorId y = g.add_op(LIBANE_OP_LOG, {x}, S(512, 128), &bogus, sizeof(bogus));
    g.mark_output(y);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

TEST_CASE("rsqrt: single-input no-weights passes", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_RSQRT, {x}, S(512, 128));
    g.mark_output(y);
    CHECK(GraphValidator::validate(g).ok());
}

TEST_CASE("rsqrt: unexpected weights fail", "[validator][unary]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    uint16_t bogus = 0x3C00;
    TensorId y = g.add_op(LIBANE_OP_RSQRT, {x}, S(512, 128), &bogus, sizeof(bogus));
    g.mark_output(y);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
}

/* ── ADD missing second input ────────────────────────────────────────────── */

TEST_CASE("add: one input fails weight check", "[validator][binary]") {
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId c = g.add_op(LIBANE_OP_ADD, {a}, S(512, 128));
    g.mark_output(c);
    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
    bool found = false;
    for (const auto& e : r.errors)
        if (e.find("two inputs") != std::string::npos) found = true;
    CHECK(found);
}

/* ── Full FFN block ──────────────────────────────────────────────────────── */

TEST_CASE("full FFN block: linear -> gelu -> linear passes", "[validator][integration]") {
    const int D = 512, FF = 2048, SP = 128;

    AneGraph g;
    TensorId x = g.add_input("x", S(D, SP));

    // Up-projection: [1, D, 1, SP] -> [1, FF, 1, SP]
    auto w1 = fp16_ones(static_cast<size_t>(D) * FF);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(FF, SP),
                           w1.data(), w1.size() * 2);

    // GELU
    TensorId a = g.add_op(LIBANE_OP_GELU, {h}, S(FF, SP));

    // Down-projection: [1, FF, 1, SP] -> [1, D, 1, SP]
    auto w2 = fp16_ones(static_cast<size_t>(FF) * D);
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {a}, S(D, SP),
                           w2.data(), w2.size() * 2);

    g.mark_output(y, "ffn_out");

    auto r = GraphValidator::validate(g);
    CHECK(r.ok());
    if (!r.ok()) {
        for (const auto& e : r.errors)
            WARN(e);
    }
}

/* ── Multiple errors accumulate ──────────────────────────────────────────── */

TEST_CASE("multiple violations all reported", "[validator]") {
    // Graph with: wrong matmul weights AND mismatched binary shapes
    AneGraph g;
    TensorId a = g.add_input("a", S(512, 128));
    TensorId b = g.add_input("b", S(256, 128)); // different C

    // Wrong weight size for matmul (give 0 instead of 512*256)
    TensorId m = g.add_op(LIBANE_OP_MATMUL, {a}, S(256, 128)); // 0 weights

    // Mismatched binary shapes
    TensorId c = g.add_op(LIBANE_OP_ADD, {a, b}, S(512, 128));

    g.mark_output(m);
    g.mark_output(c);

    auto r = GraphValidator::validate(g);
    REQUIRE_FALSE(r.ok());
    // Expect at least two errors: weight mismatch + shape mismatch
    CHECK(r.errors.size() >= 2);
}
