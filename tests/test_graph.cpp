#include <catch2/catch_test_macros.hpp>
#include "graph/ane_graph.hpp"

using namespace libane::graph;
using libane::mil::TensorShape;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static TensorShape shape(int C, int S) {
    return {1, C, 1, S};
}

/* ── add_input ───────────────────────────────────────────────────────────── */

TEST_CASE("add_input assigns sequential IDs", "[graph][ir]") {
    AneGraph g;

    TensorId a = g.add_input("x", shape(512, 128));
    TensorId b = g.add_input("y", shape(256, 128));

    CHECK(a == 0);
    CHECK(b == 1);

    CHECK(g.tensors().size() == 2);
    CHECK(g.graph_inputs().size() == 2);
    CHECK(g.graph_inputs()[0] == a);
    CHECK(g.graph_inputs()[1] == b);
}

TEST_CASE("add_input stores shape and name", "[graph][ir]") {
    AneGraph g;
    TensorId id = g.add_input("activations", shape(768, 512));

    const auto& t = g.tensor(id);
    CHECK(t.name == "activations");
    CHECK(t.shape.channels == 768);
    CHECK(t.shape.seq == 512);
    CHECK(t.producer_node_id == kInvalidNodeId);
}

TEST_CASE("add_input rejects bad ANE shape", "[graph][ir]") {
    AneGraph g;

    // seq not multiple of 8
    CHECK_THROWS_AS(g.add_input("bad", shape(512, 7)), std::invalid_argument);

    // batch != 1
    TensorShape bad_batch{2, 512, 1, 128};
    CHECK_THROWS_AS(g.add_input("bad_batch", bad_batch), std::invalid_argument);

    // height != 1
    TensorShape bad_height{1, 512, 3, 128};
    CHECK_THROWS_AS(g.add_input("bad_height", bad_height), std::invalid_argument);
}

/* ── add_op ──────────────────────────────────────────────────────────────── */

TEST_CASE("add_op returns a new tensor ID", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));

    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));

    // x is id 0, y should be id 1 (next tensor)
    CHECK(y == 1);
    CHECK(g.tensors().size() == 2);
    CHECK(g.nodes().size() == 1);
}

TEST_CASE("add_op node records op and inputs", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));

    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));

    const auto& n = g.node(0);
    CHECK(n.id == 0);
    CHECK(n.op == LIBANE_OP_GELU);
    REQUIRE(n.inputs.size() == 1);
    CHECK(n.inputs[0] == x);
    CHECK(n.output == y);
}

TEST_CASE("add_op weight-bearing node stores weights and file name", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));

    // 512×256 fp16 matmul weights
    std::vector<uint16_t> wdata(512 * 256, 0x3C00); // 1.0 in fp16
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {x}, shape(256, 128),
                           wdata.data(), wdata.size() * 2);

    const auto& n = g.node(0);
    CHECK(n.weights.size() == wdata.size() * 2);
    CHECK(!n.weight_file.empty());
    CHECK(n.weight_file == "w0.bin");
}

TEST_CASE("add_op weight-free node has empty weights and file", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));
    g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));

    const auto& n = g.node(0);
    CHECK(n.weights.empty());
    CHECK(n.weight_file.empty());
}

TEST_CASE("add_op output tensor has correct producer", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));

    CHECK(g.tensor(y).producer_node_id == 0);
}

TEST_CASE("add_op rejects bad output shape", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));

    CHECK_THROWS_AS(
        g.add_op(LIBANE_OP_GELU, {x}, shape(512, 7)),
        std::invalid_argument);
}

TEST_CASE("add_op rejects invalid input tensor ID", "[graph][ir]") {
    AneGraph g;
    g.add_input("x", shape(512, 128));

    CHECK_THROWS_AS(
        g.add_op(LIBANE_OP_GELU, {99u}, shape(512, 128)),
        std::invalid_argument);
}

/* ── Chained ops ─────────────────────────────────────────────────────────── */

TEST_CASE("chain of ops: matmul -> gelu -> matmul", "[graph][ir]") {
    AneGraph g;

    TensorId x = g.add_input("x", shape(512, 128));

    std::vector<uint16_t> w1(512 * 2048, 0x3C00);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, shape(2048, 128),
                           w1.data(), w1.size() * 2);

    TensorId a = g.add_op(LIBANE_OP_GELU, {h}, shape(2048, 128));

    std::vector<uint16_t> w2(2048 * 512, 0x3C00);
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {a}, shape(512, 128),
                           w2.data(), w2.size() * 2);

    CHECK(g.nodes().size() == 3);
    CHECK(g.tensors().size() == 4); // x, h, a, y

    // Producer chain is correct
    CHECK(g.tensor(x).producer_node_id == kInvalidNodeId);
    CHECK(g.tensor(h).producer_node_id == 0);
    CHECK(g.tensor(a).producer_node_id == 1);
    CHECK(g.tensor(y).producer_node_id == 2);

    // Node input/output linkage
    CHECK(g.node(0).inputs[0] == x);
    CHECK(g.node(0).output    == h);
    CHECK(g.node(1).inputs[0] == h);
    CHECK(g.node(1).output    == a);
    CHECK(g.node(2).inputs[0] == a);
    CHECK(g.node(2).output    == y);
}

/* ── mark_output ─────────────────────────────────────────────────────────── */

TEST_CASE("mark_output registers outputs in order", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));

    g.mark_output(y, "result");

    REQUIRE(g.graph_outputs().size() == 1);
    CHECK(g.graph_outputs()[0] == y);
    CHECK(g.tensor(y).name == "result");
}

TEST_CASE("mark_output rejects out-of-range tensor ID", "[graph][ir]") {
    AneGraph g;
    CHECK_THROWS_AS(g.mark_output(0), std::invalid_argument);
}

TEST_CASE("multiple outputs", "[graph][ir]") {
    AneGraph g;
    TensorId x  = g.add_input("x", shape(512, 128));
    TensorId y0 = g.add_op(LIBANE_OP_GELU, {x}, shape(512, 128));
    TensorId y1 = g.add_op(LIBANE_OP_SOFTMAX, {x}, shape(512, 128));

    g.mark_output(y0, "gelu_out");
    g.mark_output(y1, "softmax_out");

    CHECK(g.graph_outputs().size() == 2);
    CHECK(g.graph_outputs()[0] == y0);
    CHECK(g.graph_outputs()[1] == y1);
}

/* ── Accessors ───────────────────────────────────────────────────────────── */

TEST_CASE("tensor() throws on out-of-range ID", "[graph][ir]") {
    AneGraph g;
    CHECK_THROWS_AS(g.tensor(0), std::out_of_range);

    g.add_input("x", shape(512, 128));
    CHECK_THROWS_AS(g.tensor(1), std::out_of_range);
}

TEST_CASE("node() throws on out-of-range ID", "[graph][ir]") {
    AneGraph g;
    CHECK_THROWS_AS(g.node(0), std::out_of_range);
}

/* ── Weight file naming uniqueness ──────────────────────────────────────── */

TEST_CASE("weight_file names are unique across nodes", "[graph][ir]") {
    AneGraph g;
    TensorId x = g.add_input("x", shape(512, 128));

    std::vector<uint16_t> w(512 * 256, 0x3C00);

    TensorId y0 = g.add_op(LIBANE_OP_MATMUL, {x}, shape(256, 128),
                            w.data(), w.size() * 2);
    TensorId y1 = g.add_op(LIBANE_OP_MATMUL, {x}, shape(256, 128),
                            w.data(), w.size() * 2);

    const auto& n0 = g.node(g.tensor(y0).producer_node_id);
    const auto& n1 = g.node(g.tensor(y1).producer_node_id);

    CHECK(n0.weight_file != n1.weight_file);
}

/* ── Binary op (two inputs) ──────────────────────────────────────────────── */

TEST_CASE("binary op records two inputs", "[graph][ir]") {
    AneGraph g;
    TensorId a = g.add_input("a", shape(512, 128));
    TensorId b = g.add_input("b", shape(512, 128));

    TensorId c = g.add_op(LIBANE_OP_ADD, {a, b}, shape(512, 128));

    const auto& n = g.node(0);
    REQUIRE(n.inputs.size() == 2);
    CHECK(n.inputs[0] == a);
    CHECK(n.inputs[1] == b);
    CHECK(n.output == c);
}

/* ── Empty graph invariants ──────────────────────────────────────────────── */

TEST_CASE("empty graph has no tensors, nodes, or outputs", "[graph][ir]") {
    AneGraph g;
    CHECK(g.tensors().empty());
    CHECK(g.nodes().empty());
    CHECK(g.graph_inputs().empty());
    CHECK(g.graph_outputs().empty());
}
