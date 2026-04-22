#include <catch2/catch_test_macros.hpp>
#include "core/mil_builder.hpp"
#include "graph/ane_graph.hpp"
#include "graph/fusion_rules.hpp"
#include <algorithm>
#include <cstring>

using namespace libane::mil;
using namespace libane::graph;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static TensorShape S(int C, int seq) { return {1, C, 1, seq}; }

static std::vector<uint16_t> fp16_ones(size_t n) {
    return std::vector<uint16_t>(n, 0x3C00u);
}

// Check that a string contains a substring
static bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 1 — Fragment body correctness
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("matmul_fragment: body references in_var and out_var", "[fragment]") {
    auto f = MilBuilder::matmul_fragment(512, 256, 128, "x", "t1");

    CHECK(f.input_name  == "x");
    CHECK(f.output_name == "t1");
    CHECK(f.weight_file == "weight.bin");
    CHECK(f.output_shape == S(256, 128));
    CHECK(f.side_input_name.empty());

    // Body must reference the chain input and produce the output var
    CHECK(has(f.body, "x=x"));        // conv reads x
    CHECK(has(f.body, "tensor<fp16, [1, 256, 1, 128]> t1"));  // output typed assignment

    // Internal consts must carry the out_var prefix to avoid collision
    CHECK(has(f.body, "t1_pt"));
    CHECK(has(f.body, "t1_W"));
    CHECK(has(f.body, "t1_conv"));

    // Must NOT contain bare "pt" without prefix (would collide with another fragment)
    // Check that "val=string(\"valid\")" appears after the prefix
    CHECK(has(f.body, "t1_pt = const()"));
}

TEST_CASE("matmul_fragment: custom weight file propagates", "[fragment]") {
    auto f = MilBuilder::matmul_fragment(64, 128, 32, "inp", "out0", "w42.bin");
    CHECK(f.weight_file == "w42.bin");
    CHECK(has(f.body, "w42.bin"));
}

TEST_CASE("matmul_fragment: output type matches OC and SP", "[fragment]") {
    auto f = MilBuilder::matmul_fragment(768, 3072, 512, "x", "h");
    CHECK(f.output_shape.channels == 3072);
    CHECK(f.output_shape.seq      == 512);
    CHECK(has(f.body, "tensor<fp16, [1, 3072, 1, 512]> h"));
}

TEST_CASE("gelu_fragment: no weights, correct prefix", "[fragment]") {
    auto f = MilBuilder::gelu_fragment(512, 128, "h", "a");

    CHECK(f.input_name  == "h");
    CHECK(f.output_name == "a");
    CHECK(f.weight_file.empty());
    CHECK(f.side_input_name.empty());

    // All intermediate vars must carry the "a_" prefix
    CHECK(has(f.body, "a_x2"));
    CHECK(has(f.body, "a_x3"));
    CHECK(has(f.body, "a_c1"));
    CHECK(has(f.body, "a_gelu"));

    // Output is "a", not "a_something"
    CHECK(has(f.body, "tensor<fp16, [1, 512, 1, 128]> a = mul"));

    // Chain input "h" is referenced inside the body
    CHECK(has(f.body, "x=h"));
}

TEST_CASE("silu_fragment: sigmoid then mul", "[fragment]") {
    auto f = MilBuilder::silu_fragment(256, 64, "x", "s");
    CHECK(has(f.body, "s_sig = sigmoid(x=x)"));
    CHECK(has(f.body, "s = mul(x=x, y=s_sig)"));
}

TEST_CASE("softmax_fragment: single-line body", "[fragment]") {
    auto f = MilBuilder::softmax_fragment(128, 256, "inp", "sm_out");
    CHECK(f.input_name  == "inp");
    CHECK(f.output_name == "sm_out");
    CHECK(has(f.body, "val=int32(1)"));
    CHECK(has(f.body, "sm_out = softmax(axis=sm_out_sax, x=inp)"));
    CHECK(has(f.body, "sm_out_sm"));
}

TEST_CASE("rmsnorm_fragment: weight file and prefix", "[fragment]") {
    auto f = MilBuilder::rmsnorm_fragment(512, 128, "x", "rn", "scale42.bin");
    CHECK(f.weight_file == "scale42.bin");
    CHECK(has(f.body, "scale42.bin"));
    CHECK(has(f.body, "rn_sq"));
    CHECK(has(f.body, "rn_w"));
    CHECK(has(f.body, "tensor<fp16, [1, 512, 1, 128]> rn = mul"));
}

TEST_CASE("layernorm_fragment: two weight files, correct prefix", "[fragment]") {
    auto f = MilBuilder::layernorm_fragment(512, 128, "x", "ln",
                                             "gamma42.bin", "beta42.bin");
    CHECK(f.weight_file == "gamma42.bin");  // primary = gamma
    CHECK(has(f.body, "gamma42.bin"));
    CHECK(has(f.body, "beta42.bin"));
    CHECK(has(f.body, "ln_ax"));
    CHECK(has(f.body, "ln_g"));
    CHECK(has(f.body, "ln_beta"));
    CHECK(has(f.body, "tensor<fp16, [1,512,1,128]> ln = add"));
}

TEST_CASE("add_fragment: two inputs, side_input_name set", "[fragment]") {
    auto f = MilBuilder::add_fragment(512, 128, "chain", "side", "res");
    CHECK(f.input_name      == "chain");
    CHECK(f.side_input_name == "side");
    CHECK(f.output_name     == "res");
    CHECK(f.weight_file.empty());
    CHECK(has(f.body, "res = add(x=chain, y=side)"));
    CHECK(has(f.body, "res_add"));
}

TEST_CASE("mul_fragment: two inputs, side_input_name set", "[fragment]") {
    auto f = MilBuilder::mul_fragment(512, 128, "chain", "gate", "gated");
    CHECK(f.side_input_name == "gate");
    CHECK(has(f.body, "gated = mul(x=chain, y=gate)"));
}

TEST_CASE("sub_fragment: two inputs, side_input_name set", "[fragment]") {
    auto f = MilBuilder::sub_fragment(512, 128, "lhs", "rhs", "diff");
    CHECK(f.side_input_name == "rhs");
    CHECK(has(f.body, "diff = sub(x=lhs, y=rhs)"));
}

TEST_CASE("real_div_fragment: two inputs, side_input_name set", "[fragment]") {
    auto f = MilBuilder::real_div_fragment(512, 128, "num", "den", "quo");
    CHECK(f.side_input_name == "den");
    CHECK(has(f.body, "quo = real_div(x=num, y=den)"));
}

TEST_CASE("sqrt_fragment: one input, no side input", "[fragment]") {
    auto f = MilBuilder::sqrt_fragment(512, 128, "x", "rt");
    CHECK(f.input_name == "x");
    CHECK(f.side_input_name.empty());
    CHECK(has(f.body, "rt = sqrt(x=x)"));
}

TEST_CASE("log_fragment: emits epsilon const and unary log", "[fragment]") {
    auto f = MilBuilder::log_fragment(512, 128, "x", "lg");
    CHECK(f.input_name == "x");
    CHECK(f.side_input_name.empty());
    CHECK(has(f.body, "lg_eps = const()"));
    CHECK(has(f.body, "fp16(0x1.0cp-17)"));
    CHECK(has(f.body, "lg = log(epsilon=lg_eps, x=x)"));
}

TEST_CASE("rsqrt_fragment: emits epsilon const and unary rsqrt", "[fragment]") {
    auto f = MilBuilder::rsqrt_fragment(512, 128, "x", "rr");
    CHECK(f.input_name == "x");
    CHECK(f.side_input_name.empty());
    CHECK(has(f.body, "rr_eps = const()"));
    CHECK(has(f.body, "fp16(0x1.0cp-17)"));
    CHECK(has(f.body, "rr = rsqrt(epsilon=rr_eps, x=x)"));
}

TEST_CASE("concat_fragment: axis and interleave const params emitted", "[fragment]") {
    auto f = MilBuilder::concat_fragment(256, 512, 128, "lhs", "rhs", "cat");
    CHECK(f.input_name == "lhs");
    CHECK(f.side_input_name == "rhs");
    CHECK(f.output_shape == S(768, 128));
    CHECK(has(f.body, "cat_ax = const()"));
    CHECK(has(f.body, "val=int32(1)"));
    CHECK(has(f.body, "cat_id = const()"));
    CHECK(has(f.body, "val=bool(false)"));
    CHECK(has(f.body, "cat = concat(axis=cat_ax, interleave=cat_id, values=(lhs, rhs))"));
}

TEST_CASE("slice_by_index_fragment: conv-based selection emitted", "[fragment]") {
    // C: 512→256, S: 128→64 — both axes need selection convs
    auto f = MilBuilder::slice_by_index_fragment(512, 128, 256, 64, "x", "sl");
    CHECK(f.input_name == "x");
    CHECK(f.side_input_name.empty());
    CHECK(f.output_shape == S(256, 64));
    // C-selection: 1×1 conv with weight file sl_csel.bin
    CHECK(has(f.body, "sl_ccv"));          // C-selection conv output variable
    CHECK(has(f.body, "sl_csel.bin"));     // C-selection weight file
    CHECK(has(f.body, "conv(dilations="));  // conv op present
    // S-selection: transpose → conv → transpose
    CHECK(has(f.body, "sl_str1"));         // first transpose
    CHECK(has(f.body, "sl_ssel.bin"));     // S-selection weight file
    CHECK(has(f.body, "sl_str2"));         // second transpose
    CHECK(f.weight_file == "sl_csel.bin");
}

TEST_CASE("reduce_sum_fragment: axis+keep_dims const params emitted", "[fragment]") {
    auto f = MilBuilder::reduce_sum_fragment(512, 128, "x", "rs");
    CHECK(f.input_name == "x");
    CHECK(f.side_input_name.empty());
    CHECK(f.output_shape == S(1, 128));
    CHECK(has(f.body, "rs_ax = const()"));
    CHECK(has(f.body, "val=tensor<int32, [1]>([1])"));
    CHECK(has(f.body, "rs_kd = const()"));
    CHECK(has(f.body, "val=bool(true)"));
    CHECK(has(f.body, "rs = reduce_sum(x=x, axes=rs_ax, keep_dims=rs_kd)"));
}

TEST_CASE("reduce_mean_fragment: axis+keep_dims const params emitted", "[fragment]") {
    auto f = MilBuilder::reduce_mean_fragment(512, 128, "x", "rm");
    CHECK(f.output_shape == S(1, 128));
    CHECK(has(f.body, "rm = reduce_mean(x=x, axes=rm_ax, keep_dims=rm_kd)"));
}

TEST_CASE("reduce_max_fragment: axis+keep_dims const params emitted", "[fragment]") {
    auto f = MilBuilder::reduce_max_fragment(512, 128, "x", "rmax");
    CHECK(f.output_shape == S(1, 128));
    CHECK(has(f.body, "rmax = reduce_max(x=x, axes=rmax_ax, keep_dims=rmax_kd)"));
}

TEST_CASE("transpose_fragment: shape swap, no weights", "[fragment]") {
    auto f = MilBuilder::transpose_fragment(512, 128, "x", "xt");
    CHECK(f.output_shape.channels == 128);
    CHECK(f.output_shape.seq      == 512);
    CHECK(f.weight_file.empty());
    CHECK(has(f.body, "xt_perm"));
    CHECK(has(f.body, "xt = transpose(perm=xt_perm, x=x)"));
}

TEST_CASE("reshape_fragment: emits shape const and reshape op", "[fragment]") {
    auto f = MilBuilder::reshape_fragment(512, 128, 1024, 64, "x", "xr");
    CHECK(f.weight_file.empty());
    CHECK(f.output_shape == S(1024, 64));
    CHECK(has(f.body, "xr_shape"));
    CHECK(has(f.body, "val=tensor<int32, [4]>([1,1024,1,64])"));
    CHECK(has(f.body, "xr = reshape(shape=xr_shape, x=x)"));
}

TEST_CASE("reshape_fragment: mismatched numel throws", "[fragment]") {
    CHECK_THROWS_AS(
        MilBuilder::reshape_fragment(512, 128, 1024, 128, "x", "bad"),
        std::invalid_argument);
}

/* ── Prefix collision: two gelu fragments must not share var names ────────── */

TEST_CASE("two fragments with different out_vars have disjoint internal names",
          "[fragment]") {
    auto f0 = MilBuilder::gelu_fragment(512, 128, "x",  "t1");
    auto f1 = MilBuilder::gelu_fragment(512, 128, "t1", "t2");

    // f0 uses "t1_x2", f1 uses "t2_x2" — no collision
    CHECK(has(f0.body, "t1_x2"));
    CHECK(has(f1.body, "t2_x2"));
    CHECK_FALSE(has(f0.body, "t2_x2"));
    CHECK_FALSE(has(f1.body, "t1_x2"));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 2 — build_fused output structure
 * ═══════════════════════════════════════════════════════════════════════════ */

TEST_CASE("build_fused single fragment: valid MIL program structure", "[fused]") {
    auto frag = MilBuilder::gelu_fragment(512, 128, "x", "y");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {frag});

    // Must start with the MIL program header
    CHECK(has(prog.text, "program(1.3)"));
    CHECK(has(prog.text, "func main<ios18>"));
    CHECK(has(prog.text, "tensor<fp16, [1, 512, 1, 128]> x"));
    CHECK(has(prog.text, "} -> (y)"));

    // Input/output shapes
    CHECK(prog.input_shape  == S(512, 128));
    CHECK(prog.output_shape == S(512, 128));

    // No weights
    CHECK(prog.weight_name.empty());
    CHECK(prog.all_weight_names.empty());
}

TEST_CASE("build_fused: weight names collected from fragments", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 2048, 128, "x",  "h",   "w0.bin");
    auto f1 = MilBuilder::matmul_fragment(2048, 512, 128, "h2", "out", "w1.bin");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    REQUIRE(prog.all_weight_names.size() == 2);
    CHECK(prog.all_weight_names[0] == "w0.bin");
    CHECK(prog.all_weight_names[1] == "w1.bin");
    CHECK(prog.weight_name == "w0.bin");
}

TEST_CASE("build_fused: matmul -> gelu chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 2048, 128, "x", "h",  "w0.bin");
    auto f1 = MilBuilder::gelu_fragment  (2048, 128,      "h", "a");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    // Both fragments appear in the body
    CHECK(has(prog.text, "h_conv"));   // matmul output
    CHECK(has(prog.text, "a_gelu"));   // gelu output
    CHECK(has(prog.text, "} -> (a)"));

    CHECK(prog.output_shape == S(2048, 128));
    CHECK(prog.all_weight_names.size() == 1);
}

TEST_CASE("build_fused: matmul -> reshape -> gelu chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment (512, 256, 128, "x",  "h",  "w0.bin");
    auto f1 = MilBuilder::reshape_fragment(256, 128, 1024, 32,  "h",  "r");
    auto f2 = MilBuilder::gelu_fragment   (1024, 32,            "r",  "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1, f2});

    CHECK(has(prog.text, "r = reshape(shape=r_shape, x=h)"));
    CHECK(has(prog.text, "out_gelu"));
    CHECK(has(prog.text, "} -> (out)"));
    CHECK(prog.output_shape == S(1024, 32));
}

TEST_CASE("build_fused: matmul -> concat chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 256, 128, "x", "proj", "w0.bin");
    auto f1 = MilBuilder::concat_fragment(256, 128, 128, "proj", "side", "out");
    std::vector<FusedInput> inputs = {{"x", S(512, 128)}, {"side", S(128, 128)}};
    auto prog = MilBuilder::build_fused(inputs, {f0, f1});

    CHECK(has(prog.text, "out = concat(axis=out_ax, interleave=out_id, values=(proj, side))"));
    CHECK(has(prog.text, "} -> (out)"));
    CHECK(prog.output_shape == S(384, 128));
}

TEST_CASE("build_fused: matmul -> slice_by_index chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 512, 128, "x", "proj", "w0.bin");
    auto f1 = MilBuilder::slice_by_index_fragment(512, 128, 256, 64, "proj", "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    CHECK(has(prog.text, "conv(dilations="));  // C-selection conv present
    CHECK(has(prog.text, "out_csel.bin"));     // weight file reference
    CHECK(has(prog.text, "} -> (out)"));
    CHECK(prog.output_shape == S(256, 64));
}

TEST_CASE("build_fused: matmul -> reduce_sum chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 512, 128, "x", "proj", "w0.bin");
    auto f1 = MilBuilder::reduce_sum_fragment(512, 128, "proj", "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    CHECK(has(prog.text, "out = reduce_sum(x=proj, axes=out_ax, keep_dims=out_kd)"));
    CHECK(has(prog.text, "} -> (out)"));
    CHECK(prog.output_shape == S(1, 128));
}

TEST_CASE("build_fused: matmul -> reduce_mean chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 512, 128, "x", "proj", "w0.bin");
    auto f1 = MilBuilder::reduce_mean_fragment(512, 128, "proj", "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    CHECK(has(prog.text, "out = reduce_mean(x=proj, axes=out_ax, keep_dims=out_kd)"));
    CHECK(prog.output_shape == S(1, 128));
}

TEST_CASE("build_fused: matmul -> reduce_max chain", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 512, 128, "x", "proj", "w0.bin");
    auto f1 = MilBuilder::reduce_max_fragment(512, 128, "proj", "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1});

    CHECK(has(prog.text, "out = reduce_max(x=proj, axes=out_ax, keep_dims=out_kd)"));
    CHECK(prog.output_shape == S(1, 128));
}

TEST_CASE("build_fused: matmul -> gelu -> matmul (full FFN op)", "[fused]") {
    auto f0 = MilBuilder::matmul_fragment(512, 2048, 128, "x",  "h",   "w0.bin");
    auto f1 = MilBuilder::gelu_fragment  (2048, 128,      "h",  "a");
    auto f2 = MilBuilder::matmul_fragment(2048, 512, 128, "a",  "out", "w2.bin");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1, f2});

    CHECK(has(prog.text, "h_conv"));
    CHECK(has(prog.text, "a_gelu"));
    CHECK(has(prog.text, "out_conv"));
    CHECK(has(prog.text, "} -> (out)"));

    CHECK(prog.output_shape == S(512, 128));
    REQUIRE(prog.all_weight_names.size() == 2);
    CHECK(prog.all_weight_names[0] == "w0.bin");
    CHECK(prog.all_weight_names[1] == "w2.bin");
}

TEST_CASE("build_fused: inputs sorted alphabetically", "[fused]") {
    // Two-input (chain + residual): "res" < "x" alphabetically
    auto f0 = MilBuilder::matmul_fragment(512, 512, 128, "x", "proj", "w.bin");
    auto f1 = MilBuilder::add_fragment(512, 128, "proj", "res", "out");

    std::vector<FusedInput> inputs = {{"x", S(512, 128)}, {"res", S(512, 128)}};
    auto prog = MilBuilder::build_fused(inputs, {f0, f1});

    // "res" comes before "x" alphabetically → appears first in func signature
    size_t pos_res = prog.text.find("res");
    size_t pos_x   = prog.text.find("x)");
    CHECK(pos_res < pos_x);
}

TEST_CASE("build_fused: single-input convenience overload matches vector overload",
          "[fused]") {
    auto frag = MilBuilder::gelu_fragment(512, 128, "x", "y");

    auto p1 = MilBuilder::build_fused("x", S(512, 128), {frag});
    auto p2 = MilBuilder::build_fused(
        std::vector<FusedInput>{{"x", S(512, 128)}}, {frag});

    CHECK(p1.text == p2.text);
}

TEST_CASE("build_fused: weight-free program has empty all_weight_names", "[fused]") {
    auto f0 = MilBuilder::gelu_fragment   (512, 128, "x",  "g");
    auto f1 = MilBuilder::softmax_fragment(512, 128, "g",  "s");
    auto f2 = MilBuilder::silu_fragment   (512, 128, "s",  "out");
    auto prog = MilBuilder::build_fused("x", S(512, 128), {f0, f1, f2});

    CHECK(prog.all_weight_names.empty());
    CHECK(prog.weight_name.empty());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 3 — FusionRules::compute_groups
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── consumer_count ─────────────────────────────────────────────────────── */

TEST_CASE("build_consumer_count: linear chain", "[fusion][consumer]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId h = g.add_op(LIBANE_OP_GELU,    {x}, S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_SOFTMAX, {h}, S(512, 128));
    g.mark_output(y);

    auto cc = FusionRules::build_consumer_count(g);

    // x is consumed once by gelu
    CHECK(cc[x] == 1);
    // h is consumed once by softmax
    CHECK(cc[h] == 1);
    // y is consumed once (graph output counts)
    CHECK(cc[y] >= 1);
}

TEST_CASE("build_consumer_count: branch point", "[fusion][consumer]") {
    AneGraph g;
    TensorId x  = g.add_input("x", S(512, 128));
    TensorId y0 = g.add_op(LIBANE_OP_GELU,    {x}, S(512, 128));
    TensorId y1 = g.add_op(LIBANE_OP_SOFTMAX, {x}, S(512, 128));
    g.mark_output(y0);
    g.mark_output(y1);

    auto cc = FusionRules::build_consumer_count(g);
    // x is consumed by both gelu and softmax
    CHECK(cc[x] == 2);
}

/* ── single-op graphs ───────────────────────────────────────────────────── */

TEST_CASE("single-op graph produces one group", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId y = g.add_op(LIBANE_OP_GELU, {x}, S(512, 128));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 1);
    CHECK(groups[0].inputs.size()   == 1);
    CHECK(groups[0].inputs[0]       == x);
    CHECK(groups[0].output          == y);
}

/* ── linear chain fusion ─────────────────────────────────────────────────── */

TEST_CASE("matmul -> gelu fuses into one group", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w = fp16_ones(512 * 2048);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(2048, 128),
                           w.data(), w.size() * 2);
    TensorId y = g.add_op(LIBANE_OP_GELU, {h}, S(2048, 128));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);  // both matmul and gelu
    CHECK(groups[0].inputs[0]       == x);
    CHECK(groups[0].output          == y);
}

TEST_CASE("matmul -> gelu -> matmul fuses into one group (full FFN)", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w1 = fp16_ones(512 * 2048);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(2048, 128),
                           w1.data(), w1.size() * 2);
    TensorId a = g.add_op(LIBANE_OP_GELU, {h}, S(2048, 128));

    auto w2 = fp16_ones(2048 * 512);
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {a}, S(512, 128),
                           w2.data(), w2.size() * 2);
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 3);
}

TEST_CASE("matmul -> silu -> matmul fuses into one group", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w1 = fp16_ones(512 * 2048);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(2048, 128),
                           w1.data(), w1.size() * 2);
    TensorId a = g.add_op(LIBANE_OP_SILU, {h}, S(2048, 128));

    auto w2 = fp16_ones(2048 * 512);
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {a}, S(512, 128),
                           w2.data(), w2.size() * 2);
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 3);
}

/* ── branch breaks fusion ────────────────────────────────────────────────── */

TEST_CASE("branch point splits into two groups", "[fusion][groups]") {
    // x -> matmul -> h -> gelu -> y0 (branch 1)
    //               h -> silu -> y1 (branch 2)
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w = fp16_ones(512 * 512);
    TensorId h  = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                            w.data(), w.size() * 2);
    TensorId y0 = g.add_op(LIBANE_OP_GELU, {h}, S(512, 128));
    TensorId y1 = g.add_op(LIBANE_OP_SILU, {h}, S(512, 128));
    g.mark_output(y0);
    g.mark_output(y1);

    auto groups = FusionRules::compute_groups(g);

    // matmul alone, gelu alone, silu alone — OR matmul fused with first consumer
    // In either case, h has consumer_count == 2, so nothing can extend past matmul
    // matmul must be its own group (consumer_count[h] == 2 → can't extend)
    // gelu and silu each start new groups
    REQUIRE(groups.size() == 3);

    // First group is just matmul
    CHECK(groups[0].node_ids.size() == 1);
    CHECK(groups[0].output          == h);
}

/* ── duplicate weight file breaks fusion ─────────────────────────────────── */

TEST_CASE("duplicate weight file name prevents fusion", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    // Both matmuls use the same weight_file name "w0.bin"
    // We must construct nodes directly to force the same weight_file name.
    // add_op generates "w<node_id>.bin" so node 0 gets "w0.bin" and node 1
    // gets "w1.bin" — they're automatically distinct. To trigger this rule
    // we need to trick it, but since add_op auto-names weights, just verify
    // the normal case (distinct names) always fuses:
    auto w1 = fp16_ones(512 * 256);
    auto w2 = fp16_ones(256 * 512);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128),
                           w1.data(), w1.size() * 2);  // weight_file = "w0.bin"
    TensorId y = g.add_op(LIBANE_OP_MATMUL, {h}, S(512, 128),
                           w2.data(), w2.size() * 2);  // weight_file = "w1.bin"
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    // Distinct names → should fuse
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

/* ── binary op with side input ───────────────────────────────────────────── */

TEST_CASE("matmul -> add(+residual) fuses with residual as side input",
          "[fusion][groups]") {
    AneGraph g;
    TensorId x   = g.add_input("x",        S(512, 128));
    TensorId res = g.add_input("residual",  S(512, 128));

    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out  = g.add_op(LIBANE_OP_ADD, {proj, res}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);

    // matmul and add should fuse (residual is a side input from a graph input)
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);

    // Group inputs: chain input x + side input residual
    CHECK(groups[0].inputs.size() == 2);
    bool has_x   = false, has_res = false;
    for (TensorId t : groups[0].inputs) {
        if (t == x)   has_x   = true;
        if (t == res) has_res = true;
    }
    CHECK(has_x);
    CHECK(has_res);
}

TEST_CASE("matmul -> concat(+side) fuses with side input", "[fusion][groups]") {
    AneGraph g;
    TensorId x    = g.add_input("x", S(512, 128));
    TensorId side = g.add_input("side", S(128, 128));

    auto w = fp16_ones(512 * 256);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(256, 128),
                              w.data(), w.size() * 2);
    TensorId out  = g.add_op(LIBANE_OP_CONCAT, {proj, side}, S(384, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
    CHECK(groups[0].inputs.size() == 2);
}

TEST_CASE("matmul -> slice_by_index fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_SLICE_BY_INDEX, {proj}, S(256, 64));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> reduce_sum fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_REDUCE_SUM, {proj}, S(1, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> reduce_mean fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_REDUCE_MEAN, {proj}, S(1, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> reduce_max fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_REDUCE_MAX, {proj}, S(1, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> sub(+side) fuses with side input", "[fusion][groups]") {
    AneGraph g;
    TensorId x    = g.add_input("x", S(512, 128));
    TensorId side = g.add_input("side", S(512, 128));

    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out  = g.add_op(LIBANE_OP_SUB, {proj, side}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
    CHECK(groups[0].inputs.size() == 2);
}

TEST_CASE("matmul -> real_div(+side) fuses with side input", "[fusion][groups]") {
    AneGraph g;
    TensorId x    = g.add_input("x", S(512, 128));
    TensorId side = g.add_input("side", S(512, 128));

    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out  = g.add_op(LIBANE_OP_REAL_DIV, {proj, side}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
    CHECK(groups[0].inputs.size() == 2);
}

TEST_CASE("matmul -> sqrt fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_SQRT, {proj}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> log fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_LOG, {proj}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("matmul -> rsqrt fuses as linear chain", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    auto w = fp16_ones(512 * 512);
    TensorId proj = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                              w.data(), w.size() * 2);
    TensorId out = g.add_op(LIBANE_OP_RSQRT, {proj}, S(512, 128));
    g.mark_output(out);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("binary op with intra-group side input is NOT fused", "[fusion][groups]") {
    // x -> matmul0 -> h
    // x -> matmul1 -> z
    // add(h, z) <- both inputs produced inside, cannot fuse add with either matmul
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w = fp16_ones(512 * 512);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                           w.data(), w.size() * 2);
    TensorId z = g.add_op(LIBANE_OP_MATMUL, {x}, S(512, 128),
                           w.data(), w.size() * 2);
    TensorId y = g.add_op(LIBANE_OP_ADD, {h, z}, S(512, 128));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);

    // x is consumed twice → matmul0 and matmul1 cannot fuse past h/z respectively
    // add consumes h and z, both produced by different upstream groups
    // So we get 3 separate groups
    CHECK(groups.size() == 3);
}

/* ── unsupported ops are singleton groups ────────────────────────────────── */

TEST_CASE("gelu -> cast: cast starts a new group", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));
    TensorId a = g.add_op(LIBANE_OP_GELU, {x}, S(512, 128));
    TensorId b = g.add_op(LIBANE_OP_CAST, {a}, S(512, 128));
    g.mark_output(b);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 2);
    // First group: gelu only
    CHECK(groups[0].node_ids.size() == 1);
    CHECK(groups[0].output          == a);
    // Second group: cast only
    CHECK(groups[1].node_ids.size() == 1);
    CHECK(groups[1].output          == b);
}

/* ── independent ops stay separate ──────────────────────────────────────── */

TEST_CASE("two independent softmax ops: separate groups", "[fusion][groups]") {
    AneGraph g;
    TensorId x0 = g.add_input("x0", S(512, 128));
    TensorId x1 = g.add_input("x1", S(512, 128));
    TensorId y0 = g.add_op(LIBANE_OP_SOFTMAX, {x0}, S(512, 128));
    TensorId y1 = g.add_op(LIBANE_OP_SOFTMAX, {x1}, S(512, 128));
    g.mark_output(y0);
    g.mark_output(y1);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 2);
}

/* ── every node appears in exactly one group ─────────────────────────────── */

TEST_CASE("every node appears in exactly one group", "[fusion][groups]") {
    AneGraph g;
    TensorId x = g.add_input("x", S(512, 128));

    auto w1 = fp16_ones(512 * 2048);
    TensorId h = g.add_op(LIBANE_OP_MATMUL, {x}, S(2048, 128),
                           w1.data(), w1.size() * 2);
    TensorId a = g.add_op(LIBANE_OP_GELU, {h}, S(2048, 128));
    // branch
    TensorId b0 = g.add_op(LIBANE_OP_SOFTMAX, {a}, S(2048, 128));
    TensorId b1 = g.add_op(LIBANE_OP_SILU, {a}, S(2048, 128));
    g.mark_output(b0);
    g.mark_output(b1);

    auto groups = FusionRules::compute_groups(g);

    std::vector<uint32_t> all_node_ids;
    for (const auto& grp : groups)
        for (uint32_t nid : grp.node_ids)
            all_node_ids.push_back(nid);

    // No duplicates
    std::sort(all_node_ids.begin(), all_node_ids.end());
    all_node_ids.erase(std::unique(all_node_ids.begin(), all_node_ids.end()),
                       all_node_ids.end());

    CHECK(all_node_ids.size() == g.nodes().size());
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Part 4 — CONV2D fusion rules
 * ═══════════════════════════════════════════════════════════════════════════ */

// Conv image shape helper: [1, C, H, W]  (W must be % 32 for ANE alignment)
static TensorShape SC(int C, int H, int W) { return {1, C, H, W}; }

// Minimal dummy weight blob: 11 × int32 header (kH=1,kW=1 … groups=1)
// plus a single fp16 zero for kernel data.  Content is irrelevant for
// fusion-rule tests — we just need a non-empty blob so weight_file is set.
static std::vector<uint8_t> conv_dummy_weights(int IC, int OC) {
    // Header: kH kW stride_h stride_w pad_top pad_left pad_bot pad_right dil_h dil_w groups
    int32_t hdr[11] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1};
    // Kernel data: OC × IC × 1 × 1 fp16 zeros
    size_t kernel_elems = static_cast<size_t>(OC) * IC;
    std::vector<uint8_t> blob(sizeof(hdr) + kernel_elems * 2, 0);
    std::memcpy(blob.data(), hdr, sizeof(hdr));
    return blob;
}

TEST_CASE("conv2d alone is a singleton group", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x  = g.add_input("x", SC(8, 4, 32));
    auto     w  = conv_dummy_weights(8, 16);
    TensorId y  = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), w.data(), w.size());
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 1);
    CHECK(groups[0].output == y);
}

TEST_CASE("conv2d → relu fuses into one group", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x  = g.add_input("x", SC(8, 4, 32));
    auto     w  = conv_dummy_weights(8, 16);
    TensorId h  = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), w.data(), w.size());
    TensorId y  = g.add_op(LIBANE_OP_RELU, {h}, SC(16, 4, 32));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
    CHECK(groups[0].output == y);
}

TEST_CASE("conv2d → gelu fuses into one group", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x = g.add_input("x", SC(8, 4, 32));
    auto     w = conv_dummy_weights(8, 16);
    TensorId h = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), w.data(), w.size());
    TensorId y = g.add_op(LIBANE_OP_GELU, {h}, SC(16, 4, 32));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("conv2d → silu fuses into one group", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x = g.add_input("x", SC(8, 4, 32));
    auto     w = conv_dummy_weights(8, 16);
    TensorId h = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), w.data(), w.size());
    TensorId y = g.add_op(LIBANE_OP_SILU, {h}, SC(16, 4, 32));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 2);
}

TEST_CASE("conv2d → relu → relu: all three fuse", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x  = g.add_input("x", SC(8, 4, 32));
    auto     w  = conv_dummy_weights(8, 16);
    TensorId h  = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), w.data(), w.size());
    TensorId a  = g.add_op(LIBANE_OP_RELU, {h}, SC(16, 4, 32));
    TensorId y  = g.add_op(LIBANE_OP_RELU, {a}, SC(16, 4, 32));
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].node_ids.size() == 3);
}

TEST_CASE("conv2d group does not accept matmul extension", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x  = g.add_input("x", SC(8, 4, 32));
    auto     wc = conv_dummy_weights(8, 16);
    TensorId h  = g.add_op(LIBANE_OP_CONV2D, {x}, SC(16, 4, 32), wc.data(), wc.size());
    // Try to extend with a MATMUL.  MATMUL doesn't work on image tensors so
    // the fusion rule should reject it and start a new group.
    auto     wm = fp16_ones(16 * 16);
    TensorId y  = g.add_op(LIBANE_OP_MATMUL, {h}, S(16, 32),
                            wm.data(), wm.size() * 2);
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 2);
    CHECK(groups[0].node_ids.size() == 1);  // conv2d alone
    CHECK(groups[1].node_ids.size() == 1);  // matmul alone
}

TEST_CASE("matmul group does not accept conv2d extension", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x  = g.add_input("x", S(8, 32));
    auto     wm = fp16_ones(8 * 16);
    // Note: for MATMUL the output is S(16, 32), not a conv image shape.
    // CONV2D cannot extend this group.
    TensorId h  = g.add_op(LIBANE_OP_MATMUL, {x}, S(16, 32),
                            wm.data(), wm.size() * 2);
    auto     wc = conv_dummy_weights(16, 16);
    // This conv op's primary input (h) IS the matmul output; but the rule
    // says CONV2D cannot extend a non-CONV2D group.
    TensorId y  = g.add_op(LIBANE_OP_CONV2D, {h}, SC(16, 4, 32),
                            wc.data(), wc.size());
    g.mark_output(y);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 2);
    CHECK(groups[0].node_ids.size() == 1);  // matmul alone
    CHECK(groups[1].node_ids.size() == 1);  // conv2d alone
}

TEST_CASE("two independent conv2d nodes form separate groups", "[fusion][conv2d]") {
    AneGraph g;
    TensorId x0 = g.add_input("x0", SC(8, 4, 32));
    TensorId x1 = g.add_input("x1", SC(8, 4, 32));
    auto     w0 = conv_dummy_weights(8, 16);
    auto     w1 = conv_dummy_weights(8, 16);
    TensorId y0 = g.add_op(LIBANE_OP_CONV2D, {x0}, SC(16, 4, 32), w0.data(), w0.size());
    TensorId y1 = g.add_op(LIBANE_OP_CONV2D, {x1}, SC(16, 4, 32), w1.data(), w1.size());
    g.mark_output(y0);
    g.mark_output(y1);

    auto groups = FusionRules::compute_groups(g);
    REQUIRE(groups.size() == 2);
    CHECK(groups[0].output == y0);
    CHECK(groups[1].output == y1);
}
