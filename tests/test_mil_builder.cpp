/**
 * MIL Builder tests — validates the MIL *text* generation and weight blob format.
 *
 * These tests verify:
 *  - TensorShape ANE constraints
 *  - WeightBlob 128-byte header format (file header + chunk header + fp16 data)
 *  - MilBuilder text programs contain the right structural elements
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/mil_builder.hpp"
#include <vector>
#include <cstring>
#include <cstdint>

using namespace libane::mil;
using Catch::Matchers::ContainsSubstring;

/* ── TensorShape validation ─────────────────────────────────────────────── */

TEST_CASE("TensorShape validation — valid shapes", "[mil][shape]") {
    TensorShape s{1, 64, 1, 512};
    REQUIRE_NOTHROW(s.validate());
    CHECK(s.numel() == 64 * 512);
    CHECK(s.bytes() == 64 * 512 * 2);
}

TEST_CASE("TensorShape validation — batch must be 1", "[mil][shape]") {
    TensorShape s{2, 64, 1, 512};
    REQUIRE_THROWS_AS(s.validate(), std::invalid_argument);
}

TEST_CASE("TensorShape validation — height must be 1", "[mil][shape]") {
    TensorShape s{1, 64, 3, 512};
    REQUIRE_THROWS_AS(s.validate(), std::invalid_argument);
}

TEST_CASE("TensorShape validation — S must be multiple of 32", "[mil][shape]") {
    TensorShape s{1, 64, 1, 513};
    REQUIRE_THROWS_AS(s.validate(), std::invalid_argument);

    TensorShape s2{1, 64, 1, 32};
    REQUIRE_NOTHROW(s2.validate());

    TensorShape s3{1, 64, 1, 0};
    REQUIRE_THROWS_AS(s3.validate(), std::invalid_argument);

    // S=16 no longer valid (must be multiple of 32)
    TensorShape s4{1, 64, 1, 16};
    REQUIRE_THROWS_AS(s4.validate(), std::invalid_argument);

    // S=8 still invalid
    TensorShape s5{1, 64, 1, 8};
    REQUIRE_THROWS_AS(s5.validate(), std::invalid_argument);
}

TEST_CASE("TensorShape validation — S <= 65536", "[mil][shape]") {
    TensorShape s{1, 64, 1, 65536};
    REQUIRE_NOTHROW(s.validate());

    TensorShape s2{1, 64, 1, 65544};
    REQUIRE_THROWS_AS(s2.validate(), std::invalid_argument);
}

TEST_CASE("TensorShape validation — C <= 16384", "[mil][shape]") {
    TensorShape s{1, 16384, 1, 32};
    REQUIRE_NOTHROW(s.validate());

    TensorShape s2{1, 16385, 1, 32};
    REQUIRE_THROWS_AS(s2.validate(), std::invalid_argument);
}

/* ── WeightBlob ─────────────────────────────────────────────────────────── */

TEST_CASE("WeightBlob from_fp16 has 128-byte header + weight data", "[mil][weights]") {
    // 8 fp16 values (all 1.0 = 0x3C00)
    std::vector<uint16_t> raw(8, 0x3C00);
    size_t weight_bytes = raw.size() * 2; // 16 bytes
    auto blob = WeightBlob::from_fp16(raw.data(), weight_bytes);

    // Total size: 128-byte header + 16 bytes weight data = 144
    REQUIRE(blob.data.size() == WeightBlob::kDataOffset + weight_bytes);
    REQUIRE(blob.data.size() == 144);

    // File header: buf[0]=0x01, buf[4]=0x02
    CHECK(blob.data[0] == 0x01);
    CHECK(blob.data[4] == 0x02);

    // Chunk header at offset 64: magic = 0xDEADBEEF (little-endian)
    uint32_t magic;
    std::memcpy(&magic, blob.data.data() + 64, 4);
    CHECK(magic == 0xDEADBEEF);
    CHECK(blob.data[64 + 4]  == 0x01);
    // fp16_size at b[72..75] is uint32 LE — verify it equals weight_bytes
    uint32_t fp16_sz = 0;
    std::memcpy(&fp16_sz, blob.data.data() + 72, 4);
    CHECK(fp16_sz == static_cast<uint32_t>(weight_bytes));

    // Weight data starts at offset 128
    CHECK(std::memcmp(blob.data.data() + WeightBlob::kDataOffset,
                      raw.data(), weight_bytes) == 0);

    // Hash must be non-zero
    REQUIRE(blob.hash != 0);
}

TEST_CASE("WeightBlob kWeightDictOffset is 64", "[mil][weights]") {
    // The weight dict @"offset" key must be 64 (points past the file header,
    // into the chunk header region — Orion constraint #8).
    CHECK(WeightBlob::kWeightDictOffset == 64);
    CHECK(WeightBlob::kDataOffset == 128);
}

TEST_CASE("WeightBlob from_fp32 no transpose", "[mil][weights]") {
    std::vector<float> vals = {1.0f, 0.0f, -1.0f, 2.0f};
    // rows=2, cols=2, no transpose
    auto blob = WeightBlob::from_fp32(vals.data(), 2, 2);

    REQUIRE(blob.data.size() == WeightBlob::kDataOffset + vals.size() * 2);
    REQUIRE(blob.hash != 0);

    // 1.0f → fp16 0x3C00
    uint16_t h0;
    std::memcpy(&h0, blob.data.data() + WeightBlob::kDataOffset, 2);
    CHECK(h0 == 0x3C00);

    // 0.0f → fp16 0x0000
    uint16_t h1;
    std::memcpy(&h1, blob.data.data() + WeightBlob::kDataOffset + 2, 2);
    CHECK(h1 == 0x0000);
}

TEST_CASE("WeightBlob from_fp32 with transpose", "[mil][weights]") {
    // Matrix [2 rows, 3 cols]:
    //   1 2 3
    //   4 5 6
    // Transposed [3 rows, 2 cols]:
    //   1 4
    //   2 5
    //   3 6
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto blob = WeightBlob::from_fp32(src.data(), 2, 3, /*transpose=*/true);

    REQUIRE(blob.data.size() == WeightBlob::kDataOffset + 6 * 2);

    // After transpose: row-major output is [1, 4, 2, 5, 3, 6]
    // fp16 of 1.0 = 0x3C00, 4.0 = 0x4400
    auto* fp16_data = reinterpret_cast<const uint16_t*>(
        blob.data.data() + WeightBlob::kDataOffset);
    CHECK(fp16_data[0] == 0x3C00); // 1.0
    CHECK(fp16_data[1] == 0x4400); // 4.0
    CHECK(fp16_data[2] == 0x4000); // 2.0
}

TEST_CASE("WeightBlob hash is deterministic and data-sensitive", "[mil][weights]") {
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    auto b1 = WeightBlob::from_fp32(vals.data(), 1, 3);
    auto b2 = WeightBlob::from_fp32(vals.data(), 1, 3);
    CHECK(b1.hash == b2.hash);

    vals[0] = 99.0f;
    auto b3 = WeightBlob::from_fp32(vals.data(), 1, 3);
    CHECK(b1.hash != b3.hash);
}

/* ── MilBuilder — matmul_conv1x1 ────────────────────────────────────────── */

TEST_CASE("MilBuilder::matmul_conv1x1 produces valid MIL text", "[mil][build]") {
    auto prog = MilBuilder::matmul_conv1x1(64, 128, 512);

    // Non-empty text
    REQUIRE_FALSE(prog.text.empty());
    REQUIRE_FALSE(prog.weight_name.empty());

    // Structural checks
    CHECK_THAT(prog.text, ContainsSubstring("program(1.3)"));
    CHECK_THAT(prog.text, ContainsSubstring("func main<ios18>"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 64, 1, 512]>"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 128, 1, 512]>"));

    // Uses conv with Orion-compatible named-parameter style (no % prefix)
    CHECK_THAT(prog.text, ContainsSubstring("conv(dilations=dl"));
    CHECK_THAT(prog.text, ContainsSubstring("pad_type=pt"));
    CHECK_THAT(prog.text, ContainsSubstring("weight=W, x=x)"));

    // Weight shape is [OC, IC, kH, kW] = [OC, IC, 1, 1]
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [128,64,1,1]>"));

    // Correct BLOBFILE weight reference with offset=uint64(64)
    CHECK_THAT(prog.text, ContainsSubstring("BLOBFILE(path=string("));
    CHECK_THAT(prog.text, ContainsSubstring("offset=uint64(64)"));
    CHECK_THAT(prog.text, ContainsSubstring("@model_path/weights/"));

    // Shapes
    CHECK(prog.input_shape  == (TensorShape{1, 64,  1, 512}));
    CHECK(prog.output_shape == (TensorShape{1, 128, 1, 512}));
}

TEST_CASE("MilBuilder::matmul_conv1x1 rejects invalid shapes", "[mil][build]") {
    REQUIRE_THROWS_AS(MilBuilder::matmul_conv1x1(64, 128, 513), std::invalid_argument);
    REQUIRE_THROWS_AS(MilBuilder::matmul_conv1x1(64, 128, 0),   std::invalid_argument);
    REQUIRE_THROWS_AS(MilBuilder::matmul_conv1x1(20000, 128, 512), std::invalid_argument);
}

TEST_CASE("MilBuilder::matmul_conv1x1 custom weight filename", "[mil][build]") {
    auto prog = MilBuilder::matmul_conv1x1(32, 64, 256, "wq.bin");
    CHECK_THAT(prog.text, ContainsSubstring("wq.bin"));
    CHECK(prog.weight_name == "wq.bin");
}

/* ── MilBuilder — gelu ───────────────────────────────────────────────────── */

TEST_CASE("MilBuilder::gelu produces tanh decomposition", "[mil][build]") {
    auto prog = MilBuilder::gelu(32, 128);
    REQUIRE_FALSE(prog.text.empty());
    // gelu MIL op is invalid on ANE — must use tanh decomposition
    CHECK_FALSE(prog.text.find("gelu(") != std::string::npos);
    CHECK_FALSE(prog.text.find("tanh_approximation") != std::string::npos);
    // Must use tanh and the 0.044715 coefficient
    CHECK_THAT(prog.text, ContainsSubstring("tanh(x=g_scaled)"));
    CHECK_THAT(prog.text, ContainsSubstring("0.044715"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 32, 1, 128]>"));
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 32, 1, 128}));
    CHECK(prog.output_shape == (TensorShape{1, 32, 1, 128}));
}

TEST_CASE("MilBuilder::gelu rejects invalid shape", "[mil][build]") {
    REQUIRE_THROWS_AS(MilBuilder::gelu(32, 7), std::invalid_argument); // S not mult of 16
}

/* ── MilBuilder — softmax ───────────────────────────────────────────────── */

TEST_CASE("MilBuilder::softmax over C (channel) dimension", "[mil][build]") {
    auto prog = MilBuilder::softmax(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("softmax("));
    CHECK_THAT(prog.text, ContainsSubstring("val=int32(1)"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 8, 1, 64]>"));
    CHECK(prog.weight_name.empty());
}

/* ── MilBuilder — add ────────────────────────────────────────────────────── */

TEST_CASE("MilBuilder::add two-input elementwise", "[mil][build]") {
    auto prog = MilBuilder::add(16, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("add(x = x"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]>"));
    // Two inputs: x and y
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]> y)"));
}

TEST_CASE("MilBuilder::avg_pool lowers to identity", "[mil][build]") {
    auto prog = MilBuilder::avg_pool(16, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("identity(x=x)"));
    CHECK_FALSE(prog.text.find("avg_pool(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 16, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("MilBuilder::max_pool lowers to identity", "[mil][build]") {
    auto prog = MilBuilder::max_pool(16, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("identity(x=x)"));
    CHECK_FALSE(prog.text.find("max_pool(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 16, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("MilBuilder::logical_and lowers to cast+mul", "[mil][build]") {
    auto prog = MilBuilder::logical_and(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=x, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=y, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=xf, y=yf)"));
    CHECK_FALSE(prog.text.find("logical_and(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::logical_or lowers to cast+maximum", "[mil][build]") {
    auto prog = MilBuilder::logical_or(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=x, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=y, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("maximum(x=xf, y=yf)"));
    CHECK_FALSE(prog.text.find("logical_or(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::logical_xor lowers to cast+not_equal", "[mil][build]") {
    auto prog = MilBuilder::logical_xor(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=x, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=y, dtype=string(\"bool\"))"));
    CHECK_THAT(prog.text, ContainsSubstring("not_equal(x=xf, y=yf)"));
    CHECK_FALSE(prog.text.find("logical_xor(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::reduce_prod lowers to log+reduce_sum+exp", "[mil][build]") {
    auto prog = MilBuilder::reduce_prod(16, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("log(x=rp_x, epsilon=rp_eps)"));
    CHECK_THAT(prog.text, ContainsSubstring("reduce_sum(x=rp_l"));
    CHECK_THAT(prog.text, ContainsSubstring("exp(x=rp_s)"));
    CHECK_FALSE(prog.text.find("reduce_prod(") != std::string::npos);
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 16, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 1, 1, 64}));
}

TEST_CASE("MilBuilder::scatter_static_mask lowers to mask blend", "[mil][build]") {
    auto prog = MilBuilder::scatter_static_mask(8, 64, "mask.bin");
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("BLOBFILE(path=string(\"@model_path/weights/mask.bin\")"));
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=one, y=m)"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=base, y=inv)"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=updates, y=m)"));
    CHECK_THAT(prog.text, ContainsSubstring("add(x=xb, y=uu)"));
    CHECK(prog.weight_name == "mask.bin");
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::gather_static_mask lowers to x*mask", "[mil][build]") {
    auto prog = MilBuilder::gather_static_mask(8, 64, "gmask.bin");
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("BLOBFILE(path=string(\"@model_path/weights/gmask.bin\")"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=x, y=m)"));
    CHECK(prog.weight_name == "gmask.bin");
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::gather_dynamic_mask lowers to x*mask", "[mil][build]") {
    auto prog = MilBuilder::gather_dynamic_mask(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("func main<ios18>("));
    CHECK_THAT(prog.text, ContainsSubstring("x, tensor<fp16, [1, 8, 1, 64]> mask"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=x, y=mask)"));
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 8, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 8, 1, 64}));
}

TEST_CASE("MilBuilder::neg lowers to mul(x, -1)", "[mil][build]") {
    auto prog = MilBuilder::neg(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("fp16(-1.0)"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=x, y=n1)"));
}

TEST_CASE("MilBuilder::mod lowers to floor_div/mul/sub", "[mil][build]") {
    auto prog = MilBuilder::mod(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("floor_div(x=x, y=y)"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=q, y=y)"));
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=x, y=qy)"));
}

TEST_CASE("MilBuilder::sinh lowers to exp identities", "[mil][build]") {
    auto prog = MilBuilder::sinh(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("exp(x=x)"));
    CHECK_THAT(prog.text, ContainsSubstring("exp(x=nx)"));
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=ex, y=enx)"));
}

TEST_CASE("MilBuilder::cosh lowers to exp identities", "[mil][build]") {
    auto prog = MilBuilder::cosh(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("exp(x=x)"));
    CHECK_THAT(prog.text, ContainsSubstring("exp(x=nx)"));
    CHECK_THAT(prog.text, ContainsSubstring("add(x=ex, y=enx)"));
}

TEST_CASE("MilBuilder::tan lowers to sin/cos/real_div", "[mil][build]") {
    auto prog = MilBuilder::tan(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("sin(x=x)"));
    CHECK_THAT(prog.text, ContainsSubstring("cos(x=x)"));
    CHECK_THAT(prog.text, ContainsSubstring("real_div(x=sx, y=cxe)"));
}

TEST_CASE("MilBuilder::asin lowers to atan/sqrt with guards", "[mil][build]") {
    auto prog = MilBuilder::asin(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=one, y=xsq)"));
    CHECK_THAT(prog.text, ContainsSubstring("sqrt(x=den2e)"));
    CHECK_THAT(prog.text, ContainsSubstring("atan(x=ratio)"));
}

TEST_CASE("MilBuilder::acos lowers to pi/2 - asin path", "[mil][build]") {
    auto prog = MilBuilder::acos(8, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("fp16(1.5703125)"));
    CHECK_THAT(prog.text, ContainsSubstring("atan(x=ratio)"));
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=hp, y=asv)"));
}

/* ── MilBuilder — rmsnorm ────────────────────────────────────────────────── */

TEST_CASE("MilBuilder::rmsnorm uses reduce_sum + pow path", "[mil][build]") {
    auto prog = MilBuilder::rmsnorm(64, 512);
    REQUIRE_FALSE(prog.text.empty());
    // Must use reduce_sum (not rsqrt), then pow(x=rms_mse, -0.5)
    CHECK_FALSE(prog.text.find("rsqrt") != std::string::npos);
    CHECK_THAT(prog.text, ContainsSubstring("reduce_sum"));
    CHECK_THAT(prog.text, ContainsSubstring("pow(x=rms_mse"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 64, 1, 512]>"));
    CHECK_THAT(prog.text, ContainsSubstring("offset=uint64(64)")); // scale weight
    REQUIRE_FALSE(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 64, 1, 512}));
    CHECK(prog.output_shape == (TensorShape{1, 64, 1, 512}));
}

/* ── MilBuilder — transpose_cssc ────────────────────────────────────────── */

TEST_CASE("MilBuilder::transpose_cssc emits perm [0,3,2,1]", "[mil][build]") {
    auto prog = MilBuilder::transpose_cssc(32, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("transpose("));
    CHECK_THAT(prog.text, ContainsSubstring("[0,3,2,1]"));
    // Input [1,32,1,64] → output [1,64,1,32]
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 32, 1, 64]>"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 64, 1, 32]>"));
    CHECK(prog.input_shape  == (TensorShape{1, 32, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 64, 1, 32}));
}

TEST_CASE("MilBuilder::transpose_cssc rejects invalid dims", "[mil][build]") {
    // S and C swap roles in output: both must be valid ANE shapes
    // If C=7 → output.seq=7 which is not a multiple of 32 → invalid
    REQUIRE_THROWS_AS(MilBuilder::transpose_cssc(7, 64), std::invalid_argument);
}

/* ── MilBuilder — out_proj_add ───────────────────────────────────────────── */

TEST_CASE("MilBuilder::out_proj_add emits conv + residual add", "[mil][build]") {
    auto prog = MilBuilder::out_proj_add(64, 128, 256);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("conv(dilations=dl"));
    CHECK_THAT(prog.text, ContainsSubstring("add(x=proj"));
    CHECK_THAT(prog.text, ContainsSubstring("residual"));
    // Input x: [1,64,1,256], residual: [1,128,1,256]
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 64, 1, 256]>"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 128, 1, 256]>"));
    CHECK(prog.input_shape  == (TensorShape{1, 64,  1, 256}));
    CHECK(prog.output_shape == (TensorShape{1, 128, 1, 256}));
}

/* ── MilBuilder — mul ────────────────────────────────────────────────────── */

TEST_CASE("MilBuilder::mul two-input elementwise", "[mil][build]") {
    auto prog = MilBuilder::mul(16, 64);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=x, y=y)"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]>"));
    // Two inputs: x and y
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 16, 1, 64]> y)"));
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 16, 1, 64}));
    CHECK(prog.output_shape == (TensorShape{1, 16, 1, 64}));
}

/* ── MilBuilder — silu ───────────────────────────────────────────────────── */

TEST_CASE("MilBuilder::silu produces sigmoid + mul", "[mil][build]") {
    auto prog = MilBuilder::silu(32, 128);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("sigmoid(x=x)"));
    CHECK_THAT(prog.text, ContainsSubstring("mul(x=x, y=sig)"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 32, 1, 128]>"));
    CHECK(prog.weight_name.empty());
    CHECK(prog.input_shape  == (TensorShape{1, 32, 1, 128}));
    CHECK(prog.output_shape == (TensorShape{1, 32, 1, 128}));
}

/* ── MilBuilder — layernorm ──────────────────────────────────────────────── */

TEST_CASE("MilBuilder::layernorm emits reduce_mean + pow path", "[mil][build]") {
    auto prog = MilBuilder::layernorm(64, 512);
    REQUIRE_FALSE(prog.text.empty());
    CHECK_THAT(prog.text, ContainsSubstring("reduce_mean"));
    CHECK_THAT(prog.text, ContainsSubstring("sub(x=x, y=ln_mean)"));
    CHECK_THAT(prog.text, ContainsSubstring("pow(x=ln_veps"));
    CHECK_THAT(prog.text, ContainsSubstring("tensor<fp16, [1, 64, 1, 512]>"));
    CHECK_THAT(prog.text, ContainsSubstring("offset=uint64(64)")); // gamma weight
    REQUIRE_FALSE(prog.weight_name.empty());
    CHECK(prog.weight_name == "gamma.bin");
    CHECK(prog.input_shape  == (TensorShape{1, 64, 1, 512}));
    CHECK(prog.output_shape == (TensorShape{1, 64, 1, 512}));
}

TEST_CASE("MilBuilder::layernorm custom filenames and eps", "[mil][build]") {
    auto prog = MilBuilder::layernorm(32, 128, "my_gamma.bin", "my_beta.bin", 1e-6f);
    CHECK_THAT(prog.text, ContainsSubstring("my_gamma.bin"));
    CHECK_THAT(prog.text, ContainsSubstring("my_beta.bin"));
    CHECK(prog.weight_name == "my_gamma.bin");
}

/* ── MIL text structure ──────────────────────────────────────────────────── */

TEST_CASE("All MilBuilder programs have correct program header", "[mil][build]") {
    auto p1 = MilBuilder::matmul_conv1x1(32, 64, 128);
    auto p2 = MilBuilder::gelu(32, 128);
    auto p3 = MilBuilder::softmax(8, 64);
    auto p4 = MilBuilder::add(16, 64);
    auto p4b = MilBuilder::sub(16, 64);
    auto p4c = MilBuilder::real_div(16, 64);
    auto p4d = MilBuilder::sqrt(16, 64);
    auto p4e = MilBuilder::log(16, 64);
    auto p4f = MilBuilder::rsqrt(16, 64);
    auto p5 = MilBuilder::transpose_cssc(32, 64);
    auto p6 = MilBuilder::mul(16, 64);
    auto p7 = MilBuilder::silu(32, 128);
    auto p8 = MilBuilder::layernorm(32, 128);

    for (auto* p : {&p1, &p2, &p3, &p4, &p4b, &p4c, &p4d, &p4e, &p4f, &p5, &p6, &p7, &p8}) {
        CHECK_THAT(p->text, ContainsSubstring("program(1.3)"));
        CHECK_THAT(p->text, ContainsSubstring("buildInfo"));
        CHECK_THAT(p->text, ContainsSubstring("coremlc-component-MIL"));
        CHECK_THAT(p->text, ContainsSubstring("func main<ios18>"));
        CHECK_THAT(p->text, ContainsSubstring("} -> ("));
    }
}

TEST_CASE("core single-op builders do not emit concat", "[mil][build]") {
    auto p1 = MilBuilder::matmul_conv1x1(32, 64, 128);
    auto p2 = MilBuilder::gelu(32, 128);
    auto p3 = MilBuilder::add(16, 64);

    for (auto* p : {&p1, &p2, &p3}) {
        // These builder entrypoints should stay concat-free.
        CHECK_FALSE(p->text.find("concat") != std::string::npos);
    }
}

TEST_CASE("mul() has inputs in alphabetical order (constraint #13)", "[mil][constraints]") {
    // mul() should have two inputs named 'x' and 'y' (alphabetical order).
    // Per Orion constraint #13, inputs must be alphabetical for multi-input programs.
    auto prog = MilBuilder::mul(16, 64);

    // Check that function signature has both 'x' and 'y' as parameter names
    // MIL format: func main<ios18>(tensor<...> x, tensor<...> y) { ... }
    CHECK_THAT(prog.text, ContainsSubstring("func main<ios18>("));
    CHECK_THAT(prog.text, ContainsSubstring("> x,"));  // x appears as "> x,"
    CHECK_THAT(prog.text, ContainsSubstring("> y)"));  // y appears as "> y)"

    // Verify they appear before the '->' arrow and in correct order
    size_t x_pos = prog.text.find("> x,");
    size_t y_pos = prog.text.find("> y)");
    size_t arrow_pos = prog.text.find("->");
    CHECK(x_pos < arrow_pos);
    CHECK(y_pos < arrow_pos);

    // 'x' should come before 'y' in alphabetical order (and lexicographically in the signature)
    CHECK(x_pos < y_pos);
}

TEST_CASE("add() has inputs in alphabetical order (constraint #13)", "[mil][constraints]") {
    // add() similarly should have two inputs 'x' and 'y' in alphabetical order
    auto prog = MilBuilder::add(16, 64);

    CHECK_THAT(prog.text, ContainsSubstring("> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("> y)"));

    size_t x_pos = prog.text.find("> x,");
    size_t y_pos = prog.text.find("> y)");
    CHECK(x_pos < y_pos);  // alphabetical order
}

TEST_CASE("sub() has inputs in alphabetical order (constraint #13)", "[mil][constraints]") {
    auto prog = MilBuilder::sub(16, 64);
    CHECK_THAT(prog.text, ContainsSubstring("> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("> y)"));
    size_t x_pos = prog.text.find("> x,");
    size_t y_pos = prog.text.find("> y)");
    CHECK(x_pos < y_pos);
}

TEST_CASE("real_div() has inputs in alphabetical order (constraint #13)", "[mil][constraints]") {
    auto prog = MilBuilder::real_div(16, 64);
    CHECK_THAT(prog.text, ContainsSubstring("> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("> y)"));
    size_t x_pos = prog.text.find("> x,");
    size_t y_pos = prog.text.find("> y)");
    CHECK(x_pos < y_pos);
}

TEST_CASE("select() emits correct MIL structure", "[mil][select]") {
    auto prog = MilBuilder::select(32, 64);

    // 3 inputs alphabetically: c < x < y
    CHECK_THAT(prog.text, ContainsSubstring("> c,"));
    CHECK_THAT(prog.text, ContainsSubstring("> x,"));
    CHECK_THAT(prog.text, ContainsSubstring("> y)"));

    size_t c_pos = prog.text.find("> c,");
    size_t x_pos = prog.text.find("> x,");
    size_t y_pos = prog.text.find("> y)");
    CHECK(c_pos < x_pos);
    CHECK(x_pos < y_pos);

    // bool cast of condition
    CHECK_THAT(prog.text, ContainsSubstring("cast(x=c, dtype=string(\"bool\"))"));
    // select op present
    CHECK_THAT(prog.text, ContainsSubstring("select(cond="));
    CHECK_THAT(prog.text, ContainsSubstring("a=x"));
    CHECK_THAT(prog.text, ContainsSubstring("b=y"));
}

TEST_CASE("select_fragment() emits cast + select", "[mil][select]") {
    auto f = MilBuilder::select_fragment(16, 32, "t0", "t1", "t2", "t3");

    CHECK(f.input_name  == "t0");
    CHECK(f.output_name == "t3");
    CHECK_THAT(f.body, ContainsSubstring("cast(x=t0"));
    CHECK_THAT(f.body, ContainsSubstring("select(cond="));
    CHECK_THAT(f.body, ContainsSubstring("a=t1"));
    CHECK_THAT(f.body, ContainsSubstring("b=t2"));
}

/* ── Activation fragments ────────────────────────────────────────────────── */

TEST_CASE("relu_fragment emits relu op", "[mil][activation]") {
    auto f = MilBuilder::relu_fragment(16, 64, "x", "y");
    CHECK(f.input_name  == "x");
    CHECK(f.output_name == "y");
    CHECK_THAT(f.body, ContainsSubstring("relu(x=x)"));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("tanh_fragment emits tanh op", "[mil][activation]") {
    auto f = MilBuilder::tanh_fragment(16, 64, "x", "y");
    CHECK_THAT(f.body, ContainsSubstring("tanh(x=x)"));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("sigmoid_fragment emits sigmoid op", "[mil][activation]") {
    auto f = MilBuilder::sigmoid_fragment(16, 64, "x", "y");
    CHECK_THAT(f.body, ContainsSubstring("sigmoid(x=x)"));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("hardswish_fragment emits relu-based clamp", "[mil][activation]") {
    auto f = MilBuilder::hardswish_fragment(16, 64, "x", "y");
    CHECK_THAT(f.body, ContainsSubstring("fp16(3.0)"));
    CHECK_THAT(f.body, ContainsSubstring("fp16(6.0)"));
    CHECK_THAT(f.body, ContainsSubstring("relu("));
    CHECK_THAT(f.body, ContainsSubstring("fp16(0.16667)"));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("leaky_relu_fragment emits leaky_relu op", "[mil][activation]") {
    auto f = MilBuilder::leaky_relu_fragment(16, 64, "x", "y");
    CHECK_THAT(f.body, ContainsSubstring("leaky_relu(alpha="));
    CHECK_THAT(f.body, ContainsSubstring("fp16(0.01)"));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

TEST_CASE("elu_fragment emits exp + select", "[mil][activation]") {
    auto f = MilBuilder::elu_fragment(16, 64, "x", "y");
    CHECK_THAT(f.body, ContainsSubstring("greater(x=x"));
    CHECK_THAT(f.body, ContainsSubstring("exp(x=x)"));
    CHECK_THAT(f.body, ContainsSubstring("select(cond="));
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
}

/* ── pixel_shuffle_fragment ──────────────────────────────────────────────── */

TEST_CASE("pixel_shuffle_fragment emits reshape+transpose+reshape", "[mil][pixel_shuffle]") {
    // Input [1, 32, 1, 64] (out_C=16, r=2) → Output [1, 16, 1, 128]
    auto f = MilBuilder::pixel_shuffle_fragment(16, 64, 2, "x", "y");
    CHECK(f.input_name  == "x");
    CHECK(f.output_name == "y");
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 128}));
    CHECK_THAT(f.body, ContainsSubstring("reshape("));
    CHECK_THAT(f.body, ContainsSubstring("transpose("));
    CHECK_THAT(f.body, ContainsSubstring("[0,1,3,2]"));
    // Final output shape [1,16,1,128]
    CHECK_THAT(f.body, ContainsSubstring("1,16,1,128"));
}

TEST_CASE("pixel_shuffle_fragment rejects r=0", "[mil][pixel_shuffle]") {
    REQUIRE_THROWS_AS(MilBuilder::pixel_shuffle_fragment(16, 64, 0, "x", "y"),
                      std::invalid_argument);
}

/* ── pwl_activation_fragment ─────────────────────────────────────────────── */

TEST_CASE("pwl_activation_fragment emits piecewise linear chain", "[mil][pwl]") {
    // 3 samples = 2 segments over [-1, 1]
    float samples[] = {-0.5f, 0.0f, 0.5f};
    auto f = MilBuilder::pwl_activation_fragment(16, 64, -1.0f, 1.0f, samples, 3, "x", "y");
    CHECK(f.input_name  == "x");
    CHECK(f.output_name == "y");
    CHECK(f.output_shape == (TensorShape{1, 16, 1, 64}));
    CHECK_THAT(f.body, ContainsSubstring("mul(x=x"));
    CHECK_THAT(f.body, ContainsSubstring("less(x=x"));
    CHECK_THAT(f.body, ContainsSubstring("select(cond="));
}

TEST_CASE("pwl_activation_fragment rejects < 2 samples", "[mil][pwl]") {
    float s = 1.0f;
    REQUIRE_THROWS_AS(
        MilBuilder::pwl_activation_fragment(16, 64, -1.0f, 1.0f, &s, 1, "x", "y"),
        std::invalid_argument);
}
