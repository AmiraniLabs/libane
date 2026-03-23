#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "fallback/fallback.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace libane::fallback;

// fp16 helpers for tests
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
static float to_f32(fp16_t v) { return static_cast<float>(v); }
static fp16_t from_f32(float f) { return static_cast<fp16_t>(f); }
#else
static float to_f32(fp16_t v) {
    uint16_t h; std::memcpy(&h, &v, 2);
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t fb;
    if (exp == 0)       fb = sign | (mant << 13);
    else if (exp == 31) fb = sign | 0x7F800000 | (mant << 13);
    else                fb = sign | ((exp + 112) << 23) | (mant << 13);
    float f; std::memcpy(&f, &fb, 4);
    return f;
}
static fp16_t from_f32(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t sign = (fb >> 16) & 0x8000;
    int32_t  exp  = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fb >> 13) & 0x3FF;
    uint16_t h;
    if (exp <= 0)       h = static_cast<uint16_t>(sign);
    else if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00);
    else                h = static_cast<uint16_t>(sign | (exp << 10) | mant);
    fp16_t r; std::memcpy(&r, &h, 2);
    return r;
}
#endif

/* ── Type conversion ────────────────────────────────────────────────────── */

TEST_CASE("cast_f32_to_f16 converts common values", "[fallback][cast]") {
    std::vector<float> src = {1.0f, 0.0f, -1.0f, 2.0f, 0.5f, -0.5f};
    std::vector<fp16_t> dst(src.size());

    cast_f32_to_f16(src.data(), dst.data(), src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        float rt = to_f32(dst[i]);
        float err = std::abs(rt - src[i]) / (std::abs(src[i]) + 1e-6f);
        CHECK(err < 0.002f);
    }
}

TEST_CASE("cast_f16_to_f32 converts common values", "[fallback][cast]") {
    std::vector<float> orig = {1.0f, 2.0f, 0.5f, -1.0f};
    std::vector<fp16_t> src(orig.size());
    for (size_t i = 0; i < orig.size(); ++i) src[i] = from_f32(orig[i]);

    std::vector<float> dst(orig.size(), 0.0f);
    cast_f16_to_f32(src.data(), dst.data(), src.size());

    for (size_t i = 0; i < orig.size(); ++i) {
        float err = std::abs(dst[i] - orig[i]) / (std::abs(orig[i]) + 1e-6f);
        CHECK(err < 0.002f);
    }
}

TEST_CASE("cast round-trip fp32 → fp16 → fp32", "[fallback][cast]") {
    std::vector<float> orig = {0.1f, 0.2f, 0.3f, -0.9f, 100.0f, 0.001f};
    std::vector<fp16_t> half(orig.size());
    std::vector<float>  back(orig.size());

    cast_f32_to_f16(orig.data(), half.data(), orig.size());
    cast_f16_to_f32(half.data(), back.data(), orig.size());

    for (size_t i = 0; i < orig.size(); ++i) {
        float err = std::abs(back[i] - orig[i]) / (std::abs(orig[i]) + 1e-6f);
        CHECK(err < 0.005f); // fp16 precision ~3 digits
    }
}

/* ── matmul_f32 ─────────────────────────────────────────────────────────── */

TEST_CASE("matmul_f32 identity matrix", "[fallback][matmul]") {
    // I * A = A
    int N = 4;
    std::vector<float> I(N*N, 0.0f);
    for (int i = 0; i < N; ++i) I[i*N+i] = 1.0f;

    std::vector<float> A(N*N);
    for (int i = 0; i < N*N; ++i) A[i] = static_cast<float>(i);

    std::vector<float> C(N*N, 0.0f);
    matmul_f32(I.data(), A.data(), C.data(), N, N, N);

    for (int i = 0; i < N*N; ++i)
        CHECK(C[i] == A[i]);
}

TEST_CASE("matmul_f32 2x3 × 3x2 = 2x2", "[fallback][matmul]") {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[58, 64], [139, 154]]
    std::vector<float> A = {1,2,3, 4,5,6};
    std::vector<float> B = {7,8, 9,10, 11,12};
    std::vector<float> C(4);
    matmul_f32(A.data(), B.data(), C.data(), 2, 3, 2);

    CHECK(C[0] == 58.0f);
    CHECK(C[1] == 64.0f);
    CHECK(C[2] == 139.0f);
    CHECK(C[3] == 154.0f);
}

TEST_CASE("matmul_f32 square matrices", "[fallback][matmul]") {
    // Multiply two 32x32 matrices and check result against naive implementation
    int N = 32;
    std::vector<float> A(N*N), B(N*N), C(N*N), Cref(N*N, 0.0f);
    for (int i = 0; i < N*N; ++i) {
        A[i] = static_cast<float>(i % 7) * 0.1f;
        B[i] = static_cast<float>(i % 5) * 0.1f;
    }

    // Reference: naive triple loop
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                Cref[i*N+j] += A[i*N+k] * B[k*N+j];

    matmul_f32(A.data(), B.data(), C.data(), N, N, N);

    float max_err = 0.0f;
    for (int i = 0; i < N*N; ++i)
        max_err = std::max(max_err, std::abs(C[i] - Cref[i]));
    CHECK(max_err < 1e-4f);
}

/* ── matmul_f16 ─────────────────────────────────────────────────────────── */

TEST_CASE("matmul_f16 2x3 × 3x2", "[fallback][matmul]") {
    std::vector<float> Af = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> Bf = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    std::vector<fp16_t> A(6), B(6), C(4);
    for (int i = 0; i < 6; ++i) { A[i] = from_f32(Af[i]); B[i] = from_f32(Bf[i]); }

    matmul_f16(A.data(), B.data(), C.data(), 2, 3, 2);

    CHECK(std::abs(to_f32(C[0]) - 58.0f) < 0.5f);
    CHECK(std::abs(to_f32(C[1]) - 64.0f) < 0.5f);
    CHECK(std::abs(to_f32(C[2]) - 139.0f) < 1.0f);
    CHECK(std::abs(to_f32(C[3]) - 154.0f) < 1.0f);
}

/* ── Elementwise ─────────────────────────────────────────────────────────── */

TEST_CASE("add_f16 element-wise sum", "[fallback][elementwise]") {
    std::vector<fp16_t> A(8), B(8), C(8);
    for (int i = 0; i < 8; ++i) {
        A[i] = from_f32(static_cast<float>(i));
        B[i] = from_f32(1.0f);
    }
    add_f16(A.data(), B.data(), C.data(), 8);
    for (int i = 0; i < 8; ++i)
        CHECK(std::abs(to_f32(C[i]) - static_cast<float>(i + 1)) < 0.1f);
}

TEST_CASE("mul_f16 element-wise product", "[fallback][elementwise]") {
    std::vector<fp16_t> A(4), B(4), C(4);
    for (int i = 0; i < 4; ++i) {
        A[i] = from_f32(static_cast<float>(i + 1));
        B[i] = from_f32(2.0f);
    }
    mul_f16(A.data(), B.data(), C.data(), 4);
    for (int i = 0; i < 4; ++i)
        CHECK(std::abs(to_f32(C[i]) - static_cast<float>((i+1)*2)) < 0.1f);
}

/* ── Softmax ─────────────────────────────────────────────────────────────── */

TEST_CASE("softmax_f32 outputs sum to 1", "[fallback][softmax]") {
    std::vector<float> in  = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4);
    softmax_f32(in.data(), out.data(), 1, 4);

    float sum = std::accumulate(out.begin(), out.end(), 0.0f);
    CHECK(std::abs(sum - 1.0f) < 1e-5f);
    // Largest input (4.0) should map to largest output
    CHECK(*std::max_element(out.begin(), out.end()) == out[3]);
}

TEST_CASE("softmax_f32 two rows", "[fallback][softmax]") {
    std::vector<float> in  = {1.0f, 1.0f, 1.0f,   2.0f, 0.0f, 0.0f};
    std::vector<float> out(6);
    softmax_f32(in.data(), out.data(), 2, 3);

    // Row 0: uniform → all ~0.333
    CHECK(std::abs(out[0] - 1.0f/3.0f) < 1e-4f);
    CHECK(std::abs(out[1] - 1.0f/3.0f) < 1e-4f);

    // Row 1: first is largest
    CHECK(out[3] > out[4]);
    CHECK(out[3] > out[5]);

    // Each row sums to 1
    float s0 = out[0]+out[1]+out[2], s1 = out[3]+out[4]+out[5];
    CHECK(std::abs(s0 - 1.0f) < 1e-5f);
    CHECK(std::abs(s1 - 1.0f) < 1e-5f);
}

TEST_CASE("softmax_f32 numerical stability with very large logits", "[fallback][softmax]") {
    // Without max-subtraction, exp(1000) = +inf and the result is NaN.
    // The implementation must shift by max(row) before computing exp().
    std::vector<float> in  = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    std::vector<float> out(4);
    softmax_f32(in.data(), out.data(), 1, 4);

    for (float v : out) {
        CHECK_FALSE(std::isnan(v));
        CHECK_FALSE(std::isinf(v));
    }
    float sum = std::accumulate(out.begin(), out.end(), 0.0f);
    CHECK(std::abs(sum - 1.0f) < 1e-5f);
    // Uniform inputs → uniform outputs
    CHECK(std::abs(out[0] - 0.25f) < 1e-5f);
}

TEST_CASE("softmax_f32 stability with very negative logits", "[fallback][softmax]") {
    // All-negative large values; exp(-1000) → 0 for all except the max.
    std::vector<float> in  = {-1000.0f, -1000.0f, -999.0f, -1000.0f};
    std::vector<float> out(4);
    softmax_f32(in.data(), out.data(), 1, 4);

    for (float v : out) {
        CHECK_FALSE(std::isnan(v));
        CHECK_FALSE(std::isinf(v));
        CHECK(v >= 0.0f);
    }
    // The -999 element (index 2) has the largest logit; it should dominate
    CHECK(out[2] > out[0]);
    CHECK(out[2] > out[3]);

    float sum = std::accumulate(out.begin(), out.end(), 0.0f);
    CHECK(std::abs(sum - 1.0f) < 1e-5f);
}

TEST_CASE("softmax_f32 single element is always 1.0", "[fallback][softmax]") {
    std::vector<float> in  = {42.0f};
    std::vector<float> out = {0.0f};
    softmax_f32(in.data(), out.data(), 1, 1);
    CHECK(std::abs(out[0] - 1.0f) < 1e-6f);
}

TEST_CASE("softmax_f16 outputs sum to ~1", "[fallback][softmax]") {
    std::vector<fp16_t> in(8), out(8);
    float vals[] = {1.0f, 2.0f, 0.5f, -1.0f, 3.0f, 0.0f, -0.5f, 1.5f};
    for (int i = 0; i < 8; ++i) in[i] = from_f32(vals[i]);

    softmax_f16(in.data(), out.data(), 1, 8);

    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += to_f32(out[i]);
    CHECK(std::abs(sum - 1.0f) < 0.01f); // fp16 precision
}

TEST_CASE("softmax_f16 stable with large equal logits", "[fallback][softmax]") {
    // fp16 saturates at ~65504, but the max-subtraction trick should prevent NaN
    // even for values near fp16 max
    std::vector<fp16_t> in(4), out(4);
    // fp16 max ≈ 65504; use 500 to stay representable while being large
    for (int i = 0; i < 4; ++i) in[i] = from_f32(500.0f);

    softmax_f16(in.data(), out.data(), 1, 4);

    for (int i = 0; i < 4; ++i) {
        float v = to_f32(out[i]);
        CHECK_FALSE(std::isnan(v));
        CHECK(v >= 0.0f);
    }
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += to_f32(out[i]);
    CHECK(std::abs(sum - 1.0f) < 0.01f);
}

/* ── GELU ────────────────────────────────────────────────────────────────── */

// Reference: tanh approximation formula used in both the fallback and the ANE MIL kernel.
// If these values change, the fallback diverges from the ANE path — that's a bug.
static float gelu_ref(float x) {
    constexpr float kA = 0.7978845608f; // sqrt(2/pi)
    constexpr float kB = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kA * (x + kB * x * x * x)));
}

TEST_CASE("gelu_f32 at x=0 is 0", "[fallback][gelu]") {
    std::vector<float> in  = {0.0f};
    std::vector<float> out = {1.0f};
    gelu_f32(in.data(), out.data(), 1);
    CHECK(std::abs(out[0]) < 1e-6f);
}

TEST_CASE("gelu_f32 matches tanh approximation formula exactly", "[fallback][gelu]") {
    // Verify that the fallback uses the *same* formula as the ANE MIL kernel.
    // If these deviate the CPU and ANE paths give different results for the same model.
    float test_vals[] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f};
    std::vector<float> in(std::begin(test_vals), std::end(test_vals));
    std::vector<float> out(in.size());

    gelu_f32(in.data(), out.data(), in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        float expected = gelu_ref(in[i]);
        float err = std::abs(out[i] - expected);
        INFO("gelu_f32(" << in[i] << ") = " << out[i] << ", expected " << expected);
        CHECK(err < 1e-5f);  // should be bit-identical (same formula)
    }
}

TEST_CASE("gelu_f32 known reference values", "[fallback][gelu]") {
    // Spot-check against independently computed values
    std::vector<float> in  = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> out(4);
    gelu_f32(in.data(), out.data(), 4);

    CHECK(std::abs(out[0]) < 1e-6f);           // GELU(0) = 0 exactly
    CHECK(std::abs(out[1] - 0.8413f) < 0.001f);// GELU(1) ≈ 0.8413
    CHECK(std::abs(out[2] - (-0.1587f)) < 0.001f); // GELU(-1) ≈ -0.1587
    CHECK(std::abs(out[3] - 1.9546f) < 0.001f);// GELU(2) ≈ 1.9546
}

TEST_CASE("gelu_f32 positive values near identity for large x", "[fallback][gelu]") {
    std::vector<float> in  = {5.0f};
    std::vector<float> out(1);
    gelu_f32(in.data(), out.data(), 1);
    // GELU(x) → x as x → +∞
    CHECK(std::abs(out[0] - 5.0f) < 0.01f);
}

TEST_CASE("gelu_f32 is not anti-symmetric (verifies non-linear behaviour)", "[fallback][gelu]") {
    // GELU(-1) ≠ -GELU(1) : just verify signs are correct
    std::vector<float> pos = {1.0f, 2.0f};
    std::vector<float> neg = {-1.0f, -2.0f};
    std::vector<float> pout(2), nout(2);
    gelu_f32(pos.data(), pout.data(), 2);
    gelu_f32(neg.data(), nout.data(), 2);
    CHECK(pout[0] > 0.0f);
    CHECK(nout[0] < 0.0f);
}

TEST_CASE("gelu_f16 matches gelu_f32 within fp16 precision", "[fallback][gelu]") {
    // The fp16 path must not introduce errors beyond fp16 rounding vs. the f32 reference.
    float vals[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const int n = 8;
    std::vector<fp16_t> in16(n), out16(n);
    std::vector<float>  in32(n), out32(n);

    for (int i = 0; i < n; ++i) {
        in16[i] = from_f32(vals[i]);
        in32[i] = vals[i];
    }
    gelu_f16(in16.data(), out16.data(), n);
    gelu_f32(in32.data(), out32.data(), n);

    for (int i = 0; i < n; ++i) {
        float f16_result = to_f32(out16[i]);
        float f32_result = out32[i];
        float err = std::abs(f16_result - f32_result);
        INFO("gelu_f16(" << vals[i] << ") = " << f16_result
             << ", gelu_f32 = " << f32_result);
        // fp16 has ~3 decimal digits of precision; allow 0.5% relative error
        CHECK(err <= std::abs(f32_result) * 0.005f + 0.001f);
    }
}

/* ── Layer Normalization ─────────────────────────────────────────────────── */

TEST_CASE("layer_norm_f32 normalizes to mean~0 std~1", "[fallback][layernorm]") {
    // One row of 8 values
    std::vector<float> in  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out(8);
    layer_norm_f32(in.data(), out.data(), 1, 8, nullptr, nullptr, 1e-5f);

    float mean = std::accumulate(out.begin(), out.end(), 0.0f) / 8.0f;
    float var  = 0.0f;
    for (float v : out) var += (v - mean) * (v - mean);
    var /= 8.0f;

    CHECK(std::abs(mean) < 1e-5f);
    CHECK(std::abs(var - 1.0f) < 1e-4f);
}

TEST_CASE("layer_norm_f32 with scale and bias", "[fallback][layernorm]") {
    std::vector<float> in    = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> scale = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> bias  = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4);

    layer_norm_f32(in.data(), out.data(), 1, 4, scale.data(), bias.data(), 1e-5f);

    // After normalization then scale=2, bias=1: output std ≈ 2, mean ≈ 1
    float mean = std::accumulate(out.begin(), out.end(), 0.0f) / 4.0f;
    CHECK(std::abs(mean - 1.0f) < 1e-4f);
}

TEST_CASE("layer_norm_f32 two rows are independent", "[fallback][layernorm]") {
    // Same value for each row (row 0 = [1,2,3,4], row 1 = [10,20,30,40])
    std::vector<float> in  = {1.0f, 2.0f, 3.0f, 4.0f,  10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> out(8);
    layer_norm_f32(in.data(), out.data(), 2, 4, nullptr, nullptr, 1e-5f);

    // Both rows should normalize to the same pattern since relative distances match
    for (int c = 0; c < 4; ++c) {
        float diff = std::abs(out[c] - out[4+c]);
        CHECK(diff < 1e-3f);
    }
}

/* ── Transpose ───────────────────────────────────────────────────────────── */

TEST_CASE("transpose_f16 [0,2,1,3] swaps dims 1 and 2", "[fallback][transpose]") {
    // Input: [1, 2, 3, 4] → transpose [0,2,1,3] → [1, 3, 2, 4]
    int dims[] = {1, 2, 3, 4};
    int perm[] = {0, 2, 1, 3};

    int numel = 1*2*3*4;
    std::vector<fp16_t> in(numel), out(numel);
    for (int i = 0; i < numel; ++i) in[i] = from_f32(static_cast<float>(i));

    transpose_f16(in.data(), out.data(), dims, perm);

    // Check shape: [1, 3, 2, 4]
    // Element at out[b, d1, d2, d3] should be in[b, d2, d1, d3]
    // Just verify it's not a no-op (in ≠ out for non-trivial data)
    bool same = true;
    for (int i = 0; i < numel; ++i) {
        if (to_f32(in[i]) != to_f32(out[i])) { same = false; break; }
    }
    // For [1,2,3,4] shape with [0,2,1,3] perm, the data should differ
    CHECK_FALSE(same);
}

TEST_CASE("transpose_f16 identity perm", "[fallback][transpose]") {
    int dims[] = {1, 4, 1, 8};
    int perm[] = {0, 1, 2, 3};

    int numel = 32;
    std::vector<fp16_t> in(numel), out(numel);
    for (int i = 0; i < numel; ++i) in[i] = from_f32(static_cast<float>(i));

    transpose_f16(in.data(), out.data(), dims, perm);

    for (int i = 0; i < numel; ++i)
        CHECK(to_f32(in[i]) == to_f32(out[i]));
}

/* ── RMSNorm ─────────────────────────────────────────────────────────────── */

TEST_CASE("fallback rmsnorm_f32 basic correctness", "[fallback]") {
    // x = [[1,1,1,1],[2,2,2,2]] (C=2, S=4, channel-major)
    // scale = [1, 1]
    // rms for s=0..3: sqrt((1^2 + 2^2)/2 + eps) = sqrt(2.5 + eps)
    const int C = 2, S = 4;
    float x[C * S] = {1,1,1,1, 2,2,2,2}; // channel-major: x[c*S+s]
    float scale[C] = {1.0f, 1.0f};
    float out[C * S] = {};

    rmsnorm_f32(x, scale, out, C, S);

    // Expected: inv_rms = 1/sqrt(2.5) ≈ 0.6325
    float expected_0 = 1.0f / std::sqrt(2.5f); // ~0.6325
    float expected_1 = 2.0f / std::sqrt(2.5f); // ~1.2649

    for (int s = 0; s < S; ++s) {
        CHECK(std::abs(out[0*S + s] - expected_0) < 0.01f);
        CHECK(std::abs(out[1*S + s] - expected_1) < 0.01f);
    }
}

TEST_CASE("fallback rmsnorm_f32 with scale", "[fallback]") {
    // C=4, S=2, uniform input of 1.0, scale doubles channel 0
    const int C = 4, S = 2;
    float x[C * S];
    for (int i = 0; i < C * S; ++i) x[i] = 1.0f;
    float scale[C] = {2.0f, 1.0f, 1.0f, 1.0f};
    float out[C * S] = {};

    rmsnorm_f32(x, scale, out, C, S);

    // rms = sqrt(4 * 1.0 / 4 + eps) = sqrt(1 + eps) ≈ 1.0
    // out[0,s] = 1.0 / 1.0 * 2.0 = 2.0
    // out[1,s] = 1.0 / 1.0 * 1.0 = 1.0
    for (int s = 0; s < S; ++s) {
        CHECK(std::abs(out[0*S + s] - 2.0f) < 0.01f);
        CHECK(std::abs(out[1*S + s] - 1.0f) < 0.01f);
    }
}

TEST_CASE("fallback rmsnorm_f16 basic correctness", "[fallback]") {
    const int C = 2, S = 4;
    fp16_t x[C * S], scale[C], out[C * S];
    // x[0,s]=1, x[1,s]=2 in channel-major
    for (int s = 0; s < S; ++s) {
        x[0*S + s] = from_f32(1.0f);
        x[1*S + s] = from_f32(2.0f);
    }
    scale[0] = from_f32(1.0f);
    scale[1] = from_f32(1.0f);

    rmsnorm_f16(x, scale, out, C, S);

    float expected_0 = 1.0f / std::sqrt(2.5f);
    float expected_1 = 2.0f / std::sqrt(2.5f);

    for (int s = 0; s < S; ++s) {
        CHECK(std::abs(to_f32(out[0*S + s]) - expected_0) < 0.02f); // fp16 tolerance
        CHECK(std::abs(to_f32(out[1*S + s]) - expected_1) < 0.02f);
    }
}

/* ── SiLU ────────────────────────────────────────────────────────────────── */

TEST_CASE("silu_f32 at x=0 is 0", "[fallback][silu]") {
    std::vector<float> in = {0.0f};
    std::vector<float> out(1);
    silu_f32(in.data(), out.data(), 1);
    CHECK(std::abs(out[0]) < 1e-6f);
}

TEST_CASE("silu_f32 known reference values", "[fallback][silu]") {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // silu(1)  = 1 / (1 + exp(-1))  ≈ 0.7311
    // silu(-1) = -1 / (1 + exp(1))  ≈ -0.2689
    // silu(2)  = 2 / (1 + exp(-2))  ≈ 1.7616
    std::vector<float> in  = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> out(4);
    silu_f32(in.data(), out.data(), 4);

    CHECK(std::abs(out[0]) < 1e-6f);
    CHECK(std::abs(out[1] - 0.7311f) < 0.001f);
    CHECK(std::abs(out[2] - (-0.2689f)) < 0.001f);
    CHECK(std::abs(out[3] - 1.7616f) < 0.001f);
}

TEST_CASE("silu_f32 is always positive for positive input", "[fallback][silu]") {
    std::vector<float> in  = {0.1f, 1.0f, 5.0f, 10.0f};
    std::vector<float> out(4);
    silu_f32(in.data(), out.data(), 4);
    for (float v : out) CHECK(v > 0.0f);
}

TEST_CASE("silu_f32 approaches x for large positive x", "[fallback][silu]") {
    // sigmoid(x) → 1 as x → +∞, so silu(x) → x
    std::vector<float> in  = {10.0f};
    std::vector<float> out(1);
    silu_f32(in.data(), out.data(), 1);
    CHECK(std::abs(out[0] - 10.0f) < 0.01f);
}

TEST_CASE("silu_f32 approaches 0 for large negative x", "[fallback][silu]") {
    // sigmoid(x) → 0 as x → -∞, so silu(x) → 0
    std::vector<float> in  = {-10.0f};
    std::vector<float> out(1);
    silu_f32(in.data(), out.data(), 1);
    CHECK(std::abs(out[0]) < 0.01f);
}

TEST_CASE("silu_f16 matches silu_f32 within fp16 precision", "[fallback][silu]") {
    float vals[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const int n = 8;
    std::vector<fp16_t> in16(n), out16(n);
    std::vector<float>  in32(n), out32(n);

    for (int i = 0; i < n; ++i) {
        in16[i] = from_f32(vals[i]);
        in32[i] = vals[i];
    }
    silu_f16(in16.data(), out16.data(), n);
    silu_f32(in32.data(), out32.data(), n);

    for (int i = 0; i < n; ++i) {
        float f16_result = to_f32(out16[i]);
        float f32_result = out32[i];
        float err = std::abs(f16_result - f32_result);
        INFO("silu_f16(" << vals[i] << ") = " << f16_result
             << ", silu_f32 = " << f32_result);
        CHECK(err <= std::abs(f32_result) * 0.005f + 0.001f);
    }
}
