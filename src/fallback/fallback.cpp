#include "fallback.hpp"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <stdexcept>

using fp16_t = libane::fallback::fp16_t;

namespace libane {
namespace fallback {

/* ── Helper: fp16 ↔ fp32 via ARM intrinsics or software ─────────────────── */

static inline float f16_to_f32(fp16_t v) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
    return static_cast<float>(v);
#else
    uint16_t h; std::memcpy(&h, &v, 2);
    uint32_t sign     = (h & 0x8000) << 16;
    uint32_t exp      = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t fb;
    if (exp == 0)       fb = sign | (mantissa << 13);
    else if (exp == 31) fb = sign | 0x7F800000 | (mantissa << 13);
    else                fb = sign | ((exp + 112) << 23) | (mantissa << 13);
    float f; std::memcpy(&f, &fb, 4);
    return f;
#endif
}

static inline fp16_t f32_to_f16(float f) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
    return static_cast<fp16_t>(f);
#else
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t sign     = (fb >> 16) & 0x8000;
    int32_t  exp      = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fb >> 13) & 0x3FF;
    uint16_t h;
    if (exp <= 0)       h = static_cast<uint16_t>(sign);
    else if (exp >= 31) h = static_cast<uint16_t>(sign | 0x7C00);
    else                h = static_cast<uint16_t>(sign | (exp << 10) | mantissa);
    fp16_t r; std::memcpy(&r, &h, 2);
    return r;
#endif
}

/* ── Type conversion ─────────────────────────────────────────────────────── */

void cast_f32_to_f16(const float* src, fp16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

void cast_f16_to_f32(const fp16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

/* ── Matrix multiplication ───────────────────────────────────────────────── */

void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    // C = A × B
    // A: [M, K], B: [K, N], C: [M, N] — row-major
    // cblas_sgemm: C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,        // alpha
                A, K,
                B, N,
                0.0f,        // beta
                C, N);
}

void matmul_f16(const fp16_t* A, const fp16_t* B, fp16_t* C, int M, int K, int N) {
    // Promote to fp32, call BLAS, demote result
    std::vector<float> fA(static_cast<size_t>(M) * K);
    std::vector<float> fB(static_cast<size_t>(K) * N);
    std::vector<float> fC(static_cast<size_t>(M) * N);

    for (int i = 0; i < M * K; ++i) fA[i] = f16_to_f32(A[i]);
    for (int i = 0; i < K * N; ++i) fB[i] = f16_to_f32(B[i]);

    matmul_f32(fA.data(), fB.data(), fC.data(), M, K, N);

    for (int i = 0; i < M * N; ++i) C[i] = f32_to_f16(fC[i]);
}

/* ── Elementwise ─────────────────────────────────────────────────────────── */

void add_f16(const fp16_t* A, const fp16_t* B, fp16_t* C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        C[i] = f32_to_f16(f16_to_f32(A[i]) + f16_to_f32(B[i]));
}

void mul_f16(const fp16_t* A, const fp16_t* B, fp16_t* C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        C[i] = f32_to_f16(f16_to_f32(A[i]) * f16_to_f32(B[i]));
}

/* ── GELU ────────────────────────────────────────────────────────────────── */

static inline float gelu_approx(float x) {
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    constexpr float kAlpha = 0.7978845608f; // sqrt(2/π)
    constexpr float kBeta  = 0.044715f;
    float inner = kAlpha * (x + kBeta * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

void gelu_f32(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = gelu_approx(in[i]);
}

void gelu_f16(const fp16_t* in, fp16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = f32_to_f16(gelu_approx(f16_to_f32(in[i])));
}

/* ── SiLU ────────────────────────────────────────────────────────────────── */

static inline float silu_scalar(float x) {
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + std::exp(-x));
}

void silu_f32(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = silu_scalar(in[i]);
}

void silu_f16(const fp16_t* in, fp16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = f32_to_f16(silu_scalar(f16_to_f32(in[i])));
}

/* ── Softmax ─────────────────────────────────────────────────────────────── */

void softmax_f32(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in  = in  + r * cols;
        float*       row_out = out + r * cols;

        float max_v = row_in[0];
        for (int c = 1; c < cols; ++c) max_v = std::max(max_v, row_in[c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row_out[c] = std::exp(row_in[c] - max_v);
            sum += row_out[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) row_out[c] *= inv_sum;
    }
}

void softmax_f16(const fp16_t* in, fp16_t* out, int rows, int cols) {
    std::vector<float> f32(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows * cols; ++i) f32[i] = f16_to_f32(in[i]);
    softmax_f32(f32.data(), f32.data(), rows, cols);
    for (int i = 0; i < rows * cols; ++i) out[i] = f32_to_f16(f32[i]);
}

/* ── Layer Normalization ─────────────────────────────────────────────────── */

void layer_norm_f32(const float* in, float* out, int rows, int cols,
                    const float* scale, const float* bias, float epsilon) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in  = in  + r * cols;
        float*       row_out = out + r * cols;

        // Mean
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += row_in[c];
        mean /= cols;

        // Variance
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = row_in[c] - mean;
            var += d * d;
        }
        var /= cols;

        float inv_std = 1.0f / std::sqrt(var + epsilon);
        for (int c = 0; c < cols; ++c) {
            float v = (row_in[c] - mean) * inv_std;
            if (scale) v *= scale[c];
            if (bias)  v += bias[c];
            row_out[c] = v;
        }
    }
}

void layer_norm_f16(const fp16_t* in, fp16_t* out, int rows, int cols,
                    const fp16_t* scale, const fp16_t* bias, float epsilon) {
    std::vector<float> fi(static_cast<size_t>(rows) * cols);
    std::vector<float> fo(static_cast<size_t>(rows) * cols);

    for (int i = 0; i < rows * cols; ++i) fi[i] = f16_to_f32(in[i]);

    std::vector<float> fs, fb2;
    const float* scale_f = nullptr;
    const float* bias_f  = nullptr;

    if (scale) {
        fs.resize(cols);
        for (int c = 0; c < cols; ++c) fs[c] = f16_to_f32(scale[c]);
        scale_f = fs.data();
    }
    if (bias) {
        fb2.resize(cols);
        for (int c = 0; c < cols; ++c) fb2[c] = f16_to_f32(bias[c]);
        bias_f = fb2.data();
    }

    layer_norm_f32(fi.data(), fo.data(), rows, cols, scale_f, bias_f, epsilon);

    for (int i = 0; i < rows * cols; ++i) out[i] = f32_to_f16(fo[i]);
}

/* ── Transpose ───────────────────────────────────────────────────────────── */

void transpose_f16(const fp16_t* in, fp16_t* out,
                   const int dims[4], const int perm[4]) {
    int out_dims[4] = { dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]] };

    // Compute strides for input
    int in_stride[4];
    in_stride[3] = 1;
    in_stride[2] = dims[3];
    in_stride[1] = dims[2] * dims[3];
    in_stride[0] = dims[1] * dims[2] * dims[3];

    // Compute strides for output
    int out_stride[4];
    out_stride[3] = 1;
    out_stride[2] = out_dims[3];
    out_stride[1] = out_dims[2] * out_dims[3];
    out_stride[0] = out_dims[1] * out_dims[2] * out_dims[3];

    for (int i0 = 0; i0 < out_dims[0]; ++i0)
    for (int i1 = 0; i1 < out_dims[1]; ++i1)
    for (int i2 = 0; i2 < out_dims[2]; ++i2)
    for (int i3 = 0; i3 < out_dims[3]; ++i3) {
        // Map output index back to input
        int idx[4] = { i0, i1, i2, i3 };
        int in_idx[4];
        for (int d = 0; d < 4; ++d) in_idx[perm[d]] = idx[d];

        int src = in_idx[0] * in_stride[0] + in_idx[1] * in_stride[1]
                + in_idx[2] * in_stride[2] + in_idx[3] * in_stride[3];
        int dst = i0 * out_stride[0] + i1 * out_stride[1]
                + i2 * out_stride[2] + i3 * out_stride[3];
        out[dst] = in[src];
    }
}

/* ── RMSNorm ─────────────────────────────────────────────────────────────── */

void rmsnorm_f32(const float* x, const float* scale, float* out,
                 int C, int S, float eps) {
    for (int s = 0; s < S; ++s) {
        // Compute mean(x^2) over channels for this position
        float sum_sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = x[c * S + s];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / C + eps);
        for (int c = 0; c < C; ++c)
            out[c * S + s] = x[c * S + s] * inv_rms * scale[c];
    }
}

void rmsnorm_f16(const fp16_t* x, const fp16_t* scale, fp16_t* out,
                 int C, int S, float eps) {
    for (int s = 0; s < S; ++s) {
        // Compute mean(x^2) over channels for this position
        float sum_sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = f16_to_f32(x[c * S + s]);
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / C + eps);
        float inv_rms = 1.0f / rms;
        for (int c = 0; c < C; ++c) {
            float v     = f16_to_f32(x[c * S + s]);
            float sc    = f16_to_f32(scale[c]);
            out[c * S + s] = f32_to_f16(v * inv_rms * sc);
        }
    }
}

} // namespace fallback
} // namespace libane
