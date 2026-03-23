/**
 * CPU Fallback — Accelerate framework implementations of all libane operations.
 *
 * All operations in this module are fp16-first:
 *  - matmul uses vDSP_mmul via fp32 intermediates (BLAS sgemm via cblas)
 *  - Elementwise ops use vDSP for vectorized fp16 math
 *  - layer_norm, gelu, softmax are scalar C++ fallbacks
 *
 * The fallback is always available and is used when:
 *  1. ANE is not accessible (non-Apple-Silicon, permission denied, etc.)
 *  2. The caller has called libane_set_backend("cpu")
 *  3. The operation shape violates an ANE constraint
 *  4. Compilation fails at runtime
 *
 * Fallback ops are logged at LIBANE_LOG_DEBUG (silent at default level).
 */
#pragma once

#include <cstddef>
#include <cstdint>

// fp16 type — defined at global scope so arm_fp16.h is not included inside a namespace
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
   using libane_fp16_t = __fp16;
#else
   using libane_fp16_t = uint16_t;
#endif

namespace libane {
namespace fallback {

using fp16_t = libane_fp16_t;

/* ── Matrix multiplication ───────────────────────────────────────────────── */

/**
 * C = A × B   (row-major, fp16 in/out)
 * A: [M, K], B: [K, N], C: [M, N]
 *
 * Internally: promotes to fp32, calls cblas_sgemm, demotes to fp16.
 */
void matmul_f16(const fp16_t* A, const fp16_t* B, fp16_t* C,
                int M, int K, int N);

/**
 * C = A × B   (row-major, fp32 in/out)
 * Calls cblas_sgemm directly.
 */
void matmul_f32(const float* A, const float* B, float* C,
                int M, int K, int N);

/* ── Elementwise ─────────────────────────────────────────────────────────── */

/** C[i] = A[i] + B[i], fp16. */
void add_f16(const fp16_t* A, const fp16_t* B, fp16_t* C, size_t n);

/** C[i] = A[i] * B[i], fp16. */
void mul_f16(const fp16_t* A, const fp16_t* B, fp16_t* C, size_t n);

/* ── Activation functions ────────────────────────────────────────────────── */

/**
 * GELU (approximate, tanh variant): y = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
 * This is the only GELU variant the ANE supports; the fallback matches exactly.
 */
void gelu_f16(const fp16_t* in, fp16_t* out, size_t n);
void gelu_f32(const float* in, float* out, size_t n);

/**
 * SiLU (Swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x))
 */
void silu_f16(const fp16_t* in, fp16_t* out, size_t n);
void silu_f32(const float* in, float* out, size_t n);

/**
 * Softmax over the last dimension.
 * in/out: [rows, cols], softmax applied across cols.
 */
void softmax_f16(const fp16_t* in, fp16_t* out, int rows, int cols);
void softmax_f32(const float* in, float* out, int rows, int cols);

/* ── Normalization ───────────────────────────────────────────────────────── */

/**
 * Layer normalization over the last dimension.
 * in/out: [rows, cols]
 * scale, bias: [cols] — may be nullptr (identity scale / zero bias).
 */
void layer_norm_f16(const fp16_t* in, fp16_t* out, int rows, int cols,
                    const fp16_t* scale, const fp16_t* bias, float epsilon);

void layer_norm_f32(const float* in, float* out, int rows, int cols,
                    const float* scale, const float* bias, float epsilon);

/* ── Transpose ───────────────────────────────────────────────────────────── */

/**
 * Transpose a 4D tensor with the given permutation.
 * in/out: flat arrays, shape [d0, d1, d2, d3], perm is a 4-element array.
 */
void transpose_f16(const fp16_t* in, fp16_t* out,
                   const int dims[4], const int perm[4]);

/* ── Type conversion ─────────────────────────────────────────────────────── */

/** Convert n fp32 values to fp16. */
void cast_f32_to_f16(const float* src, fp16_t* dst, size_t n);

/** Convert n fp16 values to fp32. */
void cast_f16_to_f32(const fp16_t* src, float* dst, size_t n);

/* ── RMSNorm ─────────────────────────────────────────────────────────────── */

/**
 * RMSNorm: out[i] = (x[i] / rms(x_row)) * scale[c]
 * x and out are [C, S] in channel-major order (same as all ANE tensors).
 * scale is [C] (one value per channel).
 * eps is the stability epsilon (typically 1e-5).
 */
void rmsnorm_f16(const fp16_t* x, const fp16_t* scale, fp16_t* out,
                 int C, int S, float eps = 1e-5f);
void rmsnorm_f32(const float* x, const float* scale, float* out,
                 int C, int S, float eps = 1e-5f);

} // namespace fallback
} // namespace libane
