/**
 * libane — Apple Neural Engine Native C++ Runtime Library
 * Amirani Labs · v0.8.2
 *
 * Stable C ABI. ABI stability guaranteed across minor versions.
 * Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ── Version ─────────────────────────────────────────────────────────────── */

#define LIBANE_VERSION         "0.9.0"
#define LIBANE_VERSION_MAJOR   0
#define LIBANE_VERSION_MINOR   9
#define LIBANE_VERSION_PATCH   0

/* ── ABI visibility ──────────────────────────────────────────────────────── */

#if defined(_WIN32)
#  if defined(libane_EXPORTS)
#    define LIBANE_API __declspec(dllexport)
#  else
#    define LIBANE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define LIBANE_API __attribute__((visibility("default")))
#else
#  define LIBANE_API
#endif

/* ── fp16 portability ────────────────────────────────────────────────────── */

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
   typedef __fp16 libane_f16_t;
#else
   /* x86 fallback — library runs in CPU-fallback mode on x86 */
   typedef uint16_t libane_f16_t;
#endif

/* ── Opaque handle ───────────────────────────────────────────────────────── */

typedef struct libane_program_s* libane_handle_t;

/* ── Operation codes ─────────────────────────────────────────────────────── */

typedef enum {
    LIBANE_OP_MATMUL     = 0,
    /**
     * 2D Convolution.
     *
     * Input:  [1, IC,  H_in,  W_in]  (conv image tensor — H and W are spatial dims)
     * Output: [1, OC,  H_out, W_out]
     * Weight: [OC, IC/groups, kH, kW]  fp16, row-major, NO transpose required.
     *
     *   H_out = (H_in + pad_top + pad_bot - dilation_h*(kH-1) - 1) / stride_h + 1
     *   W_out = (W_in + pad_left + pad_right - dilation_w*(kW-1) - 1) / stride_w + 1
     *
     * W_in and W_out must be multiples of 32 (IOSurface 64-byte DMA alignment
     * for fp16 row stride).  Bias is not supported; follow with ADD if needed.
     *
     * Use libane_graph_add_conv2d() — it packs hyperparams and kernel correctly.
     * Raw weights blob format (for libane_graph_add_op):
     *   int32[11] = {kH, kW, stride_h, stride_w,
     *                pad_top, pad_left, pad_bottom, pad_right,
     *                dilation_h, dilation_w, groups}           (44 bytes)
     *   fp16[OC × IC/groups × kH × kW]                        (kernel bytes)
     */
    LIBANE_OP_CONV2D     = 1,
    LIBANE_OP_LAYER_NORM = 2,
    LIBANE_OP_GELU       = 3,
    LIBANE_OP_SOFTMAX    = 4,
    LIBANE_OP_ADD        = 5,
    LIBANE_OP_MUL        = 6,
    LIBANE_OP_TRANSPOSE  = 7,
    LIBANE_OP_CAST       = 8,
    LIBANE_OP_SILU       = 9,
    LIBANE_OP_RMSNORM    = 10,
    LIBANE_OP_LAYERNORM  = 11,
    LIBANE_OP_AVG_POOL   = 12,
    LIBANE_OP_MAX_POOL   = 13,
    LIBANE_OP_LOGICAL_AND= 14,
    LIBANE_OP_LOGICAL_OR = 15,
    LIBANE_OP_LOGICAL_XOR= 16,
    LIBANE_OP_REDUCE_PROD= 17,
    LIBANE_OP_SCATTER    = 18,
    LIBANE_OP_GATHER     = 19,
    LIBANE_OP_SCATTER_ND = 20,
    LIBANE_OP_SCATTER_ALONG_AXIS = 21,
    LIBANE_OP_NEG        = 22,
    LIBANE_OP_MOD        = 23,
    LIBANE_OP_SINH       = 24,
    LIBANE_OP_COSH       = 25,
    LIBANE_OP_TAN        = 26,
    LIBANE_OP_ASIN       = 27,
    LIBANE_OP_ACOS       = 28,
    LIBANE_OP_RESHAPE        = 29,
    LIBANE_OP_CONCAT         = 30,
    LIBANE_OP_SLICE_BY_INDEX = 31,
    LIBANE_OP_REDUCE_SUM     = 32,
    LIBANE_OP_REDUCE_MEAN    = 33,
    LIBANE_OP_REDUCE_MAX     = 34,
    LIBANE_OP_SUB            = 35,
    LIBANE_OP_REAL_DIV       = 36,
    LIBANE_OP_SQRT           = 37,
    LIBANE_OP_LOG            = 38,
    LIBANE_OP_RSQRT          = 39,
    LIBANE_OP_SELECT         = 40,

    /* ── Standalone activation ops ──────────────────────────────────────── */
    LIBANE_OP_RELU           = 41,  /**< ReLU: max(x, 0) */
    LIBANE_OP_TANH           = 42,  /**< Tanh */
    LIBANE_OP_SIGMOID        = 43,  /**< Sigmoid */
    LIBANE_OP_HARDSWISH      = 44,  /**< HardSwish: x * clamp(x+3, 0, 6) / 6 */
    LIBANE_OP_LEAKY_RELU     = 45,  /**< Leaky ReLU (alpha=0.01) */
    LIBANE_OP_ELU            = 46,  /**< ELU (alpha=1.0) */

    /* ── Spatial reorganization ──────────────────────────────────────────── */
    LIBANE_OP_PIXEL_SHUFFLE  = 47,  /**< Depth-to-space; upscale_factor encoded in weights[0..3] (int32) */

    /* ── Learnable activation ────────────────────────────────────────────── */
    LIBANE_OP_PWL_ACTIVATION = 48,  /**< Piecewise-linear custom activation; use libane_graph_add_pwl_activation() */

    /* ── General slice ───────────────────────────────────────────────────── */
    /**
     * General slice with begin, end, and stride per dimension.
     *
     * weights (optional): 8 × int32 packed as [ begin[4], stride[4] ].
     *   begin[i]  — start index for dimension i (default 0).
     *   stride[i] — step for dimension i (default 1; negative unsupported).
     * end[i] is computed as begin[i] + stride[i] * output_shape[i].
     * NULL weights → begin=[0,0,0,0], stride=[1,1,1,1] (equivalent to
     * SLICE_BY_INDEX with output_shape as the end).
     */
    LIBANE_OP_SLICE = 49,  /**< General slice; weights: int32 begin[4], stride[4] (32 bytes); NULL=origin */

    /**
     * Element-wise clamp: out = clamp(x, lo, hi).
     * weights: float32[2] = {lo, hi} (8 bytes, required).
     * Output shape must equal input shape.
     */
    LIBANE_OP_CLIP  = 50,

    /**
     * Zero-pad tensor.
     * weights: int32[8] = {pad_N0,pad_C0,pad_H0,pad_S0, pad_N1,pad_C1,pad_H1,pad_S1}
     *   where index 0 = before, 1 = after.  N and H pads must be 0.
     * output_shape must equal input_shape + total padding on each axis.
     */
    LIBANE_OP_PAD   = 51,

    /**
     * Dynamic matrix multiply: Y = X @ W^T  (both inputs are runtime tensors).
     *
     * Uses matrix tensor format (height > 1, C = 1):
     *   inputs[0] = X: [1, 1, K, M]   (height=K inner dim, seq=M output cols)
     *   inputs[1] = W: [1, 1, N, K]   (height=N output rows, seq=K inner dim)
     *   output:        [1, 1, N, M]
     *
     * No weights. K/N/M are inferred from the input shapes.
     * Constraints: K, M, N must all be multiples of 32.
     * Use this when the weight matrix changes at runtime.
     * For static weights, prefer MATMUL (3× faster via conv1x1).
     */
    LIBANE_OP_DYNAMIC_MATMUL = 52,

    /**
     * Scaled dot-product attention: out = softmax(Q @ K^T / sqrt(D)) @ V
     *
     * Uses matrix tensor format (height > 1):
     *   inputs[0] = Q:    [1, H, S, D]
     *   inputs[1] = K:    [1, H, S, D]
     *   inputs[2] = V:    [1, H, S, D]
     *   inputs[3] = mask: [1, 1, S, S]  (optional; omit for unmasked attention)
     *   output:           [1, H, S, D]
     *
     * No weights. H/S/D are inferred from the input shapes.
     * Constraints: S and D must both be multiples of 32.
     */
    LIBANE_OP_SDPA = 53,

    /**
     * W8A16 quantized matrix multiply: Y = dequant(W_int8, scales) × X
     *
     * Activations are fp16 (A16).  Weights are stored as int8 with per-output-channel
     * fp32 scale factors (W8).  Dequantization is fused into the conv1x1 path:
     *   W_fp16[ic, oc] = (float)W_int8[ic, oc] × scale[oc]
     *
     * Input:   [1, IC, 1, S]   (standard activation tensor)
     * Output:  [1, OC, 1, S]
     * Weights: [IC, OC]  int8 row-major  (same layout convention as MATMUL)
     * Scales:  [OC]      float32 per-output-channel (symmetric quantization)
     *
     * Use libane_graph_add_matmul_w8a16() — it packs hyperparams and the two
     * buffers (int8 weights + float32 scales) into the internal blob format.
     * Raw weights blob format (for libane_graph_add_op):
     *   int32[2]        = {OC, IC}                     (8 bytes)
     *   int8[IC × OC]   = quantized weights [IC, OC]   (IC*OC bytes)
     *   uint8[0..3]     = padding to next 4-byte boundary
     *   float32[OC]     = per-channel scales            (OC*4 bytes)
     *
     * Notes:
     *  - Symmetric quantization only (zero-point = 0).
     *  - S must be a multiple of 32 (same as MATMUL).
     *  - Fuses with downstream elementwise ops (GELU, SILU, ADD, etc.)
     *    exactly like MATMUL.
     *  - For asymmetric quantization, apply a bias via a subsequent ADD node.
     */
    LIBANE_OP_MATMUL_W8A16 = 54,

    /**
     * Quantized matrix multiply — int8 weights, int8 activations (W8A8).
     *
     * Both weights and input activations are stored as int8, halving memory
     * bandwidth on both sides of the multiply.  The ANE still operates in fp16;
     * dequantization is applied transparently:
     *
     *   - Weights: dequantized at compile time (offline), baked into the HWX.
     *   - Activations: dequantized at execute time before IOSurface transfer.
     *
     * Blob format (packed, little-endian):
     *   int32[2] = { OC, IC }                          (8 bytes)
     *   int8[IC × OC]                                   (IC * OC bytes)
     *   pad to 4-byte boundary
     *   float32[OC]    per-channel weight scales        (OC * 4 bytes)
     *   float32[1]     per-tensor activation scale      (4 bytes)
     *   int32[1]       per-tensor activation zero_point (4 bytes)
     *
     * Quantization equations:
     *   W_fp16[ic, oc]  = W_int8[ic, oc] × weight_scales[oc]
     *   A_fp16[n]       = (A_int8[n] - act_zero_point) × act_scale
     *
     * The caller is responsible for quantizing activations to int8 before
     * calling execute.  Use libane_quantize_i8() as a convenience helper.
     * Fuses with downstream activations (GELU, SiLU, etc.) identically to
     * MATMUL_W8A16.
     */
    LIBANE_OP_MATMUL_W8A8 = 55,

    /**
     * Grouped Query Attention (GQA) SDPA.
     *
     * Like SDPA but Q has num_q_heads heads while K and V each have
     * num_kv_heads heads (num_q_heads must be divisible by num_kv_heads).
     * K and V are tiled to num_q_heads before the scaled-dot-product-attention.
     *
     * Inputs (in order):
     *   [0] Q    [1, num_q_heads,  S, D]
     *   [1] K    [1, num_kv_heads, S, D]
     *   [2] V    [1, num_kv_heads, S, D]
     *   [3] mask [1, 1,            S, S]  (optional)
     * Output: [1, num_q_heads, S, D]
     *
     * Use libane_graph_add_sdpa_gqa() as the convenience wrapper.
     * S and D must both be multiples of 32.
     * num_q_heads % num_kv_heads == 0.
     *
     * No weights blob — num_q_heads and num_kv_heads are derived from input shapes.
     */
    LIBANE_OP_SDPA_GQA = 56,

    /* ── Elementary math ops ────────────────────────────────────────────── */
    LIBANE_OP_EXP   = 57,  /**< Element-wise natural exponential: out = exp(x) */
    LIBANE_OP_SIN   = 58,  /**< Element-wise sine (radians): out = sin(x) */
    LIBANE_OP_COS   = 59,  /**< Element-wise cosine (radians): out = cos(x) */
    LIBANE_OP_ABS   = 60,  /**< Element-wise absolute value: out = |x| */

    /**
     * Element-wise power: out = base ^ exponent.
     *
     * inputs[0] = base     [1, C, 1, S]
     * inputs[1] = exponent [1, C, 1, S]  (same shape as base)
     * output:               [1, C, 1, S]
     *
     * No weights. Both inputs must be provided at runtime.
     */
    LIBANE_OP_POW   = 61,

    LIBANE_OP_CEIL  = 62,  /**< Element-wise ceiling: out = ceil(x) */
    LIBANE_OP_FLOOR = 63,  /**< Element-wise floor:   out = floor(x) */
    LIBANE_OP_ROUND = 64,  /**< Element-wise round to nearest even: out = round(x) */
    LIBANE_OP_SIGN  = 65,  /**< Element-wise sign: out = -1/0/+1 */

    /* ── Two-input matmul ────────────────────────────────────────────────── */
    /**
     * Matrix multiplication with two live IOSurface inputs: C = A × B.
     *
     * Unlike LIBANE_OP_MATMUL (which bakes the weight into the program as a
     * conv1x1), this op takes both A and B as runtime inputs — no weight file.
     *
     * A: [1, K, 1, M]   (input 0)
     * B: [1, K, 1, N]   (input 1, treated as transposed)
     * C: [1, N, 1, M]   (output)
     *
     * On macOS 26+ (ios19 NNCompiler): accepts fp16 or int8 IOSurfaces.
     *   int8 inputs are dequantized and fused into a single quantized kernel.
     * On older macOS (ios18 NNCompiler): fp16 inputs only.
     *   Passing int8 IOSurfaces on older macOS returns LIBANE_STATUS_UNSUPPORTED.
     */
    LIBANE_OP_MATMUL_MULTI = 66,
} libane_op_t;

/* ── Shape descriptor ────────────────────────────────────────────────────── */

/**
 * ANE tensors are always [1, C, 1, S] (NCHW with H=1).
 * dims[0] = batch (always 1 for ANE), dims[1] = C, dims[2] = 1, dims[3] = S.
 * S must be a multiple of 32 (ANE activation-output stride constraint).
 *
 * For matmul(A[M,K], B[K,N]) the caller maps:
 *   A shape: {1, K, 1, M}   B shape: {1, N, 1, K}
 */
typedef struct {
    int32_t dims[4];  /**< [batch, channels, height, seq] */
    int32_t ndim;     /**< number of meaningful dimensions (≤ 4) */
} libane_shape_t;

/* ── Status codes ────────────────────────────────────────────────────────── */

typedef enum {
    LIBANE_OK                = 0,
    LIBANE_ERR_UNAVAILABLE   = -1,  /**< ANE not accessible; fallback active */
    LIBANE_ERR_INVALID_SHAPE = -2,  /**< shape violates ANE constraints */
    LIBANE_ERR_INVALID_OP    = -3,  /**< op not in supported set */
    LIBANE_ERR_COMPILE_FAILED= -4,  /**< MIL compilation failed */
    LIBANE_ERR_EXECUTE_FAILED= -5,  /**< execution failed */
    LIBANE_ERR_OOM           = -6,  /**< out of memory */
    LIBANE_ERR_INVALID_ARG   = -7,  /**< null pointer or bad argument */
} libane_status_t;

/* ── Log levels ──────────────────────────────────────────────────────────── */

typedef enum {
    LIBANE_LOG_SILENT = 0,
    LIBANE_LOG_ERROR  = 1,
    LIBANE_LOG_WARN   = 2,
    LIBANE_LOG_INFO   = 3,
    LIBANE_LOG_DEBUG  = 4,
} libane_log_level_t;

/* ── Core compile / execute / release ───────────────────────────────────── */

/**
 * Compile request descriptor for batch compilation.
 *
 * Used with libane_compile_batch() to compile multiple operations in one call.
 * Each request is compiled independently — if one fails, others may still succeed.
 */
typedef struct {
    libane_op_t op;           /**< Operation to compile */
    libane_shape_t shape;     /**< Output / activation tensor shape */
    const void* weights;      /**< Weight data (fp16). NULL for elementwise ops */
    size_t weights_len;       /**< Byte length of weights buffer */
} libane_compile_request_t;

/**
 * Compile an operation for the given shape and weights.
 *
 * @param op          Operation to compile.
 * @param shape       Output / activation tensor shape.
 * @param weights     Weight data (fp16). May be NULL for elementwise ops.
 * @param weights_len Byte length of weights buffer.
 * @return            Opaque handle, or NULL on failure. Check libane_last_error().
 *
 * Thread-safe. Compilation is cached — subsequent calls with the same
 * (op, shape, weight hash) return immediately from cache.
 */
LIBANE_API libane_handle_t libane_compile(libane_op_t op,
                                          libane_shape_t shape,
                                          const void* weights,
                                          size_t weights_len);

/**
 * Compile multiple operations in a single call.
 *
 * Convenience function for compiling a batch of operations. Each request is
 * compiled independently using the same caching as libane_compile().
 *
 * @param requests     Array of libane_compile_request_t structures (num_requests elements).
 * @param num_requests Number of requests to compile.
 * @param out_handles  Output array for handles (must have at least num_requests capacity).
 *                     On partial failure, successful handles are non-NULL, failed ones are NULL.
 *
 * @return LIBANE_OK if all compilations succeeded.
 *         LIBANE_ERR_COMPILE_FAILED if one or more compilations failed.
 *         LIBANE_ERR_INVALID_ARG if requests or out_handles is NULL or num_requests == 0.
 *
 * PARTIAL SUCCESS: On LIBANE_ERR_COMPILE_FAILED, out_handles contains both successful
 * (non-NULL) and failed (NULL) results. Callers must check individual handles or the
 * return code before using results. Successful handles are valid and usable even when
 * return code indicates partial failure.
 *
 * Thread-safe. Each request is compiled independently and may be cached.
 */
LIBANE_API libane_status_t libane_compile_batch(const libane_compile_request_t* requests,
                                                size_t num_requests,
                                                libane_handle_t* out_handles);

/**
 * Execute a compiled program.
 *
 * @param h      Handle from libane_compile().
 * @param input  fp16 input buffer (must match compiled shape).
 * @param output fp16 output buffer (caller-allocated).
 * @param shape  Runtime shape (must match compiled shape).
 * @return       LIBANE_OK on success.
 */
LIBANE_API libane_status_t libane_execute(libane_handle_t h,
                                          const void* input,
                                          void* output,
                                          libane_shape_t shape);

/**
 * Execute a compiled program with two inputs (e.g., elementwise ops, out_proj_add).
 *
 * Inputs are passed in the alphabetical order of their MIL parameter names
 * (ANE constraint #13 — alphabetical input ordering).
 *
 * @param h       Handle from libane_compile().
 * @param input0  First fp16 input buffer (alphabetically first MIL parameter).
 * @param input1  Second fp16 input buffer (alphabetically second MIL parameter).
 * @param output  fp16 output buffer (caller-allocated).
 * @param shape   Runtime shape (must match compiled shape).
 * @return        LIBANE_OK on success, negative error code on failure.
 */
LIBANE_API libane_status_t libane_execute2(libane_handle_t h,
                                           const void* input0,
                                           const void* input1,
                                           void* output,
                                           libane_shape_t shape);

/**
 * Re-load a compiled program into ANE SRAM without recompiling.
 *
 * Useful after an internal unload (e.g., to temporarily free SRAM for another
 * model) when you want to restore the program quickly.  Load-only is ~8.5×
 * faster than a full recompile (~494 ms vs ~4,200 ms).
 *
 * IMPORTANT — weight values are read-only after compilation:
 *   ANE bakes weights into the compiled HWX bytecode at compile time.
 *   This function does NOT accept new weights and cannot update them.
 *   Confirmed empirically: overwriting on-disk weight files has zero effect
 *   on execution (probe_delta_reload, M3 Pro / macOS 26.3.1, 2026-04-16).
 *   To change weights, call libane_release() + libane_compile() with new data.
 *
 * @param h  Handle from libane_compile(). Must be an ANE-compiled handle.
 * @return LIBANE_OK on success.
 *         LIBANE_ERR_UNAVAILABLE if handle is not ANE-compiled.
 *         LIBANE_ERR_EXECUTE_FAILED if the reload fails (handle is then
 *         invalid — call libane_release() + libane_compile()).
 */
LIBANE_API libane_status_t libane_delta_reload(libane_handle_t h);

/**
 * Remove a compiled program from ANE SRAM without freeing the compile slot.
 *
 * Calls unloadWithQoS: on the model but does NOT call
 * purgeCompiledModelMatchingHash:, so aned's compile-slot entry remains alive.
 * A subsequent libane_delta_reload() on the same handle will skip
 * compileWithQoS: and reload in ~1 ms (aned in-memory cache, Layer 3).
 *
 * Use this for LRU-style SRAM management when the slot limit (~16) allows it:
 *
 *   libane_unload_sram(h_a);       // free SRAM, keep compile slot
 *   libane_delta_reload(h_b);      // swap b into SRAM in ~1 ms
 *   libane_delta_reload(h_a);      // swap a back in ~1 ms (slot intact)
 *
 * Call libane_release() when you are done with the handle entirely.
 * Safe to call with NULL.
 */
LIBANE_API void libane_unload_sram(libane_handle_t h);

/**
 * Release a compiled program handle and return resources to the pool.
 * Calls unloadWithQoS: (SRAM) + purgeCompiledModelMatchingHash: (compile slot).
 * A subsequent libane_compile() with the same content will pay the
 * ANECompilerService cost (~40 ms from disk cache, or ~4000 ms cold).
 * Safe to call with NULL.
 */
LIBANE_API void libane_release(libane_handle_t h);

/* ── High-level convenience ──────────────────────────────────────────────── */

/**
 * Single-call fp16 matrix multiplication: C = A × B.
 *
 * A is [M, K], B is [K, N], C is [M, N] — all in row-major fp16.
 * Falls back to Accelerate BLAS if ANE is unavailable.
 *
 * @return LIBANE_OK on success, negative error code on failure.
 */
LIBANE_API libane_status_t libane_matmul_f16(const libane_f16_t* A,
                                              const libane_f16_t* B,
                                              libane_f16_t* C,
                                              int M, int K, int N);

/**
 * Single-call fp32 matrix multiplication: C = A × B (casts fp32→fp16→fp32).
 */
LIBANE_API libane_status_t libane_matmul_f32(const float* A,
                                              const float* B,
                                              float* C,
                                              int M, int K, int N);

/* ── Status / introspection ──────────────────────────────────────────────── */

/** @return 1 if ANE is accessible and initialized, 0 if running in fallback mode. */
LIBANE_API int libane_available(void);

/** @return Semantic version string, e.g. "0.8.2". */
LIBANE_API const char* libane_version(void);

/** @return Human-readable description of the last error on the calling thread. */
LIBANE_API const char* libane_last_error(void);

/** Set the global log level. Default: LIBANE_LOG_ERROR. */
LIBANE_API void libane_set_log_level(libane_log_level_t level);

/**
 * Force a specific backend.
 *
 * @param backend  "ane" to require ANE (returns LIBANE_ERR_UNAVAILABLE if absent),
 *                 "cpu" to force CPU fallback,
 *                 NULL to restore auto-detect (default).
 */
LIBANE_API void libane_set_backend(const char* backend);

/** Flush the compile cache and release all cached program handles. */
LIBANE_API void libane_cache_flush(void);

/**
 * Mark an explicit end-of-job boundary.
 *
 * Equivalent to libane_cache_flush(); provided as a lifecycle-oriented API for
 * long-running workloads that batch many temporary compilations.
 */
LIBANE_API void libane_end_job(void);

/** Return current cache usage in bytes. */
LIBANE_API size_t libane_cache_size_bytes(void);

/* ── Device introspection ────────────────────────────────────────────────── */

/**
 * ANE hardware capabilities, queried at initialization from _ANEDeviceInfo.
 *
 * All fields are zero/empty if _ANEDeviceInfo is unavailable on the current
 * firmware.  Check libane_device_info_t::available before using the values.
 *
 * architecture  — chip generation string, e.g. "h15g" (M3) or "h16g" (M4).
 * core_count    — number of ANE inference cores; 0 if unavailable.
 * num_anes      — number of ANE units; 0 if unavailable.
 * available     — 1 if _ANEDeviceInfo was successfully queried, 0 otherwise.
 */
typedef struct {
    char     architecture[32];  /**< e.g. "h15g" (M3), "h16g" (M4); "" if unavailable */
    uint32_t core_count;        /**< ANE inference cores (+numANECores); 0 if unavailable */
    uint32_t num_anes;          /**< number of ANE units (+numANEs); 0 if unavailable */
    int      available;         /**< 1 if _ANEDeviceInfo queried successfully, 0 otherwise */
} libane_device_info_t;

/**
 * Fill *out with ANE hardware capabilities.
 *
 * libane_available() need not be called first — this function initializes
 * the runtime if necessary.
 *
 * @return LIBANE_OK on success (even when available==0, the struct is filled
 *         with zeros and the call succeeds).
 *         LIBANE_ERR_INVALID_ARG if out is NULL.
 */
LIBANE_API libane_status_t libane_device_info(libane_device_info_t* out);

/**
 * Return the number of ANE compile slots consumed in this process lifetime.
 *
 * aned (the ANE daemon) enforces a hard per-process limit of approximately
 * 119 unique model compilations.  Exceeding it produces silent failures
 * followed by a hard crash (SIGSEGV).  libane refuses further compiles at
 * 115 (kCompileHardLimit) with a descriptive error, and prints a warning
 * to stderr at 100 (kCompileWarnAt).
 *
 * Slots are consumed only by actual `compileWithQoS:` calls — warm-path
 * hits (compiledModelExists=YES, Path C) do NOT consume a slot.
 *
 * The count is monotonically increasing within a process.  Unloading or
 * releasing a model does NOT return its slot.
 *
 * Recommended use in long-running processes or training loops:
 *   if (libane_compile_count() > 90) { // reconsider compile strategy }
 */
LIBANE_API int libane_compile_count(void);

/**
 * Return the number of compile slots remaining before the hard limit.
 *
 * Returns 0 when the budget is exhausted — further libane_compile() /
 * libane_mil_compile() calls will return NULL with a descriptive error.
 */
LIBANE_API int libane_compile_slots_remaining(void);

/**
 * Warm-path URL-reconnect cache statistics.
 *
 * Populated by libane_cache_stats().  Counters are cumulative from process
 * start (or from the last libane_cache_clear() — see cache_clear note).
 *
 * entries        — cached (hex_id → URL entry) pairs currently held.
 * bytes          — approximate memory footprint of the cache, in bytes.
 *                  Sums mil_text + weight blobs + string overhead per entry.
 * hits           — try_warm_reconnect calls that returned a program.
 * misses         — try_warm_reconnect calls that returned NULL (cache miss,
 *                  or cache hit whose aned slot had been purged).
 * cold_compiles  — cache_populate calls (successful cold compiles).
 * evictions      — stale entries dropped after a reconnect failure.
 */
typedef struct {
    size_t   entries;         /**< currently cached entries */
    size_t   capacity;        /**< max entries before LRU eviction */
    size_t   bytes;           /**< approximate footprint (mil+weights+strings) */
    uint64_t hits;            /**< try_warm_reconnect calls that returned a program */
    uint64_t misses;          /**< try_warm_reconnect calls that returned NULL */
    uint64_t cold_compiles;   /**< successful cold compiles (cache populates) */
    uint64_t evictions;       /**< stale entries dropped after reconnect failure */
    uint64_t lru_evictions;   /**< entries dropped because cache hit capacity */
} libane_cache_stats_t;

/**
 * Read cache statistics for the calling thread's MilBackend instance.
 *
 * Each thread that compiles graphs has its own MilBackend (and therefore
 * its own cache); this function reports stats for *this* thread.  Cross-
 * thread aggregation is not currently supported.
 *
 * Returns LIBANE_OK and populates *out on success.
 * Returns LIBANE_ERR if out is NULL.
 */
LIBANE_API libane_status_t libane_cache_stats(libane_cache_stats_t* out);

/**
 * Drop every entry from the calling thread's MilBackend cache.
 *
 * Subsequent compiles pay full cold cost until the cache repopulates.
 * Counters (hits/misses/cold_compiles/evictions) are preserved so
 * callers can reason about cumulative activity across clear() cycles.
 *
 * Safe to call at any time; a concurrent compile_group() on the same
 * thread is not possible by construction (MilBackend is thread_local).
 */
LIBANE_API void libane_cache_clear(void);

/**
 * Trim the calling thread's warm-path cache to at most `max_entries` via
 * LRU eviction.
 *
 * Use this when coordinating with the compile-slot budget: if
 * libane_compile_slots_remaining() is low, prune the cache aggressively
 * so a follow-up cold compile has slot headroom.  Also useful for
 * long-running processes that want periodic cache hygiene.
 *
 * If max_entries is 0, trims to the current configured capacity (default
 * 128) — effectively a "force capacity enforcement" probe.
 *
 * Returns the number of entries evicted.
 */
LIBANE_API size_t libane_cache_prune(size_t max_entries);

/**
 * Override the LRU capacity for the calling thread's warm-path cache.
 * Default is 128 entries.  Entries beyond the new capacity are evicted
 * immediately.  Capacity of 0 disables caching entirely.
 */
LIBANE_API void libane_cache_set_capacity(size_t capacity);

/**
 * Per-chip ANE tensor shape limits.
 *
 * max_seq       — maximum S dimension (must also be a multiple of seq_alignment).
 * max_channels  — maximum C dimension.
 * seq_alignment — S must be a multiple of this value (always 32).
 *
 * SRAM BUDGET WARNING:
 *   max_seq and max_channels are independent dimension caps, but the real
 *   binding constraint is on-chip SRAM.  Activations at [1, C, 1, S] consume
 *   roughly C × S × 2 bytes of SRAM per live buffer, and the ANE holds at
 *   least input + output simultaneously (2 × C × S × 2 bytes minimum).
 *   For M3 (h15g) the SRAM is approximately 32 MB.
 *
 *   A shape at max_seq × max_channels (131072 × 16384 = ~4 GB of fp16) is
 *   far beyond SRAM on any current chip.  Both max_seq and max_channels can
 *   be reached individually in isolation but NOT simultaneously.
 *
 *   The compile will fail at runtime (LIBANE_ERR_COMPILE_FAILED) if the
 *   combined tensor size spills past firmware SRAM limits — the firmware
 *   enforces Orion constraint #5.  Use these limits as per-dimension guards
 *   only; validate total tensor footprint against your known SRAM budget
 *   before submission.
 */
typedef struct {
    int32_t max_seq;        /**< maximum sequence / spatial dimension */
    int32_t max_channels;   /**< maximum channel dimension */
    int32_t seq_alignment;  /**< S must be a multiple of this (always 32) */
} libane_shape_limits_t;

/**
 * Return the tensor shape limits for the current ANE hardware.
 *
 * Limits are chip-adaptive when _ANEDeviceInfo is available, and fall back
 * to conservative universally-safe values otherwise.  See the SRAM BUDGET
 * WARNING on libane_shape_limits_t — both limits cannot be hit simultaneously.
 */
LIBANE_API libane_shape_limits_t libane_get_shape_limits(void);

/* ── Performance statistics ──────────────────────────────────────────────── */

/**
 * Per-execution ANE hardware counters sampled via IOReport.
 *
 * Populated by libane_mil_execute_stats(). Requires no entitlements or root.
 * If IOReport is unavailable (non-Apple-Silicon target), available == 0 and
 * all numeric fields are zero.
 *
 * ane_bw_utilization — fraction of time ANE DCS bus was active (0.0–1.0).
 *   0.0 may mean the op was too small to register or ran on CPU fallback.
 * avg_bw_state       — mean bandwidth histogram state (0–31 scale).
 * peak_bw_state      — highest bandwidth state observed during this execute.
 * ane_energy_units   — ANE energy in raw IOReport units (not millijoules).
 * throttle_ns        — total nanoseconds ANE spent in any throttle state.
 * available          — 1 if IOReport sampling succeeded, 0 otherwise.
 */
typedef struct {
    float ane_bw_utilization; /**< DCS bus utilization fraction (0.0–1.0) */
    float avg_bw_state;       /**< mean BW histogram state (0–31) */
    int   peak_bw_state;      /**< peak BW histogram state seen */
    long  ane_energy_units;   /**< ANE energy (raw IOReport units) */
    long  throttle_ns;        /**< total throttle residency in ns */
    int   available;          /**< 1 if IOReport sampling succeeded */
} libane_perf_stats_t;


/* ── KV Cache API ────────────────────────────────────────────────────────── */

/**
 * Opaque KV-cache handle.
 *
 * Holds CPU-side fp16 buffers for K and V at a fixed maximum sequence length.
 * Designed for autoregressive decode with SDPA/SDPA_GQA graphs compiled at
 * the maximum sequence length (static-mask strategy).
 *
 * Workflow:
 *   1. Compile your SDPA graph with K/V shape [1, num_heads, max_seq, head_dim].
 *   2. Create a KV cache that matches those dimensions.
 *   3. At each decode step:
 *        a. Pass one new token's K/V slice to libane_kv_cache_update().
 *        b. Pass libane_kv_cache_k() and libane_kv_cache_v() (the full buffers)
 *           as K and V inputs to your compiled SDPA graph.
 *        c. The buffers are zero-initialised beyond the current position,
 *           so attend over garbage is suppressed without an explicit mask.
 *           For exact causal masking, provide a mask tensor to your SDPA graph.
 *   4. Call libane_kv_cache_reset() to reuse the cache for a new sequence.
 */
typedef struct libane_kv_cache_s* libane_kv_cache_t;

/**
 * Create a KV cache for num_heads attention heads, head_dim D, and max_seq positions.
 *
 * Both K and V buffers are allocated as zeroed fp16 arrays of size
 * num_heads × max_seq × head_dim elements.  The cache layout matches the
 * matrix tensor format expected by SDPA/SDPA_GQA:
 *   [1, num_heads, max_seq, head_dim]  →  channels=num_heads, height=max_seq, seq=head_dim
 *
 * @param num_heads  Number of attention heads (= num_kv_heads for GQA).
 * @param head_dim   Head dimension D.
 * @param max_seq    Maximum sequence length to cache.
 * @return Non-null handle on success; NULL on invalid args or OOM.
 *         Must be freed with libane_kv_cache_release().
 */
LIBANE_API libane_kv_cache_t libane_kv_cache_create(int num_heads,
                                                      int head_dim,
                                                      int max_seq);

/**
 * Append one new token's K and V slices to the cache.
 *
 * new_k and new_v must each point to num_heads × head_dim fp16 elements
 * in [head, dim] row-major order (i.e. the per-token contribution from all heads).
 *
 * If pos >= max_seq the cache is full and the call returns -1 without
 * modifying the cache (the caller should stop or evict).
 *
 * @param cache   KV cache handle.
 * @param new_k   Source fp16 array [num_heads × head_dim].
 * @param new_v   Source fp16 array [num_heads × head_dim].
 * @return        New position (1-based count of filled slots) on success,
 *                or -1 if the cache is full or arguments are invalid.
 */
LIBANE_API int libane_kv_cache_update(libane_kv_cache_t cache,
                                       const void*       new_k,
                                       const void*       new_v);

/**
 * Return a read-only pointer to the full K buffer
 * (num_heads × max_seq × head_dim fp16 elements).
 */
LIBANE_API const void* libane_kv_cache_k(libane_kv_cache_t cache);

/**
 * Return a read-only pointer to the full V buffer
 * (num_heads × max_seq × head_dim fp16 elements).
 */
LIBANE_API const void* libane_kv_cache_v(libane_kv_cache_t cache);

/** Return the number of valid token positions written so far. */
LIBANE_API int libane_kv_cache_position(libane_kv_cache_t cache);

/**
 * Reset the position counter to 0 without freeing memory.
 * The K/V buffers are also zeroed so stale data cannot affect new sequences.
 */
LIBANE_API void libane_kv_cache_reset(libane_kv_cache_t cache);

/** Free a KV cache. Safe to call with NULL. */
LIBANE_API void libane_kv_cache_release(libane_kv_cache_t cache);

/* ── Graph API ───────────────────────────────────────────────────────────── */

/**
 * Sentinel value returned by libane_graph_add_input() / libane_graph_add_op()
 * on failure.
 */
#define LIBANE_INVALID_TENSOR_ID  0xFFFFFFFFu

/** Opaque graph builder handle. */
typedef struct libane_graph_s*          libane_graph_t;

/** Opaque compiled graph handle. */
typedef struct libane_compiled_graph_s* libane_compiled_graph_t;

/**
 * Create a new, empty graph builder.
 * @return Non-null handle, or NULL on OOM. Must be freed with libane_graph_release().
 */
LIBANE_API libane_graph_t libane_graph_create(void);

/**
 * Free a graph builder. Safe to call with NULL.
 * Does NOT affect any compiled graph derived from it.
 */
LIBANE_API void libane_graph_release(libane_graph_t g);

/**
 * Declare a graph input tensor.
 *
 * @param g      Graph handle.
 * @param name   Human-readable name (used in debug output).
 * @param shape  ANE tensor shape [1, C, 1, S].  S must be a multiple of 32.
 * @return       Tensor ID, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_input(libane_graph_t  g,
                                            const char*     name,
                                            libane_shape_t  shape);

/**
 * Add an operation node to the graph.
 *
 * @param g            Graph handle.
 * @param op           Operation code.
 * @param input_ids    Array of num_inputs tensor IDs consumed by this op.
 * @param num_inputs   Number of input tensor IDs.
 * @param output_shape Shape of this op's single output tensor.
 * @param weights      Raw fp16 weight bytes (may be NULL for weight-free ops).
 * @param weights_len  Byte length of weights.
 * @return             Tensor ID of the new output tensor,
 *                     or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_op(libane_graph_t       g,
                                         libane_op_t          op,
                                         const uint32_t*      input_ids,
                                         size_t               num_inputs,
                                         libane_shape_t       output_shape,
                                         const void*          weights,
                                         size_t               weights_len);

/**
 * Mark a tensor as a graph output.
 * Outputs are returned by libane_graph_execute() in the order they are marked.
 *
 * @param g         Graph handle.
 * @param tensor_id Tensor ID to mark as output.
 * @param name      Optional output name (may be NULL).
 * @return          LIBANE_OK on success, LIBANE_ERR_INVALID_ARG on bad tensor_id.
 */
LIBANE_API libane_status_t libane_graph_mark_output(libane_graph_t g,
                                                     uint32_t       tensor_id,
                                                     const char*    name);

/**
 * Add a piecewise-linear custom activation op to the graph.
 *
 * Convenience wrapper around libane_graph_add_op(LIBANE_OP_PWL_ACTIVATION).
 * Approximates any smooth activation over [x_min, x_max] using n_samples-1
 * equal-width linear segments.  Outside the range, extrapolates linearly
 * from the nearest endpoint segment.
 *
 * @param g            Graph handle.
 * @param input_id     Input tensor ID.
 * @param output_shape Must match input shape (same C and S).
 * @param x_min        Left boundary of the approximation domain.
 * @param x_max        Right boundary of the approximation domain.
 * @param samples      n_samples output values at equal x spacing from x_min to x_max.
 * @param n_samples    Number of sample points (≥ 2; 33 = 32 segments recommended).
 * @return             Tensor ID of the output, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_pwl_activation(libane_graph_t g,
                                                     uint32_t       input_id,
                                                     libane_shape_t output_shape,
                                                     float          x_min,
                                                     float          x_max,
                                                     const float*   samples,
                                                     uint32_t       n_samples);

/**
 * Add a 2D convolution op to the graph.
 *
 * Convenience wrapper around libane_graph_add_op(LIBANE_OP_CONV2D).
 * Packs the 11 integer hyperparameters and the fp16 kernel blob into the
 * internal weight format expected by the graph compiler.
 *
 * Input must be a conv image tensor: [1, IC, H_in, W_in] where H_in > 1
 * and W_in is a multiple of 32 (IOSurface 64-byte DMA alignment).
 * Output must be:  [1, OC, H_out, W_out]  where:
 *   H_out = (H_in + pad_top  + pad_bottom - dilation_h*(kH-1) - 1) / stride_h + 1
 *   W_out = (W_in + pad_left + pad_right  - dilation_w*(kW-1) - 1) / stride_w + 1
 * W_out must also be a multiple of 32.
 *
 * Bias is not supported; chain a graph ADD node if needed.
 *
 * @param g             Graph handle.
 * @param input_id      Input tensor ID.
 * @param output_shape  Output shape [1, OC, H_out, W_out].
 * @param kH            Kernel height (≥ 1).
 * @param kW            Kernel width  (≥ 1).
 * @param stride_h      Vertical stride   (≥ 1).
 * @param stride_w      Horizontal stride (≥ 1).
 * @param pad_top       Top    padding in pixels (≥ 0).
 * @param pad_left      Left   padding in pixels (≥ 0).
 * @param pad_bottom    Bottom padding in pixels (≥ 0).
 * @param pad_right     Right  padding in pixels (≥ 0).
 * @param dilation_h    Vertical dilation   (≥ 1).
 * @param dilation_w    Horizontal dilation (≥ 1).
 * @param groups        Depthwise/group factor (≥ 1; IC % groups == 0).
 * @param kernel        fp16 kernel data, shape [OC, IC/groups, kH, kW] row-major.
 * @param kernel_bytes  Byte length of kernel (must equal OC × IC/groups × kH × kW × 2).
 * @return              Tensor ID of the output, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_conv2d(libane_graph_t g,
                                             uint32_t       input_id,
                                             libane_shape_t output_shape,
                                             int            kH,
                                             int            kW,
                                             int            stride_h,
                                             int            stride_w,
                                             int            pad_top,
                                             int            pad_left,
                                             int            pad_bottom,
                                             int            pad_right,
                                             int            dilation_h,
                                             int            dilation_w,
                                             int            groups,
                                             const void*    kernel,
                                             size_t         kernel_bytes);

/**
 * Add a W8A16 quantized matrix multiply op to the graph.
 *
 * Convenience wrapper around libane_graph_add_op(LIBANE_OP_MATMUL_W8A16).
 * Packs the int8 weight matrix and float32 per-channel scale vector into the
 * internal blob format expected by the graph compiler.
 *
 * Dequantization is performed at compile time (fused into the conv1x1 path):
 *   W_fp16[ic, oc] = (float)weights[ic * OC + oc] × scales[oc]
 *
 * The resulting fp16 weight matrix is then compiled exactly like a standard
 * MATMUL, so W8A16 nodes fuse with downstream elementwise ops (GELU, ADD, …)
 * at zero extra cost.
 *
 * @param g            Graph handle.
 * @param input_id     Input tensor ID; must have shape [1, IC, 1, S].
 * @param output_shape [1, OC, 1, S]; S must match the input's S.
 * @param weights      int8 weight matrix, shape [IC, OC] row-major.
 *                     (Same layout convention as MATMUL: IC rows, OC columns.)
 * @param scales       float32 per-output-channel scales, length OC.
 *                     Symmetric quantization only (zero-point implicitly 0).
 * @param IC           Input channel count.  Must equal input tensor's channels.
 * @param OC           Output channel count. Must equal output tensor's channels.
 * @return             Tensor ID of the output, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_matmul_w8a16(libane_graph_t g,
                                                    uint32_t       input_id,
                                                    libane_shape_t output_shape,
                                                    const int8_t*  weights,
                                                    const float*   scales,
                                                    int            IC,
                                                    int            OC);

/**
 * Add a W8A8 quantized matrix multiply op to the graph.
 *
 * Like MATMUL_W8A16 but the caller also quantizes activations to int8 before
 * calling execute.  The executor dequantizes them transparently:
 *   A_fp16[n] = (A_int8[n] - act_zero_point) × act_scale
 *
 * Weight dequantization is identical to W8A16 (performed at compile time).
 *
 * @param g               Graph handle.
 * @param input_id        Input tensor ID; shape [1, IC, 1, S].
 *                        At execute time the caller passes int8 data
 *                        (IC × S bytes, not fp16).
 * @param output_shape    [1, OC, 1, S].
 * @param weights         int8 weight matrix [IC, OC] row-major.
 * @param scales          float32 per-channel weight scales [OC].
 * @param IC              Input channel count.
 * @param OC              Output channel count.
 * @param act_scale       Per-tensor activation scale.
 * @param act_zero_point  Per-tensor activation zero-point.
 * @return                Output tensor ID, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_matmul_w8a8(libane_graph_t g,
                                                   uint32_t       input_id,
                                                   libane_shape_t output_shape,
                                                   const int8_t*  weights,
                                                   const float*   scales,
                                                   int            IC,
                                                   int            OC,
                                                   float          act_scale,
                                                   int32_t        act_zero_point);

/**
 * Convenience wrapper around libane_graph_add_op(LIBANE_OP_SDPA_GQA).
 *
 * Adds a Grouped Query Attention node to the graph.  K and V tensors carry
 * num_kv_heads heads; they are tiled to num_q_heads before scaled-dot-product
 * attention is applied.
 *
 *   Q_id   : tensor [1, num_q_heads,  S, D]
 *   K_id   : tensor [1, num_kv_heads, S, D]
 *   V_id   : tensor [1, num_kv_heads, S, D]
 *   mask_id: tensor [1, 1, S, S]  or LIBANE_INVALID_TENSOR_ID for unmasked
 *
 * Output shape: [1, num_q_heads, S, D]  (derived from Q shape).
 *
 * Constraints:
 *   - num_q_heads % num_kv_heads == 0
 *   - S and D must each be a multiple of 32
 *   - all tensors use matrix format  [1, H, S, D] with H=channels, seq=D
 *
 * @param g       Graph handle.
 * @param Q_id    Query tensor ID.
 * @param K_id    Key tensor ID.
 * @param V_id    Value tensor ID.
 * @param mask_id Attention mask tensor ID, or LIBANE_INVALID_TENSOR_ID.
 * @return        Output tensor ID, or LIBANE_INVALID_TENSOR_ID on error.
 */
LIBANE_API uint32_t libane_graph_add_sdpa_gqa(libane_graph_t g,
                                               uint32_t       Q_id,
                                               uint32_t       K_id,
                                               uint32_t       V_id,
                                               uint32_t       mask_id);

/**
 * Quantize a flat fp16 array to int8 using symmetric or asymmetric quantization.
 *
 * Convenience helper for preparing activations before passing to a W8A8 graph:
 *   out[i] = clamp(round(in_fp16[i] / act_scale) + act_zero_point, -128, 127)
 *
 * @param in_fp16         Source fp16 data.
 * @param out_i8          Destination int8 buffer (must have capacity >= n).
 * @param n               Number of elements.
 * @param act_scale       Scale factor (> 0).
 * @param act_zero_point  Zero-point offset.
 */
LIBANE_API void libane_quantize_i8(const void* in_fp16,
                                    int8_t*     out_i8,
                                    size_t      n,
                                    float       act_scale,
                                    int32_t     act_zero_point);

/**
 * Validate, fuse, and compile the graph for ANE execution.
 *
 * @param g  Graph handle.  The graph is not consumed — it can be compiled again.
 * @return   Non-null compiled graph handle on success, NULL on failure
 *           (validation error, ANE unavailable, or compile error).
 *           Must be freed with libane_compiled_graph_release().
 */
LIBANE_API libane_compiled_graph_t libane_graph_compile(libane_graph_t g);

/**
 * Free a compiled graph. Safe to call with NULL.
 */
LIBANE_API void libane_compiled_graph_release(libane_compiled_graph_t cg);

/**
 * Save a compiled graph to a file for fast cold-start restoration.
 *
 * Writes MIL text, weight blobs, and compiled HWX binaries to a binary file
 * (magic "ANEG", version 1).  On reload via libane_compiled_graph_load(),
 * the expensive compileWithQoS: step is skipped — only loadWithQoS: is
 * called, which is ~8.5× faster (~494 ms vs ~4200 ms per group).
 *
 * @param cg    Non-null compiled graph to serialize.
 * @param path  Destination file path.  Created or overwritten.
 * @return      LIBANE_OK on success;
 *              LIBANE_ERR_INVALID_ARG if cg or path is NULL;
 *              LIBANE_ERR_EXECUTE_FAILED if serialization fails (e.g. I/O error).
 */
LIBANE_API libane_status_t libane_compiled_graph_save(libane_compiled_graph_t cg,
                                                       const char* path);

/**
 * Load a compiled graph from a file produced by libane_compiled_graph_save().
 *
 * Calls ane_restore_program() per group, which writes the saved HWX back to a
 * temp dir and calls loadWithQoS: — skipping recompilation for an ~8.5×
 * cold-start speedup.  Gracefully falls back to full recompilation if the
 * saved HWX is incompatible with the current macOS version.
 *
 * @param path  File path produced by libane_compiled_graph_save().
 * @return      Non-null compiled graph on success;
 *              NULL on format error, I/O error, or ANE failure.
 *              Must be freed with libane_compiled_graph_release().
 */
LIBANE_API libane_compiled_graph_t libane_compiled_graph_load(const char* path);

/**
 * Reload all ANE programs in a compiled graph into SRAM without recompiling.
 *
 * This is ~8.5× faster than recompiling and is the correct recovery operation
 * after a host suspend/resume cycle or any ANE context reset.
 *
 * It also enables LoRA-style weight hot-swap: mutate the weight blobs held by
 * each group's AneProgram, then call libane_compiled_graph_delta_reload() to
 * push the updated weights to the accelerator without rebuilding the graph.
 *
 * @param cg  Non-null compiled graph handle.
 * @return    LIBANE_OK on success; LIBANE_ERR_EXECUTE_FAILED if any reload fails;
 *            LIBANE_ERR_INVALID_ARG if cg is NULL.
 */
LIBANE_API libane_status_t libane_compiled_graph_delta_reload(libane_compiled_graph_t cg);

/**
 * Execute a compiled graph.
 *
 * @param cg           Compiled graph handle.
 * @param input_ptrs   Array of fp16 data pointers, one per graph input (in
 *                     the order inputs were added via libane_graph_add_input).
 * @param input_bytes  Byte sizes of the input buffers.
 * @param num_inputs   Length of input_ptrs / input_bytes.
 * @param output_ptrs  Caller-allocated destination buffers, one per graph
 *                     output (in the order outputs were marked).
 * @param output_bytes Byte sizes of the output buffers.
 * @param num_outputs  Length of output_ptrs / output_bytes.
 * @return             LIBANE_OK on success, negative status on failure.
 */
LIBANE_API libane_status_t libane_graph_execute(libane_compiled_graph_t cg,
                                                 const void**            input_ptrs,
                                                 const size_t*           input_bytes,
                                                 size_t                  num_inputs,
                                                 void**                  output_ptrs,
                                                 const size_t*           output_bytes,
                                                 size_t                  num_outputs);

/* ── Raw MIL probe API ───────────────────────────────────────────────────── */

/**
 * Opaque handle to a compiled raw MIL program.
 * Created by libane_mil_compile().
 * Destroyed by libane_mil_release().
 *
 * The struct is defined outside the libane namespace so the C ABI typedef
 * works without name-mangling.
 */
typedef struct libane_mil_program_s* libane_mil_handle_t;

/**
 * Compile a raw MIL text program with optional external weight files.
 *
 * @param mil_text      UTF-8 MIL source (complete program including buildInfo header).
 * @param weight_names  Array of weight filenames referenced by file() in the MIL text
 *                      (e.g. "weight.bin").  May be NULL when num_weights == 0.
 * @param weight_data   Array of pointers to raw fp16 weight data (no ANE header —
 *                      the header is added internally).  May be NULL when num_weights == 0.
 * @param weight_sizes  Byte sizes of each weight_data buffer.
 * @param num_weights   Number of weight files.
 *
 * @return  Non-null libane_mil_handle_t on success; NULL on failure.
 *          The handle must be freed with libane_mil_release().
 *          Requires ANE; returns NULL when ANE is unavailable.
 */
LIBANE_API libane_mil_handle_t libane_mil_compile(const char*   mil_text,
                                                   const char**  weight_names,
                                                   const void**  weight_data,
                                                   const size_t* weight_sizes,
                                                   size_t        num_weights);

/**
 * Execute a compiled MIL program.
 *
 * Inputs and outputs are raw fp16 buffers.  All input IOSurfaces are allocated
 * at the same size (max of all in_sizes and 49152) to satisfy Orion constraint #18.
 * All output IOSurfaces are similarly uniform (Orion constraint #2).
 *
 * @param h           Handle from libane_mil_compile().
 * @param in_data     Array of pointers to fp16 input data.
 * @param in_sizes    Byte sizes of each input buffer.
 * @param num_inputs  Length of in_data / in_sizes.
 * @param out_data    Caller-allocated destination buffers (fp16).
 * @param out_sizes   Byte sizes of each output buffer.
 * @param num_outputs Length of out_data / out_sizes.
 *
 * @return LIBANE_OK on success, negative status on failure.
 */
LIBANE_API libane_status_t libane_mil_execute(libane_mil_handle_t h,
                                               const void**  in_data,
                                               const size_t* in_sizes,
                                               size_t        num_inputs,
                                               void**        out_data,
                                               const size_t* out_sizes,
                                               size_t        num_outputs);

/**
 * Execute a compiled MIL program and return hardware performance counters.
 *
 * Identical to libane_mil_execute() but populates *stats_out after execution.
 * Pass stats_out = NULL to skip stat collection (equivalent to libane_mil_execute).
 *
 * If IOReport is unavailable, stats_out->available will be 0 and execution
 * proceeds normally (not an error).
 *
 * @param h           Handle from libane_mil_compile().
 * @param in_data     Array of pointers to fp16 input data.
 * @param in_sizes    Byte sizes of each input buffer.
 * @param num_inputs  Length of in_data / in_sizes.
 * @param out_data    Caller-allocated destination buffers (fp16).
 * @param out_sizes   Byte sizes of each output buffer.
 * @param num_outputs Length of out_data / out_sizes.
 * @param stats_out   Receives hardware counters after execution; may be NULL.
 *
 * @return LIBANE_OK on success, negative status on failure.
 */
LIBANE_API libane_status_t libane_mil_execute_stats(libane_mil_handle_t  h,
                                                     const void**         in_data,
                                                     const size_t*        in_sizes,
                                                     size_t               num_inputs,
                                                     void**               out_data,
                                                     const size_t*        out_sizes,
                                                     size_t               num_outputs,
                                                     libane_perf_stats_t* stats_out);

/**
 * Return whether the last ANE load for this program spilled to DRAM.
 *
 * After libane_mil_compile(), the firmware sets
 * _ANEInMemoryModel.intermediateBufferHandle to a non-zero IOSurface handle
 * when the model's inter-layer intermediate activations exceed on-chip SRAM
 * and must be backed by DRAM.  DRAM-backed intermediates incur ~30% throughput
 * penalty.
 *
 * NOTE: this only fires for multi-operation fused programs.  Single-layer
 * programs (one conv, one matmul) have no inter-layer intermediates and will
 * always return 0 regardless of tensor size.  Confirmed via probe_sram_spill
 * (M3 Pro / macOS 26.3.1, 2026-04-16): a [1,4096,1,4096] single conv loaded
 * cleanly with intermediateBufferHandle == 0.
 *
 * A spill cannot be resolved without redesigning the model (fewer simultaneous
 * live activations, smaller tile sizes, or splitting into multiple programs).
 *
 * @param h  Handle from libane_mil_compile().
 * @return   1 if SRAM spill was detected, 0 if not, -1 on null handle.
 */
LIBANE_API int libane_mil_sram_spill(libane_mil_handle_t h);

/**
 * Free a compiled MIL program.  Safe to call with NULL.
 */
LIBANE_API void libane_mil_release(libane_mil_handle_t h);

/**
 * Save a compiled MIL program to a file for fast cold-start restoration.
 *
 * Writes MIL source text, weight blobs, and compiled HWX binary to a
 * binary file (magic "ANEM", version 1).  On reload via libane_mil_load(),
 * the expensive compileWithQoS: step is skipped — only loadWithQoS: is
 * called, giving ~8.5× faster cold-start.
 *
 * @param h     Non-null MIL handle.
 * @param path  Destination file path (created or overwritten).
 * @return      LIBANE_OK on success;
 *              LIBANE_ERR_INVALID_ARG if h or path is NULL;
 *              LIBANE_ERR_EXECUTE_FAILED on I/O error or serialization failure.
 */
LIBANE_API libane_status_t libane_mil_save(libane_mil_handle_t h,
                                            const char*         path);

/**
 * Load a compiled MIL program saved by libane_mil_save().
 *
 * @param path  File path produced by libane_mil_save().
 * @return      Non-null handle on success;
 *              NULL on format error, I/O error, or ANE failure.
 *              Must be freed with libane_mil_release().
 */
LIBANE_API libane_mil_handle_t libane_mil_load(const char* path);

#ifdef __cplusplus
}
#endif
