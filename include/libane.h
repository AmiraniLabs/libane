/**
 * libane — Apple Neural Engine Native C++ Runtime Library
 * Amirani Labs · v0.7.1
 *
 * Stable C ABI. ABI stability guaranteed across minor versions.
 * Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
 * Not for App Store submission.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ── Version ─────────────────────────────────────────────────────────────── */

#define LIBANE_VERSION         "0.7.1"
#define LIBANE_VERSION_MAJOR   0
#define LIBANE_VERSION_MINOR   7
#define LIBANE_VERSION_PATCH   1

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
} libane_op_t;

/* ── Shape descriptor ────────────────────────────────────────────────────── */

/**
 * ANE tensors are always [1, C, 1, S] (NCHW with H=1).
 * dims[0] = batch (always 1 for ANE), dims[1] = C, dims[2] = 1, dims[3] = S.
 * S must be a multiple of 8 (ANE constraint #1).
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
libane_handle_t libane_compile(libane_op_t op,
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
libane_status_t libane_compile_batch(const libane_compile_request_t* requests,
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
libane_status_t libane_execute(libane_handle_t h,
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
libane_status_t libane_execute2(libane_handle_t h,
                                const void* input0,
                                const void* input1,
                                void* output,
                                libane_shape_t shape);

/**
 * Reload a compiled program with updated weights without recompiling.
 *
 * 8.5x faster than libane_compile() + libane_release() for weight updates
 * (Orion: ~494ms reload vs ~4,200ms recompile per layer).
 *
 * The weight data format is identical to libane_compile() — raw fp16 bytes,
 * NOT including the 128-byte ANE blob header (the library adds that).
 *
 * @param h           Handle from libane_compile(). Must be an ANE-compiled handle.
 * @param new_weights New weight data (fp16, same layout as original compile).
 * @param weights_len Byte length of new_weights.
 * @return LIBANE_OK on success.
 *         LIBANE_ERR_UNAVAILABLE if handle is not ANE-compiled.
 *         LIBANE_ERR_EXECUTE_FAILED if the reload fails (handle is then invalid — call libane_release()).
 */
libane_status_t libane_delta_reload(libane_handle_t h,
                                     const void* new_weights,
                                     size_t weights_len);

/**
 * Release a compiled program handle and return resources to the pool.
 * Safe to call with NULL.
 */
void libane_release(libane_handle_t h);

/* ── High-level convenience ──────────────────────────────────────────────── */

/**
 * Single-call fp16 matrix multiplication: C = A × B.
 *
 * A is [M, K], B is [K, N], C is [M, N] — all in row-major fp16.
 * Falls back to Accelerate BLAS if ANE is unavailable.
 *
 * @return LIBANE_OK on success, negative error code on failure.
 */
libane_status_t libane_matmul_f16(const libane_f16_t* A,
                                  const libane_f16_t* B,
                                  libane_f16_t* C,
                                  int M, int K, int N);

/**
 * Single-call fp32 matrix multiplication: C = A × B (casts fp32→fp16→fp32).
 */
libane_status_t libane_matmul_f32(const float* A,
                                  const float* B,
                                  float* C,
                                  int M, int K, int N);

/* ── Status / introspection ──────────────────────────────────────────────── */

/** @return 1 if ANE is accessible and initialized, 0 if running in fallback mode. */
int libane_available(void);

/** @return Semantic version string, e.g. "0.1.0". */
const char* libane_version(void);

/** @return Human-readable description of the last error on the calling thread. */
const char* libane_last_error(void);

/** Set the global log level. Default: LIBANE_LOG_ERROR. */
void libane_set_log_level(libane_log_level_t level);

/**
 * Force a specific backend.
 *
 * @param backend  "ane" to require ANE (returns LIBANE_ERR_UNAVAILABLE if absent),
 *                 "cpu" to force CPU fallback,
 *                 NULL to restore auto-detect (default).
 */
void libane_set_backend(const char* backend);

/** Flush the compile cache and release all cached program handles. */
void libane_cache_flush(void);

/** Return current cache usage in bytes. */
size_t libane_cache_size_bytes(void);

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
libane_graph_t libane_graph_create(void);

/**
 * Free a graph builder. Safe to call with NULL.
 * Does NOT affect any compiled graph derived from it.
 */
void libane_graph_release(libane_graph_t g);

/**
 * Declare a graph input tensor.
 *
 * @param g      Graph handle.
 * @param name   Human-readable name (used in debug output).
 * @param shape  ANE tensor shape [1, C, 1, S].  S must be a multiple of 8.
 * @return       Tensor ID, or LIBANE_INVALID_TENSOR_ID on error.
 */
uint32_t libane_graph_add_input(libane_graph_t  g,
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
uint32_t libane_graph_add_op(libane_graph_t       g,
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
libane_status_t libane_graph_mark_output(libane_graph_t g,
                                          uint32_t       tensor_id,
                                          const char*    name);

/**
 * Validate, fuse, and compile the graph for ANE execution.
 *
 * @param g  Graph handle.  The graph is not consumed — it can be compiled again.
 * @return   Non-null compiled graph handle on success, NULL on failure
 *           (validation error, ANE unavailable, or compile error).
 *           Must be freed with libane_compiled_graph_release().
 */
libane_compiled_graph_t libane_graph_compile(libane_graph_t g);

/**
 * Free a compiled graph. Safe to call with NULL.
 */
void libane_compiled_graph_release(libane_compiled_graph_t cg);

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
libane_status_t libane_graph_execute(libane_compiled_graph_t cg,
                                      const void**            input_ptrs,
                                      const size_t*           input_bytes,
                                      size_t                  num_inputs,
                                      void**                  output_ptrs,
                                      const size_t*           output_bytes,
                                      size_t                  num_outputs);

#ifdef __cplusplus
}
#endif
