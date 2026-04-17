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
libane_status_t libane_delta_reload(libane_handle_t h);

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

/* ── Device introspection ────────────────────────────────────────────────── */

/**
 * ANE hardware capabilities, queried at initialization from _ANEDeviceInfo.
 *
 * All fields are zero/empty if _ANEDeviceInfo is unavailable on the current
 * firmware.  Check libane_device_info_t::available before using the values.
 *
 * architecture  — chip generation string, e.g. "h15g" (M3) or "h16g" (M4).
 * core_count    — number of ANE inference cores; 0 if unavailable.
 * sram_bytes    — on-chip SRAM capacity in bytes; 0 if unavailable.
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
libane_status_t libane_device_info(libane_device_info_t* out);

/**
 * Per-chip ANE tensor shape limits.
 *
 * max_seq       — maximum S dimension (must also be a multiple of seq_alignment).
 * max_channels  — maximum C dimension.
 * seq_alignment — S must be a multiple of this value (always 8).
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
    int32_t seq_alignment;  /**< S must be a multiple of this (always 8) */
} libane_shape_limits_t;

/**
 * Return the tensor shape limits for the current ANE hardware.
 *
 * Limits are chip-adaptive when _ANEDeviceInfo is available, and fall back
 * to conservative universally-safe values otherwise.  See the SRAM BUDGET
 * WARNING on libane_shape_limits_t — both limits cannot be hit simultaneously.
 */
libane_shape_limits_t libane_get_shape_limits(void);

/* ── Performance statistics ──────────────────────────────────────────────── */

/**
 * Per-execution ANE hardware counters sampled via IOReport.
 *
 * Populated by libane_execute_with_stats(). Requires no entitlements or root.
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

/* ── Raw MIL probe API ───────────────────────────────────────────────────── */

/**
 * Opaque handle to a compiled raw MIL program.
 * Created by libane_mil_compile() / libane_mil_compile_with_weights().
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
libane_mil_handle_t libane_mil_compile(const char*   mil_text,
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
libane_status_t libane_mil_execute(libane_mil_handle_t h,
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
 * _ANEPerformanceStats may require private entitlements on some firmware
 * versions.  If unavailable, stats_out->hw_execution_time_ms will be -1.0
 * and execution proceeds normally (not an error).
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
libane_status_t libane_mil_execute_stats(libane_mil_handle_t h,
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
int libane_mil_sram_spill(libane_mil_handle_t h);

/**
 * Free a compiled MIL program.  Safe to call with NULL.
 */
void libane_mil_release(libane_mil_handle_t h);

#ifdef __cplusplus
}
#endif
