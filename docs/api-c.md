# C API Reference

libane exposes a stable C ABI declared in `include/libane.h`. All symbols are prefixed `libane_`. The ABI is guaranteed stable across minor versions.

---

## Version

```c
#define LIBANE_VERSION        "0.8.0"
#define LIBANE_VERSION_MAJOR  0
#define LIBANE_VERSION_MINOR  8
#define LIBANE_VERSION_PATCH  0
```

---

## Types

### `libane_handle_t`

Opaque handle to a single compiled operation. Created by `libane_compile()`, destroyed by `libane_release()`.

### `libane_shape_t`

```c
typedef struct {
    int32_t dims[4];  // [batch, channels, height, seq]
    int32_t ndim;     // number of meaningful dimensions (≤ 4)
} libane_shape_t;
```

ANE tensors are always `[1, C, 1, S]`. `dims[0]` is always 1 (batch), `dims[1]` is C (channels), `dims[2]` is always 1 (height), `dims[3]` is S (sequence). S must be a multiple of 16.

For matmul `A[M,K] × B[K,N]`: A shape is `{1, K, 1, M}`, B shape is `{1, N, 1, K}`.

### `libane_status_t`

```c
LIBANE_OK                =  0   // success
LIBANE_ERR_UNAVAILABLE   = -1   // ANE not accessible; fallback active
LIBANE_ERR_INVALID_SHAPE = -2   // shape violates ANE constraints
LIBANE_ERR_INVALID_OP    = -3   // op not in supported set
LIBANE_ERR_COMPILE_FAILED= -4   // MIL compilation failed
LIBANE_ERR_EXECUTE_FAILED= -5   // execution failed
LIBANE_ERR_OOM           = -6   // out of memory
LIBANE_ERR_INVALID_ARG   = -7   // null pointer or bad argument
```

### `libane_log_level_t`

```c
LIBANE_LOG_SILENT = 0
LIBANE_LOG_ERROR  = 1
LIBANE_LOG_WARN   = 2
LIBANE_LOG_INFO   = 3
LIBANE_LOG_DEBUG  = 4
```

---

## Runtime control

### `libane_available`

```c
int libane_available(void);
```

Returns 1 if the ANE is accessible on the current machine, 0 otherwise. Safe to call at any time; initializes the runtime on first call.

### `libane_version`

```c
const char* libane_version(void);
```

Returns the library version string, e.g. `"0.8.0"`. Pointer is valid for the lifetime of the process.

### `libane_last_error`

```c
const char* libane_last_error(void);
```

Returns the most recent error message from any libane call. Thread-local. Returns an empty string when no error has occurred.

### `libane_set_log_level`

```c
void libane_set_log_level(libane_log_level_t level);
```

Sets the minimum log level. Default is `LIBANE_LOG_ERROR`. Logs go to `stderr`.

### `libane_set_backend`

```c
void libane_set_backend(const char* backend);
```

Forces a specific backend. Values: `"ane"` (ANE only, fails if unavailable), `"cpu"` (Accelerate fallback only), `NULL` (auto-detect, default). Useful for testing CPU fallback paths.

### `libane_cache_flush`

```c
void libane_cache_flush(void);
```

Evicts all entries from the compile cache and releases cached program handles.

### `libane_cache_size_bytes`

```c
size_t libane_cache_size_bytes(void);
```

Returns current compile cache usage in bytes.

---

## Single-op API

### `libane_compile`

```c
libane_handle_t libane_compile(libane_op_t    op,
                               libane_shape_t shape,
                               const void*    weights,
                               size_t         weights_len);
```

Compiles an operation for the given shape and weights. Thread-safe. Compilation is cached — subsequent calls with the same `(op, shape, weight-hash)` tuple return from cache immediately.

Returns an opaque handle on success, `NULL` on failure. Check `libane_last_error()` on failure.

`weights` may be `NULL` for elementwise ops that carry no learned parameters (e.g. `LIBANE_OP_GELU`, `LIBANE_OP_SOFTMAX`).

### `libane_compile_batch`

```c
libane_status_t libane_compile_batch(const libane_compile_request_t* requests,
                                     size_t                          num_requests,
                                     libane_handle_t*                out_handles);
```

Compiles multiple operations in a single call. Each request is compiled independently; on partial failure `out_handles` contains both successful (non-NULL) and failed (NULL) entries. Returns `LIBANE_OK` if all compilations succeeded, `LIBANE_ERR_COMPILE_FAILED` if one or more failed.

```c
typedef struct {
    libane_op_t    op;
    libane_shape_t shape;
    const void*    weights;
    size_t         weights_len;
} libane_compile_request_t;
```

### `libane_execute`

```c
libane_status_t libane_execute(libane_handle_t h,
                               const void*    input,
                               void*          output,
                               libane_shape_t shape);
```

Executes a compiled single-input operation. `input` and `output` are fp16 buffers whose sizes are implied by `shape`. `shape` must match the shape used at compile time.

### `libane_execute2`

```c
libane_status_t libane_execute2(libane_handle_t h,
                                const void*    input0,
                                const void*    input1,
                                void*          output,
                                libane_shape_t shape);
```

Executes a compiled two-input operation (e.g. `LIBANE_OP_ADD`, `LIBANE_OP_MUL`). Inputs are in alphabetical order of their MIL parameter names (ANE constraint #13).

### `libane_delta_reload`

```c
libane_status_t libane_delta_reload(libane_handle_t h);
```

Re-loads a compiled program into ANE SRAM without recompiling. Useful after an internal unload. Load-only is ~8.5× faster than a full recompile. Does **not** accept new weights — weights are baked into the HWX bytecode at compile time and cannot be updated via reload. To change weights, call `libane_release()` + `libane_compile()`.

Returns `LIBANE_ERR_UNAVAILABLE` if the handle is not ANE-compiled. Returns `LIBANE_ERR_EXECUTE_FAILED` if the reload fails (handle is then invalid).

### `libane_release`

```c
void libane_release(libane_handle_t h);
```

Releases a compiled program handle and returns resources to the pool. Safe to call with `NULL`.

---

## Graph API

The Graph API describes full forward passes as a DAG, fuses chains of compatible ops, and compiles them into the minimum number of ANE dispatches.

### Building a graph

```c
libane_graph_t libane_graph_create(void);
```

Creates a new empty graph builder. Returns `NULL` on OOM. Must be freed with `libane_graph_release()` regardless of whether `libane_graph_compile()` was called.

```c
uint32_t libane_graph_add_input(libane_graph_t  g,
                                const char*     name,
                                libane_shape_t  shape);
```

Declares a graph input. Returns a tensor ID, or `LIBANE_INVALID_TENSOR_ID` (0xFFFFFFFF) on failure.

```c
uint32_t libane_graph_add_op(libane_graph_t       g,
                             libane_op_t          op,
                             const uint32_t*      input_ids,
                             size_t               num_inputs,
                             libane_shape_t       output_shape,
                             const void*          weights,
                             size_t               weights_len);
```

Adds an operation to the graph. `input_ids` are tensor IDs returned by prior `add_input` / `add_op` calls. Returns the output tensor ID, or `LIBANE_INVALID_TENSOR_ID` on failure.

```c
uint32_t libane_graph_add_pwl_activation(libane_graph_t g,
                                         uint32_t       input_id,
                                         libane_shape_t output_shape,
                                         float          x_min,
                                         float          x_max,
                                         const float*   samples,
                                         uint32_t       n_samples);
```

Adds a piecewise-linear custom activation. Approximates any smooth activation over `[x_min, x_max]` using `n_samples - 1` equal-width linear segments. Recommended: `n_samples = 33` (32 segments). Output shape must match input shape.

```c
libane_status_t libane_graph_mark_output(libane_graph_t g,
                                         uint32_t       tensor_id,
                                         const char*    name);
```

Marks a tensor as a graph output. Outputs are returned by `libane_graph_execute()` in the order they are marked. `name` may be `NULL`.

### Compiling and executing

```c
libane_compiled_graph_t libane_graph_compile(libane_graph_t g);
```

Validates, fuses, and compiles the graph. The graph is not consumed — it can be compiled again. Returns `NULL` on failure. Must be freed with `libane_compiled_graph_release()`.

```c
libane_status_t libane_graph_execute(libane_compiled_graph_t cg,
                                     const void**            input_ptrs,
                                     const size_t*           input_bytes,
                                     size_t                  num_inputs,
                                     void**                  output_ptrs,
                                     const size_t*           output_bytes,
                                     size_t                  num_outputs);
```

Executes a compiled graph. All buffers are fp16. Inputs are in the order they were added via `libane_graph_add_input`. Outputs are in the order they were marked via `libane_graph_mark_output`.

```c
void libane_compiled_graph_release(libane_compiled_graph_t cg);
void libane_graph_release(libane_graph_t g);
```

Both are safe to call with `NULL`.

---

## Raw MIL API

For direct MIL program submission — bypasses the graph compiler and fusion layer. Intended for research and diagnostics.

### `libane_mil_compile`

```c
libane_mil_handle_t libane_mil_compile(const char*   mil_text,
                                       const char**  weight_names,
                                       const void**  weight_data,
                                       const size_t* weight_sizes,
                                       size_t        num_weights);
```

Compiles a raw MIL text program. `mil_text` is a complete MIL source including the `buildInfo` header. `weight_names` / `weight_data` / `weight_sizes` supply external weight files referenced by `file()` in the MIL; pass `NULL` / `NULL` / `NULL` / `0` when there are no external weights.

Returns `NULL` if the ANE is unavailable or compilation fails.

### `libane_mil_execute`

```c
libane_status_t libane_mil_execute(libane_mil_handle_t h,
                                   const void**  in_data,
                                   const size_t* in_sizes,
                                   size_t        num_inputs,
                                   void**        out_data,
                                   const size_t* out_sizes,
                                   size_t        num_outputs);
```

Executes a compiled MIL program. All buffers are fp16. IOSurfaces are allocated at a uniform size (max of all `in_sizes` and 49152) to satisfy ANE constraints.

### `libane_mil_execute_stats`

```c
libane_status_t libane_mil_execute_stats(libane_mil_handle_t  h,
                                         const void**         in_data,
                                         const size_t*        in_sizes,
                                         size_t               num_inputs,
                                         void**               out_data,
                                         const size_t*        out_sizes,
                                         size_t               num_outputs,
                                         libane_perf_stats_t* stats_out);
```

Identical to `libane_mil_execute` but also populates `*stats_out` with hardware performance counters. Pass `stats_out = NULL` to skip stat collection. If IOReport sampling is unavailable, `stats_out->available` is 0 and execution proceeds normally.

### `libane_mil_sram_spill`

```c
int libane_mil_sram_spill(libane_mil_handle_t h);
```

Returns 1 if the compiled program's intermediate activations spilled to DRAM (exceeded on-chip SRAM). Returns 0 if no spill occurred. Returns -1 on a null handle.

DRAM-backed intermediates incur approximately 30% throughput penalty. A spill cannot be resolved at runtime — it requires redesigning the program (fewer simultaneous live activations, smaller tile sizes, or splitting into multiple programs).

Only meaningful for multi-operation fused programs. Single-layer programs always return 0.

### `libane_mil_release`

```c
void libane_mil_release(libane_mil_handle_t h);
```

Safe to call with `NULL`.

---

## Device introspection

See [hardware-introspection.md](hardware-introspection.md) for field-level documentation and per-chip values.

### `libane_device_info`

```c
libane_status_t libane_device_info(libane_device_info_t* out);
```

Fills `*out` with ANE hardware capabilities. Does not require `libane_available()` to be called first. Returns `LIBANE_OK` even when `out->available == 0` (struct is zero-filled). Returns `LIBANE_ERR_INVALID_ARG` if `out` is `NULL`.

```c
typedef struct {
    char     architecture[32]; // e.g. "h15g" (M3), "h16g" (M4); "" if unavailable
    uint32_t core_count;       // ANE inference cores; 0 if unavailable
    uint32_t num_anes;         // number of ANE units; 0 if unavailable
    int      available;        // 1 if queried successfully, 0 otherwise
} libane_device_info_t;
```

### `libane_get_shape_limits`

```c
libane_shape_limits_t libane_get_shape_limits(void);
```

Returns per-dimension shape limits for the current ANE hardware. Falls back to conservative universally-safe values when `_ANEDeviceInfo` is unavailable.

```c
typedef struct {
    int32_t max_seq;       // maximum S dimension
    int32_t max_channels;  // maximum C dimension
    int32_t seq_alignment; // S must be a multiple of this (always 16)
} libane_shape_limits_t;
```

**Important:** `max_seq` and `max_channels` are independent per-dimension caps. The binding constraint is on-chip SRAM — both limits cannot be reached simultaneously. See [hardware-introspection.md](hardware-introspection.md).
