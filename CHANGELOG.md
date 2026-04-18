# Changelog

## v0.8.2 — 2026-04-18

### Added

- **Python bindings parity** — all C API features now exposed in the `ane` Python package:
  - Op constants: `RELU`, `TANH`, `SIGMOID`, `HARDSWISH`, `LEAKY_RELU`, `ELU`,
    `PIXEL_SHUFFLE`, `CAST`, `CONV2D`, `SELECT`, `PWL_ACTIVATION`
  - `ane.device_info()` → dict (architecture, core_count, num_anes, available)
  - `ane.shape_limits()` → dict (max_seq, max_channels, seq_alignment)
  - `ane.compile(op, shape, weights)` → `CompiledOp`
  - `ane.compile_batch(requests)` → `list[CompiledOp | None]`
  - `CompiledOp.execute(x, shape)`, `CompiledOp.execute2(x0, x1, shape)`, `CompiledOp.delta_reload()`
  - `Graph.add_pwl_activation(input_id, output_shape, x_min, x_max, samples)`
  - `CompiledMil.sram_spill` (property)
  - `CompiledMil.run_stats(inputs, output_sizes)` → `(outputs, stats dict)`
  - Updated `ane.pyi` type stub to full parity with the C API

### Changed

- Module tagline updated to "Run ML graphs directly on the Apple Neural Engine from Python."
- Removed "Not for App Store submission" disclaimer from all files — self-evident from
  private API usage.

## v0.8.1 — 2026-04-17

### Fixed

- `libane_mil_compile` now calls `runtime::initialize()` before use — previously
  returned "ANE not available" when called from the dylib without a prior Graph API call.
- `fp16_lit` in `mil_builder.cpp` no longer emits scientific notation (`e` form) for
  near-zero PWL intercept values; firmware MIL parser rejects `e`-notation in `fp16()`
  literals. Values below the fp16 subnormal floor (`~5.96e-8`) are clamped to `0.0`.
- `LIBANE_API` visibility macro now correctly exported from dylib for all 31 public
  C API functions (regression from v0.8.0 visibility refactor).
- Example 10 (`10_throughput_sweep.py`): corrected minimum sequence length from 8 to 16
  (`S % 16 == 0` ANE constraint).

### Added

- `examples/13_c_api.c` — Graph API, `libane_compile_batch`, and `libane_execute2`
  from pure C (not C++).
- `examples/15_device_info.py` — chip architecture, core count, and SRAM budget via
  `libane_device_info`.
- `examples/16_perf_stats.py` — IOReport bandwidth utilization and energy counters via
  `libane_mil_execute_stats`; graceful fallback when IOReport is unavailable.
- `examples/17_pwl_activation.py` — piecewise-linear approximation of SELU, Mish, Swish,
  and Softplus; accuracy comparison against scipy reference.
- `examples/18_sram_spill.py` — SRAM spill detection via `libane_mil_sram_spill` and
  split-dispatch fix demonstration.

## v0.8.0 — 2026-04-17

### Added

- **Device introspection** — `libane_device_info()` returns ANE hardware capabilities
  queried from `_ANEDeviceInfo`: chip architecture string (e.g. `"h15g"` for M3,
  `"h16g"` for M4), inference core count, and ANE unit count.

- **Shape limits API** — `libane_get_shape_limits()` returns chip-adaptive
  `max_seq`, `max_channels`, and `seq_alignment` (always 16). Falls back to
  conservative universally-safe values when `_ANEDeviceInfo` is unavailable.

- **Performance statistics** — `libane_mil_execute_stats()` populates a
  `libane_perf_stats_t` after execution via IOReport: DCS bus utilization fraction,
  mean/peak bandwidth histogram state, raw energy units, and throttle residency.
  No entitlements or root required. `available == 0` when IOReport is unavailable;
  execution proceeds normally.

- **SRAM spill detection** — `libane_mil_sram_spill()` returns 1 if the compiled
  program's inter-layer intermediate activations spilled to DRAM (>30% throughput
  penalty). Checked via `_ANEInMemoryModel.intermediateBufferHandle` after compile.
  Always 0 for single-layer programs.

- **New activation ops** — `RELU`, `TANH`, `SIGMOID`, `HARDSWISH`, `LEAKY_RELU`
  (alpha=0.01), `ELU` (alpha=1.0).

- **Piecewise-linear custom activation** — `LIBANE_OP_PWL_ACTIVATION` and its
  convenience wrapper `libane_graph_add_pwl_activation()`. Approximates any
  smooth activation over `[x_min, x_max]` using equal-width linear segments.
  Recommended: 33 sample points (32 segments).

- **Trigonometric ops** — `SINH`, `COSH`, `TAN`, `ASIN`, `ACOS`.

- **Math ops** — `NEG`, `MOD`.

- **Reduce ops** — `REDUCE_SUM`, `REDUCE_MEAN`, `REDUCE_MAX`, `REDUCE_PROD`.

- **Structural ops** — `RESHAPE`, `CONCAT` (along C), `SLICE_BY_INDEX`.

- **Pooling ops** — `AVG_POOL`, `MAX_POOL`.

- **Logical ops** — `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_XOR`.

- **Scatter / gather ops** — `SCATTER`, `GATHER`, `SCATTER_ND`, `SCATTER_ALONG_AXIS`.

- **`libane_execute2()`** — two-input variant of the single-op execute path for
  elementwise ops (ADD, MUL, etc.) without going through the Graph API.

- **`libane_compile_batch()`** — compiles multiple operations in a single call.
  Supports partial success: on `LIBANE_ERR_COMPILE_FAILED`, successful handles are
  non-null and valid.

- **Documentation** — `docs/api-c.md`, `docs/api-python.md`, `docs/graph-ir.md`,
  `docs/hardware-introspection.md`.

### Fixed

- **CI: ASan on macOS ARM64** — `detect_leaks=1` is not supported on macOS ARM64
  (LSan unavailable); changed to `detect_leaks=0` to prevent all 409 tests aborting.
- **CI: clang-tidy** — Homebrew LLVM's `clang-tidy` does not inherit the macOS SDK
  path; added `xcrun --show-sdk-path` passed via `--extra-arg=--sysroot` so standard
  headers resolve.
- **Python bindings** — removed duplicate `PyMilProgram` class and helper functions
  that were a merge artifact; `ane_module.cpp` previously failed to compile with
  "redefinition of 'PyMilProgram'".
- **Graph integration tests** — `LIBANE_OP_SINH`, `LIBANE_OP_COSH`, `LIBANE_OP_TAN`,
  `LIBANE_OP_ASIN`, `LIBANE_OP_ACOS` tests now `SKIP` when the ANE compiler rejects
  the op on the current firmware (instead of `REQUIRE`-failing).

---

## v0.7.1 — 2026-04-16

### Added

- **P0 math ops** — `sub`, `real_div`, `sqrt`, `log`, `rsqrt` now lowered through
  the ANE via MIL emission. All five ops are exposed in the C API, Graph API, and
  Python bindings.
- **ANE-safe `log` and `rsqrt`** — inputs are clamped to a hardware-derived epsilon
  before the ANE op. Epsilon is `fp16(0x1.0cp-17)` (~7.63e-6), the smallest fp16
  value that produces a finite ANE output for these ops on tested silicon (H-series).
  Values below this threshold produce ±inf on hardware regardless of IEEE semantics.
- **Compile cache improvements** — internal refactor of cache key and eviction
  internals; adds 95-line Catch2 coverage for cache hit/miss/eviction paths.
- **Python runtime behavior tests** — `tests/test_runtime_behavior.py` covering
  graph execution, op dispatch, and dtype edge cases.

### Fixed

- **Python `matmul_f32`** now normalizes inputs to contiguous float32 arrays,
  fixing incorrect results for valid non-contiguous NumPy views (e.g. transposed
  matrices).
- **Compile cache stats** now use atomic counters, removing a hit/miss/eviction
  data race under concurrent access.
- **Buffer-pool lifetime safety** improved in runtime execution paths via RAII
  guards, reducing manual release-path duplication.
- Add explicit `#include <algorithm>` in `src/core/mil_builder.cpp` and
  `tests/test_fusion.cpp` (was relying on transitive include via other headers).
- Add explicit `#include <mutex>` in `src/core/compile_cache.cpp` (same issue).

---

## v0.7.0 — 2026-03-23

First public release.

### What's included

- **C API** — stable `libane_*` ABI for single-op compile/execute, delta weight
  reload, compile cache management, and ANE availability checks.
- **Graph API** — `libane_graph_*` DAG builder with automatic fusion. Describes
  full forward passes as a graph; compiles to the minimum number of ANE
  dispatches via greedy linear-chain fusion.
- **MIL builder** — internal MIL text program generator covering MATMUL
  (conv1×1), RMSNORM, LAYERNORM, GELU (tanh approximation), SILU, SOFTMAX
  (axis=1, channel dimension), ADD, MUL, TRANSPOSE.
- **Buffer manager** — IOSurface-backed fp16 buffer pool with tensor-aware
  stride-safe packing and uniform allocation size enforcement for multi-input
  ANE dispatches.
- **Compile cache** — LRU weight-hash-keyed cache to avoid recompiling programs
  with unchanged weights across calls.
- **CPU fallback** — Accelerate BLAS fallback for non-Apple targets and for ops
  rejected by the ANE compiler.
- **Python bindings** — pybind11 module (`ane`) exposing the full C API and
  Graph API with numpy integration.
- **Swift bindings** — SPM package wrapping the static library.
- **14 Python examples** + 1 C example covering hello-world through full
  transformer layer with fused ops.
- **10-file test suite** using Catch2 v3.7.1.

### Known limitations

See the [Known Limitations](README.md#known-limitations) section in the README.
