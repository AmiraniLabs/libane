# Changelog

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
