# Changelog

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
