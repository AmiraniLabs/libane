# libane

**Direct access to the Apple Neural Engine from C++ and Python.**

[![PyPI](https://img.shields.io/pypi/v/libane?label=PyPI)](https://pypi.org/project/libane/)
[![CI](https://github.com/AmiraniLabs/libane/actions/workflows/ci.yml/badge.svg)](https://github.com/AmiraniLabs/libane/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://github.com/AmiraniLabs/libane)

libane is a low-level ANE runtime and compiler interface. It exposes a Graph IR
for describing full forward passes, compiles them into the minimum number of ANE
dispatches via automatic op fusion, and executes them through a stable C ABI.
Matmul is implemented as conv1×1 for the [3× throughput advantage over MIL
matmul](https://arxiv.org/abs/2603.06728) on ANE.

> **Private framework dependency**. libane loads `AppleNeuralEngine.framework` via `dlopen`.
> It is intended for research use and low-level ANE experimentation, not production deployment.
> See [NOTICE](NOTICE) for the interoperability basis (17 U.S.C. §1201(f), *Sega v. Accolade*,
> *Sony v. Connectix*) and a statement of what libane does and does not do with Apple software.

---

## Install

```sh
pip install libane
```

Requires Apple Silicon (M1 or later) and macOS 14+. The wheel is a compiled
extension — no extra build steps needed.

---

## Requirements

| | |
|---|---|
| **Hardware** | Apple Silicon (M1 or later) |
| **OS** | macOS 14 Sonoma or later |
| **Toolchain** | Xcode 15+, CMake 3.24+, C++17 |
| **Python** | 3.10+ (optional) |

---

## Python quick-start

```python
import ane
import numpy as np

print(ane.available())   # True on Apple Silicon
print(ane.version())     # "0.8.2"

# Single-op matmul
A = np.random.randn(128, 512).astype(np.float16)
B = np.random.randn(512, 256).astype(np.float16)
C = ane.matmul(A, B)    # shape (128, 256), fp16

# Graph API — fused FFN block
D, FFN, SEQ = 512, 2048, 128
W_up   = np.random.randn(D,   FFN).astype(np.float16)
W_down = np.random.randn(FFN, D  ).astype(np.float16)
scale  = np.ones(D, dtype=np.float16)

g = ane.Graph()
x   = g.add_input("x",  [1, D,   1, SEQ])
rn  = g.add_op(ane.RMSNORM, [x],   [1, D,   1, SEQ], weights=scale)
up  = g.add_op(ane.MATMUL,  [rn],  [1, FFN, 1, SEQ], weights=W_up)
act = g.add_op(ane.GELU,    [up],  [1, FFN, 1, SEQ])
out = g.add_op(ane.MATMUL,  [act], [1, D,   1, SEQ], weights=W_down)
g.mark_output(out)

cg = g.compile()
cg.set_output_shapes([[1, D, 1, SEQ]])

x_data = np.random.randn(D, SEQ).astype(np.float16)
result = cg(x_data)
print(result.shape)   # (1, 512, 1, 128)
```

See [`examples/ffn_inference.py`](examples/ffn_inference.py) for a timed
end-to-end example.

---

## Documentation

| | |
|---|---|
| [C API reference](docs/api-c.md) | All `libane_*` functions, types, and status codes |
| [Python API reference](docs/api-python.md) | `ane.*`, `Graph`, `CompiledGraph`, `CompiledMil` |
| [Graph IR](docs/graph-ir.md) | Tensor layout, op table, fusion rules, shape limits |
| [Hardware introspection](docs/hardware-introspection.md) | Device info, shape limits, performance stats |

---

## Building from source

```sh
git clone https://github.com/AmiraniLabs/libane
cd libane
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(sysctl -n hw.logicalcpu)
ctest --test-dir build --output-on-failure
```

To build the Python module from source:

```sh
pip install pybind11 numpy scikit-build-core
pip install ./bindings/python
```

---

## C quick-start

### Single op

```c
#include "libane.h"

// Compile a matmul [1, 512, 1, 128] → [1, 256, 1, 128]
libane_shape_t shape = {.dims = {1, 256, 1, 128}, .ndim = 4};
libane_handle_t h = libane_compile(LIBANE_OP_MATMUL, shape,
                                    fp16_weights, sizeof(fp16_weights));
libane_execute(h, input, output, shape);
libane_release(h);
```

### Graph API (fused multi-op)

```c
#include "libane.h"

// Build: RMSNorm → Matmul → GELU
libane_graph_t g = libane_graph_create();

libane_shape_t in_shape  = {.dims={1,512,1,128}, .ndim=4};
libane_shape_t out_shape = {.dims={1,256,1,128}, .ndim=4};

uint32_t x  = libane_graph_add_input(g, "x", in_shape);
uint32_t rn = libane_graph_add_op(g, LIBANE_OP_RMSNORM, &x, 1,
                                   in_shape, rn_scale, rn_scale_len);
uint32_t mm = libane_graph_add_op(g, LIBANE_OP_MATMUL,  &rn, 1,
                                   out_shape, weights, weights_len);
uint32_t act= libane_graph_add_op(g, LIBANE_OP_GELU,    &mm, 1,
                                   out_shape, NULL, 0);
libane_graph_mark_output(g, act, "out");

libane_compiled_graph_t cg = libane_graph_compile(g);

// Execute
const void* in_ptrs[]   = { input_fp16 };
size_t      in_bytes[]  = { 512 * 128 * 2 };
void*       out_ptrs[]  = { output_fp16 };
size_t      out_bytes[] = { 256 * 128 * 2 };
libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);

libane_compiled_graph_release(cg);
libane_graph_release(g);
```

---

## Architecture

```
libane
├── include/libane.h          Stable C ABI
├── src/
│   ├── core/
│   │   ├── mil_builder       MIL text program generator
│   │   ├── compile_cache     LRU cache for compiled programs
│   │   └── buffer_manager    IOSurface-backed fp16 buffer pool
│   ├── graph/
│   │   ├── ane_graph         Graph IR (DAG builder)
│   │   ├── graph_validator   7-check validation pass
│   │   ├── fusion_rules      Greedy linear-chain fusion
│   │   ├── graph_compiler    build_plan() + compile()
│   │   ├── graph_executor    Per-group ANE dispatch
│   │   ├── mil_backend       MIL-path graph lowering
│   │   ├── hwx_backend       HWX-path graph lowering
│   │   ├── hwx_emitter       HWX bytecode emission
│   │   ├── espresso_backend  Espresso-path graph lowering
│   │   └── espresso_builder  Espresso program construction
│   ├── runtime/
│   │   └── ane_runtime.mm    AppleNeuralEngine.framework wrapper
│   └── fallback/             Accelerate BLAS CPU fallback
└── bindings/python/          pybind11 module (pip install libane)
```

Fusion rules compact linear chains into single ANE programs. A 6-op FFN
(RMSNorm → Matmul → SiLU, Matmul → Mul → Matmul) typically becomes 3–4 ANE
dispatches instead of 6, eliminating intermediate DRAM round-trips.

---

## Supported ops

| Category | Ops |
|---|---|
| Linear | `MATMUL` (conv1×1; ~3× faster than MIL matmul) |
| Normalization | `RMSNORM`, `LAYERNORM` |
| Activations | `GELU`, `SILU`, `RELU`, `TANH`, `SIGMOID`, `HARDSWISH`, `LEAKY_RELU`, `ELU`, `PWL_ACTIVATION` |
| Elementwise | `ADD`, `SUB`, `MUL`, `REAL_DIV`, `NEG`, `MOD` |
| Math | `SQRT`, `LOG`, `RSQRT`, `SINH`, `COSH`, `TAN`, `ASIN`, `ACOS` |
| Reduce | `SOFTMAX`, `REDUCE_SUM`, `REDUCE_MEAN`, `REDUCE_MAX`, `REDUCE_PROD` |
| Pooling | `AVG_POOL`, `MAX_POOL` |
| Structural | `TRANSPOSE`, `RESHAPE`, `CONCAT`, `SLICE_BY_INDEX` |
| Logical | `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_XOR` |
| Scatter/Gather | `SCATTER`, `GATHER`, `SCATTER_ND`, `SCATTER_ALONG_AXIS` |

Full op documentation with constraints and notes: [docs/graph-ir.md](docs/graph-ir.md).

---

## Known Limitations

- **Experimental / research-use only.** Not production-supported.
- **Private Apple framework dependency.** Uses `AppleNeuralEngine.framework` via `dlopen`. Uses private API — intentional.
- **Constrained tensor layout.** Graph API requires `[1, C, 1, S]` (NCHW with batch=1, height=1). Arbitrary shapes are not supported.
- **Channel cap.** Graph API validation enforces `C ≤ 16384`. Larger channel counts (e.g. vocabulary projections) require raw MIL emission and are not exposed through the graph API.
- **fp16 only.** No quantization (int8, int4) support. Weights and activations are fp16 throughout.
- **Some ops are compiler-sensitive.** ANE's MIL compiler accepts a strict subset of MIL. Certain op combinations or shapes may require fallback paths. See the fallback module.
- **Not a general-purpose model runner.** libane is a programmable kernel/runtime/compiler interface for ANE-native experimentation, not a drop-in inference engine.
- **macOS 14+ required.** Older systems are not tested and not supported.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

libane is not affiliated with or endorsed by Apple Inc.
