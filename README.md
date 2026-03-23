# libane

**Apple Neural Engine native C++ inference runtime.**

libane turns the ANE from a black box into a first-class compute target. It
exposes a Graph IR that lets you describe a full forward pass, compiles it into
the minimum number of ANE dispatches via automatic fusion, and executes it in a
single call. Matmul is implemented as conv1×1 for the 3× throughput advantage
documented in the [Orion paper](https://arxiv.org/abs/2603.06728).

> **Private API.** libane uses `AppleNeuralEngine.framework` via `dlopen`.
> This is intentional and documented. Do not submit to the App Store.

---

## Requirements

| | |
|---|---|
| **Hardware** | Apple Silicon (M1 or later) |
| **OS** | macOS 14 Sonoma or later |
| **Toolchain** | Xcode 15+, CMake 3.24+, C++17 |
| **Python** | 3.10+ with pybind11 and numpy (optional) |

---

## Building

```sh
git clone https://github.com/amirani-labs/libane
cd libane
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(sysctl -n hw.logicalcpu)
ctest --test-dir build --output-on-failure
```

To build the Python module:

```sh
pip install pybind11 numpy
cmake -B build -DLIBANE_BUILD_PYTHON=ON
cmake --build build --target ane -j$(sysctl -n hw.logicalcpu)
# The .so lands in build/bindings/python/
```

Or install directly with pip (builds from source):

```sh
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

## Python quick-start

```python
import ane
import numpy as np

print(ane.available())   # True on Apple Silicon
print(ane.version())     # "0.7.0"

# Single-op
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
│   │   └── graph_executor    Per-group ANE dispatch
│   ├── runtime/
│   │   └── ane_runtime.mm    AppleNeuralEngine.framework wrapper
│   └── fallback/             Accelerate BLAS CPU fallback
└── bindings/python/          pybind11 module
```

Fusion rules compact linear chains into single ANE programs. A 6-op FFN
(RMSNorm → Matmul → SiLU, Matmul → Mul → Matmul) typically becomes 3–4 ANE
dispatches instead of 6, eliminating intermediate DRAM round-trips.

---

## Supported ops

| Op | Notes |
|---|---|
| `MATMUL` | conv1×1 internally; 3× faster than MIL matmul on ANE |
| `RMSNORM` | rsqrt + mul, scale broadcast |
| `LAYERNORM` | normalise + gamma/beta affine |
| `GELU` | tanh approximation only |
| `SILU` | x × sigmoid(x) |
| `SOFTMAX` | over S (spatial) dimension |
| `ADD` / `MUL` | elementwise binary; shapes must match |
| `TRANSPOSE` | [0,3,2,1] only: [1,C,1,S] → [1,S,1,C] |

---

## License

MIT. See [LICENSE](LICENSE).

libane is not affiliated with or endorsed by Apple Inc.
