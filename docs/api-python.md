# Python API Reference

The `ane` module is a pybind11 extension exposing the full libane C API with numpy integration. Install via `pip install libane`.

```python
import ane
import numpy as np
```

All array inputs are converted to contiguous fp16 (`numpy.float16`) internally. You can pass any numeric dtype — conversion is automatic.

---

## Module-level functions

### `ane.available() → bool`

Returns `True` if the ANE is accessible on the current machine.

### `ane.version() → str`

Returns the libane version string, e.g. `"0.8.0"`.

### `ane.last_error() → str`

Returns the most recent error message. Useful after a `RuntimeError` to get more detail.

### `ane.set_log_level(level: int)`

Sets the minimum log level. Use the `ane.LOG_*` constants:

```python
ane.LOG_SILENT = 0
ane.LOG_ERROR  = 1   # default
ane.LOG_WARN   = 2
ane.LOG_INFO   = 3
ane.LOG_DEBUG  = 4
```

### `ane.set_backend(backend: str | None)`

Forces a backend: `"ane"`, `"cpu"`, or `None` (auto-detect, default).

### `ane.cache_flush()`

Evicts all entries from the compile cache.

### `ane.cache_size_bytes() → int`

Returns current compile cache usage in bytes.

---

## Single-op convenience functions

These compile-on-first-call wrappers handle memory allocation and dtype conversion.

### `ane.matmul(A, B) → np.ndarray`

fp16 matrix multiply: `C = A @ B`. Falls back to BLAS if ANE is unavailable.

- `A`: any numeric array, shape `(M, K)` — converted to fp16
- `B`: any numeric array, shape `(K, N)` — converted to fp16
- Returns fp16 array, shape `(M, N)`

### `ane.matmul_f32(A, B) → np.ndarray`

Same as `matmul` but converts inputs through fp16 and returns fp32. Useful when the caller needs fp32 output. Inputs are normalised to contiguous fp32 before conversion.

### `ane.softmax(x) → np.ndarray`

ANE softmax over the last dimension. Falls back to numpy if ANE is unavailable.

### `ane.gelu(x) → np.ndarray`

ANE GELU (tanh approximation). Falls back to numpy if ANE is unavailable.

---

## Graph API

### `ane.Graph`

Mutable graph builder.

```python
g = ane.Graph()

x  = g.add_input("x", [1, 512, 1, 128])         # returns tensor ID (int)
rn = g.add_op(ane.RMSNORM, [x], [1, 512, 1, 128], weights=scale)
mm = g.add_op(ane.MATMUL,  [rn], [1, 256, 1, 128], weights=W)
g.mark_output(mm)

cg = g.compile()   # returns CompiledGraph
```

#### `g.add_input(name: str, shape: list[int]) → int`

Declares a graph input. `shape` is `[1, C, 1, S]`. Returns a tensor ID. Raises `ValueError` on invalid shape.

#### `g.add_op(op: int, inputs: list[int], output_shape: list[int], weights=None) → int`

Adds an operation. `op` is one of the `ane.*` op constants. `inputs` is a list of tensor IDs from prior `add_input` / `add_op` calls. `weights` is an optional numpy array (any dtype; converted to fp16). Returns the output tensor ID. Raises `ValueError` on failure.

#### `g.mark_output(tensor_id: int, name: str = "")`

Marks a tensor as a graph output. Outputs are returned by `CompiledGraph.__call__` in the order they are marked.

#### `g.compile() → CompiledGraph`

Validates, fuses, and compiles the graph. The graph object is not consumed. Raises `RuntimeError` on validation or compile failure.

---

### `ane.CompiledGraph`

Compiled ANE graph. Call it directly with numpy arrays to run inference.

```python
# Single input, single output
out = cg(x)

# Multiple inputs
out = cg([a, b])

# Multiple outputs — set shapes first
cg.set_output_shapes([[1, 512, 1, 128], [1, 256, 1, 128]])
a, b = cg(x)
```

#### `cg(inputs) → np.ndarray | list[np.ndarray]`

Runs a forward pass. `inputs` is a single array or a list of arrays, one per graph input in `add_input` order. Returns a single fp16 array for single-output graphs, or a list of fp16 arrays for multi-output graphs.

Raises `RuntimeError` on execution failure.

#### `cg.set_output_shapes(shapes: list[list[int]])`

Sets expected output shapes for multi-output graphs, or when the output shape differs from the input shape. `shapes` is a list of shape lists, e.g. `[[1, 256, 1, 128], [1, 128, 1, 128]]`. Must be called before the first `__call__` for multi-output graphs.

---

## Raw MIL API

Direct MIL program submission — bypasses the graph compiler. For research and low-level experimentation.

### `ane.compile_mil(mil_text: str) → CompiledMil`

Compiles a raw MIL program with no external weights. `mil_text` is the complete MIL source including the `buildInfo` header.

Raises `RuntimeError` if the ANE is unavailable or compilation fails.

### `ane.compile_mil_with_weights(mil_text: str, weights: dict[str, np.ndarray]) → CompiledMil`

Compiles a raw MIL program with external weight files. `weights` maps filenames to fp16 arrays. Filenames must match the `file()` references in the MIL source.

Raises `RuntimeError` on failure.

### `ane.CompiledMil`

Compiled raw MIL program.

#### `prog.run(inputs: list[np.ndarray], output_sizes: list[int]) → list[np.ndarray]`

Executes the compiled MIL program.

- `inputs`: list of numpy arrays (any dtype; converted to fp16). Must be in alphabetical order of MIL parameter names (ANE constraint #13).
- `output_sizes`: list of output sizes in fp16 **elements** (not bytes).
- Returns a list of fp16 arrays, one per output.

Raises `RuntimeError` on execution failure.

---

## Op constants

These are `int` values matching the `libane_op_t` C enum.

| Constant | Op |
|---|---|
| `ane.MATMUL` | Matrix multiply (conv1×1 internally) |
| `ane.RMSNORM` | RMS normalization |
| `ane.LAYERNORM` / `ane.LAYER_NORM` | Layer normalization |
| `ane.GELU` | GELU (tanh approximation) |
| `ane.SILU` | SiLU: x × sigmoid(x) |
| `ane.SOFTMAX` | Softmax over C dimension |
| `ane.RELU` | ReLU: max(x, 0) |
| `ane.TANH` | Tanh |
| `ane.SIGMOID` | Sigmoid |
| `ane.HARDSWISH` | HardSwish: x × clamp(x+3, 0, 6) / 6 |
| `ane.LEAKY_RELU` | Leaky ReLU (alpha=0.01) |
| `ane.ELU` | ELU (alpha=1.0) |
| `ane.ADD` | Elementwise add |
| `ane.SUB` | Elementwise subtract |
| `ane.MUL` | Elementwise multiply |
| `ane.REAL_DIV` | Elementwise divide |
| `ane.NEG` | Elementwise negate |
| `ane.MOD` | Elementwise modulo |
| `ane.SQRT` | Elementwise sqrt |
| `ane.LOG` | Elementwise log (clamped input) |
| `ane.RSQRT` | Elementwise rsqrt (clamped input) |
| `ane.SINH` | Elementwise sinh |
| `ane.COSH` | Elementwise cosh |
| `ane.TAN` | Elementwise tan |
| `ane.ASIN` | Elementwise asin |
| `ane.ACOS` | Elementwise acos |
| `ane.TRANSPOSE` | Permute [1,C,1,S] → [1,S,1,C] |
| `ane.RESHAPE` | Reshape (ANE-compatible shapes) |
| `ane.CONCAT` | Concatenate along C dimension |
| `ane.SLICE_BY_INDEX` | Slice by index |
| `ane.REDUCE_SUM` | Reduce sum |
| `ane.REDUCE_MEAN` | Reduce mean |
| `ane.REDUCE_MAX` | Reduce max |
| `ane.REDUCE_PROD` | Reduce product |
| `ane.AVG_POOL` | Average pooling |
| `ane.MAX_POOL` | Max pooling |
| `ane.LOGICAL_AND` | Logical AND |
| `ane.LOGICAL_OR` | Logical OR |
| `ane.LOGICAL_XOR` | Logical XOR |
| `ane.SCATTER` | Scatter |
| `ane.GATHER` | Gather |
| `ane.SCATTER_ND` | Scatter ND |
| `ane.SCATTER_ALONG_AXIS` | Scatter along axis |

---

## Example: fused FFN block

```python
import ane
import numpy as np

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
print(result.shape)  # (1, 512, 1, 128)
```
