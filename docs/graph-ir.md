# Graph IR

The Graph API describes a full neural network forward pass as a directed acyclic graph (DAG). libane validates the graph, fuses compatible chains of ops into single ANE dispatches, and executes via the minimum number of ANE programs.

---

## Tensor layout

Most ANE tensors use the standard `[1, C, 1, S]` activation layout (NCHW, batch=1, height=1):

- **C** — channels (feature dimension)
- **S** — sequence / spatial dimension

**S must always be a multiple of 32** (ANE hardware constraint — see below). S=0 is invalid. C must be ≥ 1.

### The S%32 alignment constraint

The ANE reads and writes tensors via IOSurface DMA, which requires the row stride (bytesPerRow) to be **64-byte aligned**. For fp16 tensors (2 bytes/element) this means `S × 2` must be divisible by 64, i.e. `S % 32 == 0`. This is the theoretical basis for the constraint; the exact minimum has not been exhaustively probed at sub-64 granularity — all empirical shape-sweep data used spatial values ≥ 64. libane enforces 32 as a conservative bound consistent with the DMA requirement.

**Practical impact is minimal for most shapes.** Common values for transformer head dimensions (64, 96, 128) and sequence lengths (32, 64, 128, 256, 512 …) are all multiples of 32. If your logical S is not aligned, pad your input tensors to `ceil(S / 32) * 32` before calling libane and trim the output slice.

**Practical implication for LLM inference.** The S≥32 floor means token-by-token autoregressive decode is not viable on the ANE directly — you would pad every single-token step to S=32, running 31 wasted positions per decode step. The recommended split (which Apple uses internally for on-device LLMs) is **ANE for prefill** (long prompt, S naturally large) and **GPU/CPU for decode** (seq_len=1). libane is optimized for the prefill path.

### Matrix tensor format

`LIBANE_OP_DYNAMIC_MATMUL` and `LIBANE_OP_SDPA` use a **matrix tensor format** where `height > 1`:

| Op | Shape | Meaning |
|---|---|---|
| DYNAMIC_MATMUL X | `[1, 1, K, M]` | height=K inner dim, seq=M cols |
| DYNAMIC_MATMUL W | `[1, 1, N, K]` | height=N rows, seq=K inner dim |
| DYNAMIC_MATMUL Y | `[1, 1, N, M]` | height=N rows, seq=M cols |
| SDPA Q/K/V | `[1, H, S, D]` | H heads, S tokens, D head dim |
| SDPA mask | `[1, 1, S, S]` | attention mask (optional) |
| SDPA output | `[1, H, S, D]` | same shape as Q/K/V |

Matrix tensors require `channels=1` (DYNAMIC_MATMUL) or any H (SDPA), `seq%32 == 0`, and all dimension values must also be multiples of 32. The ANE compiler rejects any `reshape` or other op placed before `matmul` or `scaled_dot_product_attention` — inputs must be direct function parameters in the correct 4D shape.

### Static matmul layout convention

For `LIBANE_OP_MATMUL` (conv1×1 path, static weights):

- A: shape `[1, K, 1, M]` — K channels, M sequence positions
- B (weights): `[1, N, 1, K]` — N channels, K sequence positions
- Output: shape `[1, N, 1, M]`

~3× higher throughput than `DYNAMIC_MATMUL` on current silicon because the weight matrix is compiled into the program.

### Shape limits

Per-dimension limits are chip-adaptive. Check `libane_get_shape_limits()` at runtime rather than hardcoding values.

```c
libane_shape_limits_t lim = libane_get_shape_limits();
// lim.max_seq       — max S (must also be a multiple of lim.seq_alignment)
// lim.max_channels  — max C
// lim.seq_alignment — always 32
```

**SRAM budget warning.** `max_seq` and `max_channels` are independent per-dimension caps, but both cannot be reached simultaneously. Activations at `[1, C, 1, S]` consume `C × S × 2` bytes of on-chip SRAM per live buffer. The ANE holds at least input + output simultaneously. On M3 (h15g) SRAM is approximately 32 MB. A shape at `max_seq × max_channels` would require ~4 GB — far beyond any current chip. Use these limits as per-dimension guards only; validate total tensor footprint against known SRAM before submission. `libane_graph_compile()` returns `LIBANE_ERR_COMPILE_FAILED` if firmware rejects the combined size.

---

## Supported ops

| Op | Notes |
|---|---|
| `LIBANE_OP_MATMUL` | conv1×1 internally; ~3× faster than MIL matmul on ANE |
| `LIBANE_OP_RMSNORM` | rsqrt + mul; scale weights required |
| `LIBANE_OP_LAYERNORM` / `LIBANE_OP_LAYER_NORM` | normalize + gamma/beta affine |
| `LIBANE_OP_GELU` | tanh approximation |
| `LIBANE_OP_SILU` | x × sigmoid(x) |
| `LIBANE_OP_RELU` | max(x, 0) |
| `LIBANE_OP_TANH` | tanh(x) |
| `LIBANE_OP_SIGMOID` | 1 / (1 + exp(−x)) |
| `LIBANE_OP_HARDSWISH` | x × clamp(x+3, 0, 6) / 6 |
| `LIBANE_OP_LEAKY_RELU` | max(α·x, x), α=0.01 |
| `LIBANE_OP_ELU` | x≥0: x, x<0: α(exp(x)−1), α=1.0 |
| `LIBANE_OP_PWL_ACTIVATION` | piecewise-linear custom activation; use `libane_graph_add_pwl_activation()` |
| `LIBANE_OP_SOFTMAX` | over C dimension (axis=1) |
| `LIBANE_OP_ADD` | elementwise; shapes must match |
| `LIBANE_OP_SUB` | elementwise; shapes must match |
| `LIBANE_OP_MUL` | elementwise; shapes must match |
| `LIBANE_OP_REAL_DIV` | elementwise; shapes must match |
| `LIBANE_OP_NEG` | elementwise negate |
| `LIBANE_OP_MOD` | elementwise modulo |
| `LIBANE_OP_SQRT` | elementwise sqrt |
| `LIBANE_OP_LOG` | elementwise log; input clamped to ≥ε before op |
| `LIBANE_OP_RSQRT` | elementwise rsqrt; input clamped to ≥ε before op |
| `LIBANE_OP_SINH` | elementwise sinh |
| `LIBANE_OP_COSH` | elementwise cosh |
| `LIBANE_OP_TAN` | elementwise tan |
| `LIBANE_OP_ASIN` | elementwise asin |
| `LIBANE_OP_ACOS` | elementwise acos |
| `LIBANE_OP_TRANSPOSE` | `[0,3,2,1]` only: `[1,C,1,S]` → `[1,S,1,C]` |
| `LIBANE_OP_RESHAPE` | reshape to ANE-compatible shape |
| `LIBANE_OP_CONCAT` | concatenate along C dimension |
| `LIBANE_OP_SLICE_BY_INDEX` | slice by index |
| `LIBANE_OP_REDUCE_SUM` | reduce sum |
| `LIBANE_OP_REDUCE_MEAN` | reduce mean |
| `LIBANE_OP_REDUCE_MAX` | reduce max |
| `LIBANE_OP_REDUCE_PROD` | reduce product |
| `LIBANE_OP_AVG_POOL` | average pooling |
| `LIBANE_OP_MAX_POOL` | max pooling |
| `LIBANE_OP_LOGICAL_AND` | elementwise logical AND |
| `LIBANE_OP_LOGICAL_OR` | elementwise logical OR |
| `LIBANE_OP_LOGICAL_XOR` | elementwise logical XOR |
| `LIBANE_OP_SCATTER` | scatter |
| `LIBANE_OP_GATHER` | gather |
| `LIBANE_OP_SCATTER_ND` | scatter ND |
| `LIBANE_OP_SCATTER_ALONG_AXIS` | scatter along axis |
| `LIBANE_OP_DYNAMIC_MATMUL` | runtime matmul Y=X@W^T; both inputs are IOSurfaces; matrix format `[1,1,K,M]` |
| `LIBANE_OP_SDPA` | scaled dot-product attention; matrix format `[1,H,S,D]` |

### ANE-safe log and rsqrt

`LIBANE_OP_LOG` and `LIBANE_OP_RSQRT` clamp inputs to a hardware-derived epsilon (`fp16(0x1.0cp-17)` ≈ 7.63e-6) before the ANE op. Values below this threshold produce ±inf on hardware regardless of IEEE semantics — this is an ANE firmware behaviour, not a software choice.

---

## Validation rules

`libane_graph_compile()` runs 7 validation checks before fusion and compilation:

1. **No cycles** — the graph must be a DAG
2. **All inputs declared** — every op's input tensor ID must be reachable
3. **Output marked** — at least one output must be marked
4. **Shape consistency** — output shapes must be compatible with op semantics
5. **S divisibility** — S must be a multiple of 32 for all tensors (IOSurface 64-byte DMA alignment)
6. **Channel cap** — C must be ≤ 16384
7. **Weight presence** — ops that require weights (MATMUL, RMSNORM, etc.) must have non-null weight data

---

## Fusion rules

libane uses a greedy linear-chain fusion pass. Two adjacent ops in a single-input chain are fusable when:

- Both are supported by the MIL compiler on the current firmware
- Their combined tensor footprint fits in SRAM
- The intermediate tensor is not also a graph output (an output-marked tensor cannot be fused away)

Typical fusion outcomes for common patterns:

| Pattern | ANE dispatches |
|---|---|
| RMSNORM → MATMUL | 1 |
| MATMUL → GELU | 1 |
| RMSNORM → MATMUL → GELU → MATMUL | 2 |
| RMSNORM → MATMUL → SILU, MATMUL → MUL → MATMUL (SwiGLU FFN) | 3–4 |

Fusion eliminates intermediate DRAM round-trips. Each fused group is compiled into one MIL program and dispatched in a single ANE call.

---

## Multi-output graphs

Mark multiple tensor IDs as outputs before compiling. Outputs are returned in mark order.

```c
libane_graph_mark_output(g, tensor_a, "a");
libane_graph_mark_output(g, tensor_b, "b");

libane_compiled_graph_t cg = libane_graph_compile(g);

void*  out_ptrs[]  = { buf_a, buf_b };
size_t out_bytes[] = { bytes_a, bytes_b };
libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 2);
```

---

## Piecewise-linear custom activations

`libane_graph_add_pwl_activation()` approximates any smooth activation over a bounded domain using equal-width linear segments. 32 segments (33 sample points) is recommended for smooth activations. Example approximating SELU:

```c
float samples[33];
for (int i = 0; i < 33; i++) {
    float x = -3.0f + i * (6.0f / 32);
    samples[i] = x >= 0 ? 1.0507f * x : 1.0507f * 1.67326f * (expf(x) - 1.0f);
}

uint32_t selu = libane_graph_add_pwl_activation(g, input_id, shape,
                                                 -3.0f, 3.0f, samples, 33);
```
