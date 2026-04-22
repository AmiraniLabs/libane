"""
ane — Apple Neural Engine Python bindings (libane v0.9.0)

Run ML graphs directly on the Apple Neural Engine from Python.
Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
"""

from __future__ import annotations
import numpy as np
from typing import Union, Optional

# ── Op constants ──────────────────────────────────────────────────────────────

MATMUL: int
LAYERNORM: int
LAYER_NORM: int
GELU: int
SOFTMAX: int
ADD: int
MUL: int
SUB: int
REAL_DIV: int
SQRT: int
LOG: int
RSQRT: int
TRANSPOSE: int
RESHAPE: int
CONCAT: int
SLICE_BY_INDEX: int
REDUCE_SUM: int
REDUCE_MEAN: int
REDUCE_MAX: int
SILU: int
RMSNORM: int
AVG_POOL: int
MAX_POOL: int
LOGICAL_AND: int
LOGICAL_OR: int
LOGICAL_XOR: int
REDUCE_PROD: int
SCATTER: int
GATHER: int
SCATTER_ND: int
SCATTER_ALONG_AXIS: int
NEG: int
MOD: int
SINH: int
COSH: int
TAN: int
ASIN: int
ACOS: int
SELECT: int
RELU: int
TANH: int
SIGMOID: int
HARDSWISH: int
LEAKY_RELU: int
ELU: int
PIXEL_SHUFFLE: int
CAST: int
CONV2D: int
PWL_ACTIVATION: int
SLICE: int
CLIP: int
PAD: int
DYNAMIC_MATMUL: int
SDPA: int
SDPA_GQA: int
MATMUL_W8A8: int
EXP: int
SIN: int
COS: int
ABS: int
POW: int
CEIL: int
FLOOR: int
ROUND: int
SIGN: int

# ── Log level constants ───────────────────────────────────────────────────────

LOG_SILENT: int
LOG_ERROR: int
LOG_WARN: int
LOG_INFO: int
LOG_DEBUG: int

# ── Utility ───────────────────────────────────────────────────────────────────

def available() -> bool:
    """True if the Apple Neural Engine is accessible on this machine."""
    ...

def version() -> str:
    """libane version string (e.g. '0.8.2')."""
    ...

def last_error() -> str:
    """Last error message from the library."""
    ...

def device_info() -> dict:
    """
    ANE hardware capabilities.

    Returns:
        dict with keys:
            architecture (str): chip generation string, e.g. 'h15g' (M3), 'h16g' (M4).
            core_count   (int): number of ANE inference cores.
            num_anes     (int): number of ANE units.
            available   (bool): False if _ANEDeviceInfo could not be queried.
    """
    ...

def shape_limits() -> dict:
    """
    ANE tensor shape limits for the current chip.

    Returns:
        dict with keys:
            max_seq       (int): maximum S dimension.
            max_channels  (int): maximum C dimension.
            seq_alignment (int): S must be a multiple of this value (always 32).

    Note:
        max_seq and max_channels cannot be reached simultaneously — the real
        constraint is on-chip SRAM (~32 MB on M3). Use these as per-dimension
        guards only.
    """
    ...

def set_backend(backend: Optional[str]) -> None:
    """
    Force a specific backend.

    Args:
        backend: ``'ane'`` to force ANE, ``'cpu'`` to force CPU fallback,
                 or ``None`` for auto-detect.
    """
    ...

def set_log_level(level: int) -> None:
    """
    Set log verbosity.

    Args:
        level: LOG_SILENT=0, LOG_ERROR=1, LOG_WARN=2, LOG_INFO=3, LOG_DEBUG=4
    """
    ...

def cache_flush() -> None:
    """Flush the compiled program cache."""
    ...

def cache_size_bytes() -> int:
    """Return the current size of the compiled program cache in bytes."""
    ...

# ── Single-op convenience ─────────────────────────────────────────────────────

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    ANE fp16 matrix multiply: C = A @ B.

    Args:
        A: 2-D array, shape (M, K), any dtype (converted to float16 internally).
        B: 2-D array, shape (K, N), any dtype (converted to float16 internally).

    Returns:
        C: np.ndarray, shape (M, N), dtype float16. Falls back to BLAS on non-ANE.
    """
    ...

def matmul_f32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    ANE matrix multiply returning float32.

    Internally converts fp32 → fp16 → fp32 via ANE.

    Args:
        A: 2-D float32 array, shape (M, K).
        B: 2-D float32 array, shape (K, N).

    Returns:
        C: np.ndarray, shape (M, N), dtype float32.
    """
    ...

def softmax(x: np.ndarray) -> np.ndarray:
    """
    ANE softmax over the channel (C) dimension.

    Falls back to numpy for unsupported shapes.

    Args:
        x: Array of any shape; last dimension is S (must be multiple of 8 and ≤ 65536
           for ANE path). Converted to float16 internally.

    Returns:
        Softmax probabilities, same shape as input, dtype float16.
    """
    ...

def gelu(x: np.ndarray) -> np.ndarray:
    """
    ANE GELU activation (tanh approximation).

    Falls back to numpy for shapes where total elements are not a multiple of 8
    or exceed 65536.

    Args:
        x: Array of any shape. Converted to float16 internally.

    Returns:
        GELU(x), same shape as input, dtype matches input.
    """
    ...

# ── Compiled single-op handle ─────────────────────────────────────────────────

class CompiledOp:
    """
    Compiled single-op ANE program. Returned by ``compile()`` and ``compile_batch()``.

    Wraps a ``libane_handle_t``. Released on garbage collection.
    """

    def execute(self, x: np.ndarray, shape: list[int]) -> np.ndarray:
        """
        Execute with one input.

        Args:
            x:     Input array (converted to float16 internally).
            shape: Runtime shape [1, C, 1, S] — must match compiled shape.

        Returns:
            Flat np.float16 array of C*S elements.
        """
        ...

    def execute2(self, x0: np.ndarray, x1: np.ndarray, shape: list[int]) -> np.ndarray:
        """
        Execute with two inputs (e.g. elementwise ADD, MUL).

        Inputs must be in alphabetical order of their MIL parameter names
        (ANE constraint #13).

        Args:
            x0, x1: Input arrays (converted to float16 internally).
            shape:  Runtime shape [1, C, 1, S].

        Returns:
            Flat np.float16 array.
        """
        ...

    def delta_reload(self) -> None:
        """
        Reload the compiled program into ANE SRAM without recompiling.

        ~8.5× faster than a full compile. Use after an unload to restore
        execution without paying the full compile cost.

        Note: weights are baked into the compiled binary and cannot be
        changed with this call. Use ``compile()`` with new weights instead.
        """
        ...


def compile(
    op: int,
    shape: list[int],
    weights: Optional[np.ndarray] = None,
) -> CompiledOp:
    """
    Compile a single op. Returns a ``CompiledOp`` handle.

    Args:
        op:      Op constant (e.g. ``ane.SOFTMAX``, ``ane.MATMUL``).
        shape:   Output/activation shape [1, C, 1, S].
        weights: Weight array for weight-bearing ops (MATMUL, RMSNORM, etc.).
                 Converted to float16 internally.

    Returns:
        ``CompiledOp`` ready for ``.execute()`` or ``.execute2()``.

    Raises:
        RuntimeError if compilation fails. Check ``ane.last_error()``.

    Note:
        Activation ops (RELU, TANH, SIGMOID, etc.) are only available through
        the Graph API — use ``Graph.add_op()`` for those.
    """
    ...


def compile_batch(
    requests: list[tuple[int, list[int], Optional[np.ndarray]]],
) -> list[Optional[CompiledOp]]:
    """
    Compile multiple ops in one call.

    Args:
        requests: List of (op, shape, weights_or_None) tuples.

    Returns:
        List of ``CompiledOp | None`` — ``None`` for any that failed.
        Check ``ane.last_error()`` on partial failure.
    """
    ...

# ── Raw MIL API ───────────────────────────────────────────────────────────────

class CompiledMil:
    """
    Compiled raw MIL program. Returned by ``compile_mil()`` and
    ``compile_mil_with_weights()``.
    """

    sram_spill: bool
    """
    True if the compiled program requires DRAM-backed intermediate buffers.

    Non-zero ``intermediateBufferHandle`` means activations spilled past
    on-chip SRAM (~32 MB on M3). Expect ~30% throughput penalty.
    Always False for single-op programs (no inter-layer intermediates).
    """

    def run(
        self,
        inputs: list[np.ndarray],
        output_sizes: list[int],
    ) -> list[np.ndarray]:
        """
        Execute the MIL program.

        Args:
            inputs:       Input arrays (converted to float16 internally).
                          Must be in alphabetical order of MIL parameter names.
            output_sizes: Output sizes in fp16 elements (not bytes).

        Returns:
            List of np.float16 arrays, one per output.

        Raises:
            RuntimeError on ANE dispatch failure.
        """
        ...

    def save(self, path: str) -> None:
        """
        Save this compiled MIL program to a file.

        Writes MIL source, weight blobs, and the compiled HWX binary to
        a file (magic "ANEM", version 1).  On reload via :meth:`load`,
        the expensive ``compileWithQoS:`` step is skipped — ~8.5× faster
        cold start.

        Args:
            path: Destination file path (created or overwritten).

        Raises:
            RuntimeError: on I/O error or if the HWX is not available.
        """
        ...

    @staticmethod
    def load(path: str) -> "CompiledMil":
        """
        Load a compiled MIL program saved by :meth:`save`.

        Args:
            path: File path produced by :meth:`save`.

        Returns:
            :class:`CompiledMil` ready for :meth:`run`.

        Raises:
            RuntimeError: on format error, I/O error, or ANE failure.
        """
        ...

    def run_stats(
        self,
        inputs: list[np.ndarray],
        output_sizes: list[int],
    ) -> tuple[list[np.ndarray], dict]:
        """
        Execute and return IOReport hardware counters.

        Args:
            inputs, output_sizes: Same as ``run()``.

        Returns:
            (outputs, stats) where stats is a dict with keys:
                ane_bw_utilization (float): DCS bus utilization fraction (0.0–1.0).
                avg_bw_state       (float): mean bandwidth histogram state (0–31).
                peak_bw_state        (int): peak bandwidth state observed.
                ane_energy_units     (int): raw IOReport energy units.
                throttle_ns          (int): total throttle residency in nanoseconds.
                available           (bool): False if IOReport sampling failed.
        """
        ...


def compile_mil(mil_text: str) -> CompiledMil:
    """
    Compile a raw MIL program (no external weights).

    Args:
        mil_text: UTF-8 MIL source text (complete program including buildInfo header).

    Returns:
        ``CompiledMil`` ready for ``.run()`` or ``.run_stats()``.

    Raises:
        RuntimeError if ANE is unavailable or compilation fails.
    """
    ...


def compile_mil_with_weights(
    mil_text: str,
    weights: dict[str, np.ndarray],
) -> CompiledMil:
    """
    Compile a raw MIL program with external weight files.

    Args:
        mil_text: UTF-8 MIL source text.
        weights:  Mapping from filename to fp16 weight array.
                  Filenames must match ``file()`` references in the MIL text.

    Returns:
        ``CompiledMil`` ready for ``.run()`` or ``.run_stats()``.

    Raises:
        RuntimeError if ANE is unavailable or compilation fails.
    """
    ...


# ── Tensor utilities (pure numpy, no ANE execution) ──────────────────────────

def chunk(x: np.ndarray, n: int, shape: list[int]) -> list[np.ndarray]:
    """
    Split a flat ANE tensor along the channel (C) axis into n equal chunks.

    This is a pure numpy operation — no ANE execution is involved. Use it to
    split the flat fp16 output of ``CompiledGraph.__call__`` into multiple
    tensors, e.g. for QKV projections or SwiGLU gate splits.

    Args:
        x:     Flat fp16 array with C*S elements (output of ``cg(x_in)``).
        n:     Number of equal chunks. C must be divisible by n.
        shape: Full 4-D shape ``[1, C, 1, S]`` of the tensor.

    Returns:
        List of n contiguous fp16 arrays, each of size ``(C//n)*S``.

    Raises:
        ValueError: if C is not divisible by n, or shape is not length-4.

    Example::

        # QKV projection: output shape [1, 3*D, 1, S]
        qkv = cg(x)
        q, k, v = ane.chunk(qkv, 3, [1, 3 * D, 1, S])

        # SwiGLU gate split: [1, 2*H, 1, S] → two [1, H, 1, S]
        gate_proj, up_proj = ane.chunk(proj_out, 2, [1, 2 * H, 1, S])
    """
    ...


def split(
    x: np.ndarray,
    sizes: Union[int, list[int]],
    shape: list[int],
) -> list[np.ndarray]:
    """
    Split a flat ANE tensor along the channel (C) axis by the given sizes.

    Like ``torch.split``: pass a single int for equal-sized chunks, or a list
    for variable-sized chunks.

    Args:
        x:     Flat fp16 array with C*S elements.
        sizes: ``int`` — each chunk gets that many channels (repeating until C
               is exhausted); or ``list[int]`` — explicit channel count per chunk.
        shape: Full 4-D shape ``[1, C, 1, S]``.

    Returns:
        List of contiguous fp16 arrays, one per chunk.

    Raises:
        ValueError: if any size is <= 0, or list sizes sum exceeds C.

    Example::

        # Equal-size split: 512 channels, chunk size 128 → 4 arrays
        parts = ane.split(out, 128, [1, 512, 1, S])

        # Variable split: first 256 channels then 128
        a, b = ane.split(out, [256, 128], [1, 384, 1, S])
    """
    ...


def quantize_i8(
    x: np.ndarray,
    scale: float,
    zero_point: int = 0,
) -> np.ndarray:
    """
    Quantize a float16 array to int8.

    ``out[i] = clamp(round(x[i] / scale) + zero_point, -128, 127)``

    Use this to prepare activation tensors before passing them to a graph
    compiled with :meth:`Graph.add_matmul_w8a8`.

    Args:
        x:           ``np.float16`` array of any shape.
        scale:       Quantization scale (float > 0).
        zero_point:  Quantization zero-point (int, default 0).

    Returns:
        ``np.int8`` array with the same shape as ``x``.

    Example::

        x_fp16 = activations.astype(np.float16)
        x_int8 = ane.quantize_i8(x_fp16, scale=0.02, zero_point=0)
        out = cg(x_int8)
    """
    ...


def from_torch(
    module: "torch.nn.Module",
    example_input: "Union[torch.Tensor, list[torch.Tensor]]",
) -> "CompiledGraph":
    """
    Trace a ``torch.nn.Module`` with ``torch.fx`` and compile it to a
    :class:`CompiledGraph`.

    Uses ``torch.fx.symbolic_trace`` + ``ShapeProp`` to capture the graph
    topology and infer tensor shapes, then maps each node to the
    corresponding libane op.

    **Supported operations:**

    +--------------------------------------+----------------------+
    | PyTorch pattern                      | libane op            |
    +======================================+======================+
    | ``nn.Linear`` (``bias=False``)       | ``MATMUL``           |
    | ``nn.LayerNorm``                     | ``LAYERNORM``        |
    | ``nn.RMSNorm``                       | ``RMSNORM``          |
    | ``F.gelu`` / ``F.silu`` / ``F.relu`` | ``GELU/SILU/RELU``   |
    | ``F.tanh`` / ``F.sigmoid``           | ``TANH/SIGMOID``     |
    | ``F.softmax``                        | ``SOFTMAX``          |
    | ``a + b`` / ``a * b``               | ``ADD / MUL``        |
    | ``a @ b`` / ``torch.matmul``         | ``DYNAMIC_MATMUL``   |
    | ``F.scaled_dot_product_attention``   | ``SDPA / SDPA_GQA``  |
    +--------------------------------------+----------------------+

    Raises ``NotImplementedError`` for anything not in the table above.

    **Tensor shape convention:** tensors must already be in libane format
    (``[1, C, 1, S]`` for activations, ``[1, H, S, D]`` for matrix ops).
    Shape conversion is not performed automatically.

    **Bias:** ``nn.Linear(bias=True)`` is not supported.  Set ``bias=False``
    or handle the bias with a separate ``ADD`` node.

    Args:
        module:        ``torch.nn.Module`` to compile.  Must be in eval mode
                       with weights already in ``float16``.
        example_input: A single ``torch.Tensor`` or a list/tuple of tensors
                       matching the module's positional inputs.  Used only
                       for shape propagation — values are discarded.

    Returns:
        :class:`CompiledGraph` ready for repeated ANE inference.

    Raises:
        ImportError:         if ``torch`` is not installed.
        NotImplementedError: for unsupported op patterns.
        RuntimeError:        if ANE compilation fails.

    Example::

        import torch, torch.nn as nn
        import ane, numpy as np

        class FFN(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.w1 = nn.Linear(d, h, bias=False)
                self.w2 = nn.Linear(h, d, bias=False)
            def forward(self, x):
                return self.w2(torch.nn.functional.gelu(self.w1(x)))

        ffn = FFN(512, 2048).half().eval()
        x_ex = torch.zeros(1, 512, 1, 128, dtype=torch.float16)
        cg = ane.from_torch(ffn, x_ex)
        out = cg(x_ex.numpy())   # → np.float16, shape [1, 512, 1, 128]
    """
    ...


# ── Graph API ─────────────────────────────────────────────────────────────────

class Graph:
    """
    Mutable ANE graph builder.

    All tensors use layout [1, C, 1, S] (NCHW with batch=1, height=1).

    Example::

        g = ane.Graph()
        x  = g.add_input("x",  [1, 512, 1, 128])
        rn = g.add_op(ane.RMSNORM, [x],  [1, 512, 1, 128], weights=scale)
        up = g.add_op(ane.MATMUL,  [rn], [1, 2048, 1, 128], weights=W_up)
        ac = g.add_op(ane.GELU,    [up], [1, 2048, 1, 128])
        dn = g.add_op(ane.MATMUL,  [ac], [1, 512, 1, 128],  weights=W_down)
        g.mark_output(dn)
        cg = g.compile()
    """

    def __init__(self) -> None: ...

    def add_input(self, name: str, shape: list[int]) -> int:
        """
        Declare a graph input tensor.

        Args:
            name:  Logical name for the input (used in error messages).
            shape: 4-element list [1, C, 1, S].

        Returns:
            Tensor ID (uint32), passed to ``add_op`` or ``mark_output``.
        """
        ...

    def add_op(
        self,
        op: int,
        inputs: list[int],
        output_shape: list[int],
        weights: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add an operation node.

        Args:
            op:           Op constant (e.g. ``ane.MATMUL``, ``ane.RELU``).
            inputs:       List of input tensor IDs.
            output_shape: 4-element list [1, C_out, 1, S_out].
            weights:      Weight array (required for MATMUL, RMSNORM, LAYERNORM).
                          Converted to float16 internally.

        Returns:
            Output tensor ID (uint32).
        """
        ...

    def add_pwl_activation(
        self,
        input_id: int,
        output_shape: list[int],
        x_min: float,
        x_max: float,
        samples: np.ndarray,
    ) -> int:
        """
        Add a piecewise-linear custom activation.

        Approximates an arbitrary smooth function over [x_min, x_max] using
        n_samples-1 linear segments. 32 segments (33 samples) gives < 0.1%
        mean relative error for typical smooth activations.

        Args:
            input_id:     Tensor ID of the input.
            output_shape: [1, C, 1, S] — must match input shape.
            x_min, x_max: Domain of the approximation.
            samples:      np.float32 array of length n_samples.
                          Values are fn(np.linspace(x_min, x_max, n_samples)).

        Returns:
            Output tensor ID (uint32).

        Example::

            xs = np.linspace(-3, 3, 33, dtype=np.float32)
            samples = x * scipy.special.expit(x)   # Swish
            t = g.add_pwl_activation(xi, [1, C, 1, S], -3.0, 3.0, samples)
        """
        ...

    def mark_output(self, tensor_id: int, name: str = "") -> None:
        """
        Mark a tensor as a graph output.

        Args:
            tensor_id: Tensor ID returned by ``add_input`` or ``add_op``.
            name:      Optional logical name for the output.
        """
        ...

    def add_dynamic_matmul(
        self,
        x_id: int,
        w_id: int,
        output_shape: list[int],
    ) -> int:
        """
        Dynamic matrix multiply Y = X @ W^T (both inputs are runtime tensors).

        Uses matrix tensor format (height > 1, C = 1).  K/N/M are inferred
        from the input shapes — no weights argument needed.

        Use when the weight matrix changes at runtime. For static weights,
        prefer ``add_op(ane.MATMUL, ...)`` which uses the faster conv1x1 path.

        Args:
            x_id:         Tensor ID of X, shape [1, 1, K, M]  (height=K inner dim, seq=M cols).
            w_id:         Tensor ID of W, shape [1, 1, N, K]  (height=N rows, seq=K inner dim).
            output_shape: [1, 1, N, M].

        Returns:
            Output tensor ID (uint32).
        """
        ...

    def add_sdpa(
        self,
        q_id: int,
        k_id: int,
        v_id: int,
        output_shape: list[int],
        mask_id: int = -1,
    ) -> int:
        """
        Scaled dot-product attention: out = softmax(Q @ K^T / sqrt(D)) @ V

        Uses matrix tensor format (height > 1).  H/S/D are inferred from the
        input shapes — no S/D arguments needed.

        Args:
            q_id, k_id, v_id: Tensor IDs for Q, K, V — each shape [1, H, S, D].
            output_shape:     [1, H, S, D].
            mask_id:          Tensor ID of additive attention mask [1, 1, S, S],
                              or -1 for unmasked attention (default).

        Returns:
            Output tensor ID (uint32).
        """
        ...

    def add_sdpa_gqa(
        self,
        q_id: int,
        k_id: int,
        v_id: int,
        mask_id: int = -1,
    ) -> int:
        """
        Grouped Query Attention SDPA — tiles K/V from H_kv to H_q heads.

        Like :meth:`add_sdpa` but Q has ``num_q_heads`` heads while K and V each
        have ``num_kv_heads`` heads (``num_q_heads % num_kv_heads == 0``).
        K and V are tiled to ``num_q_heads`` inside the fused MIL program using
        CoreML's ``tile`` op, then passed to ``scaled_dot_product_attention``.

        The output shape is derived from Q — no ``output_shape`` argument needed.

        Args:
            q_id:    Tensor ID of Q, shape ``[1, num_q_heads,  S, D]``.
            k_id:    Tensor ID of K, shape ``[1, num_kv_heads, S, D]``.
            v_id:    Tensor ID of V, shape ``[1, num_kv_heads, S, D]``.
            mask_id: Tensor ID of additive attention mask ``[1, 1, S, S]``,
                     or -1 for unmasked attention (default).

        Returns:
            Output tensor ID, shape ``[1, num_q_heads, S, D]``.

        Constraints:
            - ``num_q_heads % num_kv_heads == 0``
            - S and D must each be a multiple of 32
            - All tensors use matrix format ``[1, H, S, D]``

        Example (Llama-3.1 8B: 32 query heads, 8 KV heads)::

            H_q, H_kv, S, D = 32, 8, 512, 128
            Q = g.add_input("Q", [1, H_q,  S, D])
            K = g.add_input("K", [1, H_kv, S, D])
            V = g.add_input("V", [1, H_kv, S, D])
            out = g.add_sdpa_gqa(Q, K, V)   # output: [1, H_q, S, D]
            g.mark_output(out)
            cg = g.compile()

        Raises:
            ValueError: if head counts are incompatible or shapes mismatch.
        """
        ...

    def add_matmul_w8a16(
        self,
        input_id: int,
        output_shape: list[int],
        weights: np.ndarray,
        scales: np.ndarray,
        IC: int,
        OC: int,
    ) -> int:
        """
        W8A16 quantized matrix multiply fused into the graph.

        Dequantizes ``int8`` weights with per-channel ``float32`` scales at
        compile time, then compiles as a standard ``conv1x1`` matmul.
        Fuses with downstream elementwise ops (GELU, SILU, ADD …) at zero
        extra cost — identical fusion behaviour to ``add_op(ane.MATMUL, …)``.

        Dequantization formula (applied once at compile time)::

            W_fp16[ic, oc] = float(weights[ic, oc]) × scales[oc]

        Args:
            input_id:     Tensor ID of the activation [1, IC, 1, S].
            output_shape: ``[1, OC, 1, S]``.
            weights:      ``np.int8`` array, shape ``[IC, OC]`` row-major.
                          (Transpose of PyTorch's ``linear.weight`` [OC, IC].)
            scales:       ``np.float32`` array, shape ``[OC]``.
                          Per-output-channel scale. Zero-point is implicitly 0
                          (symmetric quantization only).
            IC:           Input channel count.
            OC:           Output channel count.

        Returns:
            Output tensor ID (uint32).

        Example::

            # Quantize fp16 weights (PyTorch convention [OC, IC])
            W_fp16 = model.linear.weight.numpy().astype(np.float16)  # [OC, IC]
            scales = (np.max(np.abs(W_fp16), axis=1) / 127.0).astype(np.float32)
            W_int8 = np.clip(np.round(W_fp16 / scales[:, None]), -128, 127).astype(np.int8)
            W_IC_OC = np.ascontiguousarray(W_int8.T)  # [IC, OC]

            out = g.add_matmul_w8a16(x, [1, OC, 1, S],
                                     weights=W_IC_OC, scales=scales,
                                     IC=IC, OC=OC)

        Raises:
            ValueError: on invalid arguments or shape mismatch.
        """
        ...

    def add_matmul_w8a8(
        self,
        input_id: int,
        output_shape: list[int],
        weights: "np.ndarray",
        scales: "np.ndarray",
        IC: int,
        OC: int,
        act_scale: float,
        act_zero_point: int = 0,
    ) -> int:
        """
        W8A8 quantized matrix multiply — int8 weights AND int8 activations.

        Both weight and activation tensors are stored as int8, halving memory
        bandwidth.  The ANE operates in fp16; dequantization is transparent:

        - Weights: dequantized at compile time (same as ``add_matmul_w8a16``).
        - Activations: dequantized at execute time before IOSurface transfer.

        At execute time, **pass int8 numpy arrays** as graph inputs (not
        float16).  Use :func:`ane.quantize_i8` to quantize fp16 activations.

        Args:
            input_id:        Tensor ID of the int8 activation input [1, IC, 1, S].
            output_shape:    ``[1, OC, 1, S]``.
            weights:         ``np.int8`` array, shape ``[IC, OC]`` row-major.
                             (Transpose of PyTorch ``Linear.weight`` which is ``[OC, IC]``.)
            scales:          ``np.float32`` array, shape ``[OC]``.
                             Per-output-channel weight scales.
            IC:              Input channel count.
            OC:              Output channel count.
            act_scale:       Per-tensor activation scale (float > 0).
            act_zero_point:  Per-tensor activation zero-point (int, default 0).

        Returns:
            Output tensor ID (int).

        Example::

            import numpy as np
            import ane

            # Quantize weights (PyTorch convention [OC, IC])
            W_fp16   = model.linear.weight.detach().numpy()
            w_scales = np.max(np.abs(W_fp16), axis=1) / 127.0    # [OC]
            W_int8   = (W_fp16 / w_scales[:, None]).round().clip(-128,127).astype(np.int8)
            W_IC_OC  = W_int8.T.copy()                            # [IC, OC]

            act_scale = 0.02   # calibrated separately
            act_zp    = 0

            g = ane.Graph()
            x = g.add_input("x", [1, IC, 1, S])
            y = g.add_matmul_w8a8(x, [1, OC, 1, S],
                                   weights=W_IC_OC,
                                   scales=w_scales.astype(np.float32),
                                   IC=IC, OC=OC,
                                   act_scale=act_scale, act_zero_point=act_zp)
            g.mark_output(y)
            cg = g.compile()

            # Inference: quantize activations, pass int8 to execute
            x_fp16 = activations.astype(np.float16)
            x_int8 = ane.quantize_i8(x_fp16, scale=act_scale, zero_point=act_zp)
            out = cg(x_int8)   # out is np.float16

        Raises:
            ValueError: on invalid arguments or shape mismatch.
        """
        ...

    def add_conv2d(
        self,
        input_id: int,
        output_shape: list[int],
        kH: int,
        kW: int,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_top: int = 0,
        pad_left: int = 0,
        pad_bottom: int = 0,
        pad_right: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        *,
        kernel: np.ndarray,
    ) -> int:
        """
        Add a 2D convolution op to the graph.

        Conv image tensors use shape ``[1, C, H, W]`` with ``H > 1`` and
        ``W`` a multiple of 32 (IOSurface 64-byte DMA alignment for fp16).

        Output shape formula::

            H_out = (H_in + pad_top + pad_bottom - dilation_h*(kH-1) - 1) // stride_h + 1
            W_out = (W_in + pad_left + pad_right - dilation_w*(kW-1) - 1) // stride_w + 1

        Both W_in and W_out must be multiples of 32.  Bias is not supported;
        chain an ``add_op(ane.ADD, ...)`` node if needed.

        Args:
            input_id:     Tensor ID of the input ``[1, IC, H_in, W_in]``.
            output_shape: ``[1, OC, H_out, W_out]``.
            kH, kW:       Kernel spatial size (≥ 1).
            stride_h/w:   Stride in each spatial direction (default 1).
            pad_top/left/bottom/right: Explicit zero-padding in pixels (default 0).
            dilation_h/w: Dilation (default 1; 1 = standard convolution).
            groups:       Group count for group/depthwise convolution (default 1).
                          ``IC % groups`` must equal 0.
            kernel:       ``np.float16`` array, shape ``[OC, IC//groups, kH, kW]``,
                          row-major.  NO transpose required (unlike ``MATMUL``).

        Returns:
            Output tensor ID (uint32).

        Example — pointwise 1×1 conv (IC=4 → OC=8)::

            W = np.zeros((8, 4, 1, 1), dtype=np.float16)
            # fill W ...
            out = g.add_conv2d(x, [1, 8, H, W_seq], kH=1, kW=1, kernel=W)

        Example — depthwise 3×3, SAME padding, IC=OC=C::

            W = np.zeros((C, 1, 3, 3), dtype=np.float16)
            out = g.add_conv2d(x, [1, C, H, W_seq],
                               kH=3, kW=3,
                               pad_top=1, pad_left=1,
                               pad_bottom=1, pad_right=1,
                               groups=C,
                               kernel=W)

        Raises:
            ValueError: on invalid parameters or shape mismatch.
        """
        ...

    def compile(self) -> "CompiledGraph":
        """
        Compile the graph to an ANE program.

        Runs validation, fusion, and ANE compilation. Raises ``RuntimeError``
        on any failure (validation error, ANE unavailable, compile error).

        Returns:
            A ``CompiledGraph`` ready for repeated execution.
        """
        ...


class KVCache:
    """
    CPU-side KV cache for autoregressive decode with SDPA/SDPA_GQA graphs.

    Holds zero-initialised fp16 buffers for K and V at a fixed maximum
    sequence length.  Append one token at a time with :meth:`update`, then
    pass :attr:`k` and :attr:`v` (full buffers) as K/V inputs to your
    compiled SDPA graph.

    The buffer layout matches the ANE matrix-tensor format expected by
    ``add_sdpa`` / ``add_sdpa_gqa``:
      ``[1, num_heads, max_seq, head_dim]``

    Example (Llama-3.1 8B, GQA: 8 KV heads, D=128, max 512 tokens)::

        cache = ane.KVCache(num_heads=8, head_dim=128, max_seq=512)

        # at each decode step:
        pos = cache.update(k_token, v_token)  # k/v_token: float16 [8, 128]
        out = cg([q_vec, cache.k, cache.v])   # pass full buffers to graph
    """

    def __init__(self, num_heads: int, head_dim: int, max_seq: int) -> None:
        """
        Create a KV cache.

        Args:
            num_heads: Number of KV heads (``num_kv_heads`` for GQA).
            head_dim:  Head dimension D.  Must be a multiple of 32.
            max_seq:   Maximum sequence length to cache.

        Raises:
            ValueError: on invalid (non-positive) dimensions.
        """
        ...

    def update(self, k: np.ndarray, v: np.ndarray) -> int:
        """
        Append one token's K and V to the cache.

        ``k`` and ``v`` must be ``float16`` arrays with
        ``num_heads × head_dim`` elements each (any shape, contiguous).

        Args:
            k: Key slice for one token, shape broadcastable to
               ``[num_heads, head_dim]``, dtype ``float16``.
            v: Value slice for one token, same constraints.

        Returns:
            New position (1-based count of filled slots).

        Raises:
            RuntimeError: if the cache is full (``position >= max_seq``).
        """
        ...

    @property
    def k(self) -> np.ndarray:
        """Full K buffer, shape ``[1, num_heads, max_seq, head_dim]``, ``float16``."""
        ...

    @property
    def v(self) -> np.ndarray:
        """Full V buffer, shape ``[1, num_heads, max_seq, head_dim]``, ``float16``."""
        ...

    @property
    def position(self) -> int:
        """Number of valid token positions written so far."""
        ...

    @property
    def num_heads(self) -> int:
        """Number of KV heads."""
        ...

    @property
    def head_dim(self) -> int:
        """Head dimension D."""
        ...

    @property
    def max_seq(self) -> int:
        """Maximum sequence length."""
        ...

    def reset(self) -> None:
        """Reset position to 0 and zero the K/V buffers."""
        ...


class CompiledGraph:
    """
    Compiled ANE graph. Call with numpy arrays to run inference.

    Example — single input / single output::

        out = cg(x)           # x: np.float16, shape matches graph input

    Multi-input::

        out = cg([a, b])      # list of np.float16 arrays

    Multi-output — set output shapes first::

        cg.set_output_shapes([[1, 256, 1, 128], [1, 128, 1, 128]])
        a, b = cg(x)
    """

    def __call__(
        self, inputs: Union[np.ndarray, list[np.ndarray]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Run a forward pass.

        Args:
            inputs: Single np.ndarray for single-input graphs, or list of
                    np.ndarray for multi-input graphs. Arrays converted to
                    float16 contiguous layout internally.

        Returns:
            Single np.ndarray (float16) for single-output graphs, or list of
            np.ndarray for multi-output graphs.
        """
        ...

    def set_output_shapes(self, shapes: list[list[int]]) -> None:
        """
        Set expected output shapes for multi-output graphs.

        Must be called before ``__call__`` when the graph has multiple outputs
        or when the output shape differs from the input shape.

        Args:
            shapes: List of shape lists, e.g. ``[[1, 256, 1, 128], [1, 128, 1, 128]]``.
        """
        ...

    def save(self, path: str) -> None:
        """
        Save this compiled graph to a file for fast cold-start restoration.

        Writes MIL text, weight blobs, and compiled HWX binaries in a binary
        format (magic ``ANEG``, version 1).  Reload with
        :meth:`CompiledGraph.load` — skips ``compileWithQoS:`` and calls only
        ``loadWithQoS:``, which is ~8.5× faster than a full recompile.

        Typical workflow::

            # First run: compile and save
            cg = g.compile()
            cg.save("/tmp/my_model.aneg")

            # Subsequent runs: load instead of compile
            cg = ane.CompiledGraph.load("/tmp/my_model.aneg")
            out = cg(x)

        Args:
            path: Destination file path.  Created or overwritten.

        Raises:
            RuntimeError: on I/O error or if HWX extraction fails.
        """
        ...

    @staticmethod
    def load(path: str) -> "CompiledGraph":
        """
        Load a compiled graph from a file produced by :meth:`save`.

        Restores each group via ``loadWithQoS:`` with the saved HWX binary —
        skipping ``compileWithQoS:`` for an ~8.5× cold-start speedup.
        Gracefully falls back to full recompilation per group if the saved
        HWX is incompatible (e.g. after a macOS upgrade).

        Example::

            cg = ane.CompiledGraph.load("/tmp/my_model.aneg")
            out = cg(x)  # ready immediately

        Args:
            path: File path produced by :meth:`save`.

        Returns:
            A :class:`CompiledGraph` ready for inference.

        Raises:
            RuntimeError: on format error, I/O failure, or ANE unavailability.
        """
        ...

    def delta_reload(self) -> None:
        """
        Reload all ANE programs in this compiled graph into SRAM without
        recompiling (~8.5× faster than compile).

        Use after a host suspend/resume cycle or any ANE context reset to
        restore compiled programs without paying the full compile cost.

        Also enables LoRA-style weight hot-swap: mutate the underlying
        ``AneProgram`` weight blobs, then call ``delta_reload()`` to push the
        updated weights to the accelerator without rebuilding the graph.

        Example::

            cg = g.compile()
            cg(x)                 # first inference

            # ... host suspended / resumed ...
            cg.delta_reload()     # restore programs to SRAM (~8.5× vs recompile)
            cg(x)                 # inference works again

            # LoRA adapter hot-swap example:
            #   1. Overwrite weight blobs inside cg's AnePrograms
            #   2. cg.delta_reload()   ← push new weights without recompiling
            #   3. cg(x)               ← runs with adapter weights

        Raises:
            RuntimeError: if any program reload fails.
        """
        ...
