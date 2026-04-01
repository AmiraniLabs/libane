"""
ane — Apple Neural Engine Python bindings (libane v0.7.0)

ANE-accelerated ML operations with automatic CPU fallback.
Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
Not for App Store submission.
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
TRANSPOSE: int
SILU: int
RMSNORM: int
AVG_POOL: int
MAX_POOL: int
LOGICAL_AND: int
LOGICAL_OR: int
LOGICAL_XOR: int
REDUCE_PROD: int

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
    """libane version string (e.g. '0.7.0')."""
    ...

def last_error() -> str:
    """Last error message from the library."""
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

# ── Single-op API ─────────────────────────────────────────────────────────────

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

    Input is interpreted as [1, C, 1, S] where C = product of all but the last
    dimension and S = last dimension. Falls back to numpy for unsupported shapes.

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

# ── Raw MIL probe API ────────────────────────────────────────────────────────

class CompiledMil:
    """
    Compiled raw MIL program. Returned by ``compile_mil()`` and
    ``compile_mil_with_weights()``.

    Use :mod:`probe` for a higher-level interface.
    """

    def run(
        self,
        inputs: list[np.ndarray],
        output_sizes: list[int],
    ) -> list[np.ndarray]:
        """
        Execute the compiled MIL program.

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


def compile_mil(mil_text: str) -> CompiledMil:
    """
    Compile a raw MIL program (no external weights).

    Args:
        mil_text: UTF-8 MIL source text (complete program including buildInfo header).

    Returns:
        ``CompiledMil`` ready for ``.run()``.

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
        ``CompiledMil`` ready for ``.run()``.

    Raises:
        RuntimeError if ANE is unavailable or compilation fails.
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
            op:           Op constant (e.g. ``ane.MATMUL``, ``ane.RMSNORM``).
            inputs:       List of input tensor IDs.
            output_shape: 4-element list [1, C_out, 1, S_out].
            weights:      Weight array (required for MATMUL, RMSNORM, LAYERNORM).
                          Converted to float16 internally.

        Returns:
            Output tensor ID (uint32).
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

    def compile(self) -> "CompiledGraph":
        """
        Compile the graph to an ANE program.

        Runs validation, fusion, and ANE compilation. Raises ``RuntimeError``
        on any failure (validation error, ANE unavailable, compile error).

        Returns:
            A ``CompiledGraph`` ready for repeated execution.
        """
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
