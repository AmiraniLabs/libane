"""
ane.probe — ANE op probe harness for libane.

Systematically tests MIL ops for:
  1. Compilation success on the local ANE
  2. Numerical correctness vs CPU reference
  3. Compound/intermediate op discovery
  4. Parameter space exploration
  5. Cross-shape behavior

Usage::

    from ane import probe

    # Scan all known unary ops
    report = probe.scan_unary()
    probe.print_report(report)

    # Discover ops that only work as intermediates
    report = probe.scan_intermediate()

    # Try unknown op names (fuzzing)
    report = probe.explore_names(probe.CANDIDATE_NAMES)

    # Export for crowdsourced ANE research
    probe.export_json(report, "results.json")

    # CLI: python -m ane.probe [--unary] [--binary] [--composite] [--export FILE]
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

# ── MIL program builder ───────────────────────────────────────────────────────

# Standard build-info header required by the ANE compiler.
# These version strings match what maderix/ANE and ironmill use.
_BUILD_INFO = (
    '[buildInfo = dict<string, string>({'
    '{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, '
    '{"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}'
    '})]'
)

_C = 32   # default probe tensor channels
_S = 32   # default probe tensor seq length
_N = _C * _S  # total elements


def mil_program(body: str, inputs_sig: str, output: str) -> str:
    """Build a complete MIL 1.3 program from a body fragment."""
    return (
        f"program(1.3)\n"
        f"{_BUILD_INFO}\n"
        f"{{\n"
        f"    func main<ios18>({inputs_sig}) {{\n"
        f"{body}\n"
        f"    }} -> ({output});\n"
        f"}}"
    )


def _sig1(C: int = _C, S: int = _S) -> str:
    """MIL signature for one [1,C,1,S] fp16 input."""
    return f"tensor<fp16, [1,{C},1,{S}]> a_input0"


def _sig2(C: int = _C, S: int = _S) -> str:
    """MIL signature for two [1,C,1,S] fp16 inputs."""
    return (
        f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
        f"tensor<fp16, [1,{C},1,{S}]> a_input1"
    )


def _sig3(C: int = _C, S: int = _S) -> str:
    """MIL signature for three [1,C,1,S] fp16 inputs."""
    return (
        f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
        f"tensor<fp16, [1,{C},1,{S}]> a_input1, "
        f"tensor<fp16, [1,{C},1,{S}]> a_input2"
    )


def _body1(op: str, extra_args: str = "", C: int = _C, S: int = _S) -> str:
    """MIL body for a single unary op."""
    args = f"x=a_input0{extra_args}"
    return (
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
        f'{op}({args})[name=string("z_output0")];'
    )


def _body2(op: str, extra_args: str = "", C: int = _C, S: int = _S) -> str:
    """MIL body for a binary op on two fp16 inputs."""
    args = f"x=a_input0, y=a_input1{extra_args}"
    return (
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
        f'{op}({args})[name=string("z_output0")];'
    )


def _body_cmp(op: str, C: int = _C, S: int = _S) -> str:
    """MIL body for a comparison op: returns fp16 cast of bool result."""
    return (
        f'        tensor<bool, [1,{C},1,{S}]> cmp = '
        f'{op}(x=a_input0, y=a_input1)[name=string("cmp")];\n'
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
        f'cast(x=cmp, dtype=string("fp16"))[name=string("z_output0")];'
    )


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of probing one MIL op or pattern."""
    name: str
    compiled: bool
    passed: Optional[bool]   # None if compile failed or no reference
    max_err: Optional[float]
    atol: float
    compile_ms: float
    eval_ms: Optional[float]
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if not self.compiled:
            return "COMPILE_FAIL"
        if self.passed is None:
            return "NO_REF"
        return "PASS" if self.passed else "EVAL_FAIL"


# ── Numerical checking ────────────────────────────────────────────────────────

def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun polynomial approximation for erf (same as ironmill)."""
    sign = np.sign(x)
    ax = np.abs(x.astype(np.float32))
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return (sign * (1.0 - poly * np.exp(-ax * ax))).astype(np.float32)


def check(name: str,
          actual: np.ndarray,
          expected: np.ndarray,
          atol: float) -> tuple[bool, float]:
    """
    Compare f16 actual vs f32 expected. Returns (passed, max_err).
    Prints a one-line result to stdout.
    """
    a32 = actual.astype(np.float32).ravel()
    e32 = np.asarray(expected, dtype=np.float32).ravel()

    if a32.shape != e32.shape:
        print(f"    ❌ {name}: shape mismatch ({a32.shape} vs {e32.shape})")
        return False, float("inf")

    errs = np.abs(a32 - e32)
    max_err = float(errs.max())
    fail_idx = np.argmax(errs)

    if max_err > atol:
        print(
            f"    ❌ {name}: FAIL at [{fail_idx}] "
            f"got={a32[fail_idx]:.4f} exp={e32[fail_idx]:.4f} "
            f"(max_err={max_err:.6f}, atol={atol})"
        )
        return False, max_err

    print(f"    ✅ {name}: PASS (max_err={max_err:.6f}, atol={atol})")
    return True, max_err


# ── Core probe function ───────────────────────────────────────────────────────

def probe_custom(
    name: str,
    mil_text: str,
    inputs: list[np.ndarray],
    output_numel: int,
    expected: Optional[np.ndarray] = None,
    atol: float = 0.1,
    weights: Optional[dict] = None,
) -> ProbeResult:
    """
    Compile and optionally evaluate an arbitrary MIL program.

    Args:
        name:         Human-readable name for reporting.
        mil_text:     Complete MIL program text.
        inputs:       List of fp16 input arrays.
        output_numel: Number of fp16 elements in the output.
        expected:     CPU reference output (f32). If None, only compilation is tested.
        atol:         Absolute tolerance for numerical comparison.
        weights:      Optional dict of {filename: np.float16 array} for weight files.

    Returns:
        ProbeResult
    """
    try:
        import ane
    except ImportError as e:
        return ProbeResult(
            name=name, compiled=False, passed=None, max_err=None,
            atol=atol, compile_ms=0.0, eval_ms=None,
            error=f"ane module not available: {e}",
        )

    t0 = time.perf_counter()
    try:
        if weights:
            prog = ane.compile_mil_with_weights(mil_text, weights)
        else:
            prog = ane.compile_mil(mil_text)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except RuntimeError as e:
        compile_ms = (time.perf_counter() - t0) * 1000.0
        return ProbeResult(
            name=name, compiled=False, passed=None, max_err=None,
            atol=atol, compile_ms=compile_ms, eval_ms=None,
            error=str(e),
        )

    if expected is None:
        return ProbeResult(
            name=name, compiled=True, passed=None, max_err=None,
            atol=atol, compile_ms=compile_ms, eval_ms=None,
        )

    t1 = time.perf_counter()
    try:
        (out,) = prog.run(inputs, [output_numel])
        eval_ms = (time.perf_counter() - t1) * 1000.0
    except RuntimeError as e:
        eval_ms = (time.perf_counter() - t1) * 1000.0
        return ProbeResult(
            name=name, compiled=True, passed=False, max_err=None,
            atol=atol, compile_ms=compile_ms, eval_ms=eval_ms,
            error=str(e),
        )

    passed, max_err = check(name, out, expected, atol)
    return ProbeResult(
        name=name, compiled=True, passed=passed, max_err=max_err,
        atol=atol, compile_ms=compile_ms, eval_ms=eval_ms,
    )


# ── Unary op descriptors ──────────────────────────────────────────────────────
#
# Each entry: (mil_body_fn, input_fn, expected_fn, atol)
# mil_body_fn: str — the body fragment (already complete)
# input_fn: () -> np.ndarray (f32, will be cast to f16)
# expected_fn: (x_f32) -> np.ndarray (f32 reference)

def _linspace(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, _N, dtype=np.float32)


def _small_pos() -> np.ndarray:
    return (_linspace(0.0, 1.0) + 0.01).astype(np.float32)


def _safe_nonzero() -> np.ndarray:
    """Values in (-10, 10) avoiding 0."""
    x = _linspace(-5.0, 5.0)
    x[np.abs(x) < 0.5] = 0.6
    return x


_UNARY_OPS: dict[str, tuple[str, Callable, Callable, float]] = {
    # (body, input_gen, expected_fn, atol)
    "relu": (
        _body1("relu"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: np.maximum(x, 0.0),
        0.01,
    ),
    "abs": (
        _body1("abs"),
        lambda: _linspace(-5.0, 5.0),
        np.abs,
        0.01,
    ),
    "neg": (
        _body1("neg"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: -x,
        0.01,
    ),
    "sign": (
        _body1("sign"),
        lambda: np.where(
            np.abs(_linspace(-5.0, 5.0)) < 0.05, 0.1, _linspace(-5.0, 5.0)
        ).astype(np.float32),
        lambda x: np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0)).astype(np.float32),
        0.01,
    ),
    "sqrt": (
        _body1("sqrt"),
        lambda: _linspace(0.1, 10.0),
        np.sqrt,
        0.05,
    ),
    "exp": (
        _body1("exp"),
        lambda: _linspace(-5.0, 5.0) * 0.5,
        np.exp,
        0.5,
    ),
    "exp2": (
        _body1("exp2"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: np.power(2.0, x),
        0.5,
    ),
    "log": (
        # ANE requires epsilon parameter for log
        (
            '        fp16 eps = const()[name=string("eps"), val=fp16(0x1.0cp-17)];\n'
            '        tensor<fp16, [1,32,1,32]> z_output0 = '
            'log(x=a_input0, epsilon=eps)[name=string("z_output0")];'
        ),
        lambda: _linspace(0.1, 10.0),
        np.log,
        0.1,
    ),
    "sqrt": (
        _body1("sqrt"),
        lambda: _linspace(0.1, 10.0),
        np.sqrt,
        0.05,
    ),
    "rsqrt": (
        # ANE requires epsilon parameter
        (
            '        fp16 eps = const()[name=string("eps"), val=fp16(0x1.0cp-17)];\n'
            '        tensor<fp16, [1,32,1,32]> z_output0 = '
            'rsqrt(x=a_input0, epsilon=eps)[name=string("z_output0")];'
        ),
        lambda: _linspace(0.1, 10.0),
        lambda x: 1.0 / np.sqrt(x),
        0.05,
    ),
    "inverse": (
        # ANE requires epsilon parameter
        (
            '        fp16 eps = const()[name=string("eps"), val=fp16(0x1.0cp-17)];\n'
            '        tensor<fp16, [1,32,1,32]> z_output0 = '
            'inverse(x=a_input0, epsilon=eps)[name=string("z_output0")];'
        ),
        _safe_nonzero,
        lambda x: 1.0 / x,
        0.5,
    ),
    "square": (
        _body1("square"),
        lambda: _linspace(-3.0, 3.0),
        lambda x: x * x,
        0.1,
    ),
    "sigmoid": (
        _body1("sigmoid"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        0.015,
    ),
    "tanh": (
        _body1("tanh"),
        lambda: _linspace(-3.0, 3.0),
        np.tanh,
        0.01,
    ),
    "erf": (
        _body1("erf"),
        lambda: _linspace(-2.0, 2.0),
        _erf_approx,
        0.02,
    ),
    "silu": (
        _body1("silu"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: x / (1.0 + np.exp(-x)),
        0.02,
    ),
    "gelu_tanh": (
        _body1("gelu_tanh"),
        lambda: _linspace(-3.0, 3.0),
        lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        0.05,
    ),
    "softsign": (
        _body1("softsign"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: x / (1.0 + np.abs(x)),
        0.01,
    ),
    "softplus": (
        _body1("softplus"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: np.log1p(np.exp(np.clip(x, -80, 80))),
        0.05,
    ),
    "ceil": (
        _body1("ceil"),
        lambda: _linspace(-5.0, 5.0) * 3,
        lambda x: np.ceil(np.array(x, dtype=np.float16).astype(np.float32)),
        0.01,
    ),
    "floor": (
        _body1("floor"),
        lambda: _linspace(-5.0, 5.0) * 3,
        lambda x: np.floor(np.array(x, dtype=np.float16).astype(np.float32)),
        0.01,
    ),
    "round": (
        _body1("round"),
        lambda: _linspace(-5.0, 5.0) * 3,
        lambda x: np.round(np.array(x, dtype=np.float16).astype(np.float32)),
        0.01,
    ),
    "sin": (
        _body1("sin"),
        lambda: _linspace(-3.0, 3.0),
        np.sin,
        0.01,
    ),
    "cos": (
        _body1("cos"),
        lambda: _linspace(-3.0, 3.0),
        np.cos,
        0.01,
    ),
    "tan": (
        _body1("tan"),
        lambda: _linspace(-1.2, 1.2),
        np.tan,
        0.01,
    ),
    "asin": (
        _body1("asin"),
        lambda: _linspace(-0.9, 0.9),
        np.arcsin,
        0.01,
    ),
    "acos": (
        _body1("acos"),
        lambda: _linspace(-0.9, 0.9),
        np.arccos,
        0.02,
    ),
    "atan": (
        _body1("atan"),
        lambda: _linspace(-5.0, 5.0),
        np.arctan,
        0.05,
    ),
    "sinh": (
        _body1("sinh"),
        lambda: _linspace(-2.0, 2.0),
        np.sinh,
        0.05,
    ),
    "cosh": (
        _body1("cosh"),
        lambda: _linspace(-2.0, 2.0),
        np.cosh,
        0.05,
    ),
    "identity": (
        _body1("identity"),
        lambda: _linspace(-5.0, 5.0),
        lambda x: x,
        0.02,
    ),
}


# ── Binary op descriptors ─────────────────────────────────────────────────────

def _pair_ab() -> tuple[np.ndarray, np.ndarray]:
    a = _linspace(-5.0, 5.0)
    b = _linspace(0.5, 5.5)  # non-zero for div ops
    return a, b


_BINARY_OPS: dict[str, tuple[str, Callable, Callable, float]] = {
    "add": (
        _body2("add"),
        _pair_ab,
        lambda a, b: a + b,
        0.05,
    ),
    "sub": (
        _body2("sub"),
        _pair_ab,
        lambda a, b: a - b,
        0.15,
    ),
    "mul": (
        _body2("mul"),
        lambda: (_linspace(-2.0, 2.0), _linspace(-2.0, 2.0)),
        lambda a, b: a * b,
        0.5,
    ),
    "maximum": (
        _body2("maximum"),
        _pair_ab,
        lambda a, b: np.maximum(a, b),
        0.05,
    ),
    "minimum": (
        _body2("minimum"),
        _pair_ab,
        lambda a, b: np.minimum(a, b),
        0.05,
    ),
    "real_div": (
        _body2("real_div"),
        _pair_ab,
        lambda a, b: a / b,
        0.5,
    ),
    "floor_div": (
        _body2("floor_div"),
        lambda: (_linspace(1.0, 50.0), np.full(_N, 7.0, dtype=np.float32)),
        lambda a, b: np.floor(a / b),
        1.0,
    ),
    "mod": (
        _body2("mod"),
        lambda: (_linspace(1.0, 50.0), np.full(_N, 7.0, dtype=np.float32)),
        lambda a, b: a - np.floor(a / b) * b,
        1.0,
    ),
}

# Comparison ops: output is bool cast to fp16
_COMPARISON_OPS: dict[str, tuple[str, Callable, Callable, float]] = {
    "greater": (
        _body_cmp("greater"),
        lambda: (
            np.array([i % 7 for i in range(_N)], dtype=np.float32),
            np.full(_N, 3.0, dtype=np.float32),
        ),
        lambda a, b: (a > b).astype(np.float32),
        0.01,
    ),
    "greater_equal": (
        _body_cmp("greater_equal"),
        lambda: (
            np.array([i % 7 for i in range(_N)], dtype=np.float32),
            np.full(_N, 3.0, dtype=np.float32),
        ),
        lambda a, b: (a >= b).astype(np.float32),
        0.01,
    ),
    "less": (
        _body_cmp("less"),
        lambda: (
            np.array([i % 7 for i in range(_N)], dtype=np.float32),
            np.full(_N, 3.0, dtype=np.float32),
        ),
        lambda a, b: (a < b).astype(np.float32),
        0.01,
    ),
    "less_equal": (
        _body_cmp("less_equal"),
        lambda: (
            np.array([i % 7 for i in range(_N)], dtype=np.float32),
            np.full(_N, 3.0, dtype=np.float32),
        ),
        lambda a, b: (a <= b).astype(np.float32),
        0.01,
    ),
    "equal": (
        _body_cmp("equal"),
        lambda: (
            np.array([i % 5 for i in range(_N)], dtype=np.float32),
            np.full(_N, 2.0, dtype=np.float32),
        ),
        lambda a, b: (np.abs(a - b) < 1e-6).astype(np.float32),
        0.01,
    ),
    "not_equal": (
        _body_cmp("not_equal"),
        lambda: (
            np.array([i % 5 for i in range(_N)], dtype=np.float32),
            np.full(_N, 2.0, dtype=np.float32),
        ),
        lambda a, b: (np.abs(a - b) > 1e-6).astype(np.float32),
        0.01,
    ),
}


# ── Reduction op builders ─────────────────────────────────────────────────────

def _reduction_body(op: str, axis: int = -1, C: int = _C, S: int = _S) -> str:
    """Build a reduction body: reduce then tile back to full [1,C,1,S] for eval."""
    rax_val = str(axis)
    if axis == -1:
        reduced_shape = f"[1,{C},1,1]"
        tile_val = f"[1,1,1,{S}]"
    else:  # axis == 1
        reduced_shape = f"[1,1,1,{S}]"
        tile_val = f"[1,{C},1,1]"
    return (
        f'        tensor<int32, [1]> rax = const()[name=string("rax"), '
        f'val=tensor<int32, [1]>([{rax_val}])];\n'
        f'        bool kd = const()[name=string("kd"), val=bool(true)];\n'
        f'        tensor<fp16, {reduced_shape}> red = '
        f'{op}(x=a_input0, axes=rax, keep_dims=kd)[name=string("red")];\n'
        f'        tensor<int32, [4]> rep = const()[name=string("rep"), '
        f'val=tensor<int32, [4]>({tile_val})];\n'
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
        f'tile(x=red, reps=rep)[name=string("z_output0")];'
    )


# ── Generic probe functions ───────────────────────────────────────────────────

def probe_unary(
    op_name: str,
    *,
    body: Optional[str] = None,
    input_fn: Optional[Callable] = None,
    expected_fn: Optional[Callable] = None,
    atol: float = 0.1,
    C: int = _C,
    S: int = _S,
) -> ProbeResult:
    """
    Probe a unary MIL op. Looks up defaults in _UNARY_OPS if not provided.

    Args:
        op_name:     MIL op name (e.g. "relu"). Used as the result name.
        body:        MIL body fragment. Defaults to lookup in _UNARY_OPS.
        input_fn:    Callable returning f32 input array. Defaults to lookup.
        expected_fn: Callable (x_f32) -> f32 reference. Defaults to lookup.
        atol:        Absolute tolerance. Defaults to lookup.
        C, S:        Tensor shape.

    Returns:
        ProbeResult
    """
    desc = _UNARY_OPS.get(op_name)
    _body = body or (desc[0] if desc else _body1(op_name, C=C, S=S))
    _inp  = input_fn or (desc[1] if desc else lambda: _linspace(-2.0, 2.0))
    _ref  = expected_fn or (desc[2] if desc else None)
    _atol = atol if atol != 0.1 or not desc else desc[3]

    x_f32 = _inp().ravel()[:C * S].astype(np.float32)
    x_f16 = x_f32.astype(np.float16)
    expected = _ref(x_f32) if _ref else None

    mil = mil_program(_body, _sig1(C, S), "z_output0")
    return probe_custom(op_name, mil, [x_f16], C * S, expected, _atol)


def probe_binary(
    op_name: str,
    *,
    body: Optional[str] = None,
    input_fn: Optional[Callable] = None,
    expected_fn: Optional[Callable] = None,
    atol: float = 0.1,
    C: int = _C,
    S: int = _S,
) -> ProbeResult:
    """Probe a binary MIL op."""
    # Check both binary and comparison descriptors
    desc = _BINARY_OPS.get(op_name) or _COMPARISON_OPS.get(op_name)
    _body = body or (desc[0] if desc else _body2(op_name, C=C, S=S))
    _inp  = input_fn or (desc[1] if desc else _pair_ab)
    _ref  = expected_fn or (desc[2] if desc else None)
    _atol = atol if atol != 0.1 or not desc else desc[3]

    ab = _inp()
    a_f16 = ab[0].ravel()[:C * S].astype(np.float16)
    b_f16 = ab[1].ravel()[:C * S].astype(np.float16)
    expected = _ref(ab[0].ravel()[:C * S], ab[1].ravel()[:C * S]) if _ref else None

    mil = mil_program(_body, _sig2(C, S), "z_output0")
    return probe_custom(op_name, mil, [a_f16, b_f16], C * S, expected, _atol)


def probe_reduction(
    op_name: str,
    *,
    axis: int = -1,
    atol: float = 1.0,
    C: int = _C,
    S: int = _S,
) -> ProbeResult:
    """Probe a reduction op along the specified axis (-1=spatial, 1=channel)."""
    body = _reduction_body(op_name, axis, C, S)
    mil = mil_program(body, _sig1(C, S), "z_output0")

    x_f32 = _linspace(-2.0, 2.0)
    x_f16 = x_f32.astype(np.float16)

    # Build CPU reference
    x_grid = x_f32.reshape(C, S)
    if axis == -1:
        if op_name == "reduce_sum":
            red = x_grid.sum(axis=1, keepdims=True)
        elif op_name == "reduce_mean":
            red = x_grid.mean(axis=1, keepdims=True)
        elif op_name == "reduce_max":
            red = x_grid.max(axis=1, keepdims=True)
        elif op_name == "reduce_min":
            red = x_grid.min(axis=1, keepdims=True)
        elif op_name == "reduce_l2_norm":
            red = np.sqrt((x_grid ** 2).sum(axis=1, keepdims=True))
        elif op_name == "reduce_log_sum_exp":
            mv = x_grid.max(axis=1, keepdims=True)
            red = mv + np.log(np.exp(x_grid - mv).sum(axis=1, keepdims=True))
        else:
            red = None
        if red is not None:
            expected = np.tile(red, (1, S)).ravel()
        else:
            expected = None
    else:  # axis == 1
        if op_name == "reduce_sum":
            red = x_grid.sum(axis=0, keepdims=True)
        elif op_name == "reduce_mean":
            red = x_grid.mean(axis=0, keepdims=True)
        elif op_name == "reduce_max":
            red = x_grid.max(axis=0, keepdims=True)
        elif op_name == "reduce_min":
            red = x_grid.min(axis=0, keepdims=True)
        else:
            red = None
        if red is not None:
            expected = np.tile(red, (C, 1)).ravel()
        else:
            expected = None

    label = f"{op_name} axis={'spatial' if axis == -1 else 'channel'}"
    return probe_custom(label, mil, [x_f16], C * S, expected, atol)


# ── Intermediate op discovery ─────────────────────────────────────────────────

def probe_intermediate(
    op_name: str,
    *,
    op_args: str = "",
    out_shape: Optional[tuple] = None,
    C: int = _C,
    S: int = _S,
) -> ProbeResult:
    """
    Test an op as an intermediate node between two identity ops.

    This discovers ops like reshape/slice/tile that the ANE compiler rejects
    when standalone but accepts as intermediates in a multi-op graph.

    The pattern is:
        identity(input) → target_op(...) → identity(output)

    Args:
        op_name:  MIL op name.
        op_args:  Additional MIL arguments string (e.g. ' ,axes=...'). Prepend comma.
        out_shape: Output shape of target_op as [C_out, S_out]. Defaults to same as input.
        C, S:     Input tensor shape.

    Returns:
        ProbeResult (no numerical check — compile-only).
    """
    oc, os_ = out_shape if out_shape else (C, S)

    body = (
        f'        tensor<fp16, [1,{C},1,{S}]> mid0 = '
        f'identity(x=a_input0)[name=string("mid0")];\n'
        f'        tensor<fp16, [1,{oc},1,{os_}]> mid1 = '
        f'{op_name}(x=mid0{op_args})[name=string("mid1")];\n'
        f'        tensor<fp16, [1,{oc},1,{os_}]> z_output0 = '
        f'identity(x=mid1)[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(C, S), "z_output0")
    return probe_custom(f"{op_name} (intermediate)", mil,
                        [np.zeros(C * S, dtype=np.float16)], oc * os_)


# ── Composite pattern probes ──────────────────────────────────────────────────

def probe_rmsnorm() -> ProbeResult:
    """RMSNorm: x * rsqrt(mean(x²) + eps)."""
    body = (
        '        tensor<fp16, [1,32,1,32]> sq = mul(x=a_input0, y=a_input0)[name=string("sq")];\n'
        '        tensor<int32, [1]> rax = const()[name=string("rax"), val=tensor<int32, [1]>([-1])];\n'
        '        bool kd = const()[name=string("kd"), val=bool(true)];\n'
        '        tensor<fp16, [1,32,1,1]> ms = reduce_mean(x=sq, axes=rax, keep_dims=kd)[name=string("ms")];\n'
        '        fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];\n'
        '        tensor<fp16, [1,32,1,1]> mse = add(x=ms, y=eps)[name=string("mse")];\n'
        '        fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];\n'
        '        tensor<fp16, [1,32,1,1]> rrms = pow(x=mse, y=nhalf)[name=string("rrms")];\n'
        '        tensor<fp16, [1,32,1,32]> z_output0 = mul(x=a_input0, y=rrms)[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(), "z_output0")

    x_f32 = _linspace(-3.0, 3.0)
    x_f16 = x_f32.astype(np.float16)
    x_grid = x_f32.reshape(_C, _S)
    ms = (x_grid ** 2).mean(axis=1, keepdims=True)
    rrms = 1.0 / np.sqrt(ms + 1e-5)
    expected = (x_grid * rrms).ravel()

    return probe_custom("RMSNorm", mil, [x_f16], _N, expected, atol=0.1)


def probe_layer_norm() -> ProbeResult:
    """LayerNorm along spatial axis."""
    body = (
        '        tensor<int32, [1]> nax = const()[name=string("nax"), val=tensor<int32, [1]>([3])];\n'
        '        fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];\n'
        '        tensor<fp16, [1,32,1,32]> z_output0 = '
        'layer_norm(x=a_input0, axes=nax, epsilon=eps)[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(), "z_output0")

    x_f32 = _linspace(-3.0, 3.0)
    x_f16 = x_f32.astype(np.float16)
    x_grid = x_f32.reshape(_C, _S)
    mean = x_grid.mean(axis=1, keepdims=True)
    var = x_grid.var(axis=1, keepdims=True)
    expected = ((x_grid - mean) / np.sqrt(var + 1e-5)).ravel()

    return probe_custom("layer_norm", mil, [x_f16], _N, expected, atol=0.1)


def probe_softmax() -> ProbeResult:
    """Softmax over spatial axis."""
    body = (
        '        int32 ax = const()[name=string("ax"), val=int32(-1)];\n'
        '        tensor<fp16, [1,32,1,32]> z_output0 = '
        'softmax(x=a_input0, axis=ax)[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(), "z_output0")

    x_f32 = _linspace(-2.0, 2.0)
    x_f16 = x_f32.astype(np.float16)
    x_grid = x_f32.reshape(_C, _S)
    mv = x_grid.max(axis=1, keepdims=True)
    e = np.exp(x_grid - mv)
    expected = (e / e.sum(axis=1, keepdims=True)).ravel()

    return probe_custom("softmax", mil, [x_f16], _N, expected, atol=0.01)


def probe_int8_round_trip() -> ProbeResult:
    """fp16 → cast int8 → cast fp16: INT8 storage verification."""
    body = (
        '        tensor<int8, [1,32,1,32]> q = '
        'cast(x=a_input0, dtype=string("int8"))[name=string("q")];\n'
        '        tensor<fp16, [1,32,1,32]> z_output0 = '
        'cast(x=q, dtype=string("fp16"))[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(), "z_output0")

    x_f32 = np.array([i % 256 - 128 for i in range(_N)], dtype=np.float32)
    x_f16 = x_f32.astype(np.float16)
    expected = np.array([np.int8(int(v)).astype(np.float32) for v in x_f32])

    return probe_custom("INT8 round-trip", mil, [x_f16], _N, expected, atol=1.0)


def probe_int8_quant_dequant() -> ProbeResult:
    """Full affine quantize → dequantize through INT8."""
    body = (
        '        fp16 inv_scale = const()[name=string("inv_scale"), val=fp16(10.0)];\n'
        '        fp16 zp = const()[name=string("zp"), val=fp16(0.0)];\n'
        '        tensor<fp16, [1,32,1,32]> scaled = mul(x=a_input0, y=inv_scale)[name=string("scaled")];\n'
        '        tensor<fp16, [1,32,1,32]> shifted = add(x=scaled, y=zp)[name=string("shifted")];\n'
        '        tensor<fp16, [1,32,1,32]> rounded = round(x=shifted)[name=string("rounded")];\n'
        '        fp16 lo = const()[name=string("lo"), val=fp16(-128.0)];\n'
        '        fp16 hi = const()[name=string("hi"), val=fp16(127.0)];\n'
        '        tensor<fp16, [1,32,1,32]> clamped = clip(x=rounded, alpha=lo, beta=hi)[name=string("clamped")];\n'
        '        tensor<int8, [1,32,1,32]> quantized = cast(x=clamped, dtype=string("int8"))[name=string("quantized")];\n'
        '        tensor<fp16, [1,32,1,32]> back = cast(x=quantized, dtype=string("fp16"))[name=string("back")];\n'
        '        tensor<fp16, [1,32,1,32]> unshifted = sub(x=back, y=zp)[name=string("unshifted")];\n'
        '        fp16 scale = const()[name=string("scale"), val=fp16(0.1)];\n'
        '        tensor<fp16, [1,32,1,32]> z_output0 = mul(x=unshifted, y=scale)[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig1(), "z_output0")

    x_f32 = _linspace(-12.7, 12.7)
    x_f16 = x_f32.astype(np.float16)
    expected = np.array([np.int8(int(np.clip(round(v * 10), -128, 127))) * 0.1
                         for v in x_f32], dtype=np.float32)

    return probe_custom("INT8 quant→dequant", mil, [x_f16], _N, expected, atol=0.15)


def probe_concat() -> ProbeResult:
    """Concat two [1,32,1,32] tensors along channel axis → [1,64,1,32]."""
    body = (
        '        int32 cax = const()[name=string("cax"), val=int32(1)];\n'
        '        bool cid = const()[name=string("cid"), val=bool(false)];\n'
        '        tensor<fp16, [1,64,1,32]> z_output0 = '
        'concat(axis=cax, interleave=cid, values=(a_input0, a_input1))[name=string("z_output0")];'
    )
    mil = mil_program(body, _sig2(), "z_output0")

    a_f32 = _linspace(0.0, 5.0)
    b_f32 = _linspace(-5.0, 0.0)
    expected = np.concatenate([a_f32, b_f32])

    try:
        import ane
    except ImportError as e:
        return ProbeResult("concat", False, None, None, 0.05, 0.0, None, str(e))

    t0 = time.perf_counter()
    try:
        prog = ane.compile_mil(mil)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except RuntimeError as e:
        compile_ms = (time.perf_counter() - t0) * 1000.0
        return ProbeResult("concat", False, None, None, 0.05, compile_ms, None, str(e))

    t1 = time.perf_counter()
    try:
        (out,) = prog.run(
            [a_f32.astype(np.float16), b_f32.astype(np.float16)],
            [64 * _S],
        )
        eval_ms = (time.perf_counter() - t1) * 1000.0
    except RuntimeError as e:
        eval_ms = (time.perf_counter() - t1) * 1000.0
        return ProbeResult("concat", True, False, None, 0.05, compile_ms, eval_ms, str(e))

    passed, max_err = check("concat", out, expected, 0.05)
    return ProbeResult("concat", True, passed, max_err, 0.05, compile_ms, eval_ms)


def probe_matmul() -> ProbeResult:
    """Matmul: identity (I @ B = B) via matmul op on [1,32,32] tensors."""
    body = (
        '        bool bF = const()[name=string("bF"), val=bool(false)];\n'
        '        tensor<fp16, [1,32,32]> z_output0 = '
        'matmul(x=a_input0, y=a_input1, transpose_x=bF, transpose_y=bF)[name=string("z_output0")];'
    )
    sig = (
        "tensor<fp16, [1,32,32]> a_input0, "
        "tensor<fp16, [1,32,32]> a_input1"
    )
    mil = mil_program(body, sig, "z_output0")

    identity = np.eye(32, dtype=np.float32)
    b_f32 = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) * 0.01

    try:
        import ane
    except ImportError as e:
        return ProbeResult("matmul", False, None, None, 0.05, 0.0, None, str(e))

    t0 = time.perf_counter()
    try:
        prog = ane.compile_mil(mil)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except RuntimeError as e:
        compile_ms = (time.perf_counter() - t0) * 1000.0
        return ProbeResult("matmul", False, None, None, 0.05, compile_ms, None, str(e))

    t1 = time.perf_counter()
    try:
        (out,) = prog.run(
            [identity.astype(np.float16), b_f32.astype(np.float16)],
            [32 * 32],
        )
        eval_ms = (time.perf_counter() - t1) * 1000.0
    except RuntimeError as e:
        eval_ms = (time.perf_counter() - t1) * 1000.0
        return ProbeResult("matmul", True, False, None, 0.05, compile_ms, eval_ms, str(e))

    passed, max_err = check("matmul", out, b_f32.ravel(), 0.05)
    return ProbeResult("matmul", True, passed, max_err, 0.05, compile_ms, eval_ms)


# ── Parameter space exploration ───────────────────────────────────────────────

# Pre-defined epsilon values to probe for ops that require an epsilon parameter.
# ironmill discovered that rsqrt/log/inverse require epsilon; this registry
# explores which values the ANE compiler accepts and how they affect precision.
#
# The value 0x1.0cp-17 ≈ 7.63e-6 is what ironmill found works. We probe a
# range around it to characterize the full valid domain.
_EPSILON_VALUES: list[tuple[str, str]] = [
    # (label, fp16 hex literal)
    ("eps=0 (no epsilon)",    "fp16(0x0p+0)"),       # expected: rejected by ANE
    ("eps=min_pos",           "fp16(0x1p-24)"),       # fp16 smallest positive normal
    ("eps=1e-7",              "fp16(0x1.bp-24)"),
    ("eps=0x1.0cp-17 (ironmill)", "fp16(0x1.0cp-17)"),  # ironmill's known-working value
    ("eps=1e-5",              "fp16(0x1.4p-17)"),
    ("eps=1e-4",              "fp16(0x1.ap-14)"),
    ("eps=1e-3",              "fp16(0x1.06p-10)"),
    ("eps=0.01",              "fp16(0x1.47p-7)"),
    ("eps=0.1",               "fp16(0x1.99p-4)"),
    ("eps=1.0",               "fp16(0x1p+0)"),        # expected: large numerical error
]


def _eps_body(op_name: str, eps_literal: str, C: int = _C, S: int = _S) -> str:
    """Build a MIL body for an epsilon-bearing unary op."""
    return (
        f'        fp16 eps = const()[name=string("eps"), val={eps_literal}];\n'
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
        f'{op_name}(x=a_input0, epsilon=eps)[name=string("z_output0")];'
    )


# Registry of pre-defined parameter space explorations.
# Key: op name. Value: list of (label, MIL body) pairs.
PARAM_SPACES: dict[str, list[tuple[str, str]]] = {
    "rsqrt": [
        (label, _eps_body("rsqrt", eps_literal))
        for label, eps_literal in _EPSILON_VALUES
    ],
    "log": [
        (label, _eps_body("log", eps_literal))
        for label, eps_literal in _EPSILON_VALUES
    ],
    "inverse": [
        (label, _eps_body("inverse", eps_literal))
        for label, eps_literal in _EPSILON_VALUES
    ],
    # pow: probe scalar vs tensor exponent, and common exponents used in ML
    "pow": [
        ("pow scalar -0.5 (rsqrt)",
         '        fp16 sc = const()[name=string("sc"), val=fp16(-0.5)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=sc)[name=string("z_output0")];'),
        ("pow scalar 0.5 (sqrt)",
         '        fp16 sc = const()[name=string("sc"), val=fp16(0.5)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=sc)[name=string("z_output0")];'),
        ("pow scalar 2.0 (square)",
         '        fp16 sc = const()[name=string("sc"), val=fp16(2.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=sc)[name=string("z_output0")];'),
        ("pow scalar 3.0",
         '        fp16 sc = const()[name=string("sc"), val=fp16(3.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=sc)[name=string("z_output0")];'),
        ("pow tensor exponent",
         '        tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=a_input1)[name=string("z_output0")];'),
    ],
    # clip: probe different alpha/beta combinations
    "clip": [
        ("clip [-1, 1]",
         '        fp16 lo = const()[name=string("lo"), val=fp16(-1.0)];\n'
         '        fp16 hi = const()[name=string("hi"), val=fp16(1.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, alpha=lo, beta=hi)[name=string("z_output0")];'),
        ("clip [0, 6] (relu6)",
         '        fp16 lo = const()[name=string("lo"), val=fp16(0.0)];\n'
         '        fp16 hi = const()[name=string("hi"), val=fp16(6.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, alpha=lo, beta=hi)[name=string("z_output0")];'),
        ("clip no alpha (lower unbounded)",
         '        fp16 hi = const()[name=string("hi"), val=fp16(1.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, beta=hi)[name=string("z_output0")];'),
        ("clip no beta (upper unbounded)",
         '        fp16 lo = const()[name=string("lo"), val=fp16(0.0)];\n'
         '        tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, alpha=lo)[name=string("z_output0")];'),
    ],
    # leaky_relu: probe alpha parameter
    "leaky_relu": [
        (f"leaky_relu alpha={a}",
         f'        fp16 alpha = const()[name=string("alpha"), val=fp16({a})];\n'
         f'        tensor<fp16, [1,32,1,32]> z_output0 = '
         f'leaky_relu(x=a_input0, alpha=alpha)[name=string("z_output0")];')
        for a in ("0.01", "0.1", "0.2", "0.5")
    ],
    # softmax: probe axis parameter
    "softmax": [
        (f"softmax axis={ax}",
         f'        int32 ax = const()[name=string("ax"), val=int32({ax})];\n'
         f'        tensor<fp16, [1,32,1,32]> z_output0 = '
         f'softmax(x=a_input0, axis=ax)[name=string("z_output0")];')
        for ax in (-1, 1, 3)
    ],
}


def explore_op_params(
    op_name: str,
    param_variations: Optional[list[str]] = None,
    *,
    C: int = _C,
    S: int = _S,
) -> list[ProbeResult]:
    """
    Test an op with different optional parameter combinations.

    If ``param_variations`` is None, uses the pre-defined registry in
    ``PARAM_SPACES``. The highest-value targets are ``rsqrt``, ``log``, and
    ``inverse`` — ironmill discovered these require an epsilon parameter but
    didn't probe which values the ANE compiler accepts.

    Args:
        op_name:          MIL op name (e.g. "rsqrt", "log", "pow").
        param_variations: List of MIL body strings. If None, uses PARAM_SPACES.
        C, S:             Tensor shape.

    Returns:
        List of ProbeResult, one per variation.

    Example::

        # Characterise the epsilon domain for rsqrt
        results = explore_op_params("rsqrt")
        print_report(results)
    """
    if param_variations is None:
        if op_name not in PARAM_SPACES:
            raise ValueError(
                f"No pre-defined parameter space for '{op_name}'. "
                f"Available: {list(PARAM_SPACES)}. "
                f"Pass param_variations=[...] to supply your own."
            )
        variations = PARAM_SPACES[op_name]
    else:
        variations = [(f"{op_name} (variant {i})", body)
                      for i, body in enumerate(param_variations)]

    # Use positive inputs for rsqrt/log/inverse; general for others
    if op_name in ("rsqrt", "log", "inverse"):
        x_f16 = _linspace(0.1, 10.0).astype(np.float16)
        ref_fns = {
            "rsqrt":   lambda x: 1.0 / np.sqrt(x),
            "log":     np.log,
            "inverse": lambda x: 1.0 / x,
        }
        expected_f32 = ref_fns[op_name](_linspace(0.1, 10.0))
    elif op_name == "pow":
        x_f16 = _linspace(0.1, 5.0).astype(np.float16)
        expected_f32 = None  # varies by variant
    else:
        x_f16 = _linspace(-2.0, 2.0).astype(np.float16)
        expected_f32 = None

    results = []
    for label, body in variations:
        mil = mil_program(body, _sig1(C, S), "z_output0")
        # For ops with a tensor exponent (pow), pass two inputs
        if "tensor exponent" in label:
            exp_f16 = np.full(_N, 2.0, dtype=np.float16)
            exp_expected = _linspace(0.1, 5.0) ** 2.0
            results.append(probe_custom(
                label, mil,
                [x_f16, exp_f16], C * S,
                expected_f32 if expected_f32 is not None else exp_expected,
                atol=0.5,
            ))
        else:
            results.append(probe_custom(
                label, mil, [x_f16], C * S,
                expected_f32, atol=0.1,
            ))
    return results


def scan_param_spaces() -> list[ProbeResult]:
    """
    Run all pre-defined parameter space explorations.

    Probes epsilon domains for rsqrt/log/inverse, pow exponent variants,
    clip bounds, leaky_relu alpha, and softmax axis. Use this to characterise
    which parameter combinations the local ANE generation accepts.
    """
    print("── Parameter space exploration ───────────────────────────────────")
    results = []
    for op_name in PARAM_SPACES:
        print(f"  Exploring {op_name} ({len(PARAM_SPACES[op_name])} variants)...")
        results += explore_op_params(op_name)
    return results


# ── Op name fuzzing ───────────────────────────────────────────────────────────

# Op names to try when fuzzing for undocumented or generation-specific ops.
# Drawn from CoreML op taxonomy, ONNX, and common ML framework primitives.
CANDIDATE_NAMES: list[str] = [
    # Activations not in _UNARY_OPS
    "gelu", "gelu_erf", "leaky_relu", "prelu", "elu", "celu", "selu",
    "hardswish", "hardsigmoid", "hard_sigmoid", "hardtanh", "mish",
    "swish", "relu6", "crelu", "threshold_relu",
    # Unary math
    "log2", "log10", "expm1", "log1p", "cbrt", "lgamma", "digamma",
    "erfc", "erfinv",
    # Rounding
    "trunc", "fix", "clip_by_value",
    # Trig
    "atan2", "tan2", "sinc", "cis",
    # Hyperbolic
    "tanh_approx", "asinh", "acosh", "atanh",
    # Shape ops (expected to fail standalone but documented here)
    "reshape", "flatten", "expand_dims", "squeeze",
    "transpose", "permute",
    "tile", "broadcast_to",
    "slice_by_size", "gather", "scatter",
    "pad", "reflect_pad", "edge_pad",
    "depth_to_space", "space_to_depth",
    "pixel_shuffle",
    # Normalization
    "instance_norm", "group_norm", "batch_norm",
    # Pooling
    "avg_pool", "max_pool", "global_avg_pool",
    # Linear algebra
    "l2_normalize", "normalize",
    # Misc
    "where", "cond", "cumsum", "diff",
    "topk", "sort", "argsort",
    "one_hot", "embedding",
    "clip_gradient",
    "stop_gradient",
    "is_nan", "is_inf",
    "nan_to_num",
]


def explore_names(
    candidates: list[str] = CANDIDATE_NAMES,
    *,
    C: int = _C,
    S: int = _S,
) -> list[ProbeResult]:
    """
    Attempt to compile each candidate op name as a standalone unary op.

    Ops that compile are documented as newly discovered. Many will fail —
    that's expected. The interesting results are the unexpected successes.

    Args:
        candidates: List of op names to try.
        C, S:       Tensor shape.

    Returns:
        List of ProbeResult, one per candidate.
    """
    results = []
    x_f16 = _linspace(-1.0, 1.0).astype(np.float16)
    for op_name in candidates:
        body = _body1(op_name, C=C, S=S)
        mil = mil_program(body, _sig1(C, S), "z_output0")
        print(f"  Probing {op_name}...", end=" ", flush=True)
        r = probe_custom(op_name, mil, [x_f16], C * S)
        if not r.compiled:
            print("compile failed")
        else:
            print(f"COMPILED (no ref check)")
        results.append(r)
    return results


# ── Scan functions ────────────────────────────────────────────────────────────

def scan_unary(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """Scan all known unary ops."""
    print("── Unary ops ─────────────────────────────────────────────────────")
    results = []
    for op_name in _UNARY_OPS:
        print(f"  Testing {op_name}...")
        results.append(probe_unary(op_name, C=C, S=S))
    return results


def scan_binary(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """Scan all known binary and comparison ops."""
    print("── Binary ops ────────────────────────────────────────────────────")
    results = []
    for op_name in {**_BINARY_OPS, **_COMPARISON_OPS}:
        print(f"  Testing {op_name}...")
        results.append(probe_binary(op_name, C=C, S=S))
    return results


def scan_reductions(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """Scan reduction ops along both axes."""
    print("── Reductions ────────────────────────────────────────────────────")
    ops = ["reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
           "reduce_l2_norm", "reduce_log_sum_exp"]
    results = []
    for op_name in ops:
        for axis in (-1, 1):
            if op_name == "reduce_l2_norm" and axis == 1:
                continue  # skip: only well-defined on spatial axis for now
            if op_name == "reduce_log_sum_exp" and axis == 1:
                continue
            print(f"  Testing {op_name} axis={'spatial' if axis == -1 else 'channel'}...")
            results.append(probe_reduction(op_name, axis=axis, C=C, S=S))
    return results


def scan_composite() -> list[ProbeResult]:
    """Scan composite patterns used in real inference pipelines."""
    print("── Composite patterns ────────────────────────────────────────────")
    fns = [
        ("RMSNorm",          probe_rmsnorm),
        ("layer_norm",       probe_layer_norm),
        ("softmax",          probe_softmax),
        ("concat",           probe_concat),
        ("matmul",           probe_matmul),
        ("INT8 round-trip",  probe_int8_round_trip),
        ("INT8 quant→dequant", probe_int8_quant_dequant),
    ]
    results = []
    for name, fn in fns:
        print(f"  Testing {name}...")
        results.append(fn())
    return results


def scan_intermediate(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """
    Test ops as intermediate nodes (not standalone).

    These are ops the ANE compiler typically rejects as the sole op but
    accepts inside a multi-op graph. Includes shape ops (reshape, tile, etc.)
    that ironmill documents as intermediate-only.
    """
    print("── Intermediate-only ops ─────────────────────────────────────────")

    candidates = [
        # Shape ops with same-shape variants
        ("reshape",        ''),
        ("tile",           ''),
        ("transpose",      ''),
        ("slice_by_index", ''),
        ("expand_dims",    ''),
        ("squeeze",        ''),
        ("flatten",        ''),
        ("cast_fp16_int",  ''),
    ]
    results = []
    for op_name, extra_args in candidates:
        print(f"  Probing {op_name} (intermediate)...")
        results.append(probe_intermediate(op_name, op_args=extra_args, C=C, S=S))
    return results


def scan_all() -> list[ProbeResult]:
    """Run all scans and return combined results."""
    all_results: list[ProbeResult] = []
    all_results += scan_unary()
    all_results += scan_binary()
    all_results += scan_reductions()
    all_results += scan_composite()
    return all_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[ProbeResult]) -> None:
    """Print a summary table of probe results."""
    compile_pass = sum(1 for r in results if r.compiled)
    eval_pass    = sum(1 for r in results if r.passed)
    eval_fail    = sum(1 for r in results if r.passed is False)
    compile_fail = sum(1 for r in results if not r.compiled)
    total = len(results)

    print()
    print("═" * 67)
    print("  PROBE RESULTS")
    print("═" * 67)
    print(f"  Compiled:       {compile_pass}/{total}")
    print(f"  Eval correct:   {eval_pass}/{compile_pass}")
    if compile_fail:
        print(f"  Compile failures: {compile_fail}")
    if eval_fail:
        print(f"  Eval failures:    {eval_fail}")
    print()

    if compile_fail:
        print("  Compile failures:")
        for r in results:
            if not r.compiled:
                msg = f" — {r.error}" if r.error else ""
                print(f"    • {r.name}{msg}")
        print()

    if eval_fail:
        print("  Eval failures:")
        for r in results:
            if r.passed is False:
                print(f"    • {r.name}  max_err={r.max_err:.4f}  atol={r.atol}")
        print()


def export_json(results: list[ProbeResult], path: str) -> None:
    """
    Export probe results as JSON for crowdsourced ANE research.

    The JSON includes chip generation info, macOS version, libane version,
    and per-op results. Share at: https://github.com/amirani-labs/libane/discussions
    """
    try:
        import ane as _ane
        libane_version = _ane.version()
    except Exception:
        libane_version = "unknown"

    chip = _chip_info()
    payload = {
        "libane_version": libane_version,
        "platform": {
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.version(),
            "machine": platform.machine(),
            "chip": chip,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [
            {
                "name": r.name,
                "compiled": r.compiled,
                "passed": r.passed,
                "max_err": r.max_err,
                "atol": r.atol,
                "compile_ms": round(r.compile_ms, 2),
                "eval_ms": round(r.eval_ms, 2) if r.eval_ms is not None else None,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Exported {len(results)} results → {path}")


def _chip_info() -> str:
    """Best-effort Apple Silicon chip identification."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        return out
    except Exception:
        return platform.processor() or "unknown"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ane.probe",
        description="ANE op probe harness — test MIL ops on the local ANE.",
    )
    parser.add_argument("--unary",      action="store_true", help="Scan unary ops")
    parser.add_argument("--binary",     action="store_true", help="Scan binary/comparison ops")
    parser.add_argument("--reductions", action="store_true", help="Scan reduction ops")
    parser.add_argument("--composite",  action="store_true", help="Scan composite patterns")
    parser.add_argument("--intermediate", action="store_true",
                        help="Discover intermediate-only ops")
    parser.add_argument("--explore",    action="store_true",
                        help="Fuzz unknown op names")
    parser.add_argument("--all",        action="store_true",
                        help="Run all scans (unary+binary+reductions+composite)")
    parser.add_argument("--export",     metavar="FILE",
                        help="Export results to JSON file")
    args = parser.parse_args()

    try:
        import ane as _ane
        if not _ane.available():
            print("ANE not available on this machine. Exiting.")
            sys.exit(1)
        print(f"libane {_ane.version()} — ANE available ({_chip_info()})\n")
    except ImportError:
        print("Error: ane module not found. Build with: pip install -e bindings/python/")
        sys.exit(1)

    run_all = args.all or not any([
        args.unary, args.binary, args.reductions, args.composite,
        args.intermediate, args.explore,
    ])

    all_results: list[ProbeResult] = []

    if run_all or args.unary:
        all_results += scan_unary()
    if run_all or args.binary:
        all_results += scan_binary()
    if run_all or args.reductions:
        all_results += scan_reductions()
    if run_all or args.composite:
        all_results += scan_composite()
    if args.intermediate:
        all_results += scan_intermediate()
    if args.explore:
        all_results += explore_names()

    print_report(all_results)

    if args.export:
        export_json(all_results, args.export)


if __name__ == "__main__":
    _cli()
