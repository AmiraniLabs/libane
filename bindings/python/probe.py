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
import re
import sys
import time
from dataclasses import dataclass
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
_BUILD_INFO_FIELDS = {
    "coremlc-component-MIL": "3510.2.1",
    "coremlc-version": "3505.4.1",
    "coremltools-component-milinternal": "",
    "coremltools-version": "9.0",
}

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


def mil_program_with_build_info(
    body: str,
    inputs_sig: str,
    output: str,
    build_info_fields: dict[str, str],
) -> str:
    """Build MIL text with an explicit buildInfo dictionary."""
    items = ", ".join(f'{{"{k}", "{v}"}}' for k, v in build_info_fields.items())
    build_info = f'[buildInfo = dict<string, string>({{{items}}})]'
    return (
        f"program(1.3)\n"
        f"{build_info}\n"
        f"{{\n"
        f"    func main<ios18>({inputs_sig}) {{\n"
        f"{body}\n"
        f"    }} -> ({output});\n"
        f"}}"
    )


def _mil_ops_from_text(text: str) -> list[str]:
    """Extract ordered MIL op names from textual Program form."""
    return re.findall(r"=\s*([a-zA-Z_][a-zA-Z0-9_]*)\(", text)


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
    expected_compile_fail: bool = False
    expected_eval_fail: bool = False
    note: Optional[str] = None
    failure_kind: Optional[str] = None
    raw_acceptance: Optional[bool] = None
    libane_lowered_support: Optional[bool] = None

    @property
    def status(self) -> str:
        if not self.compiled and self.expected_compile_fail:
            return "XFAIL_COMPILE"
        if self.passed is False and self.expected_eval_fail:
            return "XFAIL_EVAL"
        if self.failure_kind == "compile_reject":
            return "COMPILE_REJECT"
        if self.failure_kind == "runtime_reject":
            return "RUNTIME_REJECT"
        if self.failure_kind == "semantic_mismatch":
            return "SEMANTIC_MISMATCH"
        if self.failure_kind == "numeric_mismatch":
            return "NUMERIC_MISMATCH"
        if not self.compiled:
            return "COMPILE_FAIL"
        if self.passed is None:
            return "COMPILE_ONLY"
        return "PASS" if self.passed else "EVAL_FAIL"

    @property
    def support_level(self) -> str:
        if self.compiled and self.passed is True:
            return "EVAL_VERIFIED"
        if self.compiled and self.passed is None:
            return "COMPILE_ONLY"
        return "UNSUPPORTED"


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
            failure_kind="compile_reject",
            raw_acceptance=False,
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
            failure_kind="compile_reject",
            raw_acceptance=False,
        )

    if expected is None:
        return ProbeResult(
            name=name, compiled=True, passed=None, max_err=None,
            atol=atol, compile_ms=compile_ms, eval_ms=None,
            raw_acceptance=True,
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
            failure_kind="runtime_reject",
            raw_acceptance=True,
        )

    passed, max_err = check(name, out, expected, atol)
    return ProbeResult(
        name=name, compiled=True, passed=passed, max_err=max_err,
        atol=atol, compile_ms=compile_ms, eval_ms=eval_ms,
        failure_kind=None if passed else "numeric_mismatch",
        raw_acceptance=True,
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


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Reference round-half-away-from-zero semantics."""
    x32 = np.asarray(x, dtype=np.float32)
    return np.where(x32 >= 0.0, np.floor(x32 + 0.5), np.ceil(x32 - 0.5)).astype(np.float32)


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
        _round_half_away_from_zero,
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

# Known semantic divergence list. Keep empty unless a concrete known mismatch
# is intentionally tolerated for a scan.
_UNARY_EXPECTED_EVAL_FAILS: dict[str, str] = {}


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
        r = probe_unary(op_name, C=C, S=S)
        if r.passed is False and op_name in _UNARY_EXPECTED_EVAL_FAILS:
            r.expected_eval_fail = True
            r.note = _UNARY_EXPECTED_EVAL_FAILS[op_name]
            r.failure_kind = "semantic_mismatch"
        results.append(r)
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

    # Per ironmill + local probes, these are generally non-standalone on ANE:
    # compile may fail when probed in isolation, while still working as graph
    # intermediates in larger programs.
    expected_nonstandalone = {
        "reshape",
        "tile",
        "transpose",
        "slice_by_index",
        "expand_dims",
        "flatten",
        "cast_fp16_int",
    }

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
        r = probe_intermediate(op_name, op_args=extra_args, C=C, S=S)
        if not r.compiled and op_name in expected_nonstandalone:
            r.expected_compile_fail = True
            r.note = (
                "Expected: ANE often rejects this op as standalone probe; "
                "validated via intermediate usage in graph pipelines."
            )
        elif r.compiled and op_name in expected_nonstandalone:
            r.note = "Compiled as standalone on this hardware/OS (unexpected success)."
        results.append(r)
    return results


def scan_sin_cos_ranges(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """
    Probe sin/cos across multiple input ranges to detect range-dependent behavior.
    """
    print("── Sin/Cos range sweep ───────────────────────────────────────────")
    ranges = [
        ("small", -3.0, 3.0),
        ("medium", -32.0, 32.0),
        ("large", -256.0, 256.0),
        ("xlarge", -1024.0, 1024.0),
    ]
    results: list[ProbeResult] = []
    for op_name, ref, atol in (("sin", np.sin, 0.05), ("cos", np.cos, 0.05)):
        for label, lo, hi in ranges:
            name = f"{op_name} range={label} [{lo:g},{hi:g}]"
            print(f"  Testing {name}...")
            r = probe_unary(
                name,
                body=_body1(op_name, C=C, S=S),
                input_fn=lambda lo=lo, hi=hi: _linspace(lo, hi),
                expected_fn=ref,
                atol=atol,
                C=C,
                S=S,
            )
            results.append(r)
    return results


def scan_round_semantics(C: int = 16, S: int = 64) -> list[ProbeResult]:
    """
    Characterize ANE round() tie-breaking behavior on exact fp16 half-way inputs.

    Emits one summary ProbeResult plus one ProbeResult per tie input.
    """
    print("── Round semantics ────────────────────────────────────────────────")
    ties = np.array([
        -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5,
         0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,
    ], dtype=np.float32)
    n = C * S
    x = np.zeros(n, dtype=np.float16)
    x[:len(ties)] = ties.astype(np.float16)
    for i in range(len(ties), n):
        x[i] = np.float16((i % 11) - 5)

    mil = mil_program(_body1("round", C=C, S=S), _sig1(C, S), "z_output0")

    results: list[ProbeResult] = []
    t0 = time.perf_counter()
    try:
        import ane
        prog = ane.compile_mil(mil)
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except RuntimeError as e:
        compile_ms = (time.perf_counter() - t0) * 1000.0
        results.append(ProbeResult(
            "round semantics summary", False, None, None, 0.0, compile_ms, None,
            str(e), failure_kind="compile_reject"
        ))
        return results

    t1 = time.perf_counter()
    try:
        (out,) = prog.run([x], [n])
        eval_ms = (time.perf_counter() - t1) * 1000.0
    except RuntimeError as e:
        eval_ms = (time.perf_counter() - t1) * 1000.0
        results.append(ProbeResult(
            "round semantics summary", True, False, None, 0.0, compile_ms, eval_ms,
            str(e), failure_kind="runtime_reject"
        ))
        return results

    out32 = out.astype(np.float32)
    even = np.round(x.astype(np.float32))
    away = _round_half_away_from_zero(x.astype(np.float32))
    away_matches = 0
    even_matches = 0

    for i, v in enumerate(ties):
        ane_v = float(out32[i])
        even_v = float(even[i])
        away_v = float(away[i])
        if np.isclose(ane_v, away_v, atol=0.0):
            away_matches += 1
        if np.isclose(ane_v, even_v, atol=0.0):
            even_matches += 1
        results.append(ProbeResult(
            f"round_tie input={v:+.1f}",
            True,
            bool(np.isclose(ane_v, away_v, atol=0.0)),
            float(abs(ane_v - away_v)),
            0.0,
            compile_ms if i == 0 else 0.0,
            eval_ms if i == 0 else 0.0,
            note=f"ane={ane_v:.1f}, even={even_v:.1f}, away={away_v:.1f}",
            failure_kind=None if np.isclose(ane_v, away_v, atol=0.0) else "semantic_mismatch",
        ))

    results.insert(0, ProbeResult(
        "round semantics summary",
        True,
        away_matches == len(ties),
        float(len(ties) - away_matches),
        0.0,
        compile_ms,
        eval_ms,
        note=(
            f"away_matches={away_matches}/{len(ties)}, "
            f"even_matches={even_matches}/{len(ties)}"
        ),
        failure_kind=None if away_matches == len(ties) else "semantic_mismatch",
    ))
    return results


def scan_dependency_matrix(C: int = 16, S: int = 64) -> list[ProbeResult]:
    """
    Compile-only matrix for wrapper -> intermediate target combinations.
    """
    print("── Intermediate dependency matrix ─────────────────────────────────")

    wrappers = {
        "relu": "relu",
        "add": "add(x={x}, y={x})",
        "mul": "mul(x={x}, y={x})",
        "matmul": "matmul(x={x}, y=a_input1, transpose_x=tx, transpose_y=ty)",
    }
    targets = [
        ("reshape", "reshape(x={x}, shape=sh)",
         '        tensor<int32, [4]> sh = const()[name=string("sh"), val=tensor<int32, [4]>([1,16,1,64])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("slice_by_index", "slice_by_index(x={x}, begin=b, end=e, end_mask=em)",
         '        tensor<int32, [4]> b = const()[name=string("b"), val=tensor<int32, [4]>([0,0,0,0])];\n'
         '        tensor<int32, [4]> e = const()[name=string("e"), val=tensor<int32, [4]>([1,16,1,64])];\n'
         '        tensor<bool, [4]> em = const()[name=string("em"), val=tensor<bool, [4]>([false,false,false,false])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("tile", "tile(x={x}, reps=rp)",
         '        tensor<int32, [4]> rp = const()[name=string("rp"), val=tensor<int32, [4]>([1,1,1,1])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("transpose", "transpose(x={x}, perm=pm)",
         '        tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,2,3])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("expand_dims", "expand_dims(x={x}, axes=ax)",
         '        tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([0])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("flatten", "reshape(x={x}, shape=sh)",
         '        tensor<int32, [4]> sh = const()[name=string("sh"), val=tensor<int32, [4]>([1,16,1,64])];',
         "tensor<fp16, [1,16,1,64]>"),
        ("cast_fp16_int", 'cast(x={x}, dtype=string("int8"))',
         "", "tensor<int8, [1,16,1,64]>"),
    ]

    results: list[ProbeResult] = []
    id64 = np.eye(S, dtype=np.float16).reshape(1, S, 1, S).ravel()

    for w_name, w_expr in wrappers.items():
        for t_name, t_expr, t_consts, t_ty in targets:
            extra_sig = ""
            extra_consts = ""
            inputs = [np.zeros(C * S, dtype=np.float16)]
            if w_name == "matmul":
                extra_sig = ", tensor<fp16, [1,64,1,64]> a_input1"
                extra_consts = (
                    '        bool tx = const()[name=string("tx"), val=bool(false)];\n'
                    '        bool ty = const()[name=string("ty"), val=bool(false)];\n'
                )
                inputs = [np.zeros(C * S, dtype=np.float16), id64]

            post_input = "mid1"
            cast_back = ""
            if w_name == "matmul" and t_name == "cast_fp16_int":
                cast_back = (
                    '        tensor<fp16, [1,16,1,64]> mid1f = '
                    'cast(x=mid1, dtype=string("fp16"))[name=string("mid1f")];\n'
                )
                post_input = "mid1f"

            body = (
                f"{extra_consts}"
                f'        tensor<fp16, [1,{C},1,{S}]> mid0 = {w_expr.format(x="a_input0")}[name=string("mid0")];\n'
                f"{t_consts}\n"
                f'        {t_ty} mid1 = {t_expr.format(x="mid0")}[name=string("mid1")];\n'
                f"{cast_back}"
                f'        tensor<fp16, [1,16,1,64]> z_output0 = '
                f'{w_expr.format(x=post_input)}[name=string("z_output0")];'
            )
            sig = f"tensor<fp16, [1,{C},1,{S}]> a_input0{extra_sig}"
            mil = mil_program(body, sig, "z_output0")

            t0 = time.perf_counter()
            try:
                r = probe_custom(
                    f"wrapper={w_name} target={t_name}",
                    mil,
                    inputs,
                    16 * 64,
                    expected=None,
                    atol=0.0,
                )
                if not r.compiled and not r.failure_kind:
                    r.failure_kind = "compile_reject"
                results.append(r)
            except Exception as e:
                compile_ms = (time.perf_counter() - t0) * 1000.0
                results.append(ProbeResult(
                    f"wrapper={w_name} target={t_name}",
                    False, None, None, 0.0, compile_ms, None,
                    str(e), failure_kind="compile_reject"
                ))
    return results


def scan_gap_ops(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """
    Probe pending gap ops with strict labeling:
    - Eval-verified when a trusted CPU reference is available.
    - Compile-only otherwise.
    """
    print("── Gap ops ───────────────────────────────────────────────────────")
    results: list[ProbeResult] = []
    n = C * S

    # logical_and / logical_or / logical_xor (Ironmill-equivalent MIL)
    a = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)], dtype=np.float32)
    b = np.array([1.0 if i % 3 == 0 else 0.0 for i in range(n)], dtype=np.float32)
    a_f16 = a.astype(np.float16)
    b_f16 = b.astype(np.float16)
    logical_specs = [
        ("logical_and", lambda x, y: np.where((x != 0.0) & (y != 0.0), 1.0, 0.0).astype(np.float32)),
        ("logical_or",  lambda x, y: np.where((x != 0.0) | (y != 0.0), 1.0, 0.0).astype(np.float32)),
        ("logical_xor", lambda x, y: np.where((x != 0.0) ^ (y != 0.0), 1.0, 0.0).astype(np.float32)),
    ]
    for op, ref_fn in logical_specs:
        body = (
            f'        tensor<bool, [1,{C},1,{S}]> ba = cast(x=a_input0, dtype=string("bool"))[name=string("ba")];\n'
            f'        tensor<bool, [1,{C},1,{S}]> bb = cast(x=a_input1, dtype=string("bool"))[name=string("bb")];\n'
            f'        tensor<bool, [1,{C},1,{S}]> result = {op}(x=ba, y=bb)[name=string("result")];\n'
            f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=result, dtype=string("fp16"))[name=string("z_output0")];'
        )
        mil = mil_program(body, _sig2(C, S), "z_output0")
        results.append(probe_custom(op, mil, [a_f16, b_f16], n, ref_fn(a, b), atol=0.01))

    # reduce_prod (eval)
    body_prod = _reduction_body("reduce_prod", axis=-1, C=C, S=S)
    mil_prod = mil_program(body_prod, _sig1(C, S), "z_output0")
    x = np.linspace(0.95, 1.05, n, dtype=np.float32)  # avoid overflow/underflow
    x_f16 = x.astype(np.float16)
    x_grid = x.reshape(C, S)
    prod = x_grid.prod(axis=1, keepdims=True)
    expected_prod = np.tile(prod, (1, S)).ravel()
    results.append(probe_custom("reduce_prod axis=spatial", mil_prod, [x_f16], n, expected_prod, atol=0.2))

    # avg_pool / max_pool using kernel=[1,1] => identity reference if accepted
    pool_specs = [
        ("avg_pool",
         '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
         '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
         f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
         'avg_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"), '
         'exclude_padding_from_average=bool(true))[name=string("z_output0")];'),
        ("max_pool",
         '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
         '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
         f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
         'max_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"))'
         '[name=string("z_output0")];'),
    ]
    xp = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    for name, body in pool_specs:
        mil = mil_program(body, _sig1(C, S), "z_output0")
        results.append(probe_custom(name, mil, [xp.astype(np.float16)], n, xp, atol=0.05))

    # gather (attempt eval identity via arange indices on axis=3)
    gather_body = (
        f'        tensor<int32, [{S}]> idx = const()[name=string("idx"), val=tensor<int32, [{S}]>('
        f'[{",".join(str(i) for i in range(S))}])];\n'
        '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
        f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = gather(x=a_input0, indices=idx, axis=ax)'
        '[name=string("z_output0")];'
    )
    xg = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    results.append(probe_custom(
        "gather axis=3 identity", mil_program(gather_body, _sig1(C, S), "z_output0"),
        [xg.astype(np.float16)], n, xg, atol=0.05
    ))

    # scatter family (compile probes with plausible signatures; no trusted ref yet)
    scatter_templates = [
        ("scatter",
         '        tensor<int32, [1,32,1,32]> idx = const()[name=string("idx"), '
         'val=tensor<int32, [1,32,1,32]>(0)];\n'
         '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
         f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
         'scatter(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'),
        ("scatter_nd",
         '        tensor<int32, [1,32,1,32,1]> idx = const()[name=string("idx"), '
         'val=tensor<int32, [1,32,1,32,1]>(0)];\n'
         f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
         'scatter_nd(x=a_input0, indices=idx, updates=a_input1)[name=string("z_output0")];'),
        ("scatter_along_axis",
         '        tensor<int32, [1,32,1,32]> idx = const()[name=string("idx"), '
         'val=tensor<int32, [1,32,1,32]>(0)];\n'
         '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
         f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
         'scatter_along_axis(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'),
    ]
    xs = np.zeros(n, dtype=np.float16)
    us = np.ones(n, dtype=np.float16)
    for name, body in scatter_templates:
        results.append(probe_custom(
            name, mil_program(body, _sig2(C, S), "z_output0"),
            [xs, us], n, expected=None, atol=0.0
        ))

    return results


def scan_signature_sweep(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """
    Automated signature sweep for known gap ops.

    For each op:
    - Validate canonical front-end constructability via coremltools MIL builder.
    - Sweep ANE compile forms across variant signatures, wrappers, and buildInfo variants.
    """
    print("── Signature sweep ───────────────────────────────────────────────")
    n = C * S
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32).astype(np.float16)
    x2 = np.linspace(1.0, -1.0, n, dtype=np.float32).astype(np.float16)
    b1 = (np.arange(n) % 2 == 0)
    b2 = (np.arange(n) % 3 == 0)

    build_infos = {
        "default": _BUILD_INFO_FIELDS,
        "coremltools_8_legacy": {
            "coremlc-component-MIL": "3508.0.0",
            "coremlc-version": "3503.2.0",
            "coremltools-component-milinternal": "",
            "coremltools-version": "8.0",
        },
        "coremltools_7_legacy": {
            "coremlc-component-MIL": "3400.0.0",
            "coremlc-version": "3400.0.0",
            "coremltools-component-milinternal": "",
            "coremltools-version": "7.0",
        },
    }

    # Canonical front-end checks (coremltools builder only).
    cmt_status: dict[str, str] = {}
    try:
        import coremltools as ct
        from coremltools.converters.mil.mil import Builder as mb, types

        def _cmt_ok(fn) -> str:
            try:
                _ = fn()
                return "PASS"
            except Exception as e:  # pragma: no cover - environment-specific
                return f"FAIL: {e}"

        cmt_status["logical_and"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_and(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            ))
        ))
        cmt_status["logical_or"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_or(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            ))
        ))
        cmt_status["logical_xor"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_xor(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            ))
        ))
        cmt_status["reduce_prod"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.reduce_prod(
                x=a, axes=mb.const(val=np.array([3], dtype=np.int32)), keep_dims=True
            ))
        ))
        cmt_status["avg_pool"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.avg_pool(x=a, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid"))
        ))
        cmt_status["max_pool"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.max_pool(x=a, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid"))
        ))
        cmt_status["gather"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.gather(x=a, indices=mb.const(val=np.arange(S, dtype=np.int32)), axis=3))
        ))
        cmt_status["scatter"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter(
                data=a,
                indices=mb.const(val=np.zeros((1, C, 1, S), dtype=np.int32)),
                updates=u,
                axis=3,
                mode="update",
            ))
        ))
        cmt_status["scatter_nd"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter_nd(
                data=a,
                indices=mb.const(val=np.zeros((1, C, 1, S, 1), dtype=np.int32)),
                updates=u,
                mode="update",
            ))
        ))
        cmt_status["scatter_along_axis"] = _cmt_ok(lambda: str(
            mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter_along_axis(
                data=a,
                indices=mb.const(val=np.zeros((1, C, 1, S), dtype=np.int32)),
                updates=u,
                axis=3,
                mode="update",
            ))
        ))
    except Exception as e:  # pragma: no cover - environment-specific
        for op in (
            "logical_and", "logical_or", "logical_xor", "reduce_prod",
            "avg_pool", "max_pool", "gather", "scatter", "scatter_nd",
            "scatter_along_axis",
        ):
            cmt_status[op] = f"UNAVAILABLE: {e}"

    variants: list[tuple[str, str, str, list[np.ndarray], int]] = []
    # name, base_op, mil_body, inputs, output_numel
    for op in ("logical_and", "logical_or", "logical_xor"):
        variants.append((
            f"{op}/fp16_cast_bool",
            op,
            (
                f'        tensor<bool, [1,{C},1,{S}]> ba = cast(x=a_input0, dtype=string("bool"))[name=string("ba")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> bb = cast(x=a_input1, dtype=string("bool"))[name=string("bb")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> r = {op}(x=ba, y=bb)[name=string("r")];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=r, dtype=string("fp16"))[name=string("z_output0")];'
            ),
            [x, x2], n
        ))
        variants.append((
            f"{op}/bool_inputs",
            op,
            (
                f'        tensor<bool, [1,{C},1,{S}]> r = {op}(x=a_input0, y=a_input1)[name=string("r")];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=r, dtype=string("fp16"))[name=string("z_output0")];'
            ),
            [b1, b2], n
        ))
    for opname in ("reduce_prod", "reduce_product"):
        for axis in (-1, 1):
            variants.append((
                f"{opname}/axis{axis}",
                "reduce_prod",
                (
                    f'        tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([{axis}])];\n'
                    '        bool kd = const()[name=string("kd"), val=bool(true)];\n'
                    f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                    f'{opname}(x=a_input0, axes=ax, keep_dims=kd)[name=string("z_output0")];'
                ),
                [np.linspace(0.95, 1.05, n, dtype=np.float32).astype(np.float16)], n
            ))
    variants.extend([
        (
            "avg_pool/basic", "avg_pool",
            (
                '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
                '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'avg_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"), '
                'exclude_padding_from_average=bool(true))[name=string("z_output0")];'
            ),
            [x], n
        ),
        (
            "max_pool/basic", "max_pool",
            (
                '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
                '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'max_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"))'
                '[name=string("z_output0")];'
            ),
            [x], n
        ),
        (
            "gather/axis3", "gather",
            (
                f'        tensor<int32, [{S}]> idx = const()[name=string("idx"), val=tensor<int32, [{S}]>('
                f'[{",".join(str(i) for i in range(S))}])];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = gather(x=a_input0, indices=idx, axis=ax)'
                '[name=string("z_output0")];'
            ),
            [x], n
        ),
        (
            "scatter/basic", "scatter",
            (
                f'        tensor<int32, [1,{C},1,{S}]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S}]>(0)];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'
            ),
            [x, x2], n
        ),
        (
            "scatter_nd/basic", "scatter_nd",
            (
                f'        tensor<int32, [1,{C},1,{S},1]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S},1]>(0)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter_nd(x=a_input0, indices=idx, updates=a_input1)[name=string("z_output0")];'
            ),
            [x, x2], n
        ),
        (
            "scatter_along_axis/basic", "scatter_along_axis",
            (
                f'        tensor<int32, [1,{C},1,{S}]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S}]>(0)];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter_along_axis(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'
            ),
            [x, x2], n
        ),
    ])

    wrapper_exprs = {
        "none": "{x}",
        "relu": "relu(x={x})",
        "add": "add(x={x}, y={x})",
        "mul": "mul(x={x}, y={x})",
    }

    results: list[ProbeResult] = []
    for vname, op, base_body, inputs, out_numel in variants:
        for wrap_name, wrap_expr in wrapper_exprs.items():
            if wrap_name == "none":
                body = base_body
            else:
                body = (
                    f'        tensor<fp16, [1,{C},1,{S}]> pre = '
                    f'{wrap_expr.format(x="a_input0")}[name=string("pre")];\n'
                    + base_body.replace("a_input0", "pre", 1).replace("z_output0", "mid_out")
                    + "\n"
                    + f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                    + f'{wrap_expr.format(x="mid_out")}[name=string("z_output0")];'
                )

            sig = f"tensor<fp16, [1,{C},1,{S}]> a_input0"
            if len(inputs) == 2:
                if isinstance(inputs[1], np.ndarray) and inputs[1].dtype == np.bool_:
                    sig = (
                        f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
                        f"tensor<bool, [1,{C},1,{S}]> a_input1"
                    )
                else:
                    sig = (
                        f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
                        f"tensor<fp16, [1,{C},1,{S}]> a_input1"
                    )

            for build_tag, build_info in build_infos.items():
                mil = mil_program_with_build_info(body, sig, "z_output0", build_info)
                r = probe_custom(
                    f"sigsweep/{vname}/wrap={wrap_name}/build={build_tag}",
                    mil,
                    inputs,
                    out_numel,
                    expected=None,
                    atol=0.0,
                )
                cmt = cmt_status.get(op, "N/A")
                r.note = f"coremltools_signature={cmt}"
                if not r.compiled and not r.failure_kind:
                    r.failure_kind = "compile_reject"
                results.append(r)

    return results


def scan_coreml_differential(C: int = _C, S: int = _S) -> list[ProbeResult]:
    """
    Differential scan between coremltools MIL construction/conversion and libane ANE compile.

    For each gap op candidate:
    - Build canonical program via coremltools MIL builder.
    - Record frontend and converted (mlprogram/neuralnetwork) MIL op traces.
    - Compile equivalent hand-authored MIL through libane.
    """
    print("── CoreML differential ───────────────────────────────────────────")
    n = C * S
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32).astype(np.float16)
    x2 = np.linspace(1.0, -1.0, n, dtype=np.float32).astype(np.float16)

    cases = [
        {
            "name": "logical_and",
            "sig": _sig2(C, S),
            "body": (
                f'        tensor<bool, [1,{C},1,{S}]> ba = cast(x=a_input0, dtype=string("bool"))[name=string("ba")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> bb = cast(x=a_input1, dtype=string("bool"))[name=string("bb")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> r = logical_and(x=ba, y=bb)[name=string("r")];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=r, dtype=string("fp16"))[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
        {
            "name": "logical_or",
            "sig": _sig2(C, S),
            "body": (
                f'        tensor<bool, [1,{C},1,{S}]> ba = cast(x=a_input0, dtype=string("bool"))[name=string("ba")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> bb = cast(x=a_input1, dtype=string("bool"))[name=string("bb")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> r = logical_or(x=ba, y=bb)[name=string("r")];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=r, dtype=string("fp16"))[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
        {
            "name": "logical_xor",
            "sig": _sig2(C, S),
            "body": (
                f'        tensor<bool, [1,{C},1,{S}]> ba = cast(x=a_input0, dtype=string("bool"))[name=string("ba")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> bb = cast(x=a_input1, dtype=string("bool"))[name=string("bb")];\n'
                f'        tensor<bool, [1,{C},1,{S}]> r = logical_xor(x=ba, y=bb)[name=string("r")];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = cast(x=r, dtype=string("fp16"))[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
        {
            "name": "reduce_prod",
            "sig": _sig1(C, S),
            "body": (
                '        tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([3])];\n'
                '        bool kd = const()[name=string("kd"), val=bool(true)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'reduce_prod(x=a_input0, axes=ax, keep_dims=kd)[name=string("z_output0")];'
            ),
            "inputs": [np.linspace(0.95, 1.05, n, dtype=np.float32).astype(np.float16)],
            "out_numel": n,
        },
        {
            "name": "avg_pool",
            "sig": _sig1(C, S),
            "body": (
                '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
                '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'avg_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"), '
                'exclude_padding_from_average=bool(true))[name=string("z_output0")];'
            ),
            "inputs": [x],
            "out_numel": n,
        },
        {
            "name": "max_pool",
            "sig": _sig1(C, S),
            "body": (
                '        tensor<int32, [2]> ks = const()[name=string("ks"), val=tensor<int32, [2]>([1,1])];\n'
                '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'max_pool(x=a_input0, kernel_sizes=ks, strides=st, pad_type=string("valid"))'
                '[name=string("z_output0")];'
            ),
            "inputs": [x],
            "out_numel": n,
        },
        {
            "name": "gather",
            "sig": _sig1(C, S),
            "body": (
                f'        tensor<int32, [{S}]> idx = const()[name=string("idx"), val=tensor<int32, [{S}]>('
                f'[{",".join(str(i) for i in range(S))}])];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = gather(x=a_input0, indices=idx, axis=ax)'
                '[name=string("z_output0")];'
            ),
            "inputs": [x],
            "out_numel": n,
        },
        {
            "name": "scatter",
            "sig": (
                f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
                f"tensor<fp16, [1,{C},1,1,{C},1,{S}]> a_input1"
            ),
            "body": (
                f'        tensor<int32, [1,{C},1,{S}]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S}]>(0)];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
        {
            "name": "scatter_nd",
            "sig": (
                f"tensor<fp16, [1,{C},1,{S}]> a_input0, "
                f"tensor<fp16, [1,{C},1,{S},{C},1,{S}]> a_input1"
            ),
            "body": (
                f'        tensor<int32, [1,{C},1,{S},1]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S},1]>(0)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter_nd(x=a_input0, indices=idx, updates=a_input1)[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
        {
            "name": "scatter_along_axis",
            "sig": _sig2(C, S),
            "body": (
                f'        tensor<int32, [1,{C},1,{S}]> idx = const()[name=string("idx"), val=tensor<int32, [1,{C},1,{S}]>(0)];\n'
                '        int32 ax = const()[name=string("ax"), val=int32(3)];\n'
                f'        tensor<fp16, [1,{C},1,{S}]> z_output0 = '
                'scatter_along_axis(x=a_input0, indices=idx, updates=a_input1, axis=ax)[name=string("z_output0")];'
            ),
            "inputs": [x, x2],
            "out_numel": n,
        },
    ]

    cmt_info: dict[str, dict[str, object]] = {c["name"]: {} for c in cases}
    try:
        import coremltools as ct
        from coremltools.converters.mil.mil import Builder as mb, types

        idx = np.arange(S, dtype=np.int32)
        zeros_idx = np.zeros((1, C, 1, S), dtype=np.int32)
        zeros_idx_nd = np.zeros((1, C, 1, S, 1), dtype=np.int32)

        builders = {
            "logical_and": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_and(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            )),
            "logical_or": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_or(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            )),
            "logical_xor": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, b: mb.cast(
                x=mb.logical_xor(
                    x=mb.cast(x=a, dtype="bool"),
                    y=mb.cast(x=b, dtype="bool"),
                ),
                dtype="fp16",
            )),
            "reduce_prod": lambda: mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.reduce_prod(
                x=a,
                axes=mb.const(val=np.array([3], dtype=np.int32)),
                keep_dims=True,
            )),
            "avg_pool": lambda: mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.avg_pool(
                x=a, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid"
            )),
            "max_pool": lambda: mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.max_pool(
                x=a, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid"
            )),
            "gather": lambda: mb.program(
                input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )(lambda a: mb.gather(x=a, indices=mb.const(val=idx), axis=3)),
            "scatter": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, 1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter(
                data=a, indices=mb.const(val=zeros_idx), updates=u, axis=3, mode="update"
            )),
            "scatter_nd": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter_nd(
                data=a, indices=mb.const(val=zeros_idx_nd), updates=u, mode="update"
            )),
            "scatter_along_axis": lambda: mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                    mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16),
                ],
                opset_version=ct.target.iOS18,
            )(lambda a, u: mb.scatter_along_axis(
                data=a, indices=mb.const(val=zeros_idx), updates=u, axis=3, mode="update"
            )),
        }

        for case in cases:
            name = case["name"]
            info: dict[str, object] = {}
            try:
                frontend_prog = builders[name]()
                frontend_txt = str(frontend_prog)
                info["frontend_status"] = "PASS"
                info["frontend_ops"] = _mil_ops_from_text(frontend_txt)
            except Exception as e:
                info["frontend_status"] = f"FAIL: {e}"
                info["frontend_ops"] = []
                cmt_info[name] = info
                continue

            try:
                mlp = ct.convert(
                    frontend_prog,
                    convert_to="mlprogram",
                    minimum_deployment_target=ct.target.iOS18,
                    skip_model_load=True,
                )
                lowered = str(getattr(mlp, "_mil_program", ""))
                info["mlprogram_status"] = "PASS"
                info["mlprogram_ops"] = _mil_ops_from_text(lowered)
            except Exception as e:
                info["mlprogram_status"] = f"FAIL: {e}"
                info["mlprogram_ops"] = []

            # At iOS18 target, CoreML conversion path is ML Program; NN backend is
            # not comparable and rejected by coremltools by design.
            info["neuralnetwork_status"] = "SKIPPED: not applicable at iOS18 target"
            info["neuralnetwork_ops"] = []

            cmt_info[name] = info
    except Exception as e:  # pragma: no cover - environment-specific
        for case in cases:
            cmt_info[case["name"]] = {
                "frontend_status": f"UNAVAILABLE: {e}",
                "frontend_ops": [],
                "mlprogram_status": f"UNAVAILABLE: {e}",
                "mlprogram_ops": [],
                "neuralnetwork_status": f"UNAVAILABLE: {e}",
                "neuralnetwork_ops": [],
            }

    results: list[ProbeResult] = []
    for case in cases:
        op = case["name"]
        mil = mil_program(case["body"], case["sig"], "z_output0")
        r = probe_custom(
            f"differential/{op}",
            mil,
            case["inputs"],
            case["out_numel"],
            expected=None,
            atol=0.0,
        )
        detail = {
            "coremltools": cmt_info.get(op, {}),
            "libane_source_ops": _mil_ops_from_text(mil),
            "libane_build_info": _BUILD_INFO_FIELDS,
        }
        r.note = json.dumps(detail, separators=(",", ":"))
        if not r.compiled and not r.failure_kind:
            r.failure_kind = "compile_reject"
        results.append(r)

    return results


def scan_lowered_support(C: int = 64, S: int = 512) -> list[ProbeResult]:
    """
    Probe libane graph-IR lowering support independently from raw MIL acceptance.

    This scan answers: can libane execute the op semantics via graph lowering
    on ANE, even when standalone raw MIL forms compile-reject.
    """
    print("── Lowered support ───────────────────────────────────────────────")
    n = C * S

    try:
        import ane
    except ImportError as e:
        return [
            ProbeResult(
                name="lowered/import",
                compiled=False,
                passed=None,
                max_err=None,
                atol=0.0,
                compile_ms=0.0,
                eval_ms=None,
                error=f"ane module not available: {e}",
                failure_kind="compile_reject",
                raw_acceptance=None,
                libane_lowered_support=False,
            )
        ]

    def _op_code(name: str, fallback: int) -> int:
        return int(getattr(ane, name, fallback))

    def _run_case(
        case_name: str,
        op_code: int,
        inputs: list[np.ndarray],
        expected: np.ndarray,
        atol: float,
        note: str,
        output_shape: Optional[list[int]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> ProbeResult:
        in_shape = [1, C, 1, S]
        out_shape = output_shape if output_shape is not None else in_shape
        t0 = time.perf_counter()
        try:
            g = ane.Graph()
            tids = [g.add_input(f"in{i}", in_shape) for i in range(len(inputs))]
            out = g.add_op(op_code, tids, out_shape, None if weights is None else weights)
            g.mark_output(out)
            cg = g.compile()
            if output_shape is not None:
                cg.set_output_shapes([out_shape])
            compile_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            compile_ms = (time.perf_counter() - t0) * 1000.0
            return ProbeResult(
                name=case_name,
                compiled=False,
                passed=None,
                max_err=None,
                atol=atol,
                compile_ms=compile_ms,
                eval_ms=None,
                error=str(e),
                failure_kind="compile_reject",
                note=note,
                raw_acceptance=None,
                libane_lowered_support=False,
            )

        t1 = time.perf_counter()
        try:
            call_inputs: object
            if len(inputs) == 1:
                call_inputs = inputs[0].reshape(1, C, 1, S).astype(np.float16)
            else:
                call_inputs = [x.reshape(1, C, 1, S).astype(np.float16) for x in inputs]
            out_arr = np.asarray(cg(call_inputs), dtype=np.float16).reshape(-1)
            eval_ms = (time.perf_counter() - t1) * 1000.0
        except Exception as e:
            eval_ms = (time.perf_counter() - t1) * 1000.0
            return ProbeResult(
                name=case_name,
                compiled=True,
                passed=False,
                max_err=None,
                atol=atol,
                compile_ms=compile_ms,
                eval_ms=eval_ms,
                error=str(e),
                failure_kind="runtime_reject",
                note=note,
                raw_acceptance=None,
                libane_lowered_support=False,
            )

        passed, max_err = check(case_name, out_arr, expected, atol)
        return ProbeResult(
            name=case_name,
            compiled=True,
            passed=passed,
            max_err=max_err,
            atol=atol,
            compile_ms=compile_ms,
            eval_ms=eval_ms,
            failure_kind=None if passed else "numeric_mismatch",
            note=note,
            raw_acceptance=None,
            libane_lowered_support=passed,
        )

    results: list[ProbeResult] = []

    x = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    results.append(
        _run_case(
            "lowered/avg_pool",
            _op_code("AVG_POOL", 12),
            [x],
            expected=x.astype(np.float32),
            atol=0.05,
            note="lowering=identity for 1x1/stride1 pool use-case",
        )
    )

    y = np.linspace(-1.5, 1.5, n, dtype=np.float32)
    results.append(
        _run_case(
            "lowered/max_pool",
            _op_code("MAX_POOL", 13),
            [y],
            expected=y.astype(np.float32),
            atol=0.05,
            note="lowering=identity for 1x1/stride1 pool use-case",
        )
    )

    a = np.where(np.arange(n) % 3 == 0, 0.0, 2.0).astype(np.float32)
    b = np.where(np.arange(n) % 5 == 0, 0.0, -4.0).astype(np.float32)
    exp_and = np.where((a != 0.0) & (b != 0.0), 1.0, 0.0).astype(np.float32)
    results.append(
        _run_case(
            "lowered/logical_and",
            _op_code("LOGICAL_AND", 14),
            [a, b],
            expected=exp_and,
            atol=0.05,
            note='lowering=cast(bool)->cast(fp16)->mul; out in {0,1}',
        )
    )

    exp_or = np.where((a != 0.0) | (b != 0.0), 1.0, 0.0).astype(np.float32)
    results.append(
        _run_case(
            "lowered/logical_or",
            _op_code("LOGICAL_OR", 15),
            [a, b],
            expected=exp_or,
            atol=0.05,
            note='lowering=cast(bool)->cast(fp16)->maximum; out in {0,1}',
        )
    )

    exp_xor = np.where((a != 0.0) ^ (b != 0.0), 1.0, 0.0).astype(np.float32)
    results.append(
        _run_case(
            "lowered/logical_xor",
            _op_code("LOGICAL_XOR", 16),
            [a, b],
            expected=exp_xor,
            atol=0.05,
            note='lowering=cast(bool)->cast(fp16)->not_equal->cast(fp16); out in {0,1}',
        )
    )

    xr = np.linspace(0.95, 1.05, n, dtype=np.float32)
    exp_rp = xr.reshape(1, C, 1, S).prod(axis=1, keepdims=True).astype(np.float32).reshape(-1)
    results.append(
        _run_case(
            "lowered/reduce_prod",
            _op_code("REDUCE_PROD", 17),
            [xr],
            expected=exp_rp,
            atol=0.2,
            note="lowering=exp(reduce_sum(log(x+eps))); domain expects positive inputs",
            output_shape=[1, 1, 1, S],
        )
    )

    # Static-mask scatter lowering:
    #   out = base * (1 - mask) + updates * mask
    # implemented as a graph op with mask provided as compile-time fp16 weights.
    base = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    upd = (1.0 + (np.arange(n, dtype=np.float32) % 37) * 0.125)
    mask = np.zeros((n,), dtype=np.float16)
    for c in range(C):
        idx = (7 * c + 3) % S
        mask[c * S + idx] = np.float16(1.0)
    exp_sc = base * (1.0 - mask.astype(np.float32)) + upd * mask.astype(np.float32)
    results.append(
        _run_case(
            "lowered/scatter_static_mask",
            _op_code("SCATTER", 18),
            [base, upd],
            expected=exp_sc,
            atol=0.05,
            note="lowering=base*(1-mask)+updates*mask with static fp16 mask weights",
            weights=mask,
        )
    )
    results.append(
        _run_case(
            "lowered/scatter_nd_static_mask",
            _op_code("SCATTER_ND", 20),
            [base, upd],
            expected=exp_sc,
            atol=0.05,
            note="lowering=base*(1-mask)+updates*mask with static fp16 mask weights",
            weights=mask,
        )
    )
    results.append(
        _run_case(
            "lowered/scatter_along_axis_static_mask",
            _op_code("SCATTER_ALONG_AXIS", 21),
            [base, upd],
            expected=exp_sc,
            atol=0.05,
            note="lowering=base*(1-mask)+updates*mask with static fp16 mask weights",
            weights=mask,
        )
    )

    # Gather(static): out = x * static_mask_weights
    gx = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gmask = np.zeros((n,), dtype=np.float16)
    for c in range(C):
        idx = (5 * c + 1) % S
        gmask[c * S + idx] = np.float16(1.0)
    exp_gs = gx * gmask.astype(np.float32)
    results.append(
        _run_case(
            "lowered/gather_static_mask",
            _op_code("GATHER", 19),
            [gx],
            expected=exp_gs,
            atol=0.05,
            note="lowering=mul(x, static_mask_weights)",
            weights=gmask,
        )
    )

    # Gather(dynamic): out = x * runtime_mask_input
    gm = np.where((np.arange(n) % 7) == 0, 1.0, 0.0).astype(np.float32)
    exp_gd = gx * gm
    results.append(
        _run_case(
            "lowered/gather_dynamic_mask",
            _op_code("GATHER", 19),
            [gx, gm],
            expected=exp_gd,
            atol=0.05,
            note="lowering=mul(x, runtime_mask_input)",
        )
    )

    return results


def scan_scatter_lowering_proto(C: int = 8, S: int = 32) -> list[ProbeResult]:
    """
    Prototype static-index scatter lowering via one-hot masks.

    This does not use scatter ops directly. It demonstrates a decomposition
    that uses only already-verified primitives:
      out = base * (1 - mask) + updates * mask
    where mask is a static one-hot-like tensor.
    """
    print("── Scatter lowering proto ───────────────────────────────────────")
    if S % 8 != 0:
        raise ValueError("S must be a multiple of 8")

    n = C * S

    def _run_case(name: str, mask: np.ndarray, base: np.ndarray, upd: np.ndarray) -> ProbeResult:
        tt = f"tensor<fp16, [1,{C},1,{S}]>"
        body = (
            '        fp16 one = const()[name=string("one"), val=fp16(1.0)];\n'
            f'        {tt} inv = sub(x=one, y=a_input2)[name=string("inv")];\n'
            f'        {tt} xb = mul(x=a_input0, y=inv)[name=string("xb")];\n'
            f'        {tt} uu = mul(x=a_input1, y=a_input2)[name=string("uu")];\n'
            f'        {tt} z_output0 = add(x=xb, y=uu)[name=string("z_output0")];'
        )
        mil = mil_program(body, _sig3(C, S), "z_output0")
        expected = (base * (1.0 - mask) + upd * mask).astype(np.float32)
        r = probe_custom(
            name,
            mil,
            [base.astype(np.float16), upd.astype(np.float16), mask.astype(np.float16)],
            n,
            expected=expected,
            atol=0.05,
        )
        r.raw_acceptance = None
        r.libane_lowered_support = bool(r.passed is True)
        r.note = "static-index one-hot scatter decomposition (precomputed mask input)"
        return r

    base = np.linspace(-2.0, 2.0, n, dtype=np.float32).reshape(1, C, 1, S)
    upd = (100.0 + np.arange(n, dtype=np.float32)).reshape(1, C, 1, S)

    # Case 1: one target position per channel (channel-specific static index).
    mask1 = np.zeros((1, C, 1, S), dtype=np.float32)
    for c in range(C):
        idx = (3 * c + 5) % S
        mask1[0, c, 0, idx] = 1.0

    # Case 2: two target positions per channel.
    mask2 = np.zeros((1, C, 1, S), dtype=np.float32)
    for c in range(C):
        idx0 = (2 * c + 1) % S
        idx1 = (5 * c + 7) % S
        mask2[0, c, 0, idx0] = 1.0
        mask2[0, c, 0, idx1] = 1.0

    return [
        _run_case("proto/scatter_static_onehot_single", mask1, base, upd),
        _run_case("proto/scatter_static_onehot_double", mask2, base, upd),
    ]


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
    compile_only = sum(1 for r in results if r.compiled and r.passed is None)
    eval_fail_exp = sum(1 for r in results if r.passed is False and r.expected_eval_fail)
    eval_fail_unexp = sum(1 for r in results if r.passed is False and not r.expected_eval_fail)
    compile_fail_exp = sum(1 for r in results if not r.compiled and r.expected_compile_fail)
    compile_fail_unexp = sum(1 for r in results if not r.compiled and not r.expected_compile_fail)
    compile_fail = compile_fail_exp + compile_fail_unexp
    eval_fail = eval_fail_exp + eval_fail_unexp
    total = len(results)
    compile_reject = sum(1 for r in results if r.failure_kind == "compile_reject")
    runtime_reject = sum(1 for r in results if r.failure_kind == "runtime_reject")
    numeric_mismatch = sum(1 for r in results if r.failure_kind == "numeric_mismatch")
    semantic_mismatch = sum(1 for r in results if r.failure_kind == "semantic_mismatch")
    raw_known = [r for r in results if r.raw_acceptance is not None]
    lowered_known = [r for r in results if r.libane_lowered_support is not None]
    raw_accept = sum(1 for r in raw_known if r.raw_acceptance)
    lowered_support = sum(1 for r in lowered_known if r.libane_lowered_support)

    print()
    print("═" * 67)
    print("  PROBE RESULTS")
    print("═" * 67)
    print(f"  Compiled:          {compile_pass}/{total}")
    print(f"  Eval verified:     {eval_pass}/{total}")
    print(f"  Compile-only:      {compile_only}/{total}")
    print(f"  Supported (strict): {eval_pass}/{total}")
    if raw_known:
        print(f"  Raw acceptance:    {raw_accept}/{len(raw_known)}")
    else:
        print("  Raw acceptance:    n/a")
    if lowered_known:
        print(f"  Lowered support:   {lowered_support}/{len(lowered_known)}")
    else:
        print("  Lowered support:   n/a")
    if compile_fail_unexp:
        print(f"  Compile failures: {compile_fail_unexp}")
    if eval_fail_unexp:
        print(f"  Eval failures:    {eval_fail_unexp}")
    if compile_fail_exp:
        print(f"  Expected compile-fail (XFAIL): {compile_fail_exp}")
    if eval_fail_exp:
        print(f"  Expected eval-fail (XFAIL):    {eval_fail_exp}")
    if compile_reject:
        print(f"  compile_reject: {compile_reject}")
    if runtime_reject:
        print(f"  runtime_reject: {runtime_reject}")
    if numeric_mismatch:
        print(f"  numeric_mismatch: {numeric_mismatch}")
    if semantic_mismatch:
        print(f"  semantic_mismatch: {semantic_mismatch}")
    print()

    if compile_fail_unexp:
        print("  Compile failures:")
        for r in results:
            if not r.compiled and not r.expected_compile_fail:
                msg = f" — {r.error}" if r.error else ""
                print(f"    • {r.name}{msg}")
        print()

    if eval_fail_unexp:
        print("  Eval failures:")
        for r in results:
            if r.passed is False and not r.expected_eval_fail:
                print(f"    • {r.name}  max_err={r.max_err:.4f}  atol={r.atol}")
        print()

    if compile_fail_exp:
        print("  Expected compile-fail (XFAIL):")
        for r in results:
            if not r.compiled and r.expected_compile_fail:
                msg = f" — {r.note}" if r.note else ""
                print(f"    • {r.name}{msg}")
        print()

    if eval_fail_exp:
        print("  Expected eval-fail (XFAIL):")
        for r in results:
            if r.passed is False and r.expected_eval_fail:
                msg = f" — {r.note}" if r.note else ""
                print(
                    f"    • {r.name}  max_err={r.max_err:.4f}  atol={r.atol}{msg}"
                )
        print()

    _print_scatter_gather_matrix(results)


def _print_scatter_gather_matrix(results: list[ProbeResult]) -> None:
    def _raw_for(op: str) -> str:
        vals = [r.raw_acceptance for r in results
                if r.raw_acceptance is not None and op in r.name]
        if not vals:
            return "n/a"
        return "PASS" if any(vals) else "FAIL"

    def _lowered_for(op: str) -> str:
        vals = [r.libane_lowered_support for r in results
                if r.libane_lowered_support is not None and op in r.name]
        if not vals:
            return "n/a"
        return "PASS" if any(vals) else "FAIL"

    rows = [
        ("scatter", _raw_for("differential/scatter"), _lowered_for("lowered/scatter_static_mask")),
        ("scatter_nd", _raw_for("differential/scatter_nd"), _lowered_for("lowered/scatter_nd_static_mask")),
        ("scatter_along_axis", _raw_for("differential/scatter_along_axis"), _lowered_for("lowered/scatter_along_axis_static_mask")),
        ("gather", _raw_for("differential/gather"), _lowered_for("lowered/gather")),
    ]
    if all(raw == "n/a" and low == "n/a" for _, raw, low in rows):
        return

    print("  Scatter/Gather Matrix:")
    print("    op       raw_acceptance  lowered_support")
    for op, raw, low in rows:
        print(f"    {op:<8} {raw:<15} {low}")
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

    def _matrix_val(pattern: str, field: str) -> Optional[bool]:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None and pattern in r.name]
        if not vals:
            return None
        return bool(any(vals))

    payload = {
        "libane_version": libane_version,
        "mil_build_info": _BUILD_INFO_FIELDS,
        "platform": {
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.version(),
            "machine": platform.machine(),
            "chip": chip,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scatter_gather_matrix": {
            "scatter": {
                "raw_acceptance": _matrix_val("differential/scatter", "raw_acceptance"),
                "lowered_support": _matrix_val("lowered/scatter_static_mask", "libane_lowered_support"),
            },
            "scatter_nd": {
                "raw_acceptance": _matrix_val("differential/scatter_nd", "raw_acceptance"),
                "lowered_support": _matrix_val("lowered/scatter_nd_static_mask", "libane_lowered_support"),
            },
            "scatter_along_axis": {
                "raw_acceptance": _matrix_val("differential/scatter_along_axis", "raw_acceptance"),
                "lowered_support": _matrix_val("lowered/scatter_along_axis_static_mask", "libane_lowered_support"),
            },
            "gather": {
                "raw_acceptance": _matrix_val("differential/gather", "raw_acceptance"),
                "lowered_support": _matrix_val("lowered/gather", "libane_lowered_support"),
            },
        },
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
                "expected_compile_fail": r.expected_compile_fail,
                "expected_eval_fail": r.expected_eval_fail,
                "failure_kind": r.failure_kind,
                "status": r.status,
                "support_level": r.support_level,
                "eval_verified": bool(r.passed is True),
                "raw_acceptance": r.raw_acceptance,
                "libane_lowered_support": r.libane_lowered_support,
                "note": r.note,
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
    parser.add_argument("--params",     action="append", metavar="OP",
                        help="Explore parameter variants for OP (repeatable: rsqrt/log/inverse/...)")
    parser.add_argument("--params-all", action="store_true",
                        help="Run all pre-defined parameter space explorations")
    parser.add_argument("--round-semantics", action="store_true",
                        help="Characterize round() tie-breaking semantics on fp16 half-way values")
    parser.add_argument("--dep-matrix", action="store_true",
                        help="Run wrapper->intermediate compile dependency matrix")
    parser.add_argument("--gap-ops", action="store_true",
                        help="Probe pending gap ops (logical/pool/scatter/gather/reduce_prod)")
    parser.add_argument("--sig-sweep", action="store_true",
                        help="Run automated signature+wrapper+buildInfo sweep for gap ops")
    parser.add_argument("--differential", action="store_true",
                        help="Run coremltools-vs-libane MIL differential for gap ops")
    parser.add_argument("--lowered-support", action="store_true",
                        help="Probe libane graph-lowering support (separate from raw MIL acceptance)")
    parser.add_argument("--scatter-lowering-proto", action="store_true",
                        help="Prototype static one-hot scatter decomposition using supported primitives")
    parser.add_argument("--sin-cos-sweep", action="store_true",
                        help="Run sin/cos input-range sweep")
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
        args.intermediate, args.explore, args.params, args.params_all,
        args.round_semantics, args.dep_matrix, args.gap_ops, args.sig_sweep,
        args.differential, args.lowered_support, args.scatter_lowering_proto, args.sin_cos_sweep,
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
    if args.params:
        for op_name in args.params:
            print(f"── Parameter exploration ({op_name}) ─────────────────────────────")
            all_results += explore_op_params(op_name)
    if args.params_all:
        all_results += scan_param_spaces()
    if args.round_semantics:
        all_results += scan_round_semantics()
    if args.dep_matrix:
        all_results += scan_dependency_matrix()
    if args.gap_ops:
        all_results += scan_gap_ops()
    if args.sig_sweep:
        all_results += scan_signature_sweep()
    if args.differential:
        all_results += scan_coreml_differential()
    if args.lowered_support:
        all_results += scan_lowered_support()
    if args.scatter_lowering_proto:
        all_results += scan_scatter_lowering_proto()
    if args.sin_cos_sweep:
        all_results += scan_sin_cos_ranges()

    print_report(all_results)

    if args.export:
        export_json(all_results, args.export)


if __name__ == "__main__":
    _cli()
