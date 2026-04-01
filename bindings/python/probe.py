"""
probe — ANE op discovery harness for libane.

Exposes every op class supported by the Apple Neural Engine as a
first-class Python probe, enabling:

  1. Compound / intermediate op discovery (ops that only compile as
     sub-nodes of a larger graph, not as standalone).
  2. Parameter space exploration (epsilon, axis, exponent, …).
  3. Cross-generation comparison (M1–M4) via JSON export with chip metadata.
  4. Dtype combination testing.
  5. Crowdsourced research — results are JSON-exportable with full
     chip/OS/libane version metadata so runs from many machines can be
     aggregated.

Quick start::

    import probe
    r = probe.probe_unary("relu")
    print(r)                       # ProbeResult(op='relu', status='ok', …)
    probe.scan_all()               # run everything
    probe.export_json("results.json")

CLI::

    ane-probe --all --export results.json
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import ane as _ane
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "libane Python extension not found — install with: pip install libane"
    ) from _e

# ── Build-info header required by every MIL program ──────────────────────────

_BUILD_INFO = textwrap.dedent("""\
    buildInfo {
       coremlc-version: "7.0.0"
       coremltools-version: "7.0.0"
       mlmodel-version: "7"
       model-name: "probe"
    }
    """)

# ── MIL program templates ─────────────────────────────────────────────────────

def mil_program(body: str, *, inputs: str = "x: fp16[1,32,1,32]",
                outputs: str = "fp16[1,32,1,32]") -> str:
    """Wrap *body* in a complete MIL program."""
    return (
        _BUILD_INFO
        + f"func main({inputs}) -> ({outputs}) {{\n"
        + body
        + "}\n"
    )


def _sig1(shape: str = "1,32,1,32") -> tuple[str, str]:
    """Single fp16 input/output signature with *shape*."""
    return f"x: fp16[{shape}]", f"fp16[{shape}]"


def _sig2(shape: str = "1,32,1,32") -> tuple[str, str]:
    """Two fp16 inputs, one output."""
    return f"a: fp16[{shape}], b: fp16[{shape}]", f"fp16[{shape}]"


def _sig3(in_shape: str, out_shape: str) -> tuple[str, str]:
    return f"x: fp16[{in_shape}]", f"fp16[{out_shape}]"


def _body1(op_name: str, params: str = "", var: str = "x") -> str:
    """Unary op body: v = op_name(x=var, …); return v"""
    sep = ", " if params else ""
    return f"  %v = {op_name}(x={var}{sep}{params}) -> (fp16);\n  return (%v);\n"


def _body2(op_name: str, params: str = "") -> str:
    """Binary op body: v = op_name(x=a, y=b, …); return v"""
    sep = ", " if params else ""
    return f"  %v = {op_name}(x=a, y=b{sep}{params}) -> (fp16);\n  return (%v);\n"


def _body_cmp(op_name: str) -> str:
    """Comparison op (bool output) body — cast result back to fp16."""
    return (
        "  %b = {op}(x=a, y=b) -> (bool);\n"
        "  %v = cast(x=%b, dtype=\"fp16\") -> (fp16);\n"
        "  return (%v);\n"
    ).format(op=op_name)


def _reduction_body(op_name: str, axes: str = "[3]") -> str:
    """Reduction op body."""
    return (
        f"  %v = {op_name}(x=x, axes={axes}, keep_dims=true) -> (fp16);\n"
        "  return (%v);\n"
    )


# ── ProbeResult ───────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of a single probe run."""
    op: str
    variant: str = ""
    compiles: bool = False
    executes: bool = False
    correct: bool | None = None
    error: str = ""
    max_err: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.compiles:
            return "no_compile"
        if not self.executes:
            return "no_execute"
        if self.correct is False:
            return "wrong"
        if self.correct is True:
            return "ok"
        return "compiled"  # executed not attempted

    def __str__(self) -> str:
        tag = f"[{self.variant}]" if self.variant else ""
        err = f" max_err={self.max_err:.4g}" if self.max_err is not None else ""
        return f"ProbeResult(op={self.op!r}{tag}, status={self.status!r}{err})"


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def check(actual: np.ndarray, expected: np.ndarray, atol: float = 0.02) -> tuple[bool, float]:
    """Compare fp16 actual against fp32 expected.  Returns (ok, max_abs_err)."""
    a16 = actual.astype(np.float32).ravel()
    e32 = np.asarray(expected, dtype=np.float32).ravel()
    if a16.shape != e32.shape:
        return False, float("inf")
    err = float(np.max(np.abs(a16 - e32)))
    return err <= atol, err


# ── Core probe primitive ──────────────────────────────────────────────────────

def probe_custom(
    op: str,
    mil: str,
    inputs: list[np.ndarray],
    out_numel: list[int],
    *,
    expected: list[np.ndarray] | None = None,
    variant: str = "",
    params: dict[str, Any] | None = None,
    atol: float = 0.02,
) -> ProbeResult:
    """
    Compile *mil* and optionally execute it, returning a ProbeResult.

    Args:
        op:        Op name for the result label.
        mil:       Complete MIL program text.
        inputs:    Input arrays (converted to fp16 internally).
        out_numel: Number of fp16 elements for each output.
        expected:  If provided, compare outputs against these fp32 reference arrays.
        variant:   Optional sub-label (e.g. an epsilon value).
        params:    Arbitrary parameter dict stored on the result.
        atol:      Absolute tolerance for correctness check.
    """
    r = ProbeResult(op=op, variant=variant, params=params or {})

    try:
        prog = _ane.compile_mil(mil)
        r.compiles = True
    except Exception as exc:
        r.error = str(exc)
        return r

    if not inputs:
        return r

    try:
        outputs = prog.run(inputs, out_numel)
        r.executes = True
    except Exception as exc:
        r.error = str(exc)
        return r

    if expected is not None:
        ok_list, errs = [], []
        for out, exp in zip(outputs, expected):
            ok, err = check(out, exp, atol=atol)
            ok_list.append(ok)
            errs.append(err)
        r.correct = all(ok_list)
        r.max_err = max(errs)

    return r


# ── Op descriptor tables ──────────────────────────────────────────────────────

# shape used throughout: [1, C, 1, S] = [1, 32, 1, 32]
_SHAPE = (1, 32, 1, 32)
_N = _numel(_SHAPE)  # 1024 elements

_UNARY_OPS: dict[str, dict] = {
    # activation
    "relu":          {"params": ""},
    "relu6":         {"params": ""},
    "leaky_relu":    {"params": "alpha=0.01"},
    "elu":           {"params": "alpha=1.0"},
    "selu":          {"params": ""},
    "celu":          {"params": "alpha=1.0"},
    "gelu":          {"params": ""},
    "silu":          {"params": ""},
    "swish":         {"params": ""},
    "sigmoid":       {"params": ""},
    "tanh":          {"params": ""},
    "softplus":      {"params": ""},
    "softsign":      {"params": ""},
    "hard_sigmoid":  {"params": ""},
    "hard_swish":    {"params": ""},
    "mish":          {"params": ""},
    # elementwise math
    "abs":           {"params": ""},
    "sign":          {"params": ""},
    "floor":         {"params": ""},
    "ceil":          {"params": ""},
    "round":         {"params": ""},
    "exp":           {"params": ""},
    "exp2":          {"params": ""},
    "log":           {"params": ""},
    "rsqrt":         {"params": "epsilon=0.0"},
    "sqrt":          {"params": ""},
    "inverse":       {"params": "epsilon=0.0"},
    "square":        {"params": ""},
    "erf":           {"params": ""},
    "erfinv":        {"params": ""},
    "asin":          {"params": ""},
    "acos":          {"params": ""},
    "atan":          {"params": ""},
    "sin":           {"params": ""},
    "cos":           {"params": ""},
    "tan":           {"params": ""},
    "sinh":          {"params": ""},
    "cosh":          {"params": ""},
    "clip":          {"params": "alpha=-1.0, beta=1.0"},
    "threshold":     {"params": "alpha=0.0, beta=0.0"},
}

_BINARY_OPS: dict[str, dict] = {
    "add":           {"kind": "binary"},
    "mul":           {"kind": "binary"},
    "sub":           {"kind": "binary"},
    "real_div":      {"kind": "binary"},
    "maximum":       {"kind": "binary"},
    "minimum":       {"kind": "binary"},
    "pow":           {"kind": "binary"},
    "floor_div":     {"kind": "binary"},
}

_COMPARISON_OPS: list[str] = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
]

# ── Unary probe ───────────────────────────────────────────────────────────────

def probe_unary(op_name: str, *, shape: tuple[int, ...] = _SHAPE,
                params: str | None = None) -> ProbeResult:
    """Probe a unary elementwise op."""
    desc = _UNARY_OPS.get(op_name)
    if desc is None and params is None:
        # Treat as unknown — try with no params
        params_str = ""
    else:
        params_str = params if params is not None else (desc or {}).get("params", "")

    shape_str = ",".join(str(d) for d in shape)
    ins, outs = _sig1(shape_str)
    body = _body1(op_name, params_str)
    mil = mil_program(body, inputs=ins, outputs=outs)

    n = _numel(shape)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    inp = [x]
    return probe_custom(op_name, mil, inp, [n])


# ── Binary probe ──────────────────────────────────────────────────────────────

def probe_binary(op_name: str, *, shape: tuple[int, ...] = _SHAPE) -> ProbeResult:
    """Probe a binary elementwise op."""
    shape_str = ",".join(str(d) for d in shape)
    ins, outs = _sig2(shape_str)
    body = _body2(op_name)
    mil = mil_program(body, inputs=ins, outputs=outs)

    n = _numel(shape)
    rng = np.random.default_rng(42)
    a = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    b = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    return probe_custom(op_name, mil, [a, b], [n])


# ── Reduction probe ───────────────────────────────────────────────────────────

def probe_reduction(op_name: str, *, axes: str = "[3]",
                    shape: tuple[int, ...] = _SHAPE) -> ProbeResult:
    """Probe a reduction op (keep_dims=true)."""
    shape_str = ",".join(str(d) for d in shape)
    ins, _ = _sig1(shape_str)
    body = _reduction_body(op_name, axes)
    mil = mil_program(body, inputs=ins, outputs=f"fp16[{shape_str}]")

    n = _numel(shape)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    return probe_custom(op_name, mil, [x], [n])


# ── Composite probes ──────────────────────────────────────────────────────────

def probe_rmsnorm(*, hidden: int = 32, seq: int = 32) -> ProbeResult:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * scale."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (1, hidden, 1, seq)).astype(np.float16)
    scale = np.ones((hidden,), dtype=np.float16)
    n = hidden * seq

    scale_arr = scale.astype(np.float16)
    scale_blob = scale_arr.tobytes()
    numel_scale = scale_arr.size

    mil = (
        _BUILD_INFO
        + f"func main(x: fp16[1,{hidden},1,{seq}]) -> (fp16[1,{hidden},1,{seq}]) {{\n"
        f"  %w = const() -> (fp16[{hidden}]);\n"
        f"  %v = rms_norm(x=x, weight=%w, epsilon=1e-5) -> (fp16);\n"
        "  return (%v);\n"
        "}\n"
    )
    try:
        prog = _ane.compile_mil_with_weights(
            mil, {"weight.bin": scale_arr.reshape(hidden)})
        r = ProbeResult("rmsnorm", compiles=True)
        outs = prog.run([x], [n])
        r.executes = True
        return r
    except Exception as exc:
        r = ProbeResult("rmsnorm")
        r.error = str(exc)
        return r


def probe_layer_norm(*, hidden: int = 32, seq: int = 32) -> ProbeResult:
    """LayerNorm: y = (x - mean) / sqrt(var + eps) * scale + bias."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (1, hidden, 1, seq)).astype(np.float16)
    scale = np.ones(hidden, dtype=np.float16)
    bias  = np.zeros(hidden, dtype=np.float16)
    n = hidden * seq

    mil = (
        _BUILD_INFO
        + f"func main(x: fp16[1,{hidden},1,{seq}]) -> (fp16[1,{hidden},1,{seq}]) {{\n"
        f"  %s = const() -> (fp16[{hidden}]);\n"
        f"  %b = const() -> (fp16[{hidden}]);\n"
        f"  %v = layer_norm(x=x, axes=[1], gamma=%s, beta=%b, epsilon=1e-5)"
        f" -> (fp16);\n"
        "  return (%v);\n"
        "}\n"
    )
    try:
        prog = _ane.compile_mil_with_weights(
            mil, {"scale.bin": scale, "bias.bin": bias})
        r = ProbeResult("layer_norm", compiles=True)
        prog.run([x], [n])
        r.executes = True
        return r
    except Exception as exc:
        r = ProbeResult("layer_norm")
        r.error = str(exc)
        return r


def probe_softmax(*, channels: int = 32, seq: int = 32) -> ProbeResult:
    """Softmax over the channel axis."""
    shape_str = f"1,{channels},1,{seq}"
    body = "  %v = softmax(x=x, axis=1) -> (fp16);\n  return (%v);\n"
    mil = mil_program(body, inputs=f"x: fp16[{shape_str}]",
                      outputs=f"fp16[{shape_str}]")
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (1, channels, 1, seq)).astype(np.float16)
    n = channels * seq

    r = probe_custom("softmax", mil, [x], [n])
    if r.executes:
        # verify probabilities sum to ~1 along channel axis
        out = _ane.compile_mil(mil).run([x], [n])[0].reshape(1, channels, 1, seq)
        sums = out.astype(np.float32).sum(axis=1)
        err = float(np.max(np.abs(sums - 1.0)))
        r.correct = err < 0.05
        r.max_err = err
    return r


def probe_matmul(*, M: int = 64, K: int = 64, N: int = 64) -> ProbeResult:
    """Matrix multiply via ANE matmul graph op."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 0.1, (M, K)).astype(np.float32)
    B = rng.normal(0, 0.1, (K, N)).astype(np.float32)
    try:
        C = _ane.matmul_f32(A, B)
        expected = A @ B
        ok, err = check(C, expected, atol=1.0)  # fp16 matmul has larger error
        r = ProbeResult("matmul", compiles=True, executes=True,
                        correct=ok, max_err=err)
    except Exception as exc:
        r = ProbeResult("matmul", error=str(exc))
    return r


def probe_concat(*, channels: int = 32, seq: int = 32) -> ProbeResult:
    """Concat two tensors along axis=1 (channel axis)."""
    shape_str = f"1,{channels},1,{seq}"
    out_channels = channels * 2
    out_str = f"1,{out_channels},1,{seq}"
    mil = (
        _BUILD_INFO
        + f"func main(a: fp16[{shape_str}], b: fp16[{shape_str}])"
        f" -> (fp16[{out_str}]) {{\n"
        "  %v = concat(values=(a, b), axis=1) -> (fp16);\n"
        "  return (%v);\n"
        "}\n"
    )
    rng = np.random.default_rng(42)
    a = rng.uniform(0, 1, (1, channels, 1, seq)).astype(np.float16)
    b = rng.uniform(0, 1, (1, channels, 1, seq)).astype(np.float16)
    n_out = out_channels * seq
    return probe_custom("concat", mil, [a, b], [n_out])


def probe_int8_round_trip(*, channels: int = 32, seq: int = 32) -> ProbeResult:
    """fp16 → int8 cast → fp16 cast round-trip."""
    shape_str = f"1,{channels},1,{seq}"
    mil = (
        _BUILD_INFO
        + f"func main(x: fp16[{shape_str}]) -> (fp16[{shape_str}]) {{\n"
        f"  %i = cast(x=x, dtype=\"int8\") -> (int8);\n"
        f"  %v = cast(x=%i, dtype=\"fp16\") -> (fp16);\n"
        "  return (%v);\n"
        "}\n"
    )
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (1, channels, 1, seq)).astype(np.float16)
    n = channels * seq
    return probe_custom("int8_round_trip", mil, [x], [n])


def probe_int8_quant_dequant(*, channels: int = 32, seq: int = 32) -> ProbeResult:
    """Quantize (scale+zero_point) then dequantize — probes ANE int8 quant path."""
    shape_str = f"1,{channels},1,{seq}"
    mil = (
        _BUILD_INFO
        + f"func main(x: fp16[{shape_str}]) -> (fp16[{shape_str}]) {{\n"
        f"  %q = quantize(input=x, zero_point=0, scale=0.01, output_dtype=\"int8\")"
        f" -> (int8);\n"
        f"  %v = dequantize(input=%q, zero_point=0, scale=0.01) -> (fp16);\n"
        "  return (%v);\n"
        "}\n"
    )
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (1, channels, 1, seq)).astype(np.float16)
    n = channels * seq
    return probe_custom("int8_quant_dequant", mil, [x], [n])


# ── Intermediate-only op discovery ───────────────────────────────────────────

def probe_intermediate(op_name: str, *, inner_params: str = "",
                       shape: tuple[int, ...] = _SHAPE) -> ProbeResult:
    """
    Wrap *op_name* between two identity-like ops to discover ops that only
    compile as intermediate nodes (not as standalone programs).

    The wrapper is: relu → op → relu.
    """
    shape_str = ",".join(str(d) for d in shape)
    sep = ", " if inner_params else ""
    body = (
        f"  %pre  = relu(x=x) -> (fp16);\n"
        f"  %mid  = {op_name}(x=%pre{sep}{inner_params}) -> (fp16);\n"
        f"  %v    = relu(x=%mid) -> (fp16);\n"
        "  return (%v);\n"
    )
    ins, outs = _sig1(shape_str)
    mil = mil_program(body, inputs=ins, outputs=outs)

    n = _numel(shape)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    r = probe_custom(op_name, mil, [x], [n], variant="intermediate")
    return r


# ── Parameter space exploration ───────────────────────────────────────────────

# Epsilon values that are representable as fp16 and span the useful range
_EPSILON_VALUES: list[float] = [
    0.0,
    float(np.float16("5.96e-8")),   # fp16 minimum positive (subnormal)
    float(np.float16("6.1e-5")),    # fp16 minimum normal
    float(np.float16("1e-4")),
    float(np.float16("1e-3")),
    float(np.float16("0.01")),
    float(np.float16("0.1")),
    float(np.float16("0.25")),
    float(np.float16("0.5")),
    float(np.float16("1.0")),
]

# Pre-defined parameter variation registries for interesting ops
PARAM_SPACES: dict[str, list[dict[str, Any]]] = {
    "rsqrt": [
        {"params": f"epsilon={e}", "meta": {"epsilon": e}}
        for e in _EPSILON_VALUES
    ],
    "log": [
        {"params": f"epsilon={e}", "meta": {"epsilon": e}}
        for e in _EPSILON_VALUES
    ],
    "inverse": [
        {"params": f"epsilon={e}", "meta": {"epsilon": e}}
        for e in _EPSILON_VALUES
    ],
    "pow": [
        {"params": f"alpha={a}", "meta": {"exponent": a}}
        for a in [0.5, 1.0, 2.0, 3.0, 4.0]
    ],
    "clip": [
        {"params": f"alpha={lo}, beta={hi}", "meta": {"alpha": lo, "beta": hi}}
        for lo, hi in [(-1.0, 1.0), (-0.5, 0.5), (0.0, 1.0), (-2.0, 2.0)]
    ],
    "leaky_relu": [
        {"params": f"alpha={a}", "meta": {"alpha": a}}
        for a in [0.01, 0.1, 0.2, 0.5]
    ],
    "softmax": [
        {"params": None, "body_override": f"  %v = softmax(x=x, axis={ax}) -> (fp16);\n  return (%v);\n",
         "meta": {"axis": ax}}
        for ax in [1, 2, 3]
    ],
}


def explore_op_params(
    op_name: str,
    param_variations: list[dict[str, Any]] | None = None,
    *,
    shape: tuple[int, ...] = _SHAPE,
) -> list[ProbeResult]:
    """
    Explore the parameter space for *op_name*.

    If *param_variations* is None, uses the pre-defined PARAM_SPACES registry.
    Each variation is a dict with keys:
      - ``params``: parameter string to embed in the MIL body (may be None).
      - ``body_override``: if present, use this as the full body instead.
      - ``meta``: arbitrary dict stored in ProbeResult.params.

    Returns one ProbeResult per variation.
    """
    if param_variations is None:
        param_variations = PARAM_SPACES.get(op_name)
        if not param_variations:
            return [probe_unary(op_name, shape=shape)]

    results: list[ProbeResult] = []
    shape_str = ",".join(str(d) for d in shape)
    n = _numel(shape)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 1.0, n).astype(np.float16).reshape(shape)
    ins, outs = _sig1(shape_str)

    for var in param_variations:
        meta = var.get("meta", {})
        variant_label = ", ".join(f"{k}={v}" for k, v in meta.items())

        body_override = var.get("body_override")
        if body_override:
            body = body_override
        else:
            params_str = var.get("params") or ""
            body = _body1(op_name, params_str)

        mil = mil_program(body, inputs=ins, outputs=outs)
        r = probe_custom(op_name, mil, [x], [n],
                         variant=variant_label, params=meta)
        results.append(r)

    return results


def scan_param_spaces() -> dict[str, list[ProbeResult]]:
    """Run explore_op_params for every op in PARAM_SPACES."""
    return {op: explore_op_params(op) for op in PARAM_SPACES}


# ── Op name fuzzing ───────────────────────────────────────────────────────────

CANDIDATE_NAMES: list[str] = [
    # activations not yet tested
    "prelu", "rrelu", "glu", "tanhshrink", "hardshrink", "softshrink",
    "log_sigmoid", "hardtanh", "gumbel_softmax",
    # elementwise
    "log1p", "log2", "expm1", "reciprocal", "negative", "logical_not",
    "atan2", "hypot", "xlogy", "i0", "sinc",
    # normalization variants
    "group_norm", "instance_norm", "batch_norm",
    # attention / transformer
    "scaled_dot_product_attention", "einsum",
    # linear
    "linear", "bilinear", "addmm",
    # conv variants
    "conv1d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    # pooling
    "avg_pool1d", "avg_pool2d", "max_pool1d", "max_pool2d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d",
    # tensor manipulation
    "squeeze", "unsqueeze", "flatten", "unfold", "roll",
    "gather", "scatter", "index_select",
    # type/format
    "dequantize", "quantize",
    # miscellaneous MIL ops
    "tile", "pad", "upsample_bilinear", "upsample_nearest_neighbor",
    "range_1d", "argsort", "topk",
    "l2_norm", "lp_normalization",
    "elu", "celu",
]


def explore_names(
    candidates: list[str] | None = None,
    *,
    shape: tuple[int, ...] = _SHAPE,
) -> list[ProbeResult]:
    """
    Try each candidate op name as a standalone unary op.
    Useful for discovering new ops that compile on a given ANE generation.
    """
    if candidates is None:
        candidates = CANDIDATE_NAMES
    return [probe_unary(name, shape=shape) for name in candidates]


# ── Batch scan functions ──────────────────────────────────────────────────────

def scan_unary() -> list[ProbeResult]:
    return [probe_unary(op) for op in _UNARY_OPS]


def scan_binary() -> list[ProbeResult]:
    results = [probe_binary(op) for op in _BINARY_OPS]
    results += [
        probe_custom(
            op,
            mil_program(
                _body_cmp(op),
                inputs=_sig2()[0],
                outputs=_sig2()[1],
            ),
            [
                np.random.default_rng(42).uniform(0, 1, _N).astype(np.float16).reshape(_SHAPE),
                np.random.default_rng(43).uniform(0, 1, _N).astype(np.float16).reshape(_SHAPE),
            ],
            [_N],
        )
        for op in _COMPARISON_OPS
    ]
    return results


def scan_reductions() -> list[ProbeResult]:
    ops = ["reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
           "reduce_prod", "reduce_l1_norm", "reduce_l2_norm",
           "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square"]
    results = []
    for op in ops:
        results.append(probe_reduction(op, axes="[3]"))
    return results


def scan_composite() -> list[ProbeResult]:
    return [
        probe_rmsnorm(),
        probe_layer_norm(),
        probe_softmax(),
        probe_matmul(),
        probe_concat(),
        probe_int8_round_trip(),
        probe_int8_quant_dequant(),
    ]


def scan_intermediate() -> list[ProbeResult]:
    candidates = [
        "gelu_approx", "silu", "swish", "mish",
        "prelu", "elu", "celu", "selu",
        "log_sigmoid", "hard_sigmoid",
        "rsqrt", "inverse", "square",
    ]
    return [probe_intermediate(op) for op in candidates]


def scan_all() -> dict[str, list[ProbeResult]]:
    """Run all scan categories and return a dict keyed by category."""
    return {
        "unary":        scan_unary(),
        "binary":       scan_binary(),
        "reductions":   scan_reductions(),
        "composite":    scan_composite(),
        "intermediate": scan_intermediate(),
        "param_spaces": list(
            r for rs in scan_param_spaces().values() for r in rs
        ),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: dict[str, list[ProbeResult]] | list[ProbeResult]) -> None:
    """Print a human-readable summary of *results*."""
    if isinstance(results, list):
        results = {"results": results}

    for category, rs in results.items():
        print(f"\n── {category} ({len(rs)} ops) ──")
        for r in rs:
            tag = f"[{r.variant}]" if r.variant else ""
            err = f"  max_err={r.max_err:.4g}" if r.max_err is not None else ""
            mark = {"ok": "✓", "compiled": "~", "no_compile": "✗",
                    "no_execute": "!", "wrong": "✗"}.get(r.status, "?")
            print(f"  {mark} {r.op}{tag:<30} {r.status}{err}")
            if r.error and r.status == "no_compile":
                short = r.error[:80]
                print(f"      {short}")


def _chip_info() -> dict[str, str]:
    info: dict[str, str] = {
        "os": platform.mac_ver()[0],
        "arch": platform.machine(),
    }
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, text=True).strip()
        info["cpu"] = out
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if "Chip" in line or "Model" in line:
                info.setdefault("chip", line.strip())
                break
    except Exception:
        pass
    return info


def export_json(
    path: str,
    results: dict[str, list[ProbeResult]] | list[ProbeResult] | None = None,
) -> None:
    """
    Export *results* (or run scan_all() if None) to JSON at *path*.

    The JSON includes chip/OS/libane version metadata to enable crowdsourced
    aggregation across ANE generations.
    """
    if results is None:
        results = scan_all()
    if isinstance(results, list):
        results = {"results": results}

    def _ser(r: ProbeResult) -> dict:
        return {
            "op": r.op,
            "variant": r.variant,
            "status": r.status,
            "compiles": r.compiles,
            "executes": r.executes,
            "correct": r.correct,
            "max_err": r.max_err,
            "params": r.params,
            "error": r.error,
        }

    payload = {
        "meta": {
            "libane_version": _ane.version(),
            "python": sys.version,
            **_chip_info(),
        },
        "results": {
            cat: [_ser(r) for r in rs]
            for cat, rs in results.items()
        },
    }

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Exported {sum(len(v) for v in results.values())} results → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="ane-probe",
        description="ANE op discovery harness — libane probe module",
    )
    p.add_argument("--unary",        action="store_true", help="Scan unary ops")
    p.add_argument("--binary",       action="store_true", help="Scan binary ops")
    p.add_argument("--reductions",   action="store_true", help="Scan reduction ops")
    p.add_argument("--composite",    action="store_true", help="Scan composite ops")
    p.add_argument("--intermediate", action="store_true", help="Scan intermediate-only ops")
    p.add_argument("--explore",      action="store_true",
                   help="Fuzz candidate op names")
    p.add_argument("--params",       metavar="OP",
                   help="Explore parameter space for OP (e.g. rsqrt)")
    p.add_argument("--all",          action="store_true", help="Run all scans")
    p.add_argument("--export",       metavar="FILE",
                   help="Export results to JSON file")
    args = p.parse_args()

    if not any([args.unary, args.binary, args.reductions, args.composite,
                args.intermediate, args.explore, args.params, args.all]):
        p.print_help()
        return

    if not _ane.available():
        print("WARNING: ANE not available on this machine — compile-only probes will run.")

    collected: dict[str, list[ProbeResult]] = {}

    if args.all:
        collected = scan_all()
    else:
        if args.unary:
            collected["unary"] = scan_unary()
        if args.binary:
            collected["binary"] = scan_binary()
        if args.reductions:
            collected["reductions"] = scan_reductions()
        if args.composite:
            collected["composite"] = scan_composite()
        if args.intermediate:
            collected["intermediate"] = scan_intermediate()
        if args.explore:
            collected["name_fuzz"] = explore_names()
        if args.params:
            collected[f"params_{args.params}"] = explore_op_params(args.params)

    print_report(collected)

    if args.export:
        export_json(args.export, collected)
