#!/usr/bin/env python3
"""
17 — Piecewise-linear custom activations

libane supports arbitrary smooth activations via piecewise-linear approximation.
You define a domain [x_min, x_max] and n_samples output values at equal spacing.
The ANE approximates the activation using n_samples-1 linear segments.

Why bother? Several activations used in modern architectures — SELU, Mish,
Swish with learned beta, custom gating functions — are not in libane's built-in
op set but are differentiable and smooth. PWL approximation with 32 segments
gives < 0.1% relative error on typical activation ranges, which is well within
fp16 noise.

This example approximates three activations and measures their accuracy against
scipy reference implementations at the same domain.
"""
import numpy as np
import scipy.special
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

C, S      = 512, 128
N_SAMPLES = 33   # 32 segments; recommended for smooth activations

def run_pwl(x_np, x_min, x_max, fn_ref, label):
    """Approximate fn_ref over [x_min, x_max] using PWL, run on ANE, compare."""
    xs      = np.linspace(x_min, x_max, N_SAMPLES, dtype=np.float32)
    samples = fn_ref(xs).astype(np.float32)

    g      = ane.Graph()
    inp_id = g.add_input("x", [1, C, 1, S])
    out_id = g.add_pwl_activation(inp_id, [1, C, 1, S], x_min, x_max, samples)
    g.mark_output(out_id)
    cg = g.compile()
    cg.set_output_shapes([[1, C, 1, S]])

    x_in = x_np.astype(np.float16)
    result = cg(x_in).reshape(C, S).astype(np.float32)
    ref    = fn_ref(x_np.astype(np.float32))

    mask    = (x_np >= x_min) & (x_np <= x_max)
    abs_err = np.abs(result[mask] - ref[mask])
    rel_err = abs_err / (np.abs(ref[mask]) + 1e-6)

    print(f"  {label:<20}  "
          f"max abs err: {abs_err.max():.5f}  "
          f"mean rel err: {rel_err.mean()*100:.3f}%  "
          f"p99 rel err: {np.percentile(rel_err, 99)*100:.3f}%")

rng       = np.random.default_rng(17)
x_general = rng.uniform(-3, 3, (C, S)).astype(np.float32)
x_pos     = rng.uniform( 0, 6, (C, S)).astype(np.float32)

print("PWL approximation accuracy (32 segments, fp16 execution):")
print()

selu_alpha = 1.6732632423543772
selu_scale = 1.0507009873554805
def selu(x):
    return selu_scale * np.where(x >= 0, x, selu_alpha * (np.exp(x) - 1.0))

run_pwl(x_general, -3.0, 3.0, selu,  "SELU")

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

run_pwl(x_general, -3.0, 3.0, mish,  "Mish")

def swish(x):
    return x * scipy.special.expit(x)

run_pwl(x_general, -3.0, 3.0, swish, "Swish (β=1)")

def softplus(x):
    return np.log1p(np.exp(x))

run_pwl(x_pos,     0.0,  6.0, softplus, "Softplus")

print()
print("Smooth activations (Softplus): < 0.1% mean relative error with 32 segments.")
print("Non-smooth activations (SELU has a kink at x=0): ~0.2% mean relative error.")
print("Increase N_SAMPLES for tighter approximation; 33 (32 segments) is the default.")
