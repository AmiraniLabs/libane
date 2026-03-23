#!/usr/bin/env python3
"""
FFN inference example — libane v0.7.0
Amirani Labs

Builds a SwiGLU feed-forward block, compiles it to ANE, and runs a
timed forward pass:

    x  →  gate_proj (matmul) → SiLU  ─┐
    x  →  up_proj   (matmul)          ├─ mul → down_proj (matmul) → out
                                       ┘

Model dimensions match a typical 1B-parameter transformer FFN:
  D   = 2048   (model dim)
  H   = 5632   (intermediate, ~2.75× D)
  SEQ = 128    (sequence length)

Usage:
    python examples/ffn_inference.py

On ANE-capable hardware this runs in a single compile + warm-up + timed loop.
On non-ANE machines (fallback mode) it exits with a clear message.
"""

import time
import sys
import numpy as np

try:
    import ane
except ImportError:
    sys.exit("ane module not found — build with: cmake -DLIBANE_BUILD_PYTHON=ON && cmake --build build --target ane")

# ── Config ─────────────────────────────────────────────────────────────────

D   = 2048   # model dim
H   = 5632   # FFN intermediate dim
SEQ = 128    # sequence length
N_WARMUP = 3
N_TIMED  = 20

# ── Check hardware ──────────────────────────────────────────────────────────

print(f"libane {ane.version()}  |  ANE available: {ane.available()}")
if not ane.available():
    sys.exit("ANE not available on this machine — cannot run ANE example.")

# ── Random weights (fp16) ───────────────────────────────────────────────────

rng = np.random.default_rng(42)

def rand_w(rows, cols):
    return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float16)

W_gate = rand_w(D, H)      # gate projection
W_up   = rand_w(D, H)      # up projection
W_down = rand_w(H, D)      # down projection
scale  = np.ones(D, dtype=np.float16)   # RMSNorm scale

# ── Build graph ─────────────────────────────────────────────────────────────

print("\nBuilding graph…")
g = ane.Graph()

x     = g.add_input("x",  [1, D, 1, SEQ])

# RMSNorm
rn    = g.add_op(ane.RMSNORM, [x],         [1, D, 1, SEQ], weights=scale)

# Gate path: matmul + SiLU
gate  = g.add_op(ane.MATMUL,  [rn],        [1, H, 1, SEQ], weights=W_gate)
silu  = g.add_op(ane.SILU,    [gate],      [1, H, 1, SEQ])

# Up path: matmul
up    = g.add_op(ane.MATMUL,  [rn],        [1, H, 1, SEQ], weights=W_up)

# Merge: elementwise mul
merged= g.add_op(ane.MUL,     [silu, up],  [1, H, 1, SEQ])

# Down projection
out   = g.add_op(ane.MATMUL,  [merged],    [1, D, 1, SEQ], weights=W_down)

g.mark_output(out, "ffn_out")

# ── Compile ─────────────────────────────────────────────────────────────────

print("Compiling… (first compile ~4 s, cached <1 ms)")
t0 = time.perf_counter()
cg = g.compile()
compile_ms = (time.perf_counter() - t0) * 1000
print(f"Compile time: {compile_ms:.0f} ms")

cg.set_output_shapes([[1, D, 1, SEQ]])

# ── Warm-up ─────────────────────────────────────────────────────────────────

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)
print(f"\nWarm-up ({N_WARMUP} runs)…")
for _ in range(N_WARMUP):
    result = cg(x_data)

# ── Timed run ────────────────────────────────────────────────────────────────

print(f"Timing ({N_TIMED} runs)…")
times = []
for _ in range(N_TIMED):
    t0 = time.perf_counter()
    result = cg(x_data)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
p50 = times[len(times) // 2]
p95 = times[int(len(times) * 0.95)]
mean= sum(times) / len(times)

tokens_per_sec = 1000 / p50   # SEQ tokens per second at p50

print(f"\n── Results ───────────────────────────────────────")
print(f"  Input  shape : {x_data.shape}  fp16")
print(f"  Output shape : {result.shape}  fp16")
print(f"  Latency  p50 : {p50:.2f} ms")
print(f"  Latency  p95 : {p95:.2f} ms")
print(f"  Latency mean : {mean:.2f} ms")
print(f"  Throughput   : {tokens_per_sec * SEQ:.0f} tokens/s  (at {SEQ} tok/batch)")
print(f"──────────────────────────────────────────────────")

# Basic sanity: output should be finite and non-trivially non-zero
assert np.all(np.isfinite(result.astype(np.float32))), "Output contains NaN/Inf!"
nonzero_frac = np.mean(result.astype(np.float32) != 0.0)
print(f"\n  Non-zero output fraction: {nonzero_frac:.3f}  (expected ~1.0)")
print("\nDone.")
