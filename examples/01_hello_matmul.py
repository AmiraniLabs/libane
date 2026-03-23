#!/usr/bin/env python3
"""
01 — Hello, ANE matmul

The simplest possible libane program.  Multiply two fp16 matrices on the ANE
and compare the result and timing against numpy.

Key point: the ANE uses a dedicated matrix-multiply fabric that consumes zero
GPU and near-zero CPU cycles.  Your GPU is completely free for rendering while
inference runs in parallel.
"""
import time
import numpy as np
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

M, K, N = 512, 1024, 512   # A[M×K] × B[K×N] = C[M×N]

rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(np.float16)
B = rng.standard_normal((K, N)).astype(np.float16)

# ── numpy reference ──────────────────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(50):
    C_np = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
np_ms = (time.perf_counter() - t0) / 50 * 1000

# ── ANE (warm — first call may compile) ──────────────────────────────────────
ane.matmul(A, B)             # warm-up / compile
t0 = time.perf_counter()
for _ in range(50):
    C_ane = ane.matmul(A, B)
ane_ms = (time.perf_counter() - t0) / 50 * 1000

# ── Compare ───────────────────────────────────────────────────────────────────
C_ane_f32 = C_ane.view(np.uint16).astype(np.float32)   # reinterpret uint16 → fp16 → f32
# numpy result already fp16 stored as float16
diff = np.abs(C_np.astype(np.float32) - C_ane_f32).mean()

print(f"Matrix shape:  A[{M}×{K}] × B[{K}×{N}] = C[{M}×{N}]")
print(f"numpy  (CPU):  {np_ms:.2f} ms")
print(f"libane (ANE):  {ane_ms:.2f} ms   {np_ms/ane_ms:.1f}× faster")
print(f"Mean abs diff: {diff:.4f}  (fp16 rounding expected)")
