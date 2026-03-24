#!/usr/bin/env python3
"""
07 — QKV projection: one graph, three outputs

Attention requires three projections from the same input:
    Q = x @ W_Q,  K = x @ W_K,  V = x @ W_V

Instead of three separate compile+execute calls, define a single graph that
marks all three as outputs.  libane compiles them into one CompiledGraph with
three fusion groups (one per projection) and dispatches all three in one
execute() call.

Benefits:
  - One compile call instead of three
  - Shared input IOSurface allocation
  - Potential for future fused-QKV kernel
"""
import time
import numpy as np
import ane

D    = 2048   # model dim
H    = 256    # head dim
N_H  = 8      # number of heads  → total QKV dim = H × N_H = 2048
SEQ  = 128
QKV  = H * N_H

rng = np.random.default_rng(5)
def rand(r, c): return (rng.standard_normal((r, c)) * 0.02).astype(np.float16)

W_Q = rand(D, QKV)
W_K = rand(D, QKV)
W_V = rand(D, QKV)

g = ane.Graph()
x   = g.add_input("x", [1, D, 1, SEQ])

q   = g.add_op(ane.MATMUL, [x], [1, QKV, 1, SEQ], weights=W_Q)
k   = g.add_op(ane.MATMUL, [x], [1, QKV, 1, SEQ], weights=W_K)
v   = g.add_op(ane.MATMUL, [x], [1, QKV, 1, SEQ], weights=W_V)

g.mark_output(q, "Q")
g.mark_output(k, "K")
g.mark_output(v, "V")

cg = g.compile()
cg.set_output_shapes([
    [1, QKV, 1, SEQ],
    [1, QKV, 1, SEQ],
    [1, QKV, 1, SEQ],
])

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)

# Warm-up
for _ in range(3): cg(x_data)

N = 50
times = []
for _ in range(N):
    t0 = time.perf_counter()
    Q, K, V = cg(x_data)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"QKV projection  D={D}  heads={N_H}  head_dim={H}  seq={SEQ}")
print(f"  Q shape: {Q.shape}")
print(f"  K shape: {K.shape}")
print(f"  V shape: {V.shape}")
print(f"  Latency p50: {times[N//2]:.2f} ms  (all 3 projections)")
print()

# Verify against numpy reference
ref_Q = (W_Q.astype(np.float32).T @ x_data.astype(np.float32)).astype(np.float16)
diff  = np.abs(Q.reshape(QKV, SEQ).astype(np.float32) - ref_Q.astype(np.float32))
print(f"  Q vs numpy ref mean abs err: {diff.mean():.5f}")
