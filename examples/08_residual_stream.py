#!/usr/bin/env python3
"""
08 — Residual stream: projection + skip connection

The residual (skip) connection is the backbone of every transformer:
    output = x + projection(x)

x is consumed twice (once as input to the projection, once for the add),
so fusion stops at the branch point.  The graph produces two groups:
  group 0: matmul     (projection)
  group 1: add        (residual merge — x is a side input)

This is still faster than calling libane_execute twice, because:
  - Single compile step for both ops
  - No Python overhead between dispatches
  - Pre-allocated intermediate buffer reused across calls
"""
import time
import numpy as np
import ane

D   = 2048
SEQ = 128
N   = 100

rng = np.random.default_rng(6)
W   = (rng.standard_normal((D, D)) * 0.02).astype(np.float16)

g     = ane.Graph()
x     = g.add_input("x",   [1, D, 1, SEQ])
proj  = g.add_op(ane.MATMUL, [x],        [1, D, 1, SEQ], weights=W)
out   = g.add_op(ane.ADD,    [proj, x],  [1, D, 1, SEQ])   # proj + residual
g.mark_output(out)

cg = g.compile()
cg.set_output_shapes([[1, D, 1, SEQ]])

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)

for _ in range(5): cg(x_data)   # warm-up

times = []
for _ in range(N):
    t0 = time.perf_counter()
    result = cg(x_data)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"Matmul[{D}×{D}] + residual add   seq={SEQ}")
print(f"  p50: {times[N//2]:.3f} ms")
print(f"  p95: {times[int(N*0.95)]:.3f} ms")

# Verify: output ≈ W×x + x
ref = (W.astype(np.float32).T @ x_data.astype(np.float32) + x_data.astype(np.float32)).astype(np.float16)
diff = np.abs(result.reshape(D, SEQ).astype(np.float32) - ref.astype(np.float32))
print(f"  vs numpy ref mean abs err: {diff.mean():.5f}")
print()
print("  Two fusion groups: [matmul] + [add with residual side-input]")
print("  x is read once from the IOSurface, used by both groups.")
