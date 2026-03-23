#!/usr/bin/env python3
"""
06 — SwiGLU feed-forward block  (Llama 3 / Mistral FFN)

SwiGLU replaces the classic FFN with:
    FFN(x) = (SiLU(gate_proj(x)) ⊙ up_proj(x)) @ down_proj

It requires two separate matmuls on x (gate and up), then an elementwise
multiply of their outputs.  x is consumed twice, so gate and up start in
separate fusion groups.  down_proj fuses with the multiply.

Graph topology:

    x ──→ gate_proj (matmul) → SiLU ──┐
    │                                  ├── mul → down_proj → output
    └──→ up_proj   (matmul) ───────────┘

The three ANE programs are:
    group 0: gate_proj + SiLU   (fused — linear chain)
    group 1: up_proj            (standalone — x is branched)
    group 2: mul + down_proj    (fused — linear chain)
"""
import time
import numpy as np
import ane

# Llama 3 8B dimensions
D   = 4096
H   = 14336   # SwiGLU intermediate (3.5× D)
SEQ = 128

rng = np.random.default_rng(4)
def rand(r, c): return (rng.standard_normal((r, c)) * 0.02).astype(np.float16)

W_gate = rand(D, H)
W_up   = rand(D, H)
W_down = rand(H, D)

g = ane.Graph()
x      = g.add_input("x", [1, D, 1, SEQ])

gate   = g.add_op(ane.MATMUL, [x],          [1, H, 1, SEQ], weights=W_gate)
silu   = g.add_op(ane.SILU,   [gate],        [1, H, 1, SEQ])

up     = g.add_op(ane.MATMUL, [x],           [1, H, 1, SEQ], weights=W_up)

merged = g.add_op(ane.MUL,    [silu, up],    [1, H, 1, SEQ])
out    = g.add_op(ane.MATMUL, [merged],      [1, D, 1, SEQ], weights=W_down)

g.mark_output(out)

print("Compiling SwiGLU graph…")
cg = g.compile()
cg.set_output_shapes([[1, D, 1, SEQ]])

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)

# Warm-up
for _ in range(3): cg(x_data)

# Timed run
N = 50
times = []
for _ in range(N):
    t0 = time.perf_counter()
    result = cg(x_data)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"\nSwiGLU FFN  D={D}  H={H}  seq={SEQ}")
print(f"  Parameters : {(D*H + D*H + H*D) / 1e6:.0f} M  (gate + up + down)")
print(f"  Latency p50: {times[N//2]:.2f} ms")
print(f"  Latency p95: {times[int(N*0.95)]:.2f} ms")
print(f"  Output shape: {result.shape}")
print(f"  Output finite: {np.all(np.isfinite(result.astype(np.float32)))}")
