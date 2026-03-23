#!/usr/bin/env python3
"""
04 — Fusion benefit: 3 dispatches vs 1

Without fusion, a Matmul → RMSNorm → GELU chain requires three separate ANE
dispatches.  Each dispatch writes the intermediate result to DRAM, then the
next dispatch reads it back — wasting memory bandwidth.

libane's fusion engine detects the linear chain and compiles all three ops
into a single MIL program with a single dispatch.  The intermediates live in
ANE SRAM and never touch DRAM.

This example measures both paths directly.
"""
import time
import numpy as np
import ane

D, SEQ = 512, 128
N_RUNS = 100

rng = np.random.default_rng(2)
W   = rng.standard_normal((D, D)).astype(np.float16)
scl = np.ones(D, dtype=np.float16)
x   = rng.standard_normal((D, SEQ)).astype(np.float16)

# ── Unfused: compile 3 separate programs, dispatch 3 times ──────────────────
g_mm  = ane.Graph(); xi = g_mm.add_input("x", [1,D,1,SEQ])
t1 = g_mm.add_op(ane.MATMUL,  [xi], [1,D,1,SEQ], weights=W)
g_mm.mark_output(t1)
cg_mm = g_mm.compile(); cg_mm.set_output_shapes([[1,D,1,SEQ]])

g_rn  = ane.Graph(); xi = g_rn.add_input("x", [1,D,1,SEQ])
t2 = g_rn.add_op(ane.RMSNORM, [xi], [1,D,1,SEQ], weights=scl)
g_rn.mark_output(t2)
cg_rn = g_rn.compile(); cg_rn.set_output_shapes([[1,D,1,SEQ]])

g_ge  = ane.Graph(); xi = g_ge.add_input("x", [1,D,1,SEQ])
t3 = g_ge.add_op(ane.GELU,    [xi], [1,D,1,SEQ])
g_ge.mark_output(t3)
cg_ge = g_ge.compile(); cg_ge.set_output_shapes([[1,D,1,SEQ]])

# warm-up
for _ in range(5):
    tmp = cg_mm(x); tmp = cg_rn(tmp); cg_ge(tmp)

t0 = time.perf_counter()
for _ in range(N_RUNS):
    tmp = cg_mm(x)
    tmp = cg_rn(tmp)
    cg_ge(tmp)
unfused_ms = (time.perf_counter() - t0) / N_RUNS * 1000

# ── Fused: one graph, one dispatch ───────────────────────────────────────────
g = ane.Graph()
xi  = g.add_input("x",  [1, D, 1, SEQ])
t1  = g.add_op(ane.MATMUL,  [xi], [1, D, 1, SEQ], weights=W)
t2  = g.add_op(ane.RMSNORM, [t1], [1, D, 1, SEQ], weights=scl)
t3  = g.add_op(ane.GELU,    [t2], [1, D, 1, SEQ])
g.mark_output(t3)
cg  = g.compile()
cg.set_output_shapes([[1, D, 1, SEQ]])

for _ in range(5): cg(x)    # warm-up

t0 = time.perf_counter()
for _ in range(N_RUNS):
    cg(x)
fused_ms = (time.perf_counter() - t0) / N_RUNS * 1000

print(f"Chain: Matmul[{D}×{D}] → RMSNorm → GELU   (seq={SEQ})")
print()
print(f"  Unfused (3 dispatches, 2 DRAM round-trips): {unfused_ms:.3f} ms")
print(f"  Fused   (1 dispatch,   0 DRAM round-trips): {fused_ms:.3f} ms")
print(f"  Speedup: {unfused_ms/fused_ms:.2f}×")
print()
print("  Fusion eliminates intermediate DRAM writes — the outputs of Matmul")
print("  and RMSNorm stay in ANE SRAM and flow directly into the next op.")
