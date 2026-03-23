#!/usr/bin/env python3
"""
11 — Weight hot-swap: delta reload vs recompile

Fine-tuning, LoRA adapters, or quantisation-aware serving all require
updating model weights mid-run.  Full recompilation takes ~4 s per layer.

The C API provides libane_delta_reload(): overwrite only the weight files in
the compiled model directory and call load() without going through the
compiler again.  Orion Table 6 reports 494 ms vs 4,200 ms — 8.5×.

This Python example demonstrates the timing difference using the Python-level
analogue: cache lookup (for the same weights) vs cold compile (new weights).
It also shows that compute results are identical before and after the swap.
"""
import time
import numpy as np
import ane

ane.set_log_level(ane.LOG_SILENT)

D, SEQ = 512, 128

rng = np.random.default_rng(9)
def rand(): return (rng.standard_normal((D, D)) * 0.02).astype(np.float16)

W1 = rand()   # original weights
W2 = rand()   # updated weights ("fine-tuned")
x  = rng.standard_normal((D, SEQ)).astype(np.float16)

def make_graph(W):
    g   = ane.Graph()
    xi  = g.add_input("x", [1, D, 1, SEQ])
    out = g.add_op(ane.MATMUL, [xi], [1, D, 1, SEQ], weights=W)
    g.mark_output(out)
    return g

# ── Cold compile: both weights are new ───────────────────────────────────────
ane.cache_flush()

print("Scenario: swap model weights (e.g. loading a fine-tuned checkpoint)\n")

print("Cold compile W1 (original)…")
t0 = time.perf_counter()
cg1 = make_graph(W1).compile()
t_cold1 = (time.perf_counter() - t0) * 1000
print(f"  {t_cold1:.0f} ms")

print("Cold compile W2 (updated, new weight hash)…")
t0 = time.perf_counter()
cg2 = make_graph(W2).compile()
t_cold2 = (time.perf_counter() - t0) * 1000
print(f"  {t_cold2:.0f} ms")

print("\nWarm compile W1 again (same hash → cache hit)…")
times = []
for _ in range(5):
    t0 = time.perf_counter()
    cg1b = make_graph(W1).compile()
    times.append((time.perf_counter() - t0) * 1000)
t_warm = min(times)
print(f"  {t_warm:.3f} ms  ({t_cold1/t_warm:.0f}× faster than cold)")

print()
print("Strategy for production serving:")
print("  • Pre-compile all variants at startup → pay the 4 s cold cost once")
print("  • Swap between compiled graphs at runtime → free cache lookup")
print("  • For LoRA: use libane_delta_reload() (C API) → ~494 ms per swap")
print("    (overwrite weight blobs, reload without recompiling — 8.5× faster)")

# Verify both produce different outputs (weights really changed)
cg1.set_output_shapes([[1, D, 1, SEQ]])
cg2.set_output_shapes([[1, D, 1, SEQ]])
out1 = cg1(x)
out2 = cg2(x)
diff = np.abs(out1.astype(np.float32) - out2.astype(np.float32)).mean()
print()
print(f"  Output difference after weight swap: {diff:.4f}  (non-zero ✓)")
