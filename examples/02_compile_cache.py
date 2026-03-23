#!/usr/bin/env python3
"""
02 — Compile cache: cold vs warm

ANE compilation is slow (~4 s) because it runs a full hardware compiler.
libane caches every compiled program keyed on (op, shape, weight-hash).
Subsequent calls with the same signature return in under a millisecond.

This example makes the cache timing concrete so you can plan your startup
strategy: compile all layers once at boot, then execute at full speed.
"""
import time
import numpy as np
import ane

ane.set_log_level(ane.LOG_SILENT)   # suppress error noise

D, SEQ = 256, 128
W = np.random.randn(D, D).astype(np.float16)
x = np.random.randn(D, SEQ).astype(np.float16)

g = ane.Graph()
inp = g.add_input("x", [1, D, 1, SEQ])
out = g.add_op(ane.MATMUL, [inp], [1, D, 1, SEQ], weights=W)
g.mark_output(out)

print("Cold compile (first time, hardware compilation)…")
ane.cache_flush()
t0 = time.perf_counter()
cg = g.compile()
cold_ms = (time.perf_counter() - t0) * 1000
print(f"  {cold_ms:.0f} ms")

cg.set_output_shapes([[1, D, 1, SEQ]])
cg(x)  # warm-up execute

print("\nWarm compile (same graph, cache hit)…")
times = []
for _ in range(10):
    t0 = time.perf_counter()
    cg2 = g.compile()
    times.append((time.perf_counter() - t0) * 1000)
    cg2.set_output_shapes([[1, D, 1, SEQ]])

warm_ms = min(times)
print(f"  {warm_ms:.3f} ms   ({cold_ms/warm_ms:.0f}× faster than cold)")

print("\nExecution latency (compiled graph, hot path)…")
exec_times = []
for _ in range(100):
    t0 = time.perf_counter()
    cg(x)
    exec_times.append((time.perf_counter() - t0) * 1000)
exec_times.sort()
print(f"  p50: {exec_times[50]:.3f} ms")
print(f"  p99: {exec_times[99]:.3f} ms")

print(f"\nSummary:")
print(f"  Cold compile : {cold_ms:.0f} ms  — pay once at startup")
print(f"  Warm compile : {warm_ms:.3f} ms  — free after first call")
print(f"  Execute p50  : {exec_times[50]:.3f} ms  — hot-path cost")
