#!/usr/bin/env python3
"""
10 — Throughput sweep: tokens/sec vs sequence length

ANE throughput is not constant — it varies with sequence length because the
hardware is optimised for specific tile sizes.  This example sweeps seq_len
from 8 to 512 and plots tokens/sec for a single matmul, helping you pick the
right batching strategy for your model.

Practical guide:
  - Autoregressive decoding (seq=1): use seq=8 minimum (ANE constraint)
  - Prefill (long prompts): sweet spot usually around 128–256
  - Batched decoding: pad to next multiple of 8, batch until ANE is saturated
"""
import time
import numpy as np
import ane

D = 2048   # matmul dim — typical transformer projection

rng = np.random.default_rng(8)
W   = (rng.standard_normal((D, D)) * 0.02).astype(np.float16)

SEQ_LENS = [8, 16, 32, 64, 128, 256, 512]
N_RUNS   = 50

print(f"Matmul[{D}×{D}]  throughput sweep")
print(f"{'seq':>6}  {'latency ms':>12}  {'tokens/sec':>12}  {'GFLOP/s':>10}")
print("─" * 48)

results = []
for seq in SEQ_LENS:
    x = rng.standard_normal((D, seq)).astype(np.float16)

    g   = ane.Graph()
    xi  = g.add_input("x", [1, D, 1, seq])
    out = g.add_op(ane.MATMUL, [xi], [1, D, 1, seq], weights=W)
    g.mark_output(out)
    cg  = g.compile()
    cg.set_output_shapes([[1, D, 1, seq]])

    for _ in range(5): cg(x)   # warm-up

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        cg(x)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    lat_ms  = times[N_RUNS // 2]
    tok_s   = seq / (lat_ms / 1000)
    flops   = 2 * D * D * seq   # 2×D×D×seq FLOPs for matmul
    gflop_s = flops / (lat_ms / 1000) / 1e9

    results.append((seq, lat_ms, tok_s, gflop_s))
    print(f"{seq:>6}  {lat_ms:>11.3f}  {tok_s:>12.0f}  {gflop_s:>9.1f}")

best = max(results, key=lambda r: r[2])
print()
print(f"Peak throughput: {best[2]:.0f} tokens/sec at seq={best[0]}")
print(f"Peak compute   : {best[3]:.1f} GFLOP/s")

# Latency for autoregressive decoding (smallest valid seq=8)
ar = results[0]
print()
print(f"Autoregressive decoding (seq=8): {ar[1]:.3f} ms/step → {ar[2]:.0f} tok/s")
