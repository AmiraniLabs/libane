#!/usr/bin/env python3
"""
16 — Performance statistics via IOReport

libane can sample ANE hardware counters during execution: DCS bus utilization,
bandwidth histogram states, energy units, and throttle residency — all without
entitlements or root, using IOReport directly.

Key insight: ane_bw_utilization is the most actionable field.
  - Near 0.0 → kernel too small to saturate the ANE, or ran on CPU fallback
  - Near 1.0 → ANE DCS bus saturated; compute-bound dispatch

This example runs the same op at two tensor sizes and compares the counters.
Small tensors (D=64, S=16) barely register. Large tensors (D=2048, S=256)
drive the ANE hard and show meaningful utilization.
"""
import time
import numpy as np
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

C_S_CONFIGS = [
    (64,   16,  "small  (C=64,   S=16 )"),
    (512,  128, "medium (C=512,  S=128)"),
    (2048, 256, "large  (C=2048, S=256)"),
]

N_RUNS = 10

print(f"{'config':<28}  {'lat ms':>7}  {'BW util%':>9}  {'avg BW':>7}  "
      f"{'peak BW':>8}  {'energy':>10}")
print("─" * 78)

for C, S, label in C_S_CONFIGS:
    mil = f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1,{C},1,{S}]> a_input0) {{
        tensor<fp16, [1,{C},1,{S}]> z_output0 =
            relu(x=a_input0)[name=string("z_output0")];
    }} -> (z_output0);
}}
"""
    prog = ane.compile_mil(mil)

    n_elem = C * S
    inp    = np.ones(n_elem, dtype=np.float16)

    # warm-up
    prog.run([inp], [n_elem])

    bw_util_sum = avg_bw_sum = energy_sum = 0.0
    peak_bw_max = 0
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _, stats = prog.run_stats([inp], [n_elem])
        bw_util_sum += stats["ane_bw_utilization"]
        avg_bw_sum  += stats["avg_bw_state"]
        peak_bw_max  = max(peak_bw_max, stats["peak_bw_state"])
        energy_sum  += stats["ane_energy_units"]
    elapsed_ms = (time.perf_counter() - t0) / N_RUNS * 1000

    if not stats["available"]:
        print(f"  IOReport unavailable — stats not collected")
        break

    print(f"  {label:<26}  {elapsed_ms:>7.3f}  "
          f"{bw_util_sum / N_RUNS * 100:>8.1f}%  "
          f"{avg_bw_sum / N_RUNS:>7.1f}  "
          f"{peak_bw_max:>8d}  "
          f"{energy_sum // N_RUNS:>10d}")

print()
print("BW util ≈ 0% on small tensors: kernel completes before IOReport samples.")
print("BW util rises with tensor size as ANE DCS bus stays active longer.")
print("energy units are raw IOReport counts — useful for relative comparison only.")
print("throttle_ns > 0 would indicate thermal/power throttling during the dispatch.")
