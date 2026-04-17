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

Perf stats are available through libane_mil_execute_stats() in the C API.
This example uses ctypes against the locally-built dylib.
Build first: cmake --build build -j$(sysctl -n hw.logicalcpu)
"""
import ctypes, os, time
import numpy as np
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

# ── Load dylib ────────────────────────────────────────────────────────────────

_build = os.path.join(os.path.dirname(__file__), "..", "build", "libane.dylib")
_build = os.path.normpath(_build)
if not os.path.exists(_build):
    print(f"dylib not found at {_build}")
    print("Build first: cmake --build build -j$(sysctl -n hw.logicalcpu)")
    raise SystemExit(1)

lib = ctypes.CDLL(_build)

# ── Bind structs and functions ────────────────────────────────────────────────

class PerfStats(ctypes.Structure):
    _fields_ = [
        ("ane_bw_utilization", ctypes.c_float),
        ("avg_bw_state",       ctypes.c_float),
        ("peak_bw_state",      ctypes.c_int),
        ("ane_energy_units",   ctypes.c_long),
        ("throttle_ns",        ctypes.c_long),
        ("available",          ctypes.c_int),
    ]

lib.libane_mil_compile.restype  = ctypes.c_void_p
lib.libane_mil_compile.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
]
lib.libane_mil_execute_stats.restype  = ctypes.c_int
lib.libane_mil_execute_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(PerfStats),
]
lib.libane_mil_release.restype  = None
lib.libane_mil_release.argtypes = [ctypes.c_void_p]

# ── MIL template for a single relu (lightweight, no weights) ─────────────────

MIL_TEMPLATE = """\
program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1,{C},1,{S}]> a_input0) {{
        tensor<fp16, [1,{C},1,{S}]> z_output0 =
            relu(x=a_input0)[name=string("z_output0")];
    }} -> (z_output0);
}}
"""

def run_with_stats(C, S, n_runs=10):
    """Compile and execute a relu dispatch; return averaged PerfStats."""
    mil = MIL_TEMPLATE.format(C=C, S=S).encode()
    handle = lib.libane_mil_compile(mil, None, None, None, 0)
    if not handle:
        raise RuntimeError(f"compile failed for C={C} S={S}")

    n_elem  = C * S
    inp     = np.ones(n_elem, dtype=np.float16)
    out_buf = np.empty(n_elem, dtype=np.float16)
    nbytes  = ctypes.c_size_t(n_elem * 2)

    inp_p = inp.ctypes.data_as(ctypes.c_void_p)
    out_p = out_buf.ctypes.data_as(ctypes.c_void_p)

    in_ptrs  = (ctypes.c_void_p * 1)(inp_p)
    out_ptrs = (ctypes.c_void_p * 1)(out_p)
    in_sizes = (ctypes.c_size_t * 1)(nbytes)
    out_sizes= (ctypes.c_size_t * 1)(nbytes)

    # warm-up (first call loads into SRAM)
    stats = PerfStats()
    lib.libane_mil_execute_stats(handle, in_ptrs, in_sizes, 1,
                                  out_ptrs, out_sizes, 1, ctypes.byref(stats))

    # timed runs
    bw_util_sum = 0.0
    avg_bw_sum  = 0.0
    peak_bw_max = 0
    energy_sum  = 0
    t0 = time.perf_counter()
    for _ in range(n_runs):
        s = PerfStats()
        lib.libane_mil_execute_stats(handle, in_ptrs, in_sizes, 1,
                                      out_ptrs, out_sizes, 1, ctypes.byref(s))
        bw_util_sum += s.ane_bw_utilization
        avg_bw_sum  += s.avg_bw_state
        peak_bw_max  = max(peak_bw_max, s.peak_bw_state)
        energy_sum  += s.ane_energy_units
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000

    lib.libane_mil_release(handle)

    avg = PerfStats()
    avg.ane_bw_utilization = bw_util_sum / n_runs
    avg.avg_bw_state       = avg_bw_sum  / n_runs
    avg.peak_bw_state      = peak_bw_max
    avg.ane_energy_units   = energy_sum  // n_runs
    avg.available          = stats.available
    return avg, elapsed_ms

# ── Run at two sizes ──────────────────────────────────────────────────────────

configs = [
    (64,   16,  "small  (C=64,   S=16 )"),
    (512,  128, "medium (C=512,  S=128)"),
    (2048, 256, "large  (C=2048, S=256)"),
]

print(f"{'config':<28}  {'lat ms':>7}  {'BW util%':>9}  {'avg BW':>7}  "
      f"{'peak BW':>8}  {'energy':>10}")
print("─" * 78)

for C, S, label in configs:
    stats, lat_ms = run_with_stats(C, S)
    if not stats.available:
        print(f"  IOReport unavailable — stats not collected")
        break
    print(f"  {label:<26}  {lat_ms:>7.3f}  "
          f"{stats.ane_bw_utilization*100:>8.1f}%  "
          f"{stats.avg_bw_state:>7.1f}  "
          f"{stats.peak_bw_state:>8d}  "
          f"{stats.ane_energy_units:>10d}")

print()
print("BW util ≈ 0% on small tensors: kernel completes before IOReport samples.")
print("BW util rises with tensor size as ANE DCS bus stays active longer.")
print("energy units are raw IOReport counts — useful for relative comparison only.")
print("throttle_ns > 0 would indicate thermal/power throttling during the dispatch.")
