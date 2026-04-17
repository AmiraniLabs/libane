#!/usr/bin/env python3
"""
15 — Device info & shape limits

libane can query the ANE hardware directly: chip architecture generation,
inference core count, and the per-dimension shape limits that constrain
how large a single tensor dispatch can be.

Key point: shape limits are chip-adaptive. max_seq and max_channels are
independent per-dimension caps — NOT simultaneous limits. The real binding
constraint is on-chip SRAM. Both limits can be hit in isolation but not
together. See the SRAM budget section below.

This information is useful when sizing graphs for a target chip, or when
you want to report hardware context alongside benchmark results.
"""
import ctypes
import os
import numpy as np
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

# ── Load the C library directly for the introspection APIs ───────────────────
# (device_info and shape_limits are C API only in v0.8.0)

_build = os.path.join(os.path.dirname(__file__), "..", "build", "libane.dylib")
_build = os.path.normpath(_build)
if not os.path.exists(_build):
    print(f"dylib not found at {_build}")
    print("Build first: cmake -B build && cmake --build build -j$(sysctl -n hw.logicalcpu)")
    raise SystemExit(1)

lib = ctypes.CDLL(_build)

# ── Bind libane_device_info ───────────────────────────────────────────────────

class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("architecture", ctypes.c_char * 32),
        ("core_count",   ctypes.c_uint32),
        ("num_anes",     ctypes.c_uint32),
        ("available",    ctypes.c_int),
    ]

lib.libane_device_info.restype  = ctypes.c_int
lib.libane_device_info.argtypes = [ctypes.POINTER(DeviceInfo)]

# ── Bind libane_get_shape_limits ──────────────────────────────────────────────

class ShapeLimits(ctypes.Structure):
    _fields_ = [
        ("max_seq",       ctypes.c_int32),
        ("max_channels",  ctypes.c_int32),
        ("seq_alignment", ctypes.c_int32),
    ]

lib.libane_get_shape_limits.restype  = ShapeLimits
lib.libane_get_shape_limits.argtypes = []

# ── Query ─────────────────────────────────────────────────────────────────────

info = DeviceInfo()
lib.libane_device_info(ctypes.byref(info))

print("── ANE device info ─────────────────────────────────────────────────────")
if info.available:
    arch = info.architecture.decode()
    print(f"  Architecture : {arch}")
    print(f"  Core count   : {info.core_count}")
    print(f"  ANE units    : {info.num_anes}")

    # Map known architecture strings to chip names
    arch_map = {
        "h11g": "A14 / M1",
        "h12g": "A15 / M2",
        "h13g": "A16",
        "h14g": "A17 / M3 (early)",
        "h15g": "M3",
        "h16g": "M4",
    }
    chip = arch_map.get(arch, "unknown — new chip?")
    print(f"  Chip         : {chip}")
else:
    print("  _ANEDeviceInfo unavailable on this firmware (fields zeroed)")
print()

lim = lib.libane_get_shape_limits()

print("── Shape limits ────────────────────────────────────────────────────────")
print(f"  max_seq       : {lim.max_seq:,}")
print(f"  max_channels  : {lim.max_channels:,}")
print(f"  seq_alignment : {lim.seq_alignment}  (S must be a multiple of this)")
print()

# ── SRAM budget analysis ──────────────────────────────────────────────────────

print("── SRAM budget ─────────────────────────────────────────────────────────")
print("  max_seq and max_channels are INDEPENDENT per-dimension caps.")
print("  The real constraint is on-chip SRAM. The ANE holds at least")
print("  input + output simultaneously: 2 × C × S × 2 bytes minimum.")
print()

# Typical SRAM budgets (approximate)
sram_map = {"h11g": 8, "h12g": 16, "h15g": 32, "h16g": 48}
sram_mb  = sram_map.get(info.architecture.decode(), None)

if sram_mb:
    print(f"  Approximate on-chip SRAM for {chip}: {sram_mb} MB")
    print()

# Show what shapes are feasible within the limits
print(f"  {'C':>8}  {'S':>8}  {'single buf MB':>14}  {'in+out MB':>10}  {'fits?':>6}")
print(f"  {'─'*8}  {'─'*8}  {'─'*14}  {'─'*10}  {'─'*6}")

test_shapes = [
    (512,              128),
    (2048,             128),
    (4096,             512),
    (8192,             1024),
    (lim.max_channels, 128),
    (512,              lim.max_seq),
]

for C, S in test_shapes:
    buf_mb    = C * S * 2 / 1024**2
    inout_mb  = buf_mb * 2
    fits      = "yes" if (sram_mb and inout_mb < sram_mb) else ("?" if not sram_mb else "no")
    c_str = str(C) if C != lim.max_channels else f"{C} (max)"
    s_str = str(S) if S != lim.max_seq      else f"{S} (max)"
    print(f"  {c_str:>8}  {s_str:>8}  {buf_mb:>13.1f}  {inout_mb:>9.1f}  {fits:>6}")

print()
print("  Rule: validate C × S × 4 bytes (input + output) < SRAM before dispatch.")
print("  libane_graph_compile() returns LIBANE_ERR_COMPILE_FAILED if firmware")
print("  rejects the shape at runtime.")
