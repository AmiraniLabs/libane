#!/usr/bin/env python3
"""
18 — SRAM spill detection

When a fused MIL program's inter-layer intermediate activations exceed the
ANE's on-chip SRAM, the firmware spills them to DRAM. This incurs roughly
a 30% throughput penalty and is invisible without explicit detection.

libane_mil_sram_spill() checks _ANEInMemoryModel.intermediateBufferHandle
after compile: non-zero means the firmware allocated a DRAM-backed IOSurface
for intermediates.

Key constraint: spill detection only fires for MULTI-OPERATION fused programs.
Single-layer programs (one matmul, one relu) have no inter-layer intermediates
and always return 0 regardless of tensor size.

This example:
  1. Compiles a single-op program (always 0) to confirm baseline
  2. Compiles a fused multi-op program at increasing sizes until spill is detected
  3. Shows how to fix it by splitting into two dispatches

Build first: cmake --build build -j$(sysctl -n hw.logicalcpu)
"""
import ctypes, os
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
lib.libane_mil_compile.restype  = ctypes.c_void_p
lib.libane_mil_compile.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
]
lib.libane_mil_sram_spill.restype  = ctypes.c_int
lib.libane_mil_sram_spill.argtypes = [ctypes.c_void_p]
lib.libane_mil_release.restype  = None
lib.libane_mil_release.argtypes = [ctypes.c_void_p]

# ── MIL templates ─────────────────────────────────────────────────────────────

def _header(C, S):
    return (f'program(1.3)\n[buildInfo = dict<string, string>('
            f'{{{{"coremlc-component-MIL", "3510.2.1"}}, '
            f'{{"coremlc-version", "3505.4.1"}}, '
            f'{{"coremltools-component-milinternal", ""}}, '
            f'{{"coremltools-version", "9.0"}}}})]'
            f'\n{{\n'
            f'    func main<ios18>(tensor<fp16, [1,{C},1,{S}]> a_input0) {{\n')

def SINGLE_OP(C, S):
    return (_header(C, S) +
            f'        tensor<fp16, [1,{C},1,{S}]> z_output0 =\n'
            f'            relu(x=a_input0)[name=string("z_output0")];\n'
            f'    }} -> (z_output0);\n}}\n')

def FUSED_THREE(C, S):
    return (_header(C, S) +
            f'        tensor<fp16, [1,{C},1,{S}]> v0 =\n'
            f'            relu(x=a_input0)[name=string("v0")];\n'
            f'        tensor<fp16, [1,{C},1,{S}]> v1 =\n'
            f'            relu(x=v0)[name=string("v1")];\n'
            f'        tensor<fp16, [1,{C},1,{S}]> z_output0 =\n'
            f'            relu(x=v1)[name=string("z_output0")];\n'
            f'    }} -> (z_output0);\n}}\n')

def compile_and_check(mil: str, label: str) -> int:
    handle = lib.libane_mil_compile(mil.encode(), None, None, None, 0)
    if not handle:
        print(f"  {label:<40}  compile failed")
        return -1
    spill = lib.libane_mil_sram_spill(handle)
    lib.libane_mil_release(handle)
    return spill

# ── Part 1: single-op baseline ────────────────────────────────────────────────

print("── Part 1: single-op always returns 0 ─────────────────────────────────")
for C, S in [(512, 128), (4096, 512), (4096, 4096)]:
    sram_mb = C * S * 2 / 1024**2
    mil     = SINGLE_OP(C, S)
    spill   = compile_and_check(mil, f"single relu  C={C:>5} S={S:>5} ({sram_mb:.1f} MB)")
    status  = "spill" if spill == 1 else ("0 (no spill)" if spill == 0 else "error")
    print(f"  single relu  C={C:>5} S={S:>5} ({sram_mb:>5.1f} MB/buf)  →  {status}")

print()

# ── Part 2: fused multi-op — sweep until spill ────────────────────────────────

print("── Part 2: fused 3-layer relu — sweep size until SRAM spill ───────────")
print(f"  {'C':>6}  {'S':>6}  {'per-buf MB':>10}  {'total est MB':>13}  spill?")
print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*6}")

spill_found = False
for C in [512, 1024, 2048, 4096, 8192]:
    for S in [128, 256, 512, 1024]:
        sram_per_buf = C * S * 2 / 1024**2
        # 3-layer fused: firmware holds ~2 intermediates simultaneously
        total_est   = sram_per_buf * 3
        mil   = FUSED_THREE(C, S)
        spill = compile_and_check(mil, "")
        status = "YES ← spill!" if spill == 1 else ("no" if spill == 0 else "error")
        print(f"  {C:>6}  {S:>6}  {sram_per_buf:>9.1f}  {total_est:>12.1f}  {status}")
        if spill == 1:
            spill_found = True
            spill_C, spill_S = C, S
            break
    if spill_found:
        break

print()

# ── Part 3: fix by splitting ──────────────────────────────────────────────────

if spill_found:
    print("── Part 3: fix — split the fused program into two dispatches ───────────")
    print(f"  Spill detected at C={spill_C}, S={spill_S}.")
    print(f"  Split: dispatch 1 = relu → relu, dispatch 2 = relu")
    print()

    SPLIT_PART1 = (_header(spill_C, spill_S) +
                   f'        tensor<fp16, [1,{spill_C},1,{spill_S}]> v0 =\n'
                   f'            relu(x=a_input0)[name=string("v0")];\n'
                   f'        tensor<fp16, [1,{spill_C},1,{spill_S}]> z_output0 =\n'
                   f'            relu(x=v0)[name=string("z_output0")];\n'
                   f'    }} -> (z_output0);\n}}\n')

    SPLIT_PART2 = SINGLE_OP(spill_C, spill_S)

    s1 = compile_and_check(SPLIT_PART1, "split part 1 (relu→relu)")
    s2 = compile_and_check(SPLIT_PART2, "split part 2 (relu)")
    print(f"  split part 1 (relu → relu):  spill = {s1}")
    print(f"  split part 2 (relu):         spill = {s2}")
    print()
    if s1 == 0 and s2 == 0:
        print("  Both parts fit in SRAM. Split resolved the spill.")
        print("  Trade-off: one extra DRAM round-trip for the intermediate,")
        print("  but ~30% better throughput vs the DRAM-backed fused version.")
    else:
        print("  Part 1 still spills — reduce C or S further, or use 3 dispatches.")
else:
    print("No spill detected in the tested range — your chip has enough SRAM.")
    print("Try larger C × S values to find the boundary.")
