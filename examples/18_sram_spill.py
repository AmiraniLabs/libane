#!/usr/bin/env python3
"""
18 — SRAM spill detection

When a fused MIL program's inter-layer intermediate activations exceed the
ANE's on-chip SRAM, the firmware spills them to DRAM. This incurs roughly
a 30% throughput penalty and is invisible without explicit detection.

CompiledMil.sram_spill checks _ANEInMemoryModel.intermediateBufferHandle
after compile: True means the firmware allocated a DRAM-backed IOSurface
for intermediates.

Key constraint: spill detection only fires for MULTI-OPERATION fused programs.
Single-layer programs (one matmul, one relu) have no inter-layer intermediates
and always return False regardless of tensor size.

This example:
  1. Compiles a single-op program (always False) to confirm baseline
  2. Compiles a fused multi-op program at increasing sizes until spill is detected
  3. Shows how to fix it by splitting into two dispatches
"""
import ane
import numpy as np

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

def make_mil(C, S, ops):
    """Build a MIL program with a chain of relu ops."""
    body = f'    func main<ios18>(tensor<fp16, [1,{C},1,{S}]> a_input0) {{\n'
    prev = "a_input0"
    for i, name in enumerate(ops[:-1]):
        body += (f'        tensor<fp16, [1,{C},1,{S}]> {name} =\n'
                 f'            relu(x={prev})[name=string("{name}")];\n')
        prev = name
    last = ops[-1]
    body += (f'        tensor<fp16, [1,{C},1,{S}]> {last} =\n'
             f'            relu(x={prev})[name=string("{last}")];\n')
    body += f'    }} -> ({last});\n}}\n'

    header = (f'program(1.3)\n'
              f'[buildInfo = dict<string, string>('
              f'{{{{"coremlc-component-MIL", "3510.2.1"}}, '
              f'{{"coremlc-version", "3505.4.1"}}, '
              f'{{"coremltools-component-milinternal", ""}}, '
              f'{{"coremltools-version", "9.0"}}}})]'
              f'\n{{\n')
    return header + body

# ── Part 1: single-op baseline ────────────────────────────────────────────────

print("── Part 1: single-op always returns False ──────────────────────────────")
for C, S in [(512, 128), (4096, 512), (4096, 4096)]:
    sram_mb = C * S * 2 / 1024**2
    prog    = ane.compile_mil(make_mil(C, S, ["z_output0"]))
    print(f"  single relu  C={C:>5} S={S:>5} ({sram_mb:>5.1f} MB/buf)  "
          f"→  {'spill' if prog.sram_spill else '0 (no spill)'}")

print()

# ── Part 2: fused multi-op — sweep until spill ────────────────────────────────

print("── Part 2: fused 3-layer relu — sweep size until SRAM spill ───────────")
print(f"  {'C':>6}  {'S':>6}  {'per-buf MB':>10}  {'total est MB':>13}  spill?")
print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*6}")

spill_found = False
for C in [512, 1024, 2048, 4096, 8192]:
    for S in [128, 256, 512, 1024]:
        sram_per_buf = C * S * 2 / 1024**2
        total_est    = sram_per_buf * 3
        prog  = ane.compile_mil(make_mil(C, S, ["v0", "v1", "z_output0"]))
        spill = prog.sram_spill
        print(f"  {C:>6}  {S:>6}  {sram_per_buf:>9.1f}  {total_est:>12.1f}  "
              f"{'YES ← spill!' if spill else 'no'}")
        if spill:
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

    p1 = ane.compile_mil(make_mil(spill_C, spill_S, ["v0", "z_output0"]))
    p2 = ane.compile_mil(make_mil(spill_C, spill_S, ["z_output0"]))
    print(f"  split part 1 (relu → relu):  sram_spill = {p1.sram_spill}")
    print(f"  split part 2 (relu):         sram_spill = {p2.sram_spill}")
    print()
    if not p1.sram_spill and not p2.sram_spill:
        print("  Both parts fit in SRAM. Split resolved the spill.")
        print("  Trade-off: one extra DRAM round-trip for the intermediate,")
        print("  but ~30% better throughput vs the DRAM-backed fused version.")
    else:
        print("  Part 1 still spills — reduce C or S further, or use 3 dispatches.")
else:
    print("No spill detected in the tested range — your chip has enough SRAM.")
    print("Try larger C × S values to find the boundary.")
