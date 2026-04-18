#!/usr/bin/env python3
"""
15 — Device introspection

Query ANE hardware capabilities: chip architecture, inference core count,
ANE unit count, and tensor shape limits.

The architecture string maps to Apple chip generations:
  h13g → A15 / M2    h14g → A16 / M2 Pro
  h15g → M3 family   h16g → M4 family

Shape limits report the maximum per-dimension values. Note that max_seq
and max_channels cannot be reached simultaneously — the real constraint
is on-chip SRAM (~32 MB on M3). Use limits as per-dimension guards only.
"""
import ane

print(f"libane {ane.version()}  ANE available: {ane.available()}")
print()

# ── Device info ───────────────────────────────────────────────────────────────

info = ane.device_info()

if not info["available"]:
    print("_ANEDeviceInfo unavailable on this machine.")
else:
    arch = info["architecture"]
    chip = {
        "h13g": "A15 / M2",
        "h14g": "A16 / M2 Pro",
        "h15g": "M3 family",
        "h16g": "M4 family",
    }.get(arch, "unknown chip")

    print(f"Architecture   : {arch}  ({chip})")
    print(f"Inference cores: {info['core_count']}")
    print(f"ANE units      : {info['num_anes']}")
    print()

# ── Shape limits ──────────────────────────────────────────────────────────────

lim = ane.shape_limits()

print(f"Shape limits:")
print(f"  max_seq       : {lim['max_seq']}")
print(f"  max_channels  : {lim['max_channels']}")
print(f"  seq_alignment : {lim['seq_alignment']}  (S must be a multiple of this)")
print()
print(f"SRAM budget note: max_seq × max_channels × 2 bytes = "
      f"{lim['max_seq'] * lim['max_channels'] * 2 / 1024**3:.1f} GB — "
      f"far beyond on-chip SRAM. Both limits cannot be reached simultaneously.")
