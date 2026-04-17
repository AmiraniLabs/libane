# Hardware Introspection

libane exposes three introspection surfaces: ANE device capabilities, tensor shape limits, and per-execution performance counters. None require entitlements or root.

---

## Device info

```c
libane_device_info_t info;
libane_device_info(&info);

if (info.available) {
    printf("architecture: %s\n", info.architecture); // e.g. "h15g"
    printf("core_count:   %u\n", info.core_count);
    printf("num_anes:     %u\n", info.num_anes);
}
```

### `libane_device_info_t` fields

| Field | Type | Description |
|---|---|---|
| `architecture` | `char[32]` | Chip generation string from `_ANEDeviceInfo`. `""` if unavailable. |
| `core_count` | `uint32_t` | Number of ANE inference cores (`+numANECores`). 0 if unavailable. |
| `num_anes` | `uint32_t` | Number of ANE units (`+numANEs`). 0 if unavailable. |
| `available` | `int` | 1 if `_ANEDeviceInfo` was queried successfully, 0 otherwise. |

`libane_device_info()` returns `LIBANE_OK` even when `available == 0` — the struct is zero-filled and the call is not an error. Check `available` before using the values.

### Known architecture strings

| String | Chip |
|---|---|
| `"h11g"` | A14 / M1 |
| `"h12g"` | A15 / M2 |
| `"h13g"` | A16 |
| `"h14g"` | A17 / M3 early |
| `"h15g"` | M3 |
| `"h16g"` | M4 |

These are firmware-reported values from `_ANEDeviceInfo`. New chips may report strings not listed here.

---

## Shape limits

```c
libane_shape_limits_t lim = libane_get_shape_limits();
printf("max_seq:      %d\n", lim.max_seq);
printf("max_channels: %d\n", lim.max_channels);
printf("seq_alignment:%d\n", lim.seq_alignment);  // always 16
```

### `libane_shape_limits_t` fields

| Field | Description |
|---|---|
| `max_seq` | Maximum S dimension. Must also be a multiple of `seq_alignment`. |
| `max_channels` | Maximum C dimension. |
| `seq_alignment` | S must be a multiple of this value. Always 16. |

Limits are chip-adaptive when `_ANEDeviceInfo` is available. When unavailable, conservative universally-safe values are returned.

### SRAM budget

`max_seq` and `max_channels` are **independent per-dimension caps**, not simultaneous limits. The binding constraint is on-chip SRAM.

A `[1, C, 1, S]` activation buffer uses `C × S × 2` bytes. The ANE holds at least input + output simultaneously, so the minimum SRAM requirement for any single dispatch is `2 × C × S × 2` bytes.

| Chip | Approx. SRAM |
|---|---|
| M1 | ~8 MB |
| M2 | ~16 MB |
| M3 | ~32 MB |
| M4 | ~48 MB |

A shape at `max_seq × max_channels` would require gigabytes — far beyond any current chip. At the same time, either dimension can be reached in isolation for small values of the other. Use these limits as per-dimension guards and validate total tensor footprint against your known budget before submission.

`libane_graph_compile()` returns `LIBANE_ERR_COMPILE_FAILED` if firmware rejects the combined size at compile time.

---

## Performance statistics

`libane_mil_execute_stats()` populates a `libane_perf_stats_t` after execution using IOReport hardware counters. No entitlements or root required.

```c
libane_perf_stats_t stats = {0};
libane_mil_execute_stats(h, in_ptrs, in_sizes, n_in,
                            out_ptrs, out_sizes, n_out,
                            &stats);

if (stats.available) {
    printf("BW utilization: %.2f%%\n", stats.ane_bw_utilization * 100.0f);
    printf("avg BW state:   %.1f\n",   stats.avg_bw_state);
    printf("peak BW state:  %d\n",     stats.peak_bw_state);
    printf("energy units:   %ld\n",    stats.ane_energy_units);
    printf("throttle ns:    %ld\n",    stats.throttle_ns);
}
```

### `libane_perf_stats_t` fields

| Field | Type | Description |
|---|---|---|
| `ane_bw_utilization` | `float` | Fraction of time the ANE DCS bus was active (0.0–1.0). |
| `avg_bw_state` | `float` | Mean bandwidth histogram state (0–31 scale). |
| `peak_bw_state` | `int` | Highest bandwidth state observed during this execution. |
| `ane_energy_units` | `long` | ANE energy in raw IOReport units (not millijoules). |
| `throttle_ns` | `long` | Total nanoseconds the ANE spent in any throttle state. |
| `available` | `int` | 1 if IOReport sampling succeeded, 0 otherwise. |

### Interpreting the fields

**`ane_bw_utilization`** is the most directly actionable field. A value near 0.0 typically means one of:
- The op was too small to register meaningful ANE activity
- The op ran on CPU fallback (check `libane_available()` and `libane_last_error()`)
- The kernel was too short-lived for IOReport to sample

A value near 1.0 indicates the ANE DCS bus was saturated — the dispatch was compute-bound.

**`avg_bw_state` / `peak_bw_state`** are raw IOReport histogram states on a 0–31 scale. Higher values indicate higher bandwidth states. These are hardware-specific; there is no universal mapping to GB/s.

**`ane_energy_units`** are raw IOReport units and are not calibrated to millijoules. They are useful for relative comparisons between dispatches on the same machine, not absolute energy measurement.

**`throttle_ns`** is non-zero when the chip was thermally or power-throttled during the dispatch. Sustained non-zero throttle indicates the workload is hitting thermal limits.

**`available == 0`** means IOReport sampling was unavailable on this system or firmware version. Execution still succeeded — stats collection is best-effort and does not affect correctness.

---

## SRAM spill detection

After compiling a multi-operation MIL program, call `libane_mil_sram_spill()` to check whether intermediate activations spilled to DRAM:

```c
libane_mil_handle_t h = libane_mil_compile(mil_text, ...);
if (libane_mil_sram_spill(h)) {
    // Intermediate activations exceed SRAM — expect ~30% throughput penalty
}
```

| Return | Meaning |
|---|---|
| `1` | SRAM spill detected; intermediates are DRAM-backed |
| `0` | No spill; intermediates fit on-chip |
| `-1` | Null handle |

A spill cannot be resolved at runtime. Options: reduce the number of simultaneous live activations, use smaller tile sizes, or split into multiple programs with separate dispatches.

Single-layer programs always return 0 — they have no inter-layer intermediates regardless of tensor size.

---

## Python

Device info and shape limits are not yet exposed in the Python bindings. Use the C API via ctypes or cffi for these in the interim, or check `ane.available()` + `ane.version()` for basic runtime checks.

Performance stats are available through `libane_mil_execute_stats` via the C API only in v0.8.2.
