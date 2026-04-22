# Probes

Standalone ObjC diagnostic binaries. Each targets one specific aned /
ANE behavior. They do **not** link `libane`; they use Apple private
frameworks directly to isolate findings from libane's own abstractions.

## Probes

- `probe_compile_budget/` — exercises and measures the per-process
  compile-slot limit (~119). Drove the `kCompileHardLimit` guard in
  `ane_compile()`.

- `probe_delta_reload/` — probes whether weight-blob modifications
  between `compileWithQoS:` and `loadWithQoS:` affect the loaded
  program. (Result: no effect — aned snapshots at compile time.)

- `probe_device_info/` — enumerates ANE device properties visible to
  the client (compute units, firmware version, etc.).

- `probe_dynamic_ops/` — tests which MIL ops accept dynamic input
  shapes vs require static compile-time shapes.

- `probe_perf_stats/` — inspects `_ANEProgramForEvaluation`'s perf
  counters (cycles, SRAM usage, tile count) after execution.

- `probe_qos/` — sweep of `compileWithQoS:` / `loadWithQoS:` /
  `evaluateWithQoS:` across QoS classes to find which the ANE
  actually respects.

- `probe_sram_spill/` — detects SRAM→DRAM spill via
  `intermediateBufferHandle`. Confirmed the ~32 MB SRAM threshold.

## Building

Each probe is self-contained. Build pattern (from the probe's own dir):

```
clang -fobjc-arc -framework Foundation \
  -framework IOSurface -framework CoreFoundation \
  probe_NAME.m -o probe_NAME
```

Some probes also link `AppleNeuralEngine.framework`:

```
clang -fobjc-arc -framework Foundation \
  -F /System/Library/PrivateFrameworks \
  -framework AppleNeuralEngine \
  probe_NAME.m -o probe_NAME
```

Binaries are gitignored — commit only the `.m` source.
