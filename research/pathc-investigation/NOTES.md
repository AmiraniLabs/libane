# Path-C Investigation — Notes

Investigations into aned's per-process compile table, URL reconnect timing,
precompiled-model loading, compiler options, QoS behavior, and AOT binary
storage on macOS 26. Findings landed in production as `ane_reconnect`,
`ane_load_hwx`, `compileWithQoS:` skip logic, and the `kCompileHardLimit`
guard.

## Files

- `test_pathc_probe.mm` — initial Path-C discovery. Established that
  `compiledModelExists` returns YES after a compile and that a fresh
  `_ANEInMemoryModel` with the same hexID can `loadWithQoS:` without
  a compile.

- `test_pathc3_reconnect.mm` — timing probes for the three reconnect
  strategies (initWithModelIdentifier, setProgramHandle:, fresh model).
  Result: fresh-model + setModelURL: + loadWithQoS: is ~0.722ms. This
  became `ane_reconnect()`.

- `test_precompiled_path.mm` — tested aned's `isPreCompiled` code path
  by pre-staging a binary and setting flags. Failed pre-XPC-18: the
  missing piece was writing `model.mil` at the same location. Superseded
  by XPC-18.

- `test_compiler_options_probe.mm` — enumerated `_ANEInMemoryModel`'s
  `compilerOptionsWithOptions:isCompiledModelCached:` output across
  compile states. Revealed `kANEFInMemoryModelIsCachedKey` and
  `kANEFIsInMemoryModelTypeKey` option keys. Findings fed into XPC-17.

- `test_aot_binary_probe.mm` — search for where aned writes the AOT
  compiled binary on macOS 26. Checked ModelAssetsCache, modelDataVault,
  systemModelsCacheDirectory. Conclusion: nothing observable from the
  client side — the binary lives exclusively in aned's IPC table.
  Complemented by `hwx_capture_inline` (inline compile via `dlopen`)
  which provides the bytes via a parallel path.

- `test_hwx_search.mm` — filesystem scan for `.hwx` files during and
  after compile. Confirmed no `.hwx` is written to any client-visible
  location on macOS 26. Motivated the inline-compile approach.

- `test_qos_sweep.mm` — tested `compileWithQoS:` behavior across QoS
  classes. Probe 4 empirically found the per-process ~119 compile-slot
  limit; exceeding it causes silent failures then SIGSEGV. Led to
  `kCompileHardLimit` guard in `ane_compile()`.

- `test_compile_diag.cpp` — debug-logging compile for manual diagnostics
  when a compile fails unexpectedly. Not a regression test.

## Production counterparts

- `libane::runtime::ane_reconnect()` — `test_pathc3_reconnect.mm`
- `libane::runtime::ane_load_hwx()` — `test_precompiled_path.mm` (+
  XPC-18/19; see `../xpc-investigation/NOTES.md`)
- `libane::runtime::ane_compile()` slot-limit guard — `test_qos_sweep.mm`
- `hwx_capture_inline()` — `test_aot_binary_probe.mm`,
  `test_hwx_search.mm`

## Key invariants observed

- aned enforces a per-process ~119 compile-slot budget. Exceeding it
  is unrecoverable without process restart.
- `compiledModelExists` tracks aned's in-process cache. A hard-unload
  (via `unloadWithQoS:` + release) purges the slot; a soft-unload
  (release without unload call) keeps it alive.
- ANECompilerService has its own disk cache (Layer 2, ~40ms hits).
  aned's in-process cache is Layer 3 (~1ms hits). Cold compile is
  Layer 1 (~4200ms first run, ~111ms subsequent cold-but-cached).
