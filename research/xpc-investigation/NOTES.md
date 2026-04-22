# XPC Investigation — Notes

Chronological probes into `com.apple.ANECompilerService` XPC surface and
adjacent aned behaviors on macOS 26. Each XPC-N is a `TEST_CASE` in
`test_xpc_protocol.mm`.

## Summary of findings

**XPC-1..8 — protocol discovery.** Established that `_ANEClient` already
holds an XPC connection to `ANECompilerService`. Read the
`remoteObjectInterface` off that connection to recover the real protocol
and its method signatures from ObjC runtime metadata.

**XPC-9, XPC-10 — in-process compile.** Confirmed `ANECompilerService.xpc`
can be `dlopen`'d into the caller. `_ANEMILCompiler.compileModelAt:` is
callable as a class method directly, producing `model.hwx` in a
caller-specified output directory. Same class method the XPC service
invokes internally — just a shorter code path that avoids the XPC
round-trip.

**XPC-11 — argument shape.** `compileModelAt:` expects a **directory** URL
plus a `modelName:` string. Passing a file URL caused `InvalidCompilationParam`.
The fixed shape is `src_url` + `@"model.mil"`.

**XPC-12 — URL vs hexID cache, three outcomes.** Tested whether aned reads
from the model URL or from its hexID-keyed in-process table. Outcome 3
confirmed: with RELU in cache, a `setModelURL:` pointing to TANH-patched
bytes followed by `loadWithQoS:` loaded successfully but executed RELU.
aned uses its hexID cache; the URL is not read at load time.

**XPC-13 — inline compile and aned's compile table.** `_ANEMILCompiler`
called inline does **not** register results in `com.apple.aned`'s compile
table. `compiledModelExists` returns NO after inline compile. `loadWithQoS:`
with the inline output dir triggers `ANECCompile() FAILED` because aned
goes down the recompile path (no `model.mil` in that dir).

**XPC-14 — `ane_load_mlmodelc` with pre-staged HWX.** All probes failed
with `_ANEEspressoIRTranslator : error Cannot load network
'.../model.espresso.net'`. `_ANEClient.compileModel:` is on the Espresso
stack — a different compilation pipeline. Not applicable to MIL-compiled
models.

**XPC-15 — `_ANEMILCompiler` output inventory.** The compiler writes only
`model.hwx` + `model.src` (63 bytes: the `saveSourceURL` path). No Espresso
files. Confirms XPC-14 is a dead end for MIL programs.

**XPC-16 — `_ANECoreMLModelCompiler` probe.** `pathsForModelURL:` resolves
to `model.espresso.net`. `compileModelAt:csIdentity:key:...` also requires
Espresso source. Same pipeline as `_ANEClient`; same dead end.

**XPC-17 — `_ANEInMemoryModel` attributes.** Discovered that
`localModelPath` is deterministic: `TempDir/{hexID}`, known **before**
`compileWithQoS:`. `compilerOptionsWithOptions:isCompiledModelCached:`
returns `kANEFInMemoryModelIsCachedKey=1` + `kANEFIsInMemoryModelTypeKey`
= the hexID. `setModelAttributes:` with `isPrecompiledModel=YES` had no
effect on its own — model.mil was missing from localModelPath.

**XPC-18 — the breakthrough.** Pre-staging **both** `model.mil` (target
op) **and** `model.hwx` (patched bytes) at `localModelPath` before
`compileWithQoS:` causes aned to skip `ANECCompile()` and use the staged
binary. Four probes:

| Probe | MIL | HWX | Result |
|---|---|---|---|
| A | tanh | tanh | TANH ✓ (baseline) |
| B | tanh | RELU-template + TANH op config | **TANH ✓** (cross-op patch accepted) |
| C | relu | tanh | RELU (aned recompiled — mismatch rejected) |
| D | tanh | (none) | TANH ✓ (used model.hwx from Probe A, cached on disk) |

**XPC-19 — end-to-end validation.** The full `ane_load_hwx` flow:
`hwx_capture_inline` → `HwxEmitter` → MIL-builder text → `ane_load_hwx`.
Output `0x2E5D` = TANH at 37.6ms (vs ~111ms cold compile).

## Production counterpart

`tests/test_ane_load_hwx.mm` — three regression tests that pin the
current behavior discovered here (baseline load, cross-op patch, argument
validation).

## Key invariants observed

- aned validates MIL ↔ HWX as **functionally consistent**, not
  byte-identical. Different-sized binaries pass if they compute the same
  op (Probe B: 49152b patched passed as tanh despite tanh's own compile
  producing 65536b).
- `localModelPath = TempDir/{hexID}` — stable across process restarts
  for the same MIL + weights.
- `hexID` is derived server-side by aned. The client cannot spoof it.
