# ANE Runtime Boundary: Two-Path Model

## Summary

The Apple Neural Engine exposes two fundamentally different runtime paths,
separated by a firmware resource management gate. libane operates entirely on
one side of that gate. This document maps what is and is not accessible from
each path, and why.

---

## The Two Paths

### Path A — In-Memory / MIL Compile (libane's path)

Entry: `_ANEInMemoryModelDescriptor` → `_ANEInMemoryModel.compileWithQoS:` →
`_ANEInMemoryModel.loadWithQoS:`

The model is compiled from MIL text at runtime, producing an E5 FlatBuffer
written to a temp directory. The compiled program is loaded directly into ANE
SRAM without going through `_ANEClient`.

Internal state after load:
- `programHandle` — non-zero (loaded program)
- `intermediateBufferHandle` — always **0** (never allocated)
- `queueDepth` — 127

### Path B — Espresso / Client Load (CoreML's path)

Entry: Espresso IR (`model.espresso.net`) → `_ANEClient.loadModel:` →
`_ANEModel` with UUID

The model is loaded through `_ANEClient`, which registers the model's buffer
layout with the ANE firmware's IOMMU mapping table. This registration step
allocates `intermediateBufferHandle` and pre-maps IOSurface IOVAs.

---

## The Gate: `intermediateBufferHandle`

`intermediateBufferHandle` is the firmware-side resource handle that gates
access to the ANE's pre-registered buffer system. Every higher-level feature
that requires firmware-side knowledge of buffer layout checks this handle:

| Feature | Requires intermediateBufferHandle | Accessible from Path A |
|---|---|---|
| `processRequest:model:qos:...` | No | **Yes** |
| `evaluateWithQoS:options:request:` | No | **Yes** |
| `_ANERequest.completionHandler` | No | **Yes** |
| Shared IOSurface zero-copy chaining | No | **Yes** |
| `_ANESharedEvents` signal events | Yes | No (crash: nil C++ vtable) |
| `_ANESharedEvents` wait events (condition met) | Yes | No (crash: nil C++ vtable) |
| `_ANESharedEvents` wait events (condition unmet) | Yes | No (silently ignored) |
| `mapIOSurfacesWithRequest:cacheInference:` | Yes | No (error 0x12) |
| `_ANEIOSurfaceObject.startOffset` honored by DMA | Yes | No (ignored) |
| `_ANEChainingRequest` / `prepareChainingWithModel:` | Yes | No (error 15) |
| `processInputBuffers:` / `processOutputSet:` | Yes | No (silent fail) |

---

## What startOffset Actually Is

`_ANEIOSurfaceObject.startOffset` is a buffer descriptor field consumed by
`_ANEProgramIOSurfacesMapper` when it registers surface layouts with the
firmware. When registration succeeds (Path B only), the firmware stores an
IOVA mapping that includes the offset. At DMA time the firmware uses that
stored mapping — not the CPU-visible surface base address — to DMA correctly
into the offset sub-region.

On Path A, `mapIOSurfacesWithRequest:cacheInference:` always fails with
error 13 / 0x12 (`Program IOSurfaces map failure`). The error is identical
regardless of whether the AIO objects carry offset=0 or offset=N, meaning the
mapper never reaches offset validation — it fails before that on the missing
`intermediateBufferHandle`. With no IOVA mapping registered, `processRequest:`
DMAes from the IOSurface base address unconditionally.

Verified: output appears at byte 0 of the surface regardless of `startOffset`
value, with `sum(|data at slotT|) = 0.00` across all offset values tested
(1, 64, 128, 256, 512, 1K, 4K, 8K, 16K).

---

## What libane Can Do (Path A capabilities)

### Confirmed working

**`processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:`**
Direct dispatch on `_ANEProgramForEvaluation`, bypassing `_ANEInMemoryModel`
dispatch overhead. Measured 13% lower latency than `evaluateWithQoS:` (0.365
vs 0.413 ms/eval). Wired into `ane_runtime.mm` as the default dispatch path.

**Zero-copy sequential chaining via shared IOSurfaces**
Running kernel A with output surface S, then kernel B with S as input surface,
produces coherent data with no CPU touch between calls. Verified at 8 KB
(64×32) and 768 KB (768×256) tensor scale.

- 8 KB: coherent (max_diff = 0.000469, fp16 rounding only)
- 768 KB: coherent (max_diff = 0.000484); shared surface is 6.4% faster than
  explicit CPU copy baseline; copy overhead ~34 μs per inter-kernel transition

The existing libane architecture (separate IOSurface per tensor, sequential
`ane_execute` calls) already implements zero-copy chaining correctly. This
is confirmed, not an approximation.

**`_ANERequest.completionHandler`**
A block property on `_ANERequest` that fires asynchronously on
`ANEServicesThread` after `evaluateWithQoS:` returns. The IOSurface output
buffer is coherent at the time the handler fires. Unlike `_ANESharedEvents`,
this mechanism requires no C++ event infrastructure and works on Path A.

Pattern for async ANE dispatch on Path A:
```objc
id req = /* build request */;
[req setCompletionHandler:^{
    /* runs on ANEServicesThread, ~0.2ms after evaluateWithQoS: returns */
    /* IOSurface output is coherent here */
    dispatch_semaphore_signal(done);
}];
[model evaluateWithQoS:kQoS options:@{} request:req error:nil];
/* caller can be on any thread; evaluateWithQoS: blocks until hardware done */
/* handler fires shortly after, independently */
```

Timing (64×32 model): eval wall ≈ 0.1–0.4 ms; handler fires ≈ 0.2 ms later
(XPC completion notification roundtrip from ANE daemon to client).

---

## What libane Cannot Do Without a CoreML Detour

- Packing multiple tensors into one IOSurface (startOffset ignored by DMA)
- True firmware-level kernel pipelining (_ANEChainingRequest)
- IOSurface pre-mapping for repeated inference (mapIOSurfaces)
- Loopback buffer support (processInputBuffers / processOutputSet)
- `_ANESharedEvents` signal/wait for cross-process or Metal GPU↔ANE sync

All of these require `intermediateBufferHandle != 0`, which requires
`_ANEClient.loadModel:`, which requires `model.espresso.net` format — a
completely different compilation pipeline incompatible with the MIL text
path.

### _ANESharedEvents crash analysis

`_ANESharedEvents` crashes on Path A because the `processRequest:` completion
block maintains a C++ event infrastructure pointer for event processing. That
pointer is nil when `intermediateBufferHandle=0`. At completion, the block
attempts C++ virtual dispatch on the nil pointer:

```
ldr x9, [x8, #0x10]!   ;; x8 = nil → EXC_BAD_ACCESS at address 0x10
blraa x9, x8            ;; C++ vtable virtual call with self=nil
```

(crash frame: `__100-[_ANEProgramForEvaluation processRequest:...]_block_invoke + 1320`)

Crash conditions:
- Any `_ANESharedSignalEvent` on the request → always crashes
- Any `_ANESharedWaitEvent` whose wait condition is already met → crashes
- Any `_ANESharedWaitEvent` whose condition is never met → silently ignored (no crash)
- Empty `_ANESharedEvents` (@[], @[]) → no crash (nothing to process)

The crash cannot be worked around from user space. `_ANERequest.completionHandler`
is the correct async notification mechanism for Path A.

---

## Legitimate Optimization: Buffer Pool Reuse

Since tensor packing is unavailable, the next best approach is to eliminate
IOSurface allocation overhead by pre-allocating a fixed-size pool of surfaces
and recycling them across forward passes. IOSurface allocation has non-trivial
kernel overhead; reusing surfaces eliminates it entirely. This requires no
private API and fits within Path A's capabilities.

---

## Investigated During

- Branch `experiment/chaining-request`: `_ANEChainingRequest`,
  `_ANEProgramForEvaluation`, zero-copy chaining confirmation,
  `processRequest:` optimization, `intermediateBufferHandle` root cause
- Branch `experiment/iosurface-startoffset`: `startOffset` mechanism,
  alignment sweep, functional test, `mapIOSurfacesWithRequest:` mapper
  analysis, `vm_remap` reasoning
- Branch `experiment/ane-shared-events`: `_ANESharedEvents` crash root cause,
  `_ANESharedSignalEvent` / `_ANESharedWaitEvent` ivar layout, crash condition
  matrix, `_ANERequest.completionHandler` as working alternative
