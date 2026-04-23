# Compile-Path Tiers

libane's graph compiler routes each fusion group through a cascade of three
warm-path tiers before falling back to full cold compilation. The tiers form
a strictly ascending cost ladder: the cheapest viable path is tried first,
and each tier's failure cost is bounded (no speculative work).

This document describes the tier model, the mechanism behind each tier,
and measured latency on a reference machine.

---

## Architecture

All URL-reconnect state lives in `MilBackend`. `HwxBackend` delegates same-op
reconnect hits to `MilBackend::try_warm_reconnect()` before invoking its own
cross-op machinery, so every compile path — activation ops, weight-bearing
ops, attention, conv2d — shares one cache, keyed by aned's own hexID
equivalence class.

```
HwxBackend::compile_group(group):
    mil_text = MilBuilder::build_fused(target_op_fragment)

    Tier 1 — mil_->try_warm_reconnect(mil_text, {}, ...)
        hexID lookup → ane_reconnect on hit
        returns program or nullptr

    Tier 2 — HwxEmitter.emit(C, S, op) + ane_load_hwx(bytes, mil_text, ...)
        cross-op patch of cached shape template
        returns program or falls through

    Tier 3 — mil_->compile_group(graph, group, ...)
        full ane_compile() via ANECompilerService
        populates MilBackend cache + seeds HwxEmitter on success
```

Weight-bearing ops (matmul, rmsnorm, softmax, layernorm, conv2d, sdpa_*) are
owned by `MilBackend` directly. They see Tier 1 and Tier 3 only; Tier 2 is
activation-only because only weight-free shapes can be cross-op patched
safely. Tier 1 hits the same cache regardless.

---

## Tier 1 — URL reconnect (~1.3–1.7 ms)

**When it hits:** same `(mil_text, weights)` has been compiled before and
aned still holds the compile slot (`compiledModelExists = YES`).

**Mechanism:** `_ANEInMemoryModel.hexStringIdentifier` is computed without
compiling via `ane_compute_hex_id()`. If the hexID is in
`MilBackend::url_cache_`, the cached `model_url` is injected into a fresh
model via `setModelURL:` and `loadWithQoS:` binds the existing compile slot
without re-entering `compileWithQoS:`.

**Miss modes:** cache miss (first time compiling this MIL), or hit but the
aned slot was purged (entry is evicted, caller falls through to Tier 3).

## Tier 2 — HwxEmitter cross-op patch (~25–40 ms)

**When it hits:** a different op at the same shape has been compiled before.
The shape template is cached in `HwxEmitter`; the target op's op-config
words are patched into a copy of the template.

**Mechanism:** `HwxEmitter::emit(C, S, op)` produces a BEEFFACE HWX binary
structurally based on some prior op (typically RELU) but with `op`'s
op-config words substituted at known offsets. `ane_load_hwx()` writes the
patched binary as `model.hwx` at `localModelPath` alongside the target
op's `model.mil`, then calls `compileWithQoS:`. aned's `compileAsNeeded`
path sees `model.hwx` already present and skips `ANECCompile()`,
registering the provided binary in its in-process cache.

**Miss modes:** eligible op set is activation-only (RELU, TANH, SIGMOID,
HARDSWISH, LEAKY_RELU, ELU); no prior compile at this shape means no
template to patch; aned's functional validation (MIL op must match what
the binary computes) rejects mismatched pairs and falls through.

## Tier 3 — cold compile (~15–70 ms)

**When it hits:** first-time compile of a new `(mil_text, weights)` pair.

**Mechanism:** full `ane_compile()` through `compileWithQoS:` →
ANECompilerService → aned. On success, `model_url` + `hex_id` are captured
into `MilBackend::url_cache_` and (for activation ops) the compiled HWX is
captured into `HwxEmitter` via either `capture_from_model_dir` (macOS ≤25)
or `hwx_capture_inline` (macOS 26+).

**Latency varies with op complexity and aned's own disk cache.** Activation
ops at mid scale land around 65 ms. Matmul at small scale lands around
15 ms. Attention and conv2d are heavier. The first-ever compile of a given
shape on a given aned binary version pays an additional one-off
ANECompilerService startup cost (can be several hundred ms).

---

## Measured latencies

Measured on M3 Pro / macOS 26.3.1 via `tests/test_warmpath_tiers_bench.cpp`.

| Workload | Shape | Tier 3 (cold) | Tier 2 (cross-op) | Tier 1 (reconnect) |
|---|---|---:|---:|---:|
| RELU  | C=64, S=512 | 66 ms | — | 1.7 ms |
| TANH  | C=64, S=512 | — (Tier 2 used) | 29 ms | 1.3 ms |
| MATMUL | 64×64, SP=128 | 17 ms | n/a | 1.6 ms |

Speedups on repeat compiles of the same graph node:

- RELU: cold → reconnect ≈ **38×**
- TANH: cross-op → reconnect ≈ **22×**
- MATMUL: cold → reconnect ≈ **10×**

Numbers are end-to-end `compile_group()` latency, not just the
ANE-facing call. Tier 1 in particular includes the MIL-text build step
(microseconds) plus the hexID compute + cache lookup + `ane_reconnect`
round-trip. Just the `ane_reconnect` call itself lands at ~0.7 ms
(measured separately in `tests/test_pathc_warmpath.cpp`).

---

## Compile-slot budget

aned enforces a per-process hard limit of ~115 cold compiles. Tier 1 hits
do **not** consume slots. Tier 2 does consume a slot (it goes through
`compileWithQoS:`, aned just skips the actual `ANECCompile()` work). Tier 3
consumes a slot.

See `libane_compile_count()` and `libane_compile_slots_remaining()` for
introspection.
