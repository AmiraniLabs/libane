# ANE Research

Investigations into Apple Neural Engine internals that informed production
code in `libane`. Each artifact here is a time-stamped diagnostic: it reflects
the state of macOS / aned / ANE private frameworks at the moment it was
written, and is kept for re-running when a future OS release changes behavior.

## Layout

- `probes/` — standalone diagnostic binaries (ObjC/C) that poke at specific
  aned behaviors (compile budget, QoS, SRAM spill, device info, etc.). Each
  is built with its own `build_and_run.sh` or manually via `clang`.

- `xpc-investigation/` — the XPC-1..XPC-17 probe chain that mapped out the
  ANECompilerService protocol surface and discovered how to load arbitrary
  HWX binaries on macOS 26. See `xpc-investigation/NOTES.md`.

- `pathc-investigation/` — investigations into aned's per-process compile
  table, `compiledModelExists` / URL-reconnect behavior, precompiled-model
  loading paths, compiler options, QoS sweeps, and AOT binary storage. See
  `pathc-investigation/NOTES.md`.

## What's here vs in `tests/`

Files in `research/` are **not** part of the regression test suite. They were
load-bearing for the investigation that produced a feature, but once the
feature is shipped and has its own regression test in `tests/`, the original
investigation stays here as historical record.

Production behavior is pinned by tests in `tests/` — e.g.
`tests/test_ane_load_hwx.mm` holds the regression for the mechanism XPC-18
and XPC-19 discovered.

## Rerunning

When an OS update breaks something, the probes here are the fastest way to
diagnose the change. They are intentionally standalone (no libane dependency
where possible) so they keep working even if `libane` itself is broken.

The XPC and path-C test files still link against the in-tree Catch2 and
libane_static, but they are not built by default. To rerun:

```
cd research/xpc-investigation
clang++ -std=c++17 -fobjc-arc -I../../include -I../../src \
  -framework Foundation -framework IOSurface \
  test_xpc_protocol.mm -L../../build -lane -o /tmp/xpc && /tmp/xpc
```

(Adjust as needed; these were originally wired through `tests/CMakeLists.txt`
before extraction.)
