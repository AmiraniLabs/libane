# Pass 9 — LOAD_PROGRAM orchestrator decoded (full section taxonomy)

Disassembled the function at `0x54f78` in M3 (Erebus, H15) — the
top-level program-load orchestrator that walks the program-context
struct, dumps each section, and dispatches per-section validators.

## The big correction

This pass **invalidates the section taxonomy from Passes 7–8**. The
program-context struct has **9 sections**, not 6 or 7, and the names
M1 used in its `[VERIFICATION]` strings were partially **legacy
names** that M3's verbose builder renames.

### Definitive section table (M3 verbose section-name strings)

| Slot | Offset    | Name (M3)              | M1 legacy name |
|------|-----------|------------------------|----------------|
| 0    | `+0x008`  | `genericSection`       | Generic Section |
| 1    | `+0x038`  | `kernelSection`        | (sub-pointer on M1) |
| 2    | `+0x068`  | `textSection`          | **TD Section** |
| 3    | `+0x098`  | `operationSection`     | Operation Section |
| 4    | `+0x0c8`  | `procedureSection`     | Procedure Section |
| 5    | `+0x0f8`  | `kernelPropSection`    | KernelProp |
| 6    | `+0x128`  | `textPropSection`      | **TD Prop Section** |
| 7    | `+0x158`  | `opDbgSection`         | (new in M3) |
| 8    | `+0x188`  | `procPropSection`      | (new in M3) |

Each descriptor is a uniform 48 bytes (0x30); table runs +0x008..+0x1b8
contiguously. The "Generic is double-sized" claim from Pass 7 was
wrong — what looked like a 0x60 first slot was actually two adjacent
0x30 slots (`genericSection` and `kernelSection`).

### Why M1 names differ from M3

The "TD Section" / "TD Prop Section" terminology M1 uses is **legacy
from the H11/H13-era nomenclature**. The actual data is the same
(task descriptors), but M2+ renamed to "text" / "textProp" terminology
— consistent with the broader M1→M2 refactor that eliminated the
standalone `CSneTDDrv` driver and folded TD handling into the engine
(seen in Pass 3 cross-gen diff).

`opDbgSection` (operation debug info) and `procPropSection` (procedure
properties) are **new in M3** — neither appears in M1's section list.
These are post-M1 additions, probably tied to:
- `opDbgSection` — the per-Op debug events surfaced via
  `CAneDebugEventsManager`
- `procPropSection` — extra procedure metadata for the cache-request
  call variants added after M1

## The orchestrator function (0x54f78)

Reconstructed C:

```cpp
void loadProgramOrchestrator(Logger* logger, ProgramCtx* prog) {
    // Phase 1: dump every section descriptor for build-info logging.
    sectionDumpOrLog(logger, &prog->genericSection,    "genericSection");
    sectionDumpOrLog(logger, &prog->kernelSection,     "kernelSection");
    sectionDumpOrLog(logger, &prog->textSection,       "textSection");
    sectionDumpOrLog(logger, &prog->operationSection,  "operationSection");
    sectionDumpOrLog(logger, &prog->procedureSection,  "procedureSection");
    sectionDumpOrLog(logger, &prog->kernelPropSection, "kernelPropSection");
    sectionDumpOrLog(logger, &prog->textPropSection,   "textPropSection ");  // sic — trailing space
    sectionDumpOrLog(logger, &prog->opDbgSection,      "opDbgSection");
    sectionDumpOrLog(logger, &prog->procPropSection,   "procPropSection");

    // Phase 2: per-section validators.
    validateGeneric        (logger, &prog->genericSection);          // 0x553b0
    validateOperation      (logger, &prog->operationSection);        // 0x5550c → 0x64338
    validateProcedure      (logger, &prog->procedureSection);        // 0x55694
    validateKernelProp     (logger, &prog->kernelPropSection);       // 0x557a0
    validateTextAndTextProp(logger, &prog->textPropSection,
                                    &prog->textSection);              // 0x558ac (tail)
}
```

### Validator entry points (M3)

| PC          | Validates                                  | Notes |
|-------------|--------------------------------------------|-------|
| `0x553b0`   | genericSection                             | (next-pass target) |
| `0x5550c`   | operationSection (wrapper around 0x64338)  | decoded this pass |
| `0x55694`   | procedureSection                           | (next-pass target) |
| `0x557a0`   | kernelPropSection                          | (next-pass target) |
| `0x558ac`   | textSection + textPropSection (combined)   | tail-called; cross-validates TID linkages |
| `0x64338`   | inner Operation array validator            | decoded in Pass 6 |

**No explicit validator for** `kernelSection`, `opDbgSection`,
`procPropSection`. These are dumped but not validated in this
orchestrator — they're either:
- validated as a side-effect of another section's validator (e.g.,
  Procedure validator probably touches procPropSection cross-fields)
- pure metadata that doesn't need structural validation (debug info)

## Section descriptor layout — confirmed

The Operation-section validator helper at `0x5550c` reads its input
descriptor as:

```asm
0x55528: ldrb w8, [x1]          ; section.valid       (+0x00, byte)
0x55530: ldr  x0, [x1, #0x18]   ; section.dataPtr     (+0x18, u64)
0x5553c: ldr  x1, [x1, #0x20]   ; section.size        (+0x20, u64)
```

→ **Definitive `SectionDescriptor` layout:**

```cpp
struct SectionDescriptor {           // sizeof == 0x30 (48)
    uint8_t   valid;                 // +0x00 — bit 0 set if section present
    uint8_t   _pad0[0x17];           // +0x01..+0x18
    void*     dataPtr;               // +0x18 — host-shared section data ptr
    uint64_t  size;                  // +0x20 — declared section data size
    uint8_t   _pad1[0x08];           // +0x28..+0x30 — likely flags/type/version
};
```

This **confirms Pass 7's hypothesized shape** for the descriptor
(valid at +0, ptr at +0x18, size at +0x20), but the slot count and
naming were wrong.

## The Operation-section validator wrapper (0x5550c)

This wraps the inner validator at `0x64338` with **two passes** and
result-flag bookkeeping:

```cpp
bool validateOperationSectionWrapper(Logger* log, SectionDescriptor* sec) {
    int outFlag = 0;
    if (!sec) goto err;
    if (!(sec->valid & 1)) return /* skip */;
    if (!sec->dataPtr) return;

    // First pass: validate.
    bool ok = validateOperationSection(sec->dataPtr, sec->size, &outFlag);

    if (!ok) {
        log_msg(log, "...", LINE_278);

        // Second pass: re-validate (probably revalidates with corrected/clipped data).
        ok = validateOperationSection(sec->dataPtr, sec->size, &outFlag);
        if (!ok) goto fatal_err;
    }

    if (outFlag) return /* with reported failure code */;

    log_msg(log, "...", LINE_280);
    if (outFlag == 0) return /* success path */;
    /* ... */
}
```

The two-pass pattern is unusual. Most likely interpretations:
- The validator both **validates and records sub-failures** via the
  `outFlag` (so even when the function returns true, individual issues
  may be flagged). The retry logs failures.
- OR firmware patches the data between the two calls (trimming an
  out-of-range count, etc.) and revalidates.

The third entry to `0x64338` we noted in Pass 8 (at `0x64f30`) is
a separate runtime re-checker, distinct from this load-time wrapper.

## Implications for libane (more)

22. **A program has 9 named sections, not 7.** Any libane code that
    builds a program needs to set up all nine descriptor slots even
    if some are zero-valid. The Pass 4 list ("seven sections") was
    incomplete.
23. **`textSection` and `textPropSection` are the modern names.** Map
    libane's internal naming to these (current firmware terminology).
    "TD" / "TdProp" are M1-era names; expect the public Apple
    framework headers to use one or the other depending on age.
24. **`kernelSection` got promoted to a top-level slot in M3.** On M1
    it was reachable via a sub-pointer chain. If we ever pull data
    *out* of a Kernel section, the access path differs by gen.
25. **`opDbgSection` and `procPropSection` are M3-only additions.** A
    libane program built only for M3 can use them; portable programs
    should leave them empty (`valid=0`).
26. **Each per-section validator has its own entry point** — a
    failure can be attributed to one specific validator, which makes
    error mapping much cleaner.
27. **Validators run twice on the load path.** Don't expect to
    "succeed once" — the firmware does pre-DMA + post-DMA passes.

## Field-offset table — corrected and consolidated (v3)

| Struct                | Field                       | Offset    | Type   |
|-----------------------|-----------------------------|-----------|--------|
| `Program`             | genericSection              | `+0x008`  | (0x30) |
| `Program`             | kernelSection               | `+0x038`  | (0x30) |
| `Program`             | textSection                 | `+0x068`  | (0x30) |
| `Program`             | operationSection            | `+0x098`  | (0x30) |
| `Program`             | procedureSection            | `+0x0c8`  | (0x30) |
| `Program`             | kernelPropSection           | `+0x0f8`  | (0x30) |
| `Program`             | textPropSection             | `+0x128`  | (0x30) |
| `Program`             | opDbgSection                | `+0x158`  | (0x30) |
| `Program`             | procPropSection             | `+0x188`  | (0x30) |
| `Program`             | (unmapped)                  | `+0x1b8`..`+0x204` | wrapper state |
| `Program`             | totalBufferNbr              | `+0x204`  | u32    |
| `Program`             | buffer[0]                   | `+0x208`  | (0x30) |
| `SectionDescriptor`   | valid                       | `+0x000`  | u8     |
| `SectionDescriptor`   | dataPtr                     | `+0x018`  | u64    |
| `SectionDescriptor`   | size                        | `+0x020`  | u64    |
| `OperationSectionData`| count                       | `+0x000`  | u32    |
| `OperationSectionData`| ops[]                       | `+0x004`  | stride 1036 |
| `Operation`           | opType                      | `+0x000`  | u32    |
| `Operation`           | nbrOfNe                     | `+0x004`  | u16    |
| `Operation`           | nbrOfLocalbarSetup          | `+0x008`  | u32    |
| `Bar`                 | index                       | `+0x000`  | u32    |

## Next-pass candidates

1. **Decode validator at `0x55694`** (procedureSection) — get
   procedure-array layout and the content-type matrix in detail.
2. **Decode validator at `0x557a0`** (kernelPropSection) — confirm
   the overlap-detection and offset-bounds logic.
3. **Decode validator at `0x558ac`** (text + textProp combined) —
   the cross-section TID linkage check.
4. **Decode validator at `0x553b0`** (genericSection) — the
   buffer-table validator (we already know each entry but not the
   full per-buffer field layout beyond `valid` + `type`).
5. **Trace the wrapper struct at `+0x1c0`** — Pass 6 saw it loaded
   as a back-pointer; finishing that decode would explain the
   "two struct views" relationship from Pass 8.
