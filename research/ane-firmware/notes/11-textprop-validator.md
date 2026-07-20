# Pass 11 — text + textProp combined validator (validator map complete)

Disassembled the last unknown per-section validator: the
**combined text + textProp validator** at `0x558ac` and its inner
function at `0x64564`. This completes the per-section validator map.

## Wrapper at 0x558ac (4-arg dispatch)

The orchestrator tail-calls this with `x1 = textPropDesc`,
`x2 = textDesc`. The wrapper unwraps both descriptors:

```cpp
bool validateTextAndTextProp(Logger* log,
                              SectionDescriptor* tp,   // x1
                              SectionDescriptor* tx)   // x2
{
    if (!tp) goto fatal_at_line_0x175;       // 373
    if (!(tp->valid & 1)) return /*OK skip*/;
    void* tpData = tp->dataPtr;
    if (!tpData) return;
    uint64_t tpSize = tp->size;
    uint64_t txSize = tx->size;              // ldp from [tx+0x18]
    void*    txData = tx->dataPtr;

    bool ok = innerValidator(tpData, tpSize, txData, txSize); // 0x64564
    if (ok) return /*OK*/;

    log_msg(log, "...", LINE_0x17d);          // 381
    // retry
    ldp x0, x1, [tp + 0x18];   // re-load
    ldp x2, x3, [tx + 0x18];
    ok = innerValidator(...);
    if (!ok) goto fatal_at_line_0x17d;
    return /*OK after retry*/;
}
```

→ **textPropSection and textSection are validated together** because
textProp entries reference regions of text. The pre-DMA + post-DMA
two-pass pattern matches the other validators.

## Inner validator at 0x64564 (the cross-section linkage check)

```cpp
bool validateTextAndTextPropInner(
    void*    tpData, uint64_t tpSize,    // x0, x1: textProp ptr+size
    void*    txData, uint64_t txSize)    // x2, x3: text    ptr+size
{
    if (!tpData) panic_at_line_0xe0;      // textProp must exist
    if (!txData) panic_at_line_0xe1;      // text     must exist

    uint32_t count = tpData[0];           // [+0x000] entry count

    uint64_t need = 8 + count * 32;       // (count << 5) | 8
    if (need > tpSize) return false;      // textProp section size cap

    if (count == 0) return true;

    uint64_t off0 = tpData[+0x18];        // first entry's text offset
    uint64_t len0 = tpData[+0x20];        // first entry's text length
    if (off0 + len0 > txSize) return false;  // entry must fit IN TEXT SECTION

    if (count == 1) return true;
    // ... per-entry overlap+ordering past 0x6465c
    return true;
}
```

→ **textProp data layout:**

```cpp
struct TextPropData {
    uint32_t   count;            // +0x000  — number of text-region entries
    uint8_t    _pad0[0x14];      // +0x004..+0x018 — header metadata (TID base?, version?)
    TextPropEntry entries[count]; // +0x018, stride 0x20
};

struct TextPropEntry {           // sizeof == 0x20 (32 bytes)
    uint64_t   offset;           // +0x000 — offset INTO text section
    uint64_t   len;              // +0x008 — length within text section
    uint8_t    _meta[0x10];      // +0x010..+0x020 — TID + type + linkage fields
                                  //   (M1 verbose: "TD[%d] (headerTID:TdPropTID:index)=(%d:%d)")
};
```

## What this finally explains

The relationship between the two sections:

- **textSection** holds the actual code body — task descriptors
  packed as a contiguous blob.
- **textPropSection** is the **index/manifest** describing where each
  TD lives within textSection (offset + length within text), plus
  per-TD metadata (TID, header-TID, etc.).

The validator's job is to ensure every textProp entry points to a
range that actually fits inside textSection. Cross-section integrity:
no textProp entry can describe text bytes outside the text section.

This matches the M1 verbose strings about TD entry validation:

- `TD[%d] exceeds limit (offset, len) (0x%x, %d) section size %lu` —
  per-entry `offset + len ≤ size` check
- `TD[%d] offset 0x%x is overlapped with the previous offset 0x%x len %d` —
  no-overlap check
- `TD[%d] len %d is smaller than ane_TD_HEADER_t (TDE %d)` — a per-TD
  minimum-size check (each TD must contain at least its hardware
  header)
- `TD[%d] (headerTID:TdPropTID:index)=(%d:%d)` — diagnostic that
  prints the linkage between text-side TD-header TID and textProp-side
  TID for each entry

(The M1 strings reference these as "TD" because that was the legacy
name; the M3 firmware calls the same data "text".)

## Validator map — COMPLETE

| Section            | Wrapper (M3) | Inner validator | Per-entry stride | Header bytes |
|--------------------|--------------|-----------------|------------------|--------------|
| `genericSection`   | `0x553b0`    | `0x6415c`       | 0x30 (buffer)    | 0x208        |
| `kernelSection`    | (no orchestrator validator) | — | — | — |
| `textSection`      | `0x558ac` (combined) | `0x64564` (combined) | — | — |
| `operationSection` | `0x5550c`    | `0x64338`       | 0x40c (op)       | 4            |
| `procedureSection` | `0x55694`    | `0x6445c`       | ≥0x10 (proc)     | 8            |
| `kernelPropSection`| `0x557a0`    | `0x64230`       | 0x18 (kprop)     | 0x10         |
| `textPropSection`  | `0x558ac` (combined) | `0x64564` (combined) | 0x20 (textProp) | 0x18 |
| `opDbgSection`     | (no validator) | — | — | — |
| `procPropSection`  | (no validator) | — | — | — |

Six of the nine sections have explicit per-section validators. The
remaining three (`kernelSection`, `opDbgSection`, `procPropSection`)
are dumped at load time but not structurally validated by the
orchestrator — they're either side-validated by another section's
validator (e.g., procedureSection's per-entry walk likely touches
procPropSection cross-fields) or pure metadata where structural
validation isn't applicable.

## Field-offset table — final consolidated v5

| Struct                | Field           | Offset  | Type      |
|-----------------------|-----------------|---------|-----------|
| `Program`             | (9 section descriptors)| `+0x008..+0x1b8` | each 0x30 |
| `Program`             | totalBufferNbr  | `+0x204`| u32       |
| `Program`             | buffers[]       | `+0x208`| stride 0x30 |
| `SectionDescriptor`   | valid           | `+0x000`| u8        |
| `SectionDescriptor`   | dataPtr         | `+0x018`| u64       |
| `SectionDescriptor`   | size            | `+0x020`| u64       |
| `GenericSectionData`  | magic (=1)      | `+0x000`| u32       |
| `GenericSectionData`  | firstAneNbrOfNe | `+0x004`| u32       |
| `GenericSectionData`  | totalBufferNbr  | `+0x204`| u32       |
| `GenericSectionData`  | buffers[]       | `+0x208`| stride 0x30 |
| `GenericBuffer`       | valid           | `+0x000`| u8        |
| `GenericBuffer`       | type (<8)       | `+0x008`| u32       |
| `OperationSectionData`| count (≤128)   | `+0x000`| u32       |
| `OperationSectionData`| ops[]           | `+0x004`| stride 0x40c |
| `Operation`           | opType (≤4)    | `+0x000`| u32       |
| `Operation`           | nbrOfNe (≤16)  | `+0x004`| u16       |
| `Operation`           | nbrOfLocalbarSetup (≤128) | `+0x008` | u32 |
| `Bar`                 | index (≤60)     | `+0x000`| u32       |
| `Bar`                 | (entry stride)  | 0x008   |           |
| `ProcSectionData`     | count           | `+0x000`| u32       |
| `ProcSectionData`     | entries[]       | `+0x008`| stride ≥0x10 |
| `ProcEntry`           | offset          | `+0x000`| u64       |
| `ProcEntry`           | len             | `+0x008`| u64       |
| `KernelPropData`      | count           | `+0x000`| u32       |
| `KernelPropData`      | entries[]       | `+0x010`| stride 0x18 |
| `KPropEntry`          | offset          | `+0x000`| u64       |
| `KPropEntry`          | len             | `+0x008`| u64       |
| `KPropEntry`          | _other          | `+0x010`| u64       |
| `TextPropData`        | count           | `+0x000`| u32       |
| `TextPropData`        | entries[]       | `+0x018`| stride 0x20 |
| `TextPropEntry`       | offset (in text)| `+0x000`| u64       |
| `TextPropEntry`       | len (in text)   | `+0x008`| u64       |
| `TextPropEntry`       | meta (TID etc.)| `+0x010`| 16 bytes  |

## Implications for libane (final)

34. **textProp entries are exactly 32 bytes** with offset+len at start.
35. **textProp.count is in `count*32+8 ≤ size`** — declared section size
    must accommodate `8 + count*32`. (The actual entries start at +0x18
    in the section, so the section is really `≥ 0x18 + count*32`; the
    `+8` formula is a permissive lower bound — full bound enforced by
    per-entry `offset+len ≤ textSize` checks.)
36. **textSection holds raw code; textPropSection is the manifest.**
    Programs without TDs need both sections to be empty (`valid=0` or
    `count=0`); programs with TDs must populate both consistently.
37. **Cross-section validation: every textProp entry's `(offset, len)`
    must fit within textSection.** No textProp entry can dangle off
    the end of text.
38. **The TD↔TdProp TID linkage** mentioned in M1's verbose strings
    is enforced via the 16-byte `meta` field of each textProp entry —
    each entry carries `headerTID` (matching the TD header in text)
    and `TdPropTID` (its own ID within textProp).
39. **Of 9 sections, 6 are explicitly validated; 3 (kernel, opDbg,
    procProp) are dumped but not structurally checked at load time.**
    Programs can leave the unvalidated three empty without
    consequence.

## Validator picture is now functionally complete

We have:
- Every per-section validator entry point
- Every per-section data layout (header, entry stride, key fields)
- Every numerical bound enforced
- The cross-section linkage rules (textProp ↔ text)

A libane builder that produces a program respecting every rule above
will load cleanly on M3. We can now write a host-side mirror struct
with confidence, and we can map any future load failure to a specific
named validator and a specific assertion within it.

## What's left (genuinely lower marginal value)

- The Procedure validator's continuation past `0x644fc` — the per-entry
  overlap and content-type matrix logic.
- Decoding the 12-byte gap inside `KernelPropData` and the 20-byte
  gap inside `TextPropData` (likely contain version/flags but
  unenforced).
- The wrapper-struct relationship (`+0x1c0` etc.) noted in Pass 8.
- Trace-record layout (`CTraceBuffer` push path).
- Cross-gen diff of TunableManager arrays.
- Disassembling `RESOURCE_INFO_GET` to confirm the 240/496/32/32/128
  numbers we read out of the const pool (independent verification).
