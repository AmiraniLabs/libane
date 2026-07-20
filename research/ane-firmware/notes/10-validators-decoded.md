# Pass 10 — Generic, Procedure, KernelProp validators decoded

Disassembled three more inner-validator entry points found via the
LOAD_PROGRAM orchestrator (Pass 9). Each follows the same shape: a
wrapper that handles descriptor-level validity and retry, then an
inner function that walks the section data.

## Generic-section inner validator (0x6415c)

```cpp
bool validateGenericSection(GenericSectionData* data, uint64_t sectionSize) {
    if (!data) panic();
    if (data->magic != 1)              return false;   // [+0x000] must == 1
    if (data->firstAneNbrOfNe > 16)    return false;   // [+0x004] ≤ 16
    if (data->totalBufferNbr - 0x201 + 0x200 > 0x401) return false;  // ∈ [1, 511]
                                                       // [+0x204]
    uint64_t need = data->totalBufferNbr * 0x30 + 0x208;
    if (need > sectionSize)            return false;   // section big enough?
    return validateBufferTable(data, data->totalBufferNbr);  // jumps to 0x64204
}
```

→ **Generic section data layout:**

```cpp
struct GenericSectionData {
    uint32_t  magic;             // +0x000  must be 1
    uint32_t  firstAneNbrOfNe;   // +0x004  ≤ 16
    uint8_t   _gap[0x1FC];       // +0x008..+0x204 (508 bytes — per-ANE config?)
    uint32_t  totalBufferNbr;    // +0x204  ∈ [1, 511]
    GenericBuffer buffers[totalBufferNbr];  // +0x208, stride 0x30
};
```

Notable: there's a **508-byte gap** between the two header fields and
the buffer count. That space holds per-ANE configuration (likely
8 ANE entries × 64 bytes each = 512 — close enough). The very first
of those 8 ANE entries is just one u32 (`nbrOfNe`) read by the
validator; the rest of the per-ANE data isn't validated structurally.

## Procedure-section inner validator (0x6445c)

```cpp
bool validateProcedureSection(ProcSectionData* data, uint64_t sectionSize) {
    if (!data) panic();
    if (data->count == 0) return true;            // empty OK

    uint64_t off0 = data->entries[0].offset;      // [+0x008]
    uint64_t len0 = data->entries[0].len;         // [+0x010]
    if (off0 + len0 > sectionSize) return false;  // first entry fits

    if (data->count == 1) return true;            // single-entry OK
    // ... for count ≥ 2: per-entry overlap detection (continues past 0x644fc)
    return true;
}
```

→ **Procedure section data layout:**

```cpp
struct ProcSectionData {
    uint32_t   count;            // +0x000
    uint32_t   _pad;             // +0x004
    ProcEntry  entries[count];   // +0x008
};

struct ProcEntry {               // (per-entry head; full struct probably larger)
    uint64_t   offset;           // +0x000
    uint64_t   len;              // +0x008
    // ... additional fields TBD
};
```

The `entries[i].offset + len ≤ sectionSize` and **no-overlap** rules
(seen in M1 verbose: "Procedure[%d] exceeds limit", "Procedure[%d]
offset is overlapped with previous") are enforced past `0x644fc` —
the function continues with per-entry compares I haven't fully traced.

## KernelProp-section inner validator (0x64230)

```cpp
bool validateKernelPropSection(KernelPropData* data, uint64_t sectionSize) {
    if (!data) panic();
    if (data->count == 0) return true;            // empty OK

    uint64_t off0 = data->entries[0].offset;      // [+0x010]
    uint64_t len0 = data->entries[0].len;         // [+0x018]
    if (off0 + len0 > sectionSize) return false;

    if (data->count == 1) return true;

    uint64_t off1 = data->entries[1].offset;      // [+0x028]  ← stride 0x18 confirmed
    if (off1 < off0 + len0) return false;         // no-overlap; entries[1] starts ≥ entries[0].end
    // ... continues for count ≥ 3
    return true;
}
```

→ **KernelProp section data layout:**

```cpp
struct KernelPropData {
    uint32_t   count;            // +0x000
    uint8_t    _pad[0x0C];       // +0x004..+0x010
    KPropEntry entries[count];   // +0x010, stride 0x18
};

struct KPropEntry {              // sizeof == 0x18 (24)
    uint64_t   offset;           // +0x000
    uint64_t   len;              // +0x008
    uint64_t   _other;           // +0x010 (TBD)
};
```

The 12-byte gap between `count` and `entries[0]` is unusual (vs.
Procedure's 4-byte gap). Probably holds additional metadata fields
(e.g. `version`, `flags`, `entrySize`).

## Confirmed: all per-section inner validators are siblings

Same prologue, same NULL-check + count==0 short-circuit, same per-entry
loop pattern, same `return w0 = bool` ABI. Five known so far:

| Inner validator | Section data type | Per-entry stride | Notes |
|-----------------|-------------------|-------------------|-------|
| `0x6415c`       | Generic           | 0x30 (buffer)     | header at +0/+4/+0x204 |
| `0x64230`       | KernelProp        | 0x18              | header gap to +0x10 |
| `0x64338`       | Operation         | 0x40c             | per-Op deep validation |
| `0x6445c`       | Procedure         | 0x10+ (head)      | offset+len at start |
| (TBD)           | text + textProp   | ?                 | combined; cross-validates TID |

## Validator table for libane consumption

If we mirror the firmware section data shapes host-side, here's the
operational checklist for each section:

### genericSection
- `data[0:4]` = 1 (literal one)
- `data[4:8]` ≤ 16 (first ANE NE count)
- `data[0x204:0x208]` = N where 1 ≤ N ≤ 511
- declared `size` ≥ 0x208 + N × 0x30
- per-buffer (`data + 0x208 + i*0x30`):
  - `[+0x00]` valid bit 0 set
  - `[+0x08]` type < 8

### operationSection
- `data[0:4]` = M ≤ 128 (op count)
- declared `size` ≥ 4 + M × 1036
- per-op (`data + 4 + i*1036`):
  - `[+0x00]` opType ≤ 4
  - if opType == 0: `[+0x04]` (u16) nbrOfNe ≤ 16; `[+0x08]` nbrOfLocalbarSetup ≤ 128;
    each bar entry's first u32 ≤ 60

### procedureSection
- `data[0:4]` = K (proc count); 0 OK
- per-entry (starting at `data + 8`):
  - `offset + len` within `size`
  - `entry[i].offset >= entry[i-1].offset + entry[i-1].len` (no overlap)

### kernelPropSection
- `data[0:4]` = J (kprop count); 0 OK
- per-entry (starting at `data + 0x10`, stride 0x18):
  - `offset + len` within `size`
  - non-overlapping with previous

### kernelSection / opDbgSection / procPropSection / textSection / textPropSection
- Not validated by the orchestrator's named validators (text+textProp
  combined validator at `0x558ac` is still TBD).

## Field-offset table (v4)

Add to running:

| Struct                    | Field             | Offset    | Type   |
|---------------------------|-------------------|-----------|--------|
| `GenericSectionData`      | magic             | `+0x000`  | u32 (=1) |
| `GenericSectionData`      | firstAneNbrOfNe   | `+0x004`  | u32      |
| `GenericSectionData`      | (per-ANE region)  | `+0x008`..`+0x204` | 508 B |
| `GenericSectionData`      | totalBufferNbr    | `+0x204`  | u32      |
| `GenericSectionData`      | buffers[]         | `+0x208`  | stride 0x30 |
| `ProcSectionData`         | count             | `+0x000`  | u32      |
| `ProcSectionData`         | entries[]         | `+0x008`  | stride ≥0x10 |
| `ProcEntry`               | offset            | `+0x000`  | u64      |
| `ProcEntry`               | len               | `+0x008`  | u64      |
| `KernelPropData`          | count             | `+0x000`  | u32      |
| `KernelPropData`          | entries[]         | `+0x010`  | stride 0x18 |
| `KPropEntry`              | offset            | `+0x000`  | u64      |
| `KPropEntry`              | len               | `+0x008`  | u64      |

## Implications for libane (final batch)

28. **`genericSection.magic == 1` is mandatory.** Any program with a
    different value at the first 4 bytes is rejected.
29. **The 508-byte per-ANE region** in `genericSection` is mostly
    opaque to the validator — only the first `nbrOfNe` u32 is checked.
    The rest is forwarded to runtime configuration (FSM init, etc.).
30. **Procedure entries are at minimum 16 bytes each** (`offset` +
    `len`). Anything smaller cannot encode an entry.
31. **KernelProp entries are exactly 24 bytes each** (3 × u64).
32. **Section data `count` of 0 is always OK** — empty Procedure /
    KernelProp / etc. sections do not fail validation. Only Generic
    requires `totalBufferNbr ≥ 1`.
33. **No-overlap is a hard rule** for Procedure and KernelProp
    entries — entries must be sorted by `offset` and contiguous (or
    gapped, but never overlapping).

## Next-pass candidates

1. Decode validator at **`0x558ac`** (text + textProp combined) — last
   unknown validator; expected to spell out TD ↔ TdProp TID linkage
   rules.
2. Walk past `0x6445c+0x60` (procedure validator's loop continuation)
   to confirm the per-entry overlap+sort logic.
3. Decode the remaining `_pad` regions in `KernelPropData`'s 12-byte
   gap and `SectionDescriptor`'s 8-byte tail (likely flags / version).
4. Disassemble the genericSection data dump path (the buffer-dump
   loop at the end of `0x553b0`) — what auxiliary info does the firmware
   surface per buffer for debug?
