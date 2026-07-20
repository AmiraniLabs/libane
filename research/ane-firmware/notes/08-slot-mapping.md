# Pass 8 — definitive slot→name mapping (and a Pass-7 correction)

Used the M1 `[No] X Section` anchor strings to pin down which struct
offset holds which named section. Each anchor's xref site immediately
precedes a `ldrb w8, [x0, #N]` — that's the validity byte for the
section the message names.

## M1 (H13) — DEFINITIVE slot mapping

| Validity byte | Pointer (valid+0x18) | Section name |
|---------------|----------------------|--------------|
| `+0x008`      | `+0x020`             | **Generic**  |
| `+0x068`      | `+0x080`             | **TD**       |
| `+0x098`      | `+0x0b0`             | **Operation**|
| `+0x0c8`      | `+0x0e0`             | **Procedure**|
| `+0x0f8`      | `+0x110`             | **KernelProp**|
| `+0x128`      | `+0x140`             | **TdProp**   |

(The `+0x18` ptr offset relative to the validity byte is consistent:
`+0x8→+0x20`, `+0x68→+0x80`, `+0x98→+0xb0`, `+0xc8→+0xe0`, `+0xf8→+0x110`,
`+0x128→+0x140`.)

The plain `Kernel` section (distinct from `KernelProp`) is reached
through an indirect pointer chain (`ldrb w9, [x8, #0x38]` from a
secondary base), not through the main descriptor table — likely
because Kernel is a sub-resource of one of the named sections rather
than a top-level descriptor.

### Correction to Pass 7

Pass 7 listed the slot mapping as a tentative guess. The actual M1
ordering has **Operation at `+0x98`** (not `+0xc8` as guessed), and
**KernelProp at `+0xf8`** (not Kernel). The corrected table above is
authoritative for M1.

## M3 (H15) — same offsets for the named sections

M3's presence-check loop at `0x64eb4..0x64efc` loads the same
`(validity_byte, ptr)` pairs in the same offsets:

```
ldr  x8, [x0, #0x20]    ; Generic.ptr
ldrb w8, [x0, #0x8]     ; Generic.valid
ldr  x8, [x0, #0x80]    ; TD.ptr
ldrb w8, [x0, #0x68]    ; TD.valid
ldr  x8, [x0, #0x140]   ; TdProp.ptr
ldrb w8, [x0, #0x128]   ; TdProp.valid
ldr  x8, [x0, #0xb0]    ; Operation.ptr
ldrb w8, [x0, #0x98]    ; Operation.valid
ldr  x8, [x0, #0xe0]    ; Procedure.ptr
ldrb w8, [x0, #0xc8]    ; Procedure.valid
ldr  x8, [x0, #0x1c0]   ; (slot 6 — KernelProp moved to +0x1c0?)
```

→ **Generic, TD, TdProp, Operation, Procedure are at the same offsets
on M1 and M3.** Section descriptor layout for these five is gen-stable.

The M3 presence check visits `+0x1c0` last and does *not* visit `+0xf8`.
This is consistent with KernelProp having been **relocated** in M3,
either to `+0x1c0` or factored out entirely. Two readings:

- *Hypothesis A*: M3 moved KernelProp to `+0x1c0` (rearranged).
- *Hypothesis B*: M3 dropped KernelProp from the main descriptor table
  (now reached via a sub-pointer like Kernel was on M1).

Either way: libane code that mirrors the firmware layout for the
**five stable sections** is portable M1↔M3↔M4. KernelProp specifically
needs per-gen handling.

## Operation validator function (decoded)

Function entry: `0x64338` in M3. Signature (inferred from caller):

```cpp
bool validateOperationSection(
    OperationData* opData,    // x0: section data — first u32 is count
    uint64_t       sectionSz  // x1: declared section size
);
```

Body (reconstructed):
```cpp
if (opData == nullptr) return false;
uint32_t count = opData->count;            // [x0]
if (count > 128) return false;             // MAX Operation count
uint64_t need = count * 1036 + 4;
if (need > sectionSz) return false;        // "section smaller than actual"

for (uint32_t i = 0; i < count; ++i) {
    Operation* op = &opData->ops[i];       // base + 4 + i*1036
    if (op->opType > 4) return false;
    if (op->opType != 0) continue;         // only fully-validate type 0

    if (op->nbrOfNe > 16) return false;
    if (op->nbrOfLocalbarSetup > 128) return false;

    for (uint32_t j = 0; j < op->nbrOfLocalbarSetup; ++j) {
        Bar* bar = &op->bars[j];           // stride 8
        if (bar->index > 60) return false;
    }
}
return true;
```

→ **Operation section data layout:**

```cpp
struct OperationSectionData {
    uint32_t  count;             // +0x000  (≤ 128)
    Operation ops[count];        // +0x004, stride 1036
};

struct Operation {               // sizeof == 1036 = 0x40c
    uint32_t opType;             // +0x000  (0..4; only type 0 has further validation)
    uint16_t nbrOfNe;            // +0x004  (≤ 16)
    uint16_t _pad;               // +0x006
    uint32_t nbrOfLocalbarSetup; // +0x008  (≤ 128)
    Bar      bars[nbrOfLocalbarSetup];  // start TBD; each 8 B
    // ... remaining 0x40c - <bars region> bytes hold per-Op metadata
};

struct Bar {                     // sizeof == 8
    uint32_t index;              // +0x000  (≤ 60)
    uint32_t _other;             // +0x004 (TBD)
};
```

## Caller side: where the Operation validator gets its args

There are **three call sites** of `0x64338` in M3:

| Caller PC   | x0 source                  | x1 source                 | Context |
|-------------|----------------------------|---------------------------|---------|
| `0x55544`   | `add x2, sp, #0xc`         | (set above)               | Probably called from `LOAD_PROGRAM` handler |
| `0x55588`   | `add x2, sp, #0xc`         | (set above)               | Probably the same handler, second pass |
| `0x64f30`   | `ldr x0, [x20, #0x1d0]`    | `ldr x1, [x20, #0xb8]`    | The internal post-load checker |

The validator function itself is shared infrastructure — called from
both the IPC `LOAD_PROGRAM` handler at `0x55540ish` and from a
program-context post-load checker at `0x64f30`.

→ Note the inconsistency: the **presence-check loop** finds the
Operation validity byte at `program +0x98` (and ptr at `+0xb0`), while
**the second internal caller** at `0x64f30` reads `op_data` from
`[x20, #0x1d0]` and `size` from `[x20, #0xb8]`. The simplest
explanation: there are two struct views in play at different layers
of the firmware, and `x20` at `0x64f30` is a **wrapper struct** that
contains the program (the descriptor table view) at one offset plus
side-tables of the same data at other offsets, set up at load time.

This is consistent with the firmware's overall pattern of preparing
"fast" view structures from the program metadata so that hot-path
validators don't re-walk the descriptor table on every check.

## Pass-7 caveats updated

- **Slot mapping in Pass 7 was speculative; the table in this pass
  supersedes it.** Specifically: Operation is at `+0x98`, KernelProp
  was at `+0xf8` on M1 and is somewhere else on M3.
- **The `+0x1c0` offset** referenced in Pass 6 as "Operation section
  ptr" is actually a wrapper-side pointer (and on M3 may instead point
  at the relocated KernelProp). The Pass-6 finding about the `0x40c`
  Operation stride is still correct because the validator function
  itself is the same regardless of how the caller reaches it.

## Field-offset table — corrected and consolidated

| Struct                       | Field                       | Offset    | Type     |
|------------------------------|-----------------------------|-----------|----------|
| `Program` (M1+M3)            | Generic.valid               | `+0x008`  | u8       |
| `Program` (M1+M3)            | Generic.ptr                 | `+0x020`  | u64      |
| `Program` (M1+M3)            | TD.valid                    | `+0x068`  | u8       |
| `Program` (M1+M3)            | TD.ptr                      | `+0x080`  | u64      |
| `Program` (M1+M3)            | Operation.valid             | `+0x098`  | u8       |
| `Program` (M1+M3)            | Operation.ptr               | `+0x0b0`  | u64      |
| `Program` (M1+M3)            | Procedure.valid             | `+0x0c8`  | u8       |
| `Program` (M1+M3)            | Procedure.ptr               | `+0x0e0`  | u64      |
| `Program` (M1 only at +0xf8) | KernelProp.valid            | `+0x0f8`  | u8       |
| `Program` (M1 only at +0xf8) | KernelProp.ptr              | `+0x110`  | u64      |
| `Program` (M1+M3)            | TdProp.valid                | `+0x128`  | u8       |
| `Program` (M1+M3)            | TdProp.ptr                  | `+0x140`  | u64      |
| `Program` (M3-only side ptr) | (KernelProp? or wrapper)    | `+0x1c0`  | u64      |
| `Program` (Generic data)     | totalBufferNbr              | `+0x204`  | u32      |
| `Program` (Generic data)     | buffer[0]                   | `+0x208`  | (0x30)   |
| `OperationSectionData`       | count                       | `+0x000`  | u32      |
| `OperationSectionData`       | ops[]                       | `+0x004`  | stride 1036 |
| `Operation`                  | opType                      | `+0x000`  | u32      |
| `Operation`                  | nbrOfNe                     | `+0x004`  | u16      |
| `Operation`                  | nbrOfLocalbarSetup          | `+0x008`  | u32      |
| `Bar`                        | index                       | `+0x000`  | u32      |
| `Bar`                        | (per-entry stride)          | `0x008`   |          |

## Implications for libane (additions)

17. **Five sections (Generic, TD, Operation, Procedure, TdProp) have
    gen-stable offsets across M1↔M3.** A libane-side struct mirroring
    these will work on every chip M1→M4.
18. **KernelProp is not gen-stable** — was `+0xf8` on M1, moved (or
    factored out) on M3. Don't hardcode this offset.
19. **Operation section data has a 4-byte count header** before the
    op array. Total declared section size must be `≥ 4 + count*1036`.
20. **Three independent code paths call the Operation validator** —
    the IPC `LOAD_PROGRAM` handler validates twice (likely one
    pre-DMA pass and one post-DMA verification), and there's a
    runtime checker that re-validates from a wrapper view.
21. **The firmware builds wrapper/shadow views** of the program at
    load time. If we ever observe load-time behavior that doesn't
    match a single descriptor field, it may be because firmware is
    consulting a derived view rather than the raw descriptor.

## Open questions still

- **Where does M3 actually store KernelProp?** Most likely `+0x1c0`
  but needs confirmation by tracing what the +0x1c0 pointer is used
  for downstream.
- **Decode the 0x18 gap inside each section descriptor** — likely
  contains `size: u64` and `flags: u32` (we see size loaded at
  `[x20, #0xb8]` for Operation, which is `+0x18 + 0x10 = +0x28`
  from validity byte at `+0x98`, suggesting size lives at
  validity + 0x20).
- **Disassemble the LOAD_PROGRAM handler** at `0x55540ish` — that
  gives us the IPC payload-to-program-context translation, which
  would round out the load-path picture.
