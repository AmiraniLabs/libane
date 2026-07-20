# OIDN → libane port

Initial scoping notes for porting Intel's [OpenImageDenoise](https://github.com/RenderKit/oidn)
neural denoiser to ANE via libane.  Pitch: same denoise quality as
OIDN's CPU/Metal backends, executed on the ANE so the GPU stays free
for the renderer.  Target audience: 3D / VFX / game-rendering creative
software (Blender Cycles, Final Cut, DaVinci Resolve, Cinema 4D),
shipping for WWDC 2026.

Reference clone in `/Users/sheldon/libane/reference/oidn` (gitignored,
upstream commit pinned by clone date 2026-04-23).

---

## 1. Weight file format (TZA)

`.tza` = "tensor archive."  Source of truth: `training/tza.py` (213 lines).

Layout:

```
+--------------------+
|  uint16 magic 0x41D7
|  uint8  major version (=2)
|  uint8  minor version (=0)
|  uint64 table offset
+--------------------+
|  tensor blobs      ← variable size, each 64-byte aligned
|    ...
+--------------------+
|  table offset starts here:
|  uint32 num_tensors
|  for each tensor:
|    uint16 name_len; name (utf-8)
|    uint8  ndims
|    uint32 dim[0..ndims-1]
|    char   layout[ndims]  ← e.g. "oihw" or "x"
|    char   dtype          ← 'f'=fp32, 'h'=fp16, 'b'=int8, 'B'=uint8
|    uint64 offset (into blob region)
+--------------------+
```

dtype encoding: ASCII single char.  Layouts observed in `training/export.py`:
- `oihw` for conv kernels: [out_channels, in_channels, kH, kW]
- `x` for biases: [out_channels]

**Implication for libane:** trivial converter.  ~150 lines of Python to
read TZA → write libane weight blobs.  Layout matches what
`MilBuilder::conv2d_fragment` expects after re-stride (libane wants
fp16 weights in a specific stride; needs verification against
existing CONV2D op's expected layout).

---

## 2. U-Net architectures

OIDN ships four variants in `training/model.py`:

| Variant | enc channels | bottleneck | dec channels | params (approx) |
|---|---|---|---|---|
| `unet_small` | 32-32-32-32 | 32 | 64-64-32-32 | ~250K |
| `unet` (default) | 32-48-64-80 | 96 | 112-96-64-32 | ~700K |
| `unet_large` | 64-96-128-192 | 256 | 192-128-96-64 | ~2.5M |
| `unet_xl` | 96-128-192-256 | 384 | 256-192-128-96 | ~6M |

All four share the same topology, only channel counts differ:

```
encoder:
  conv0  : ic → ec1, ReLU
  conv1  : ec1 → ec1, ReLU,  → pool1 → 2x downsample
  conv2  : ec1 → ec2, ReLU,  → pool2 → 2x downsample
  conv3  : ec2 → ec3, ReLU,  → pool3 → 2x downsample
  conv4  : ec3 → ec4, ReLU,  →         2x downsample
bottleneck:
  conv5a : ec4 → ec5, ReLU
  conv5b : ec5 → ec5, ReLU
decoder:
  upsample (2x nearest), concat(pool3)
  conv4a : ec5+ec3 → dc4, ReLU
  conv4b : dc4 → dc4, ReLU
  upsample (2x nearest), concat(pool2)
  conv3a : dc4+ec2 → dc3, ReLU
  conv3b : dc3 → dc3, ReLU
  upsample (2x nearest), concat(pool1)
  conv2a : dc3+ec1 → dc2a, ReLU
  conv2b : dc2a → dc2b, ReLU
  upsample (2x nearest), concat(input)
  conv1a : dc2b+ic → dc1a, ReLU
  conv1b : dc1a → dc1b, ReLU
  conv0  : dc1b → oc       (no ReLU on final)
```

All conv layers are 3×3 with padding=1.  All pool layers are 2×2 stride 2.
All upsamples are 2× nearest-neighbor.

**Image alignment requirement: input H and W must be multiples of 16.**
Renderer-side pre-pad to nearest multiple of 16, post-crop the output.

---

## 3. Input/output features

Determined by feature flags (`training/dataset.py::get_channels`):

| Features | Input channels | Output channels |
|---|---:|---:|
| `hdr` only | 3 | 3 |
| `hdr,alb` (HDR + albedo) | 6 | 3 |
| `hdr,alb,nrm` (full RT denoise) | 9 | 3 |
| `ldr` only | 3 | 3 |
| `ldr,alb,nrm` | 9 | 3 |

Default Blender / Cycles config: `hdr,alb,nrm` → 9-channel input, 3-channel output.

---

## 4. Op coverage check against libane (current 0.9.0 ops)

| OIDN op | libane op | Status |
|---|---|---|
| 3×3 conv (padding=1) | `LIBANE_OP_CONV2D` | ✓ shipped 0.9.0 |
| ReLU | `LIBANE_OP_RELU` | ✓ |
| 2×2 max pool stride 2 | `LIBANE_OP_MAX_POOL` | ✓ — verify config |
| **2× nearest-neighbor upsample** | (none) | **✗ MISSING** |
| Channel concatenation | `LIBANE_OP_CONCAT` | ✓ |

### The one gap: nearest-neighbor 2× upsample

OIDN's decoder uses `F.interpolate(x, scale_factor=2, mode='nearest')`
between every decoder block.  libane has `PIXEL_SHUFFLE` (channel→spatial
rearrangement) which is **not equivalent** — pixel shuffle reduces
channels, nearest-upsample preserves them and just replicates pixels.

Three options:

**A. Add `LIBANE_OP_UPSAMPLE_NEAREST` as a new first-class op.**
   MIL has `upsample_nearest_neighbor` as a primitive; emit it directly.
   ~100 lines following existing op patterns: enum entry, MilBuilder
   fragment, MilBackend dispatch, fusion-rule entry, validator entry,
   C API wrapper, Python binding, test.  Matches every other op the
   library ships.

**B. Synthesize via reshape + tile.**
   Conceptually: [N,C,H,W] → reshape [N,C,H,1,W,1] → tile [1,1,1,2,1,2]
   → reshape [N,C,2H,2W].  Requires `tile`/`repeat` op which libane also
   doesn't have.  Doesn't actually save us from adding ops; less clean.

**C. Use a transposed conv with a fixed nearest-neighbor kernel.**
   Works with existing CONV2D infrastructure.  Wastes ~9× compute per
   output pixel for what should be a memory copy.  Bad tradeoff.

**Recommendation: option A.**  This is the single missing capability for
OIDN; adding it unlocks the entire port.

---

## 5. Concrete weight-converter plan

Inputs:
- `.tza` file from OIDN's weights repo (or produced by `training/export.py`)
- Variant identifier (`unet_small` / `unet` / `unet_large` / `unet_xl`)
- Feature config (`hdr,alb,nrm` etc.)

Outputs:
- A directory of fp16 `.bin` weight blobs in libane's expected stride:
  one per conv layer, named e.g. `enc_conv0_w.bin`, `enc_conv0_b.bin`.
- A Python module that builds the libane `Graph` matching the architecture,
  binding the weight blobs to the correct nodes.

Steps:

1. **Read TZA** using the bundled `tza.py` reader (or our own — 70 lines).
2. **Validate names** against the variant's expected layer set.  Fail
   loud if a layer is missing or a shape mismatches.
3. **Convert oihw fp32 → libane stride fp16.** Cast precision, copy to
   the layout libane's CONV2D op expects.  Verify against
   `src/graph/mil_backend.cpp` conv path.
4. **Write blobs** to a model directory.
5. **Emit graph builder** that reproduces the exact architecture using
   libane Python bindings (`ane.Graph()`, `.add_input`, `.add_op` with
   `LIBANE_OP_CONV2D`, etc.) and binds weight files to each conv.

Then a thin runtime wrapper:
```python
denoiser = ane.OIDNDenoiser.load("path/to/weights.tza", variant="unet")
denoised = denoiser(noisy_hdr, albedo, normal)  # H, W must be multiples of 16
```

---

## 6. Open questions / risks

1. **Memory at production resolutions.**  OIDN at 1080p + 9 input channels
   + ~700K params: intermediate feature maps blow past ANE's 32 MB SRAM
   threshold.  Will trigger SRAM spill (`intermediateBufferHandle != 0`),
   ~30% throughput hit.  Mitigation: **tile-based inference with halo**
   regions.  Standard practice; OIDN itself does this on CPU/Metal at
   high res.  Tiling is host-side orchestration above libane, not a
   libane feature.

2. **HDR preprocessing.**  OIDN preprocesses HDR inputs with a log-style
   transform before the network, and reverses it on output.  This is
   pure host-side math (no ANE), but needs to be matched bit-for-bit
   for output parity.  Reference: `core/color.cpp` in OIDN.

3. **Weight stride compatibility.**  libane's CONV2D expects a specific
   weight layout — verify by reading
   `src/graph/mil_backend.cpp::conv2d_fragment` before writing the
   converter.  Possible reshape/permute needed.

4. **Output parity.**  "Same quality as OIDN" means PSNR within ~0.1 dB
   of OIDN reference on a standard benchmark image set (OIDN ships a
   test suite).  We need a comparison harness as part of this port,
   not an afterthought.

---

## 7. Next concrete step

Add `LIBANE_OP_UPSAMPLE_NEAREST` to libane (~100 lines, mirrors existing
op patterns).  Once shipped, the OIDN port is pure plumbing:

1. Add `LIBANE_OP_UPSAMPLE_NEAREST` op (~100 lines + tests)
2. Write TZA → libane weight converter (~150 lines Python)
3. Write OIDN-UNet graph builder (~200 lines Python)
4. End-to-end forward pass on a tiny noisy image (~50 lines validation)
5. Comparison harness vs OIDN reference (~100 lines validation)
6. Tile-based wrapper for full-resolution images (~150 lines wrapper)

Total: ~750 lines of new code + the new op.  Two weeks of focused work to
something demoable.

WWDC window allowance (today is 2026-04-23, WWDC is 2026-06-09):
~6½ weeks.  Comfortable buffer for parity benchmarking, multi-chip
validation, and the writeup.
