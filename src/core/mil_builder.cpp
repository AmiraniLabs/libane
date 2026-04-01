/**
 * MIL Builder implementation — generates MIL text programs for ANE.
 *
 * The ANE private API (_ANEInMemoryModelDescriptor) takes MIL *text* (UTF-8
 * string), not CoreML protobuf. This file generates that text.
 *
 * All linear projections use conv 1×1 (3× faster than matmul on ANE per
 * Orion §5.2). Weight layout for conv1x1 is [OC, 1, 1, IC] stored row-major.
 *
 * Weight blob format (from maderix/ANE ane_bridge.m):
 *   [0..63]   64-byte file header  (buf[0]=0x01, buf[4]=0x02)
 *   [64..127] 64-byte chunk header (magic=0xDEADBEEF LE, buf[4]=0x01, buf[8..11]=fp16_size uint32)
 *   [128..]   raw fp16 weight data (row-major)
 *
 * The weight dict @"offset" key = 64 (points into blob past the file header).
 */
#include "mil_builder.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <sstream>
#include <iomanip>

// fp16 conversion helpers (software fallback — also used for blob building)
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
#endif

namespace libane {
namespace mil {

/* ── TensorShape::validate ───────────────────────────────────────────────── */

void TensorShape::validate() const {
    if (batch != 1)
        throw std::invalid_argument("ANE: batch must be 1, got " +
                                    std::to_string(batch));
    if (height != 1)
        throw std::invalid_argument("ANE: height must be 1, got " +
                                    std::to_string(height));
    if (seq <= 0 || seq % 8 != 0)
        throw std::invalid_argument("ANE: S must be > 0 and multiple of 8, got " +
                                    std::to_string(seq));
    if (seq > 65536)
        throw std::invalid_argument("ANE: S must be ≤ 65536, got " +
                                    std::to_string(seq));
    if (channels <= 0 || channels > 16384)
        throw std::invalid_argument("ANE: C must be in [1, 16384], got " +
                                    std::to_string(channels));
}

/* ── WeightBlob ──────────────────────────────────────────────────────────── */

// Minimal xxhash64 — used for cache keying
static uint64_t xxhash64(const uint8_t* data, size_t len, uint64_t seed = 0) {
    constexpr uint64_t P1 = 11400714785074694791ULL;
    constexpr uint64_t P2 = 14029467366897019727ULL;
    constexpr uint64_t P3 =  1609587929392839161ULL;
    constexpr uint64_t P4 =  9650029242287828579ULL;
    constexpr uint64_t P5 =  2870177450012600261ULL;
    auto rotl = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };

    uint64_t h;
    const uint8_t* p   = data;
    const uint8_t* end = data + len;

    if (len >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2;
        uint64_t v3 = seed,           v4 = seed - P1;
        do {
            uint64_t lane;
            memcpy(&lane, p, 8); v1 = rotl(v1 + lane * P2, 31) * P1; p += 8;
            memcpy(&lane, p, 8); v2 = rotl(v2 + lane * P2, 31) * P1; p += 8;
            memcpy(&lane, p, 8); v3 = rotl(v3 + lane * P2, 31) * P1; p += 8;
            memcpy(&lane, p, 8); v4 = rotl(v4 + lane * P2, 31) * P1; p += 8;
        } while (p <= end - 32);
        h  = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h  = (h ^ (rotl(v1 * P2, 31) * P1)) * P1 + P4;
        h  = (h ^ (rotl(v2 * P2, 31) * P1)) * P1 + P4;
        h  = (h ^ (rotl(v3 * P2, 31) * P1)) * P1 + P4;
        h  = (h ^ (rotl(v4 * P2, 31) * P1)) * P1 + P4;
    } else {
        h = seed + P5;
    }
    h += static_cast<uint64_t>(len);
    while (p + 8 <= end) {
        uint64_t lane; memcpy(&lane, p, 8);
        h ^= rotl(lane * P2, 31) * P1; h = rotl(h, 27) * P1 + P4; p += 8;
    }
    if (p + 4 <= end) {
        uint32_t lane; memcpy(&lane, p, 4);
        h ^= static_cast<uint64_t>(lane) * P1; h = rotl(h, 23) * P2 + P3; p += 4;
    }
    while (p < end) { h ^= static_cast<uint64_t>(*p++) * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

// Software fp32 → fp16 (bit-correct IEEE 754)
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t fb; memcpy(&fb, &f, 4);
    uint32_t sign     = (fb >> 16) & 0x8000;
    int32_t  exp      = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fb >> 13) & 0x3FF;
    if (exp <= 0)       return static_cast<uint16_t>(sign);
    if (exp >= 31)      return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
}

// Build the 128-byte ANE blob header and return an appropriately-sized blob.
// Format exactly matches Orion's make_blobfile():
//   [0]      = 0x01  (file type)
//   [4]      = 0x02  (file version)
//   [64..67] = 0xDEADBEEF LE  (chunk magic)
//   [68]     = 0x01  (chunk type)
//   [72..75] = fp16_size (uint32 LE) — byte count of actual weight data
//   [80..83] = 128 (uint32 LE) — offset from start of blob where data begins
//   [128..]  = raw fp16 weight data
static WeightBlob make_blob(size_t weight_bytes) {
    WeightBlob b;
    b.data.resize(WeightBlob::kDataOffset + weight_bytes, 0);

    // File header [0..63]
    b.data[0] = 0x01;
    b.data[4] = 0x02;

    // Chunk header [64..127]
    constexpr uint32_t kMagic = 0xDEADBEEF;
    std::memcpy(b.data.data() + 64, &kMagic, 4);     // magic at b[64]
    b.data[64 + 4]  = 0x01;                             // chunk type at b[68]
    uint32_t fp16_size = static_cast<uint32_t>(weight_bytes);
    std::memcpy(b.data.data() + 72, &fp16_size, 4);    // data byte count at b[72..75]
    uint32_t data_offset = 128;
    std::memcpy(b.data.data() + 80, &data_offset, 4);  // data start offset at b[80]

    return b;
}

void WeightBlob::compute_hash() {
    if (data.size() <= kDataOffset) { hash = 0; return; }
    hash = xxhash64(data.data() + kDataOffset, data.size() - kDataOffset);
}

WeightBlob WeightBlob::from_fp16(const void* src, size_t weight_bytes) {
    auto b = make_blob(weight_bytes);
    std::memcpy(b.data.data() + kDataOffset, src, weight_bytes);
    b.compute_hash();
    return b;
}

WeightBlob WeightBlob::from_fp16_transposed(const void* src, int rows, int cols) {
    // Transpose [rows, cols] → [cols, rows] in-place conversion
    size_t count = static_cast<size_t>(rows) * cols;
    auto b = make_blob(count * 2);
    const auto* s = reinterpret_cast<const uint16_t*>(src);
    auto*       d = reinterpret_cast<uint16_t*>(b.data.data() + kDataOffset);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            d[static_cast<size_t>(c) * rows + r] = s[static_cast<size_t>(r) * cols + c];
    b.compute_hash();
    return b;
}

WeightBlob WeightBlob::from_fp32(const float* src, int rows, int cols, bool transpose) {
    size_t count = static_cast<size_t>(rows) * cols;
    auto b = make_blob(count * 2);
    auto* dst = reinterpret_cast<uint16_t*>(b.data.data() + kDataOffset);

    if (transpose) {
        // Input is [rows, cols] row-major → output is [cols, rows] row-major
        // (e.g., weight matrix B[IC, OC] → W[OC, IC] for conv1x1)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                dst[static_cast<size_t>(c) * rows + r] = fp32_to_fp16(src[static_cast<size_t>(r) * cols + c]);
    } else {
        for (size_t i = 0; i < count; ++i)
            dst[i] = fp32_to_fp16(src[i]);
    }
    b.compute_hash();
    return b;
}

/* ── MilBuilder helpers ──────────────────────────────────────────────────── */

// Standard MIL program preamble.
// Must include all 4 buildInfo keys — ANE compiler requires coremlc-version.
// Keys must be on ONE logical line (no newlines between entries).
std::string MilBuilder::header() {
    return
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n";
}

// Render "tensor<fp16, [batch, C, height, S]>"
std::string MilBuilder::tensor_type(const TensorShape& s) {
    return "tensor<fp16, [" +
           std::to_string(s.batch)    + ", " +
           std::to_string(s.channels) + ", " +
           std::to_string(s.height)   + ", " +
           std::to_string(s.seq)      + "]>";
}

// Render BLOBFILE weight tensor literal for use inside a const() node.
// Shape is the WEIGHT shape [OC, IC, kH, kW] (standard conv ordering).
// Offset 64 = skip the 64-byte file-level header to reach the chunk descriptor.
std::string MilBuilder::file_ref(const std::string& filename, uint64_t offset,
                                  const TensorShape& shape) {
    // Standard conv weight shape: [OC, IC, kH, kW].
    // shape.batch=OC, shape.channels=IC, shape.height=kH, shape.seq=kW
    return "tensor<fp16, [" +
           std::to_string(shape.batch)    + "," +
           std::to_string(shape.channels) + "," +
           std::to_string(shape.height)   + "," +
           std::to_string(shape.seq)      + "]>"
           "(BLOBFILE(path=string(\"@model_path/weights/" + filename +
           "\"), offset=uint64(" + std::to_string(offset) + ")))";
}

/* ── matmul_conv1x1 ──────────────────────────────────────────────────────── */

MilProgram MilBuilder::matmul_conv1x1(int IC, int OC, int SP,
                                       const std::string& weight_file) {
    TensorShape in {1, IC, 1, SP};
    TensorShape out{1, OC, 1, SP};
    in.validate();
    out.validate();

    // Conv1x1 weight shape: [OC, IC, kH, kW] = [OC, IC, 1, 1].
    // batch=OC, channels=IC, height=1, seq=1 for the BLOBFILE tensor type.
    TensorShape wshape{OC, IC, 1, 1};

    // Orion-compatible conv syntax:
    //   - Each conv parameter is a typed const node referenced by name
    //   - pad_type is a string const
    //   - strides/pad/dilations are typed int32 tensor consts
    //   - groups is int32 scalar const
    //   - weight tensor literal uses BLOBFILE(...)
    //   - All ops carry [name=string("...")] attribute
    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(in) + " x) {\n";
    t += "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n";
    t += "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    t += "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
    // Weight const: type W = const()[name=..., val=type(BLOBFILE(...))];
    std::string wtype = "tensor<fp16, [" +
        std::to_string(OC) + "," + std::to_string(IC) + ",1,1]>";
    std::string wval  = file_ref(weight_file, WeightBlob::kWeightDictOffset, wshape);
    t += "        " + wtype + " W = const()[name=string(\"W\"), val=" + wval + "];\n";
    t += "        " + tensor_type(out) + " y = conv("
         "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
         "weight=W, x=x)[name=string(\"y\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text         = std::move(t);
    p.weight_name  = weight_file;
    p.input_shape  = in;
    p.output_shape = out;
    return p;
}

/* ── qkv_proj ────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::qkv_proj(int IC, int head_dim, int n_heads, int SP,
                                  const QKVWeights& w) {
    int OC = head_dim * n_heads;
    TensorShape in {1, IC, 1, SP};
    TensorShape out{1, OC, 1, SP};
    in.validate();
    out.validate();

    // Conv1x1 weight shape: [OC, IC, 1, 1]
    TensorShape wshape{OC, IC, 1, 1};

    // Q-only projection using correct Orion conv syntax (no % prefix, named consts)
    std::string wtype = "tensor<fp16, [" +
        std::to_string(OC) + "," + std::to_string(IC) + ",1,1]>";
    std::string wval = file_ref(w.q_file, WeightBlob::kWeightDictOffset, wshape);

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(in) + " x) {\n";
    t += "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n";
    t += "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    t += "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
    t += "        " + wtype + " W = const()[name=string(\"W\"), val=" + wval + "];\n";
    t += "        " + tensor_type(out) + " y = conv("
         "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
         "weight=W, x=x)[name=string(\"y\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = w.q_file; // primary weight (K/V are separate dispatches)
    p.input_shape  = in;
    p.output_shape = out;
    return p;
}

/* ── rmsnorm ─────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::rmsnorm(int C, int SP,
                                const std::string& scale_file) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    float inv_c = 1.0f / static_cast<float>(C);

    // Format inv_c as a string suitable for fp16() literal
    char inv_c_buf[32];
    std::snprintf(inv_c_buf, sizeof(inv_c_buf), "%.10g", (double)inv_c);

    std::string tt  = tensor_type(shape);                     // [1,C,1,SP]
    std::string tss = "tensor<fp16, [1,1,1," + std::to_string(SP) + "]>";  // [1,1,1,SP]

    std::string t = header();
    t += "    func main<ios18>(" + tt + " x) {\n";
    // x^2
    t += "        " + tt + " rms_sq = mul(x=x, y=x)[name=string(\"rms_sq\")];\n";
    // reduce_sum along axis=1 (channels)
    t += "        tensor<int32, [1]> rms_ax = const()[name=string(\"rms_ax\"), val=tensor<int32, [1]>([1])];\n";
    t += "        bool rms_kd = const()[name=string(\"rms_kd\"), val=bool(true)];\n";
    t += "        " + tss + " rms_ss = reduce_sum(x=rms_sq, axes=rms_ax, keep_dims=rms_kd)[name=string(\"rms_ss\")];\n";
    // multiply by 1/C
    t += "        fp16 rms_invd = const()[name=string(\"rms_invd\"), val=fp16(" + std::string(inv_c_buf) + ")];\n";
    t += "        " + tss + " rms_ms = mul(x=rms_ss, y=rms_invd)[name=string(\"rms_ms\")];\n";
    // add eps
    t += "        fp16 rms_eps = const()[name=string(\"rms_eps\"), val=fp16(9.765625e-4)];\n";
    t += "        " + tss + " rms_mse = add(x=rms_ms, y=rms_eps)[name=string(\"rms_mse\")];\n";
    // pow(mse, -0.5)  =  1/sqrt(mse)
    t += "        fp16 rms_nhalf = const()[name=string(\"rms_nhalf\"), val=fp16(-0.5)];\n";
    t += "        " + tss + " rms_rrms = pow(x=rms_mse, y=rms_nhalf)[name=string(\"rms_rrms\")];\n";
    // x * rrms
    t += "        " + tt + " rms_xr = mul(x=x, y=rms_rrms)[name=string(\"rms_xr\")];\n";
    // scale weight [1,C,1,1]
    std::string wref = "tensor<fp16, [1," + std::to_string(C) + ",1,1]>(BLOBFILE(path=string(\"@model_path/weights/" +
                       scale_file + "\"), offset=uint64(" + std::to_string(WeightBlob::kWeightDictOffset) + ")))";
    t += "        tensor<fp16, [1," + std::to_string(C) + ",1,1]> rms_w = const()[name=string(\"rms_w\"), val=" + wref + "];\n";
    t += "        " + tt + " out = mul(x=rms_xr, y=rms_w)[name=string(\"out\")];\n";
    t += "    } -> (out);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = scale_file;
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── gelu ────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::gelu(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    // gelu is NOT a valid ANE MIL op (Orion constraint #10).
    // Decompose GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // using primitive MIL ops that ANE supports.
    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    std::string tt = tensor_type(shape);
    t += "        " + tt + " g_x2 = mul(x=x, y=x)[name=string(\"g_x2\")];\n";
    t += "        " + tt + " g_x3 = mul(x=g_x2, y=x)[name=string(\"g_x3\")];\n";
    t += "        fp16 g_c1 = const()[name=string(\"g_c1\"), val=fp16(0.044715)];\n";
    t += "        " + tt + " g_cx3 = mul(x=g_x3, y=g_c1)[name=string(\"g_cx3\")];\n";
    t += "        " + tt + " g_inner = add(x=x, y=g_cx3)[name=string(\"g_inner\")];\n";
    t += "        fp16 g_c2 = const()[name=string(\"g_c2\"), val=fp16(0.7979)];\n";
    t += "        " + tt + " g_scaled = mul(x=g_inner, y=g_c2)[name=string(\"g_scaled\")];\n";
    t += "        " + tt + " g_th = tanh(x=g_scaled)[name=string(\"g_th\")];\n";
    t += "        fp16 g_one = const()[name=string(\"g_one\"), val=fp16(1.0)];\n";
    t += "        " + tt + " g_onep = add(x=g_th, y=g_one)[name=string(\"g_onep\")];\n";
    t += "        fp16 g_half = const()[name=string(\"g_half\"), val=fp16(0.5)];\n";
    t += "        " + tt + " g_hx = mul(x=x, y=g_half)[name=string(\"g_hx\")];\n";
    t += "        " + tt + " y = mul(x=g_hx, y=g_onep)[name=string(\"y\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── softmax ─────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::softmax(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    t += "        int32 sm_ax = const()[name=string(\"sm_ax\"), val=int32(1)];\n";
    t += "        " + tensor_type(shape) + " y = softmax(axis=sm_ax, x=x)[name=string(\"sm\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── add ─────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::add(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    // Two-input elementwise add. Inputs are named "x" and "y" in ANE ordering
    // (alphabetical order for IOSurface assignment).
    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x, " +
                                   tensor_type(shape) + " y) {\n";
    t += "        " + tensor_type(shape) + " z = add(x = x, y = y)[name=string(\"add\")];\n";
    t += "    } -> (z);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── transpose_cssc ──────────────────────────────────────────────────────── */

MilProgram MilBuilder::transpose_cssc(int C, int SP) {
    // Input:  [1, C, 1, S]
    // Output: [1, S, 1, C]  (perm [0, 3, 2, 1])
    TensorShape in {1, C,  1, SP};
    TensorShape out{1, SP, 1, C };
    in.validate();
    out.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(in) + " x) {\n";
    t += "        tensor<int32, [4]> perm = const()[name=string(\"perm\"), val=tensor<int32, [4]>([0,3,2,1])];\n";
    t += "        " + tensor_type(out) + " y = transpose(perm=perm, x=x)[name=string(\"tr\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = in;
    p.output_shape = out;
    return p;
}

/* ── out_proj_add ────────────────────────────────────────────────────────── */

MilProgram MilBuilder::out_proj_add(int IC, int OC, int SP,
                                     const std::string& weight_file) {
    // residual: [1, OC, 1, SP]   (skip connection — first input alphabetically)
    // x:        [1, IC, 1, SP]   (post-attention activations — second input alphabetically)
    // W:        [OC, IC, 1, 1]   (projection weight)
    // out = conv(x, W) + residual
    // IMPORTANT: inputs are ordered alphabetically ("residual" < "x") per ANE constraint #13.
    TensorShape in      {1, IC, 1, SP};
    TensorShape residual{1, OC, 1, SP};
    TensorShape wshape  {OC, IC, 1, 1};
    in.validate();
    residual.validate();

    std::string wtype = "tensor<fp16, [" +
        std::to_string(OC) + "," + std::to_string(IC) + ",1,1]>";
    std::string wval  = file_ref(weight_file, WeightBlob::kWeightDictOffset, wshape);

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(residual) + " residual, " +
                                   tensor_type(in) + " x) {\n";
    t += "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n";
    t += "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    t += "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
    t += "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
    t += "        " + wtype + " W = const()[name=string(\"W\"), val=" + wval + "];\n";
    t += "        " + tensor_type(residual) + " proj = conv("
         "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
         "weight=W, x=x)[name=string(\"proj\")];\n";
    t += "        " + tensor_type(residual) + " out = add(x=proj, y=residual)[name=string(\"out\")];\n";
    t += "    } -> (out);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = weight_file;
    p.input_shape  = in;
    p.output_shape = residual;
    return p;
}

/* ── mul ─────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::mul(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    // Two-input elementwise mul. Inputs named "x" and "y" (alphabetical order).
    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x, " +
                                   tensor_type(shape) + " y) {\n";
    t += "        " + tensor_type(shape) + " z = mul(x=x, y=y)[name=string(\"mul\")];\n";
    t += "    } -> (z);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── sub ─────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::sub(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x, " +
                                   tensor_type(shape) + " y) {\n";
    t += "        " + tensor_type(shape) + " z = sub(x=x, y=y)[name=string(\"sub\")];\n";
    t += "    } -> (z);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── real_div ────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::real_div(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x, " +
                                   tensor_type(shape) + " y) {\n";
    t += "        " + tensor_type(shape) + " z = real_div(x=x, y=y)[name=string(\"real_div\")];\n";
    t += "    } -> (z);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── sqrt ────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::sqrt(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    t += "        " + tensor_type(shape) + " y = sqrt(x=x)[name=string(\"sqrt\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── log ─────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::log(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    // ANE compiler requires explicit epsilon parameter for log/rsqrt.
    // Standard MIL treats epsilon as optional but ANE silently rejects
    // programs without it. Discovered by ironmill op verification.
    // Value 0x1.0cp-17 ≈ 7.63e-6 (fp16).
    t += "        fp16 lg_eps = const()[name=string(\"lg_eps\"), val=fp16(0x1.0cp-17)];\n";
    t += "        " + tensor_type(shape) + " y = log(epsilon=lg_eps, x=x)[name=string(\"log\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── rsqrt ───────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::rsqrt(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    // ANE compiler requires explicit epsilon parameter for log/rsqrt.
    // Standard MIL treats epsilon as optional but ANE silently rejects
    // programs without it. Discovered by ironmill op verification.
    // Value 0x1.0cp-17 ≈ 7.63e-6 (fp16).
    t += "        fp16 rs_eps = const()[name=string(\"rs_eps\"), val=fp16(0x1.0cp-17)];\n";
    t += "        " + tensor_type(shape) + " y = rsqrt(epsilon=rs_eps, x=x)[name=string(\"rsqrt\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── silu ────────────────────────────────────────────────────────────────── */

MilProgram MilBuilder::silu(int C, int SP) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    // SiLU(x) = x * sigmoid(x)
    std::string t = header();
    t += "    func main<ios18>(" + tensor_type(shape) + " x) {\n";
    std::string tt = tensor_type(shape);
    t += "        " + tt + " sig = sigmoid(x=x)[name=string(\"sig\")];\n";
    t += "        " + tt + " y = mul(x=x, y=sig)[name=string(\"silu\")];\n";
    t += "    } -> (y);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = "";
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── layernorm ───────────────────────────────────────────────────────────── */

MilProgram MilBuilder::layernorm(int C, int SP,
                                  const std::string& gamma_file,
                                  const std::string& beta_file,
                                  float eps) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    // Format eps as a string suitable for fp16() literal
    char eps_buf[32];
    std::snprintf(eps_buf, sizeof(eps_buf), "%.6g", (double)eps);

    std::string tt  = tensor_type(shape);                               // [1,C,1,SP]
    std::string tss = "tensor<fp16, [1,1,1," + std::to_string(SP) + "]>";  // [1,1,1,SP]
    std::string tc  = "tensor<fp16, [1," + std::to_string(C) + ",1," + std::to_string(SP) + "]>";

    // Weight shape [1, C, 1, 1] for gamma and beta
    std::string tw = "tensor<fp16, [1," + std::to_string(C) + ",1,1]>";

    auto wref = [&](const std::string& filename) {
        return tw + "(BLOBFILE(path=string(\"@model_path/weights/" + filename +
               "\"), offset=uint64(" + std::to_string(WeightBlob::kWeightDictOffset) + ")))";
    };

    std::string t = header();
    t += "    func main<ios18>(" + tt + " x) {\n";
    t += "        tensor<int32, [1]> ln_ax = const()[name=string(\"ln_ax\"), val=tensor<int32, [1]>([1])];\n";
    t += "        bool ln_kd = const()[name=string(\"ln_kd\"), val=bool(true)];\n";
    t += "        " + tss + " ln_mean = reduce_mean(x=x, axes=ln_ax, keep_dims=ln_kd)[name=string(\"ln_mean\")];\n";
    t += "        " + tc + " ln_cent = sub(x=x, y=ln_mean)[name=string(\"ln_cent\")];\n";
    t += "        " + tc + " ln_sq = mul(x=ln_cent, y=ln_cent)[name=string(\"ln_sq\")];\n";
    t += "        " + tss + " ln_var = reduce_mean(x=ln_sq, axes=ln_ax, keep_dims=ln_kd)[name=string(\"ln_var\")];\n";
    t += "        fp16 ln_eps = const()[name=string(\"ln_eps\"), val=fp16(" + std::string(eps_buf) + ")];\n";
    t += "        " + tss + " ln_veps = add(x=ln_var, y=ln_eps)[name=string(\"ln_veps\")];\n";
    t += "        fp16 ln_nhalf = const()[name=string(\"ln_nhalf\"), val=fp16(-0.5)];\n";
    t += "        " + tss + " ln_rstd = pow(x=ln_veps, y=ln_nhalf)[name=string(\"ln_rstd\")];\n";
    t += "        " + tc + " ln_norm = mul(x=ln_cent, y=ln_rstd)[name=string(\"ln_norm\")];\n";
    t += "        " + tw + " ln_g = const()[name=string(\"ln_g\"), val=" + wref(gamma_file) + "];\n";
    t += "        " + tc + " ln_scaled = mul(x=ln_norm, y=ln_g)[name=string(\"ln_scaled\")];\n";
    t += "        " + tw + " ln_beta = const()[name=string(\"ln_beta\"), val=" + wref(beta_file) + "];\n";
    t += "        " + tc + " out = add(x=ln_scaled, y=ln_beta)[name=string(\"out\")];\n";
    t += "    } -> (out);\n";
    t += "}\n";

    MilProgram p;
    p.text        = std::move(t);
    p.weight_name = gamma_file;   // primary weight
    p.input_shape  = shape;
    p.output_shape = shape;
    return p;
}

/* ── WeightBlob::kWeightDictOffset ──────────────────────────────────────── */
// These are constexpr defined in the header; no out-of-line definition needed.

/* ── Fragment emitters ───────────────────────────────────────────────────── */
//
// Each fragment uses the convention:
//   prefix = out_var + "_"
// so all internal variables are named "<out_var>_<suffix>", guaranteeing
// uniqueness when multiple fragments share a MIL program scope.

MilFragment MilBuilder::matmul_fragment(int IC, int OC, int SP,
                                         const std::string& in_var,
                                         const std::string& out_var,
                                         const std::string& weight_file) {
    TensorShape in {1, IC, 1, SP};
    TensorShape out{1, OC, 1, SP};
    in.validate();
    out.validate();

    const std::string p = out_var + "_";
    std::string wtype = "tensor<fp16, [" +
        std::to_string(OC) + "," + std::to_string(IC) + ",1,1]>";
    TensorShape wshape{OC, IC, 1, 1};

    std::string body;
    body += "        string " + p + "pt = const()[name=string(\"" + p + "pt\"), val=string(\"valid\")];\n";
    body += "        tensor<int32, [2]> " + p + "st = const()[name=string(\"" + p + "st\"), val=tensor<int32, [2]>([1,1])];\n";
    body += "        tensor<int32, [4]> " + p + "pd = const()[name=string(\"" + p + "pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    body += "        tensor<int32, [2]> " + p + "dl = const()[name=string(\"" + p + "dl\"), val=tensor<int32, [2]>([1,1])];\n";
    body += "        int32 " + p + "gr = const()[name=string(\"" + p + "gr\"), val=int32(1)];\n";
    body += "        " + wtype + " " + p + "W = const()[name=string(\"" + p + "W\"), val=" +
            file_ref(weight_file, WeightBlob::kWeightDictOffset, wshape) + "];\n";
    body += "        " + tensor_type(out) + " " + out_var +
            " = conv(dilations=" + p + "dl, groups=" + p + "gr, pad=" + p + "pd"
            ", pad_type=" + p + "pt, strides=" + p + "st"
            ", weight=" + p + "W, x=" + in_var + ")[name=string(\"" + p + "conv\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.weight_file  = weight_file;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::rmsnorm_fragment(int C, int SP,
                                          const std::string& in_var,
                                          const std::string& out_var,
                                          const std::string& scale_file) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    float inv_c = 1.0f / static_cast<float>(C);
    char inv_c_buf[32];
    std::snprintf(inv_c_buf, sizeof(inv_c_buf), "%.10g", (double)inv_c);

    const std::string p   = out_var + "_";
    std::string tt  = tensor_type(shape);
    std::string tss = "tensor<fp16, [1,1,1," + std::to_string(SP) + "]>";

    std::string wref = "tensor<fp16, [1," + std::to_string(C) + ",1,1]>"
                       "(BLOBFILE(path=string(\"@model_path/weights/" + scale_file +
                       "\"), offset=uint64(" + std::to_string(WeightBlob::kWeightDictOffset) + ")))";

    std::string body;
    body += "        " + tt  + " " + p + "sq = mul(x=" + in_var + ", y=" + in_var + ")[name=string(\"" + p + "sq\")];\n";
    body += "        tensor<int32, [1]> " + p + "ax = const()[name=string(\"" + p + "ax\"), val=tensor<int32, [1]>([1])];\n";
    body += "        bool " + p + "kd = const()[name=string(\"" + p + "kd\"), val=bool(true)];\n";
    body += "        " + tss + " " + p + "ss = reduce_sum(x=" + p + "sq, axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "ss\")];\n";
    body += "        fp16 " + p + "invd = const()[name=string(\"" + p + "invd\"), val=fp16(" + std::string(inv_c_buf) + ")];\n";
    body += "        " + tss + " " + p + "ms = mul(x=" + p + "ss, y=" + p + "invd)[name=string(\"" + p + "ms\")];\n";
    body += "        fp16 " + p + "eps = const()[name=string(\"" + p + "eps\"), val=fp16(9.765625e-4)];\n";
    body += "        " + tss + " " + p + "mse = add(x=" + p + "ms, y=" + p + "eps)[name=string(\"" + p + "mse\")];\n";
    body += "        fp16 " + p + "nh = const()[name=string(\"" + p + "nh\"), val=fp16(-0.5)];\n";
    body += "        " + tss + " " + p + "rr = pow(x=" + p + "mse, y=" + p + "nh)[name=string(\"" + p + "rr\")];\n";
    body += "        " + tt  + " " + p + "xr = mul(x=" + in_var + ", y=" + p + "rr)[name=string(\"" + p + "xr\")];\n";
    body += "        tensor<fp16, [1," + std::to_string(C) + ",1,1]> " + p + "w = const()[name=string(\"" + p + "w\"), val=" + wref + "];\n";
    body += "        " + tt  + " " + out_var + " = mul(x=" + p + "xr, y=" + p + "w)[name=string(\"" + p + "out\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.weight_file  = scale_file;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::layernorm_fragment(int C, int SP,
                                            const std::string& in_var,
                                            const std::string& out_var,
                                            const std::string& gamma_file,
                                            const std::string& beta_file,
                                            float eps) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    char eps_buf[32];
    std::snprintf(eps_buf, sizeof(eps_buf), "%.6g", (double)eps);

    const std::string p   = out_var + "_";
    std::string tt  = tensor_type(shape);
    std::string tss = "tensor<fp16, [1,1,1," + std::to_string(SP) + "]>";
    std::string tc  = "tensor<fp16, [1," + std::to_string(C) + ",1," + std::to_string(SP) + "]>";
    std::string tw  = "tensor<fp16, [1," + std::to_string(C) + ",1,1]>";

    auto wref = [&](const std::string& fname) {
        return tw + "(BLOBFILE(path=string(\"@model_path/weights/" + fname +
               "\"), offset=uint64(" + std::to_string(WeightBlob::kWeightDictOffset) + ")))";
    };

    std::string body;
    body += "        tensor<int32, [1]> " + p + "ax = const()[name=string(\"" + p + "ax\"), val=tensor<int32, [1]>([1])];\n";
    body += "        bool " + p + "kd = const()[name=string(\"" + p + "kd\"), val=bool(true)];\n";
    body += "        " + tss + " " + p + "mean = reduce_mean(x=" + in_var + ", axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "mean\")];\n";
    body += "        " + tc  + " " + p + "cent = sub(x=" + in_var + ", y=" + p + "mean)[name=string(\"" + p + "cent\")];\n";
    body += "        " + tc  + " " + p + "sq = mul(x=" + p + "cent, y=" + p + "cent)[name=string(\"" + p + "sq\")];\n";
    body += "        " + tss + " " + p + "var = reduce_mean(x=" + p + "sq, axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "var\")];\n";
    body += "        fp16 " + p + "eps = const()[name=string(\"" + p + "eps\"), val=fp16(" + std::string(eps_buf) + ")];\n";
    body += "        " + tss + " " + p + "veps = add(x=" + p + "var, y=" + p + "eps)[name=string(\"" + p + "veps\")];\n";
    body += "        fp16 " + p + "nh = const()[name=string(\"" + p + "nh\"), val=fp16(-0.5)];\n";
    body += "        " + tss + " " + p + "rstd = pow(x=" + p + "veps, y=" + p + "nh)[name=string(\"" + p + "rstd\")];\n";
    body += "        " + tc  + " " + p + "norm = mul(x=" + p + "cent, y=" + p + "rstd)[name=string(\"" + p + "norm\")];\n";
    body += "        " + tw  + " " + p + "g = const()[name=string(\"" + p + "g\"), val=" + wref(gamma_file) + "];\n";
    body += "        " + tc  + " " + p + "sc = mul(x=" + p + "norm, y=" + p + "g)[name=string(\"" + p + "sc\")];\n";
    body += "        " + tw  + " " + p + "beta = const()[name=string(\"" + p + "beta\"), val=" + wref(beta_file) + "];\n";
    body += "        " + tc  + " " + out_var + " = add(x=" + p + "sc, y=" + p + "beta)[name=string(\"" + p + "out\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.weight_file  = gamma_file;   // primary weight; beta is the second
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::gelu_fragment(int C, int SP,
                                       const std::string& in_var,
                                       const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + p + "x2 = mul(x=" + in_var + ", y=" + in_var + ")[name=string(\"" + p + "x2\")];\n";
    body += "        " + tt + " " + p + "x3 = mul(x=" + p + "x2, y=" + in_var + ")[name=string(\"" + p + "x3\")];\n";
    body += "        fp16 " + p + "c1 = const()[name=string(\"" + p + "c1\"), val=fp16(0.044715)];\n";
    body += "        " + tt + " " + p + "cx3 = mul(x=" + p + "x3, y=" + p + "c1)[name=string(\"" + p + "cx3\")];\n";
    body += "        " + tt + " " + p + "inn = add(x=" + in_var + ", y=" + p + "cx3)[name=string(\"" + p + "inn\")];\n";
    body += "        fp16 " + p + "c2 = const()[name=string(\"" + p + "c2\"), val=fp16(0.7979)];\n";
    body += "        " + tt + " " + p + "sc = mul(x=" + p + "inn, y=" + p + "c2)[name=string(\"" + p + "sc\")];\n";
    body += "        " + tt + " " + p + "th = tanh(x=" + p + "sc)[name=string(\"" + p + "th\")];\n";
    body += "        fp16 " + p + "one = const()[name=string(\"" + p + "one\"), val=fp16(1.0)];\n";
    body += "        " + tt + " " + p + "onep = add(x=" + p + "th, y=" + p + "one)[name=string(\"" + p + "onep\")];\n";
    body += "        fp16 " + p + "half = const()[name=string(\"" + p + "half\"), val=fp16(0.5)];\n";
    body += "        " + tt + " " + p + "hx = mul(x=" + in_var + ", y=" + p + "half)[name=string(\"" + p + "hx\")];\n";
    body += "        " + tt + " " + out_var + " = mul(x=" + p + "hx, y=" + p + "onep)[name=string(\"" + p + "gelu\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::silu_fragment(int C, int SP,
                                       const std::string& in_var,
                                       const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + p + "sig = sigmoid(x=" + in_var + ")[name=string(\"" + p + "sig\")];\n";
    body += "        " + tt + " " + out_var + " = mul(x=" + in_var + ", y=" + p + "sig)[name=string(\"" + p + "silu\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::softmax_fragment(int C, int SP,
                                          const std::string& in_var,
                                          const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    const std::string ax_var = p + "sax";
    body += "        int32 " + ax_var + " = const()[name=string(\"" + ax_var + "\"), val=int32(1)];\n";
    body += "        " + tt + " " + out_var + " = softmax(axis=" + ax_var + ", x=" + in_var + ")[name=string(\"" + p + "sm\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::add_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& side_var,
                                      const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + out_var + " = add(x=" + in_var + ", y=" + side_var + ")[name=string(\"" + p + "add\")];\n";

    MilFragment f;
    f.body            = std::move(body);
    f.input_name      = in_var;
    f.side_input_name = side_var;
    f.output_name     = out_var;
    f.output_shape    = shape;
    return f;
}

MilFragment MilBuilder::mul_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& side_var,
                                      const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + out_var + " = mul(x=" + in_var + ", y=" + side_var + ")[name=string(\"" + p + "mul\")];\n";

    MilFragment f;
    f.body            = std::move(body);
    f.input_name      = in_var;
    f.side_input_name = side_var;
    f.output_name     = out_var;
    f.output_shape    = shape;
    return f;
}

MilFragment MilBuilder::sub_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& side_var,
                                      const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + out_var + " = sub(x=" + in_var + ", y=" + side_var + ")[name=string(\"" + p + "sub\")];\n";

    MilFragment f;
    f.body            = std::move(body);
    f.input_name      = in_var;
    f.side_input_name = side_var;
    f.output_name     = out_var;
    f.output_shape    = shape;
    return f;
}

MilFragment MilBuilder::real_div_fragment(int C, int SP,
                                           const std::string& in_var,
                                           const std::string& side_var,
                                           const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + out_var + " = real_div(x=" + in_var + ", y=" + side_var + ")[name=string(\"" + p + "rdiv\")];\n";

    MilFragment f;
    f.body            = std::move(body);
    f.input_name      = in_var;
    f.side_input_name = side_var;
    f.output_name     = out_var;
    f.output_shape    = shape;
    return f;
}

MilFragment MilBuilder::sqrt_fragment(int C, int SP,
                                       const std::string& in_var,
                                       const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        " + tt + " " + out_var + " = sqrt(x=" + in_var + ")[name=string(\"" + p + "sqrt\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::log_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    body += "        fp16 " + p + "eps = const()[name=string(\"" + p + "eps\"), val=fp16(0x1.0cp-17)];\n";
    body += "        " + tt + " " + out_var +
            " = log(epsilon=" + p + "eps, x=" + in_var + ")[name=string(\"" + p + "log\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::rsqrt_fragment(int C, int SP,
                                        const std::string& in_var,
                                        const std::string& out_var) {
    TensorShape shape{1, C, 1, SP};
    shape.validate();

    const std::string p  = out_var + "_";
    std::string tt = tensor_type(shape);

    std::string body;
    // ANE compiler requires explicit epsilon parameter for log/rsqrt.
    // Standard MIL treats epsilon as optional but ANE silently rejects
    // programs without it. Discovered by ironmill op verification.
    // Value 0x1.0cp-17 ≈ 7.63e-6 (fp16).
    body += "        fp16 " + p + "eps = const()[name=string(\"" + p + "eps\"), val=fp16(0x1.0cp-17)];\n";
    body += "        " + tt + " " + out_var +
            " = rsqrt(epsilon=" + p + "eps, x=" + in_var + ")[name=string(\"" + p + "rsqrt\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = shape;
    return f;
}

MilFragment MilBuilder::concat_fragment(int in0_C, int in1_C, int SP,
                                         const std::string& in_var,
                                         const std::string& side_var,
                                         const std::string& out_var) {
    TensorShape in0{1, in0_C, 1, SP};
    TensorShape in1{1, in1_C, 1, SP};
    TensorShape out{1, in0_C + in1_C, 1, SP};
    in0.validate();
    in1.validate();
    out.validate();

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        int32 " + p + "ax = const()[name=string(\"" + p + "ax\"), val=int32(1)];\n";
    body += "        bool " + p + "id = const()[name=string(\"" + p + "id\"), val=bool(false)];\n";
    body += "        " + tout + " " + out_var +
            " = concat(axis=" + p + "ax, interleave=" + p + "id, values=(" +
            in_var + ", " + side_var + "))[name=string(\"" + p + "cat\")];\n";

    MilFragment f;
    f.body            = std::move(body);
    f.input_name      = in_var;
    f.side_input_name = side_var;
    f.output_name     = out_var;
    f.output_shape    = out;
    return f;
}

MilFragment MilBuilder::slice_by_index_fragment(int in_C, int in_SP,
                                                 int out_C, int out_SP,
                                                 const std::string& in_var,
                                                 const std::string& out_var) {
    TensorShape in {1, in_C, 1, in_SP};
    TensorShape out{1, out_C, 1, out_SP};
    in.validate();
    out.validate();

    if (out_C > in_C || out_SP > in_SP) {
        throw std::invalid_argument(
            "slice_by_index_fragment: output shape must be <= input shape");
    }

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [4]> " + p + "bg = const()[name=string(\"" + p + "bg\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    body += "        tensor<int32, [4]> " + p + "ed = const()[name=string(\"" + p + "ed\"), val=tensor<int32, [4]>([1," +
            std::to_string(out_C) + ",1," + std::to_string(out_SP) + "])];\n";
    body += "        tensor<int32, [4]> " + p + "st = const()[name=string(\"" + p + "st\"), val=tensor<int32, [4]>([1,1,1,1])];\n";
    body += "        " + tout + " " + out_var +
            " = slice_by_index(begin=" + p + "bg, end=" + p + "ed, strides=" + p + "st, x=" + in_var +
            ")[name=string(\"" + p + "sbi\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::reduce_sum_fragment(int in_C, int SP,
                                             const std::string& in_var,
                                             const std::string& out_var) {
    TensorShape in {1, in_C, 1, SP};
    TensorShape out{1, 1, 1, SP};
    in.validate();
    out.validate();

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [1]> " + p + "ax = const()[name=string(\"" + p + "ax\"), val=tensor<int32, [1]>([1])];\n";
    body += "        bool " + p + "kd = const()[name=string(\"" + p + "kd\"), val=bool(true)];\n";
    body += "        " + tout + " " + out_var +
            " = reduce_sum(x=" + in_var + ", axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "rs\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::reduce_mean_fragment(int in_C, int SP,
                                              const std::string& in_var,
                                              const std::string& out_var) {
    TensorShape in {1, in_C, 1, SP};
    TensorShape out{1, 1, 1, SP};
    in.validate();
    out.validate();

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [1]> " + p + "ax = const()[name=string(\"" + p + "ax\"), val=tensor<int32, [1]>([1])];\n";
    body += "        bool " + p + "kd = const()[name=string(\"" + p + "kd\"), val=bool(true)];\n";
    body += "        " + tout + " " + out_var +
            " = reduce_mean(x=" + in_var + ", axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "rm\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::reduce_max_fragment(int in_C, int SP,
                                             const std::string& in_var,
                                             const std::string& out_var) {
    TensorShape in {1, in_C, 1, SP};
    TensorShape out{1, 1, 1, SP};
    in.validate();
    out.validate();

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [1]> " + p + "ax = const()[name=string(\"" + p + "ax\"), val=tensor<int32, [1]>([1])];\n";
    body += "        bool " + p + "kd = const()[name=string(\"" + p + "kd\"), val=bool(true)];\n";
    body += "        " + tout + " " + out_var +
            " = reduce_max(x=" + in_var + ", axes=" + p + "ax, keep_dims=" + p + "kd)[name=string(\"" + p + "rmax\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::transpose_fragment(int C, int SP,
                                            const std::string& in_var,
                                            const std::string& out_var) {
    TensorShape in {1, C,  1, SP};
    TensorShape out{1, SP, 1, C };
    in.validate();
    out.validate();

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [4]> " + p + "perm = const()[name=string(\"" + p + "perm\"), val=tensor<int32, [4]>([0,3,2,1])];\n";
    body += "        " + tout + " " + out_var + " = transpose(perm=" + p + "perm, x=" + in_var + ")[name=string(\"" + p + "tr\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

MilFragment MilBuilder::reshape_fragment(int in_C, int in_SP,
                                          int out_C, int out_SP,
                                          const std::string& in_var,
                                          const std::string& out_var) {
    TensorShape in {1, in_C, 1, in_SP};
    TensorShape out{1, out_C, 1, out_SP};
    in.validate();
    out.validate();

    if (in.numel() != out.numel()) {
        throw std::invalid_argument(
            "reshape_fragment: input and output must have identical element counts");
    }

    const std::string p  = out_var + "_";
    std::string tout = tensor_type(out);

    std::string body;
    body += "        tensor<int32, [4]> " + p + "shape = const()[name=string(\"" + p + "shape\"), val=tensor<int32, [4]>([1," +
            std::to_string(out_C) + ",1," + std::to_string(out_SP) + "])];\n";
    body += "        " + tout + " " + out_var + " = reshape(shape=" + p + "shape, x=" + in_var + ")[name=string(\"" + p + "rs\")];\n";

    MilFragment f;
    f.body         = std::move(body);
    f.input_name   = in_var;
    f.output_name  = out_var;
    f.output_shape = out;
    return f;
}

/* ── build_fused ─────────────────────────────────────────────────────────── */

MilProgram MilBuilder::build_fused(const std::vector<FusedInput>&  inputs,
                                    const std::vector<MilFragment>& fragments) {
    if (inputs.empty())
        throw std::invalid_argument("build_fused: inputs must not be empty");
    if (fragments.empty())
        throw std::invalid_argument("build_fused: fragments must not be empty");

    // Sort inputs alphabetically — ANE constraint #13
    std::vector<FusedInput> sorted = inputs;
    std::sort(sorted.begin(), sorted.end(),
              [](const FusedInput& a, const FusedInput& b) {
                  return a.var_name < b.var_name;
              });

    std::string t = header();

    // func signature
    t += "    func main<ios18>(";
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i > 0) t += ", ";
        t += tensor_type(sorted[i].shape) + " " + sorted[i].var_name;
    }
    t += ") {\n";

    // Fragment bodies
    for (const auto& frag : fragments)
        t += frag.body;

    // Return last output
    t += "    } -> (" + fragments.back().output_name + ");\n";
    t += "}\n";

    // Collect weight names in order; skip duplicates and empties
    std::vector<std::string> wnames;
    for (const auto& frag : fragments) {
        if (!frag.weight_file.empty()) {
            bool already = false;
            for (const auto& w : wnames)
                if (w == frag.weight_file) { already = true; break; }
            if (!already) wnames.push_back(frag.weight_file);
        }
    }

    // input_shape = the alphabetically-first input's shape (conventionally
    // the chain input for single-input groups; callers may inspect directly)
    MilProgram p;
    p.text             = std::move(t);
    p.all_weight_names = wnames;
    p.weight_name      = wnames.empty() ? "" : wnames[0];
    p.input_shape      = sorted[0].shape;
    p.output_shape     = fragments.back().output_shape;
    return p;
}

MilProgram MilBuilder::build_fused(const std::string& input_name,
                                    TensorShape        input_shape,
                                    const std::vector<MilFragment>& fragments) {
    return build_fused(std::vector<FusedInput>{{input_name, input_shape}}, fragments);
}

} // namespace mil
} // namespace libane
