/**
 * HwxEmitter implementation — BEEFFACE HWX binary cache and patcher.
 *
 * Format references:
 *   reference/beefface/docs/HWX_BYTE_MAP.md   — full byte-level format
 *   reference/beefface/src/hwx_format.py      — patch point constants
 *   reference/beefface/src/zin_builder.py     — template patching logic
 *
 * Apple's authoritative names (confirmed from framework strings):
 *   ANECIR  — in-memory IR  (mlir::anec dialect)
 *   HWX     — file format   (model.hwx, magic 0xBEEFFACE)
 */
#include "hwx_emitter.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <cstdio>

namespace libane {
namespace graph {

namespace {

// ── HWX format constants ──────────────────────────────────────────────────

static constexpr uint32_t kBeeffaceMagic = 0xBEEFFACEu;

// File offsets for section-header cross-references (from HWX_BYTE_MAP.md)
static constexpr size_t kTextSectSizeOff  = 0x0208;  // __text size in LC[3]  (uint64 LE)
static constexpr size_t kConstVmaddrOff   = 0x0250;  // __const vmaddr        (uint64 LE)
static constexpr size_t kConstFileoffOff  = 0x0260;  // __const fileoff       (uint32 LE)
static constexpr size_t kTdInstrCntOff    = 0x0AE4;  // TD instruction count  (uint32 LE)

// __text section starts at fixed file offset 0x4000
static constexpr size_t kTextFileOff  = 0x4000;

// __const section: 64-byte aligned after __text, always 0x4000 bytes
static constexpr size_t kConstAlign   = 64;
static constexpr size_t kConstSize    = 0x4000;

// ANE VM base for __TEXT segment (0x30008000)
static constexpr uint64_t kTextVmBase = 0x30008000ULL;

// Op-specific word file offsets (fixed within __text header region)
static constexpr size_t kPP_ProgramSize  = 0x4010;  // Word[4]
static constexpr size_t kPP_StageFlags   = 0x4030;  // Word[12]
static constexpr size_t kPP_Opcode       = 0x404C;  // Word[19]
static constexpr size_t kPP_ExtraConfig  = 0x405C;  // Word[23]

// Minimum valid file size (header + first few load commands)
static constexpr size_t kMinHwxSize = kTextFileOff + 0x60;

// ── Little-endian helpers ─────────────────────────────────────────────────

static uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static void wr32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
static void wr64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }

} // namespace

// ── HwxEmitter::valid_hwx ────────────────────────────────────────────────

bool HwxEmitter::valid_hwx(const std::vector<uint8_t>& hwx) {
    if (hwx.size() < kMinHwxSize) return false;
    return rd32(hwx.data()) == kBeeffaceMagic;
}

// ── HwxEmitter::read_hwx_from_dir ────────────────────────────────────────

std::vector<uint8_t> HwxEmitter::read_hwx_from_dir(const std::string& dir) {
    namespace fs = std::filesystem;

    std::string hwx_path;
    try {
        for (const auto& e : fs::recursive_directory_iterator(
                 dir, fs::directory_options::skip_permission_denied)) {
            if (e.is_regular_file() && e.path().extension() == ".hwx") {
                hwx_path = e.path().string();
                break;
            }
        }
    } catch (...) {
        return {};
    }
    if (hwx_path.empty()) return {};

    FILE* f = fopen(hwx_path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return {}; }
    std::vector<uint8_t> data((size_t)sz);
    bool ok = fread(data.data(), 1, (size_t)sz, f) == (size_t)sz;
    fclose(f);
    if (!ok) return {};

    if (!valid_hwx(data)) return {};
    return data;
}

// ── HwxEmitter::extract_op_config ────────────────────────────────────────

HwxEmitter::OpConfig HwxEmitter::extract_op_config(const std::vector<uint8_t>& hwx) {
    OpConfig cfg;
    if (hwx.size() < kMinHwxSize) return cfg;
    cfg.program_size = rd32(hwx.data() + kPP_ProgramSize);
    cfg.stage_flags  = rd32(hwx.data() + kPP_StageFlags);
    cfg.opcode       = rd32(hwx.data() + kPP_Opcode);
    cfg.extra_config = rd32(hwx.data() + kPP_ExtraConfig);
    cfg.num_words    = static_cast<uint32_t>(
        rd64(hwx.data() + kTextSectSizeOff) / 4);
    return cfg;
}

// ── HwxEmitter::patch_op ─────────────────────────────────────────────────

std::vector<uint8_t> HwxEmitter::patch_op(std::vector<uint8_t> hwx,
                                            const OpConfig& new_cfg,
                                            const OpConfig& old_cfg) {
    if (!valid_hwx(hwx)) return hwx;

    // Patch the 4 op-specific words (always at fixed offsets in __text header)
    wr32(hwx.data() + kPP_ProgramSize, new_cfg.program_size);
    wr32(hwx.data() + kPP_StageFlags,  new_cfg.stage_flags);
    wr32(hwx.data() + kPP_Opcode,      new_cfg.opcode);
    wr32(hwx.data() + kPP_ExtraConfig, new_cfg.extra_config);

    if (new_cfg.num_words == old_cfg.num_words) return hwx;  // no layout change

    // __text size changed — relocate __const and update all cross-references
    size_t old_text  = old_cfg.num_words * 4;
    size_t new_text  = new_cfg.num_words * 4;

    size_t old_const = (kTextFileOff + old_text + kConstAlign - 1) & ~(kConstAlign - 1);
    size_t new_const = (kTextFileOff + new_text + kConstAlign - 1) & ~(kConstAlign - 1);

    uint64_t old_vm = kTextVmBase + (old_const - kTextFileOff);
    uint64_t new_vm = kTextVmBase + (new_const - kTextFileOff);

    // Save __const before modifying the buffer
    std::vector<uint8_t> const_data;
    if (old_const + kConstSize <= hwx.size())
        const_data.assign(hwx.data() + old_const, hwx.data() + old_const + kConstSize);

    // Ensure buffer is large enough for new layout
    size_t needed = new_const + kConstSize;
    if (hwx.size() < needed) hwx.resize(needed, 0);

    // Zero the old __text tail if shrinking
    if (new_text < old_text)
        std::fill(hwx.data() + kTextFileOff + new_text,
                  hwx.data() + kTextFileOff + old_text, 0u);

    // Move __const to new offset
    if (old_const != new_const && !const_data.empty()) {
        std::fill(hwx.data() + old_const, hwx.data() + old_const + kConstSize, 0u);
        std::copy(const_data.begin(), const_data.end(), hwx.data() + new_const);
    }

    // Update LC[3] __text section size
    wr64(hwx.data() + kTextSectSizeOff, (uint64_t)new_text);

    // Update LC[3] __const vmaddr
    wr64(hwx.data() + kConstVmaddrOff, new_vm);

    // Update LC[3] __const fileoff
    wr32(hwx.data() + kConstFileoffOff, (uint32_t)new_const);

    // Replace all occurrences of the old __const vmaddr in the load command
    // region (covers the TD flavor=4 master config and any duplicates)
    uint8_t old_vm_b[8], new_vm_b[8];
    wr64(old_vm_b, old_vm);
    wr64(new_vm_b, new_vm);

    constexpr size_t kLcEnd = kTextFileOff;
    for (size_t i = 0x20; i + 8 <= kLcEnd; ++i) {
        if (std::memcmp(hwx.data() + i, old_vm_b, 8) == 0) {
            std::memcpy(hwx.data() + i, new_vm_b, 8);
            i += 7;
        }
    }

    // Update TD instruction word count
    wr32(hwx.data() + kTdInstrCntOff, new_cfg.num_words);

    return hwx;
}

// ── HwxEmitter public API ─────────────────────────────────────────────────

bool HwxEmitter::capture_from_model_dir(const std::string& model_dir,
                                         int channels, int seq, libane_op_t op) {
    auto hwx = read_hwx_from_dir(model_dir);
    if (hwx.empty()) return false;

    ShapeKey key{channels, seq};
    if (!shape_cache_.count(key))
        shape_cache_[key] = hwx;

    int iop = static_cast<int>(op);
    if (!op_cfg_cache_.count(iop))
        op_cfg_cache_[iop] = extract_op_config(hwx);

    return true;
}

bool HwxEmitter::has_shape(int channels, int seq) const {
    return shape_cache_.count(ShapeKey{channels, seq}) > 0;
}

bool HwxEmitter::has_op_config(libane_op_t op) const {
    return op_cfg_cache_.count(static_cast<int>(op)) > 0;
}

bool HwxEmitter::can_emit(int channels, int seq, libane_op_t op) const {
    return has_shape(channels, seq) && has_op_config(op);
}

std::vector<uint8_t> HwxEmitter::emit(int channels, int seq, libane_op_t op) const {
    if (!can_emit(channels, seq, op)) return {};

    const auto& tmpl    = shape_cache_.at(ShapeKey{channels, seq});
    const auto& new_cfg = op_cfg_cache_.at(static_cast<int>(op));
    const OpConfig old_cfg = extract_op_config(tmpl);

    return patch_op(tmpl, new_cfg, old_cfg);
}

} // namespace graph
} // namespace libane
