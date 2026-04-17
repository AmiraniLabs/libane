/**
 * HwxEmitter — in-process cache and patcher for BEEFFACE HWX binaries.
 *
 * Supports ANE activation ops by maintaining two caches:
 *   shape_cache_:   (channels, seq) → reference HWX bytes from any compiled op
 *   op_cfg_cache_:  op → 5 extracted op-specific words from __text
 *
 * First compilation of a new (C, S, op): MilBackend produces the HWX via
 * aned; HwxBackend captures it with capture_from_model_dir().  Subsequent
 * requests with cached (C, S) + any known op use emit() to cross-patch the
 * shape template without recompilation.
 *
 * The patching is correct because the ANE __text section mixes shape-specific
 * configuration (buffer sizes, strides — unchanged across ops for fixed shape)
 * with 4 op-specific words at fixed offsets:
 *
 *   Word[ 4] @ __text+0x10 : program_size  — pipeline entry count
 *   Word[12] @ __text+0x30 : stage_flags   — stage enable bitmask
 *   Word[19] @ __text+0x4C : opcode        — NE operation selector
 *   Word[23] @ __text+0x5C : extra_config  — additional pipeline config
 *
 * The __const section (16 KB pipeline config) is identical across all
 * activation ops for the same tensor shape (beefface research finding),
 * so only __text changes and the 4 words above control the operation.
 *
 * When num_words changes between ops (e.g., relu=65, abs=63), __const
 * must be relocated: section headers and TD cross-references are updated.
 */
#pragma once

#include "../../include/libane.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace libane {
namespace graph {

class HwxEmitter {
public:
    HwxEmitter() = default;

    struct OpConfig {
        uint32_t program_size = 0;  // Word[4]
        uint32_t stage_flags  = 0;  // Word[12]
        uint32_t opcode       = 0;  // Word[19]
        uint32_t extra_config = 0;  // Word[23]
        uint32_t num_words    = 0;  // __text size / 4
    };

    /**
     * Scan model_dir recursively for the first *.hwx file produced by
     * ane_compile / MilBackend and cache it.  Also extracts and caches
     * the op config.  Returns true on success.
     */
    bool capture_from_model_dir(const std::string& model_dir,
                                int channels, int seq, libane_op_t op);

    /**
     * Emit patched HWX bytes for (channels, seq, op).
     *
     * Returns empty vector if the required data is not yet cached
     * (caller should fall back to MilBackend + capture_from_model_dir).
     */
    std::vector<uint8_t> emit(int channels, int seq, libane_op_t op) const;

    bool can_emit(int channels, int seq, libane_op_t op) const;
    bool has_shape(int channels, int seq) const;
    bool has_op_config(libane_op_t op) const;

private:
    struct ShapeKey {
        int channels, seq;
        bool operator==(const ShapeKey& o) const {
            return channels == o.channels && seq == o.seq;
        }
    };
    struct ShapeKeyHash {
        size_t operator()(const ShapeKey& k) const noexcept {
            return std::hash<uint64_t>{}(
                (uint64_t)(uint32_t)k.channels << 32 | (uint32_t)k.seq);
        }
    };

    std::unordered_map<ShapeKey, std::vector<uint8_t>, ShapeKeyHash> shape_cache_;
    std::unordered_map<int, OpConfig>                                 op_cfg_cache_;

    static std::vector<uint8_t> read_hwx_from_dir(const std::string& dir);
    static OpConfig              extract_op_config(const std::vector<uint8_t>& hwx);
    static std::vector<uint8_t> patch_op(std::vector<uint8_t>  hwx,
                                          const OpConfig&       new_cfg,
                                          const OpConfig&       old_cfg);
    static bool valid_hwx(const std::vector<uint8_t>& hwx);
};

} // namespace graph
} // namespace libane
