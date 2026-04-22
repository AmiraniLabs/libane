#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace libane {
namespace graph {

/**
 * hwx_capture_inline — compile MIL via _ANEMILCompiler in-process and return
 * the resulting BEEFFACE HWX bytes.
 *
 * model_dir must be a directory containing model.mil (the format produced by
 * ane_compile / MilBackend's temp dir).  On success returns non-empty bytes.
 * Returns empty on failure (macOS < 26 where the class isn't present, or if
 * dlopen fails).
 *
 * Only available on Apple targets.
 */
std::vector<uint8_t> hwx_capture_inline(const std::string& model_dir);

} // namespace graph
} // namespace libane
