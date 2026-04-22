/**
 * Graph executor — runs a CompiledGraph on the ANE.
 *
 * execute() performs a single forward pass:
 *
 *  For each fusion group (in topological order):
 *   1. Gather input IOSurfaces:
 *      - Graph inputs:    acquire a buffer from the global pool, copy fp16 data in.
 *      - Intermediates:   use the pre-allocated buffer stored in CompiledGraph.
 *   2. Dispatch via ane_execute_multi().
 *   3. Release temporary graph-input buffers back to the pool.
 *
 *  After the last group, copy each graph-output tensor from its pre-allocated
 *  AneBuffer to the caller-supplied destination pointer.
 *
 * Thread safety:
 *  A single CompiledGraph may be executed from only one thread at a time.
 *  The CompiledGraph's internal AneBuffers are mutated on every call.
 */
#pragma once

#include "graph_compiler.hpp"

#include <vector>
#include <cstddef>

namespace libane {
namespace graph {

class GraphExecutor {
public:
    /**
     * Run a compiled graph.
     *
     * @param cg           Compiled graph (from GraphCompiler::compile).
     * @param input_ptrs   fp16 data pointers; one per graph_input_ids() entry.
     * @param input_bytes  Byte size of each input buffer (must match tensor size).
     * @param output_ptrs  Destination buffers; one per graph_output_ids() entry.
     * @param output_bytes Byte size of each output buffer.
     *
     * @return true on success; false if any ane_execute_multi call fails or if
     *         input/output counts do not match the graph's declared I/O.
     */
    static bool execute(const CompiledGraph&            cg,
                        const std::vector<const void*>& input_ptrs,
                        const std::vector<size_t>&      input_bytes,
                        const std::vector<void*>&       output_ptrs,
                        const std::vector<size_t>&      output_bytes);

    /**
     * Reload all ANE programs in a compiled graph into SRAM without
     * recompiling.  Equivalent to calling ane_delta_reload() on each
     * group's program in order.
     *
     * Use this after the host has been suspended / resumed, or after any
     * ANE context reset, to restore the compiled weights without paying the
     * full compile cost (~8.5× faster than recompiling).
     *
     * Also enables LoRA-style weight hot-swap workflows: rewrite the weight
     * blobs in the AneProgram SRAM buffers, then call delta_reload() to
     * push the new weights to the accelerator without rebuilding the graph.
     *
     * @param cg  Compiled graph whose programs are to be reloaded.
     * @return    true if every ane_delta_reload() call succeeded;
     *            false on the first failure (remaining groups are skipped).
     */
    static bool delta_reload(const CompiledGraph& cg);
};

} // namespace graph
} // namespace libane
