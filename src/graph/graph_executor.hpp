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
};

} // namespace graph
} // namespace libane
