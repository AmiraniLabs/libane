/**
 * GraphExecutor implementation.
 */
#include "graph_executor.hpp"

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

#include <unordered_map>
#include <memory>

namespace libane {
namespace graph {

bool GraphExecutor::execute(const CompiledGraph&            cg,
                             const std::vector<const void*>& input_ptrs,
                             const std::vector<size_t>&      input_bytes,
                             const std::vector<void*>&       output_ptrs,
                             const std::vector<size_t>&      output_bytes) {
    // Validate I/O counts
    if (input_ptrs.size()  != cg.graph_input_ids().size())  return false;
    if (input_bytes.size() != cg.graph_input_ids().size())  return false;
    if (output_ptrs.size() != cg.graph_output_ids().size()) return false;
    if (output_bytes.size()!= cg.graph_output_ids().size()) return false;

    // Temporary IOSurface-backed buffers for graph inputs.
    // Wrapped in a RAII guard so they return to the pool on every exit path.
    std::unordered_map<TensorId, std::unique_ptr<AneBuffer>> tmp_input_bufs;

    struct PoolGuard {
        std::unordered_map<TensorId, std::unique_ptr<AneBuffer>>& bufs;
        ~PoolGuard() {
            for (auto& kv : bufs)
                if (kv.second)
                    global_buffer_pool().release(std::move(kv.second));
        }
    } guard{tmp_input_bufs};

    // Acquire and fill one buffer per graph input
    for (size_t i = 0; i < cg.graph_input_ids().size(); ++i) {
        TensorId tid = cg.graph_input_ids()[i];
        if (!cg.tensor_bytes().count(tid)) return false;
        const auto* tshape = cg.tensor_shape(tid);
        if (!tshape) return false;

        // Input pointers are contiguous logical [1,C,1,S] fp16 tensors.
        // Acquire tensor-aware buffers so copy_from applies ANE stride-safe packing.
        auto buf = global_buffer_pool().acquire_tensor_padded(
            tshape->channels, tshape->seq, cg.io_alloc_bytes());
        if (!buf) return false;
        buf->copy_from(input_ptrs[i], input_bytes[i]);
        if (!buf) return false;
        tmp_input_bufs[tid] = std::move(buf);
    }

#ifdef __APPLE__
    for (const auto& group : cg.groups()) {
        std::vector<IOSurfaceRef> in_surfs;
        in_surfs.reserve(group.inputs.size());

        for (TensorId tid : group.inputs) {
            AneBuffer* buf = nullptr;
            auto tmp_it = tmp_input_bufs.find(tid);
            if (tmp_it != tmp_input_bufs.end())
                buf = tmp_it->second.get();
            else
                buf = cg.ane_buf(tid);
            if (!buf) return false;
            in_surfs.push_back(buf->iosurface());
        }

        AneBuffer* out_buf = cg.ane_buf(group.output);
        if (!out_buf) return false;
        std::vector<IOSurfaceRef> out_surfs = { out_buf->iosurface() };

        if (!runtime::ane_execute_multi(group.program, in_surfs, out_surfs))
            return false;
    }
#else
    return false;  // Non-Apple: execution not supported
#endif

    // Copy graph outputs to caller-supplied destination pointers.
    // Pool release happens via guard destructor after this.
    for (size_t i = 0; i < cg.graph_output_ids().size(); ++i) {
        TensorId   tid = cg.graph_output_ids()[i];
        AneBuffer* buf = cg.ane_buf(tid);
        if (!buf) return false;
        buf->copy_to(output_ptrs[i], output_bytes[i]);
    }

    return true;
}

} // namespace graph
} // namespace libane
