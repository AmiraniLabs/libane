/**
 * GraphExecutor implementation.
 */
#include "graph_executor.hpp"

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

#include <unordered_map>
#include <memory>
#include <vector>
#include <cstring>

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

    // Acquire and fill one buffer per graph input.
    // If the tensor has int8 quant params (from a W8A8 op), the caller has
    // supplied int8 data — dequantize to fp16 before packing the IOSurface.
    for (size_t i = 0; i < cg.graph_input_ids().size(); ++i) {
        TensorId tid = cg.graph_input_ids()[i];
        if (!cg.tensor_bytes().count(tid)) return false;
        const auto* tshape = cg.tensor_shape(tid);
        if (!tshape) return false;

        auto buf = global_buffer_pool().acquire_tensor_padded(
            tshape->channels, tshape->height * tshape->seq, cg.io_alloc_bytes());
        if (!buf) return false;

        auto qp_it = cg.quant_params().find(tid);
        if (qp_it != cg.quant_params().end()) {
            // int8 input: dequantize to fp16 before IOSurface copy
            const QuantParams& qp    = qp_it->second;
            const int8_t*      src   = static_cast<const int8_t*>(input_ptrs[i]);
            size_t             n_elems = input_bytes[i]; // bytes == elements for int8
            std::vector<uint16_t> fp16_buf(n_elems);
            for (size_t k = 0; k < n_elems; ++k) {
                float f = (static_cast<float>(src[k]) - static_cast<float>(qp.zero_point))
                          * qp.scale;
                // fp32 → fp16 via bit manipulation
                uint32_t fb; std::memcpy(&fb, &f, 4);
                uint32_t s = (fb >> 16) & 0x8000u;
                int32_t  e = static_cast<int32_t>((fb >> 23) & 0xFFu) - 127 + 15;
                uint32_t m = (fb >> 13) & 0x3FFu;
                uint16_t h;
                if (e <= 0)       h = static_cast<uint16_t>(s);
                else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00u);
                else              h = static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | m);
                fp16_buf[k] = h;
            }
            buf->copy_from(fp16_buf.data(), n_elems * sizeof(uint16_t));
        } else {
            // fp16 input: normal path
            buf->copy_from(input_ptrs[i], input_bytes[i]);
        }

        if (!buf) return false;
        tmp_input_bufs[tid] = std::move(buf);
    }

#ifdef __APPLE__
    for (const auto& group : cg.groups()) {
        runtime::AneProgram* program = group.program;
        if (!program) return false;

        // Path B — _ANEClient warm-path dispatch
        if (program->objc_client_model) {
            TensorId in_tid = group.inputs[0];
            const auto* in_shape  = cg.tensor_shape(in_tid);
            const auto* out_shape = cg.tensor_shape(group.output);
            if (!in_shape || !out_shape) return false;

            const int in_ch      = in_shape->channels;
            const int out_ch     = out_shape->channels;
            const int g_in_seq   = in_shape->seq;
            const int g_out_seq  = out_shape->seq;
            const int cl_in_seq  = program->client_in_seq;
            const int cl_out_seq = program->client_out_seq;

            // Read full graph input buffer: [in_ch][g_in_seq] fp16
            size_t full_in_bytes = static_cast<size_t>(in_ch) * g_in_seq * 2;
            std::vector<uint8_t> full_in(full_in_bytes);
            {
                auto tmp_it = tmp_input_bufs.find(in_tid);
                if (tmp_it != tmp_input_bufs.end())
                    tmp_it->second->copy_to(full_in.data(), full_in_bytes);
                else {
                    AneBuffer* buf = cg.ane_buf(in_tid);
                    if (!buf) return false;
                    buf->copy_to(full_in.data(), full_in_bytes);
                }
            }

            // Stride-extract: take first cl_in_seq fp16s from each channel.
            // Converts [in_ch][g_in_seq] → [in_ch][cl_in_seq] (column-0 slice).
            const size_t cl_in_ch_bytes = static_cast<size_t>(cl_in_seq) * 2;
            std::vector<uint8_t> cl_in(static_cast<size_t>(in_ch) * cl_in_ch_bytes);
            for (int c = 0; c < in_ch; ++c)
                std::memcpy(cl_in.data()  + c * cl_in_ch_bytes,
                            full_in.data() + c * (static_cast<size_t>(g_in_seq) * 2),
                            cl_in_ch_bytes);

            // Execute via _ANEClient
            const size_t cl_out_ch_bytes = static_cast<size_t>(cl_out_seq) * 2;
            std::vector<uint8_t> cl_out(static_cast<size_t>(out_ch) * cl_out_ch_bytes);
            if (!runtime::ane_execute_client(program, cl_in.data(), cl_out.data()))
                return false;

            // Stride-insert: write cl_out_seq fp16s back to position 0 of each channel,
            // zero-fill the remainder. Converts [out_ch][cl_out_seq] → [out_ch][g_out_seq].
            size_t full_out_bytes = static_cast<size_t>(out_ch) * g_out_seq * 2;
            std::vector<uint8_t> full_out(full_out_bytes, 0);
            for (int j = 0; j < out_ch; ++j)
                std::memcpy(full_out.data() + j * (static_cast<size_t>(g_out_seq) * 2),
                            cl_out.data()   + j * cl_out_ch_bytes,
                            cl_out_ch_bytes);

            AneBuffer* out_buf = cg.ane_buf(group.output);
            if (!out_buf) return false;
            out_buf->copy_from(full_out.data(), full_out_bytes);
            continue;
        }

        // Path A — IOSurface-based dispatch
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

bool GraphExecutor::delta_reload(const CompiledGraph& cg) {
    for (const auto& group : cg.groups()) {
        if (!group.program) continue;
        if (!runtime::ane_delta_reload(group.program))
            return false;
    }
    return true;
}

} // namespace graph
} // namespace libane
