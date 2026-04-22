#include "hwx_backend.hpp"
#include "hwx_inline_capture.hpp"
#include "mil_backend.hpp"
#include "../core/mil_builder.hpp"
#include "../runtime/ane_runtime.hpp"

#include <string>

namespace libane {
namespace graph {

static std::string tvar(TensorId id) {
    return "t" + std::to_string(id);
}

bool HwxBackend::is_hwx_eligible(libane_op_t op) {
    switch (op) {
    case LIBANE_OP_RELU:
    case LIBANE_OP_TANH:
    case LIBANE_OP_SIGMOID:
    case LIBANE_OP_HARDSWISH:
    case LIBANE_OP_LEAKY_RELU:
    case LIBANE_OP_ELU:
        return true;
    default:
        return false;
    }
}

bool HwxBackend::owns(const AneGraph& graph, const FusionGroup& group) const {
    if (group.node_ids.size() != 1) return false;
    return is_hwx_eligible(graph.node(group.node_ids[0]).op);
}

runtime::AneProgram* HwxBackend::compile_group(const AneGraph&    graph,
                                                const FusionGroup& group,
                                                const std::string& debug_name) {
    const GraphNode&        node      = graph.node(group.node_ids[0]);
    const mil::TensorShape& out_shape = graph.tensor(node.output).shape;
    const int C = out_shape.channels;
    const int S = out_shape.seq;

    // Warm path A (macOS 26+): URL reconnect — injects stored model URL into a
    // fresh _ANEInMemoryModel and calls loadWithQoS: directly, skipping
    // compileWithQoS: entirely (~0.722ms when aned still holds the slot).
    ShapeOpKey key{C, S, (int)node.op};
    auto url_it = url_cache_.find(key);
    if (url_it != url_cache_.end()) {
        runtime::AneProgram* prog = runtime::ane_reconnect(
            url_it->second.mil_text, {}, url_it->second.model_url, debug_name);
        if (prog) return prog;
        // Slot was purged — fall through to cold compile and refresh cache entry
    }

    // Warm path B: HWX binary patching via HwxEmitter.
    // On macOS ≤25: capture_from_model_dir extracts the .hwx from the temp dir.
    // On macOS 26+: hwx_capture_inline provides bytes; ane_load_hwx pre-stages
    // model.hwx + model.mil in localModelPath before compileWithQoS: — aned's
    // compileAsNeeded logic uses the existing binary rather than ANECCompile()
    // (~38ms vs ~111ms cold compile).  Confirmed by XPC-18/XPC-19 (2026-04-22).
    if (emitter_.can_emit(C, S, node.op)) {
        auto hwx = emitter_.emit(C, S, node.op);
        if (!hwx.empty()) {
            const std::string in_var  = tvar(group.inputs[0]);
            const std::string out_var = tvar(node.output);
            const mil::TensorShape shape{1, C, 1, S};
            mil::MilFragment frag = [&]() -> mil::MilFragment {
                switch (node.op) {
                case LIBANE_OP_RELU:      return mil::MilBuilder::relu_fragment(C, S, in_var, out_var);
                case LIBANE_OP_TANH:      return mil::MilBuilder::tanh_fragment(C, S, in_var, out_var);
                case LIBANE_OP_SIGMOID:   return mil::MilBuilder::sigmoid_fragment(C, S, in_var, out_var);
                case LIBANE_OP_HARDSWISH: return mil::MilBuilder::hardswish_fragment(C, S, in_var, out_var);
                case LIBANE_OP_LEAKY_RELU:return mil::MilBuilder::leaky_relu_fragment(C, S, in_var, out_var);
                case LIBANE_OP_ELU:       return mil::MilBuilder::elu_fragment(C, S, in_var, out_var);
                default:                  return mil::MilBuilder::relu_fragment(C, S, in_var, out_var);
                }
            }();
            std::string mil_text = mil::MilBuilder::build_fused(
                in_var, shape, {frag}).text;

            runtime::AneProgram* prog = runtime::ane_load_hwx(
                hwx, mil_text, C, S, in_var, out_var, debug_name);
            if (prog) return prog;
        }
    }

    // Cold path: full MilBackend compile.  Captures URL (macOS 26) and HWX
    // bytes (macOS ≤25) for subsequent warm-path hits.
    MilBackend mil;
    runtime::AneProgram* prog = mil.compile_group(graph, group, debug_name);
    if (prog) {
        bool captured = false;
        if (!prog->model_dir.empty())
            captured = emitter_.capture_from_model_dir(prog->model_dir, C, S, node.op);

        // macOS 26: model_dir has no .hwx (aned withholds it).  Call
        // _ANEMILCompiler inline — same binary but via dlopen, no XPC gate.
        if (!captured && !prog->model_dir.empty()) {
            auto hwx = hwx_capture_inline(prog->model_dir);
            if (!hwx.empty())
                emitter_.capture_from_bytes(hwx, C, S, node.op);
        }

        if (!prog->model_url.empty())
            url_cache_[key] = {prog->model_url, prog->mil_text};
    }
    return prog;
}

} // namespace graph
} // namespace libane
