#include "hwx_backend.hpp"
#include "hwx_inline_capture.hpp"
#include "../core/mil_builder.hpp"
#include "../runtime/ane_runtime.hpp"

#include <string>

namespace libane {
namespace graph {

static std::string tvar(TensorId id) {
    return "t" + std::to_string(id);
}

HwxBackend::HwxBackend()
    : owned_mil_(std::make_unique<MilBackend>()), mil_(owned_mil_.get()) {}

HwxBackend::HwxBackend(MilBackend* shared_mil)
    : owned_mil_(nullptr), mil_(shared_mil) {}

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

    // Build target-op MIL text up front — needed by both the warm-reconnect
    // lookup (Phase 1 cache on MilBackend) and the HwxEmitter cross-op
    // patch path (ane_load_hwx takes mil_text + hwx_bytes).
    const std::string in_var  = tvar(group.inputs[0]);
    const std::string out_var = tvar(node.output);
    const mil::TensorShape shape{1, C, 1, S};
    mil::MilFragment frag = [&]() -> mil::MilFragment {
        switch (node.op) {
        case LIBANE_OP_RELU:       return mil::MilBuilder::relu_fragment(C, S, in_var, out_var);
        case LIBANE_OP_TANH:       return mil::MilBuilder::tanh_fragment(C, S, in_var, out_var);
        case LIBANE_OP_SIGMOID:    return mil::MilBuilder::sigmoid_fragment(C, S, in_var, out_var);
        case LIBANE_OP_HARDSWISH:  return mil::MilBuilder::hardswish_fragment(C, S, in_var, out_var);
        case LIBANE_OP_LEAKY_RELU: return mil::MilBuilder::leaky_relu_fragment(C, S, in_var, out_var);
        case LIBANE_OP_ELU:        return mil::MilBuilder::elu_fragment(C, S, in_var, out_var);
        default:                   return mil::MilBuilder::relu_fragment(C, S, in_var, out_var);
        }
    }();
    const std::string mil_text = mil::MilBuilder::build_fused(
        in_var, shape, {frag}).text;

    // Tier 1: MilBackend URL reconnect — same op, aned slot still alive.
    // ~0.7 ms when hot.  Hands back a ready-to-execute AneProgram.
    if (auto* warm = mil_->try_warm_reconnect(mil_text, {}, debug_name))
        return warm;

    // Tier 2: HwxEmitter cross-op patch — different op than what's cached
    // but same shape template exists.  Patches op-config words and pre-
    // stages the binary in localModelPath for compileWithQoS:'s
    // compileAsNeeded path (~20–40 ms).
    if (emitter_.can_emit(C, S, node.op)) {
        auto hwx = emitter_.emit(C, S, node.op);
        if (!hwx.empty()) {
            if (auto* prog = runtime::ane_load_hwx(
                    hwx, mil_text, C, S, in_var, out_var, debug_name))
                return prog;
        }
    }

    // Tier 3: delegate to MilBackend for a full cold compile.  MilBackend
    // populates its own URL cache on success so the next call for this
    // same (mil_text, weights) hits try_warm_reconnect.
    runtime::AneProgram* prog = mil_->compile_group(graph, group, debug_name);
    if (prog) {
        // Seed HwxEmitter with the fresh HWX bytes so future cross-op
        // hits at this shape skip the cold path.  macOS ≤25 reads from
        // the model temp dir; macOS 26 uses inline _ANEMILCompiler.
        bool captured = false;
        if (!prog->model_dir.empty())
            captured = emitter_.capture_from_model_dir(prog->model_dir, C, S, node.op);
        if (!captured && !prog->model_dir.empty()) {
            auto hwx = hwx_capture_inline(prog->model_dir);
            if (!hwx.empty())
                emitter_.capture_from_bytes(hwx, C, S, node.op);
        }
    }
    return prog;
}

} // namespace graph
} // namespace libane
