#include "hwx_backend.hpp"
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

runtime::AneProgram* HwxBackend::compile_group(const AneGraph&    graph,
                                                const FusionGroup& group,
                                                const std::string& debug_name) {
    // Only single-node groups with weight-free activation ops qualify for Path C
    if (group.node_ids.size() != 1)
        return mil_.compile_group(graph, group, debug_name);

    const GraphNode& node = graph.node(group.node_ids[0]);
    if (!is_hwx_eligible(node.op))
        return mil_.compile_group(graph, group, debug_name);

    const mil::TensorShape& out_shape = graph.tensor(node.output).shape;
    const int C = out_shape.channels;
    const int S = out_shape.seq;

    // Path C: emit from cache and load without recompilation
    if (emitter_.can_emit(C, S, node.op)) {
        auto hwx = emitter_.emit(C, S, node.op);
        if (!hwx.empty()) {
            runtime::AneProgram* prog = runtime::ane_load_hwx(
                hwx, C, S,
                tvar(group.inputs[0]),
                tvar(node.output),
                debug_name);
            if (prog) return prog;
            // ane_load_hwx failure → fall through to MilBackend
        }
    }

    // Bootstrap: compile via MilBackend, then capture the HWX for future use
    runtime::AneProgram* prog = mil_.compile_group(graph, group, debug_name);
    if (prog && !prog->model_dir.empty())
        emitter_.capture_from_model_dir(prog->model_dir, C, S, node.op);
    return prog;
}

} // namespace graph
} // namespace libane
