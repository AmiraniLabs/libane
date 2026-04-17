#include "hwx_backend.hpp"
#include "mil_backend.hpp"
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

    // Warm path: emit patched HWX from cache
    if (emitter_.can_emit(C, S, node.op)) {
        auto hwx = emitter_.emit(C, S, node.op);
        if (!hwx.empty()) {
            runtime::AneProgram* prog = runtime::ane_load_hwx(
                hwx, C, S,
                tvar(group.inputs[0]),
                tvar(node.output),
                debug_name);
            if (prog) return prog;
        }
    }

    // Cold path: bootstrap via MilBackend, capture resulting HWX for future use
    MilBackend mil;
    runtime::AneProgram* prog = mil.compile_group(graph, group, debug_name);
    if (prog && !prog->model_dir.empty())
        emitter_.capture_from_model_dir(prog->model_dir, C, S, node.op);
    return prog;
}

} // namespace graph
} // namespace libane
