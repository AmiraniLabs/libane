/**
 * GraphCompiler implementation.
 */
#include "graph_compiler.hpp"
#include "graph_validator.hpp"
#include "hwx_backend.hpp"
#include "mil_backend.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

namespace libane {
namespace graph {

/* ── CompiledGraph ───────────────────────────────────────────────────────── */

CompiledGraph::~CompiledGraph() {
    for (auto& g : groups_)
        runtime::ane_unload(g.program);
}

AneBuffer* CompiledGraph::ane_buf(TensorId id) const {
    auto it = ane_bufs_.find(id);
    return it == ane_bufs_.end() ? nullptr : it->second.get();
}

/* ── GraphCompiler::build_plan ───────────────────────────────────────────── */

ExecutionPlan GraphCompiler::build_plan(const AneGraph& graph) {
    ValidationResult vr = GraphValidator::validate(graph);
    if (!vr.ok()) {
        std::string msg = "Graph validation failed:";
        for (const auto& e : vr.errors)
            msg += "\n  " + e;
        throw std::runtime_error(msg);
    }

    ExecutionPlan plan;
    plan.groups           = FusionRules::compute_groups(graph);
    plan.graph_input_ids  = graph.graph_inputs();
    plan.graph_output_ids = graph.graph_outputs();

    for (const auto& t : graph.tensors())
        plan.tensor_bytes[t.id] = t.shape.bytes();

    for (const auto& g : plan.groups)
        plan.buffer_ids.push_back(g.output);

    return plan;
}

/* ── GraphCompiler::compile (backend-parameterised) ─────────────────────── */

std::unique_ptr<CompiledGraph> GraphCompiler::compile(const AneGraph& graph,
                                                       CompilerBackend& backend) {
    ExecutionPlan plan;
    try {
        plan = build_plan(graph);
    } catch (...) {
        return nullptr;
    }

    if (runtime::initialize() == runtime::AneState::Fallback)
        return nullptr;

    auto cg = std::make_unique<CompiledGraph>();
    cg->tensor_bytes_     = plan.tensor_bytes;
    cg->graph_input_ids_  = plan.graph_input_ids;
    cg->graph_output_ids_ = plan.graph_output_ids;
    for (const auto& t : graph.tensors())
        cg->tensor_shapes_[t.id] = t.shape;

    // Runtime enforces uniform IOSurface allocation size across multi-input
    // dispatches; choose one per-graph allocation target for all I/O tensors.
    for (const auto& kv : plan.tensor_bytes)
        cg->io_alloc_bytes_ = std::max(cg->io_alloc_bytes_, kv.second);

    auto round_up = [](size_t x, size_t m) {
        return ((x + m - 1) / m) * m;
    };

    // ANE raw channel stride constraints:
    // - Must be at least S*2 bytes for each tensor that shares this alloc size.
    // - Keep 64-byte aligned to match hardware vector length.
    if (cg->io_alloc_bytes_ > 0) {
        size_t target = std::max(cg->io_alloc_bytes_, static_cast<size_t>(49152));
        target = round_up(target, static_cast<size_t>(64));

        bool ok = false;
        for (int iter = 0; iter < 40960 && !ok; ++iter) {
            ok = true;
            for (TensorId tid : plan.graph_input_ids) {
                const auto& s = graph.tensor(tid).shape;
                size_t stride = target / static_cast<size_t>(s.channels);
                if (stride < static_cast<size_t>(s.seq) * 2 || (stride % 64) != 0) {
                    ok = false; break;
                }
            }
            if (ok) {
                for (TensorId tid : plan.graph_output_ids) {
                    const auto& s = graph.tensor(tid).shape;
                    size_t stride = target / static_cast<size_t>(s.channels);
                    if (stride < static_cast<size_t>(s.seq) * 2 || (stride % 64) != 0) {
                        ok = false; break;
                    }
                }
            }
            if (!ok) target += 64;
        }
        cg->io_alloc_bytes_ = target;
    }

    for (const auto& group : plan.groups) {
        std::string debug = "group_" + std::to_string(cg->groups_.size());

        runtime::AneProgram* program = backend.compile_group(graph, group, debug);
        if (!program) return nullptr;

        CompiledPlanGroup cpg;
        cpg.node_ids = group.node_ids;
        cpg.program  = program;
        cpg.inputs   = group.inputs;
        cpg.output   = group.output;
        cg->groups_.push_back(std::move(cpg));

        const auto& out_shape = graph.tensor(group.output).shape;
        cg->ane_bufs_[group.output] = global_buffer_pool().acquire_tensor_padded(
            out_shape.channels, out_shape.seq, cg->io_alloc_bytes_);
    }

    return cg;
}

/* ── RoutingBackend ──────────────────────────────────────────────────────── */

// Priority-ordered list of backends.  For each FusionGroup, the first backend
// whose owns() returns true receives the group.  Never exposed in the header —
// it is an implementation detail of the default compile() overload.
class RoutingBackend final : public CompilerBackend {
public:
    explicit RoutingBackend(std::vector<CompilerBackend*> backends)
        : backends_(std::move(backends)) {}

    bool owns(const AneGraph&, const FusionGroup&) const override {
        return true;  // top-level router owns everything
    }

    runtime::AneProgram* compile_group(const AneGraph&    graph,
                                       const FusionGroup& group,
                                       const std::string& debug_name) override {
        for (CompilerBackend* b : backends_)
            if (b->owns(graph, group))
                return b->compile_group(graph, group, debug_name);
        return nullptr;  // unreachable: MilBackend is always last and owns all
    }

private:
    std::vector<CompilerBackend*> backends_;
};

/* ── GraphCompiler::compile (default routed overload) ───────────────────── */

std::unique_ptr<CompiledGraph> GraphCompiler::compile(const AneGraph& graph) {
    thread_local HwxBackend      hwx;
    thread_local MilBackend      mil;
    thread_local RoutingBackend  router({&hwx, &mil});
    return compile(graph, router);
}

} // namespace graph
} // namespace libane
