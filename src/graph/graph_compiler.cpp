/**
 * GraphCompiler implementation.
 */
#include "graph_compiler.hpp"
#include "graph_validator.hpp"
#include "hwx_backend.hpp"
#include "espresso_backend.hpp"
#include "mil_backend.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstring>

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

        auto check_stride = [&](TensorId tid) -> bool {
            const auto& s = graph.tensor(tid).shape;
            size_t eff_seq = static_cast<size_t>(s.height) * s.seq;
            size_t stride  = target / static_cast<size_t>(s.channels);
            return stride >= eff_seq * 2 && (stride % 64) == 0;
        };

        bool ok = false;
        for (int iter = 0; iter < 40960 && !ok; ++iter) {
            ok = true;
            for (TensorId tid : plan.graph_input_ids) {
                if (!check_stride(tid)) { ok = false; break; }
            }
            if (ok) {
                for (TensorId tid : plan.graph_output_ids) {
                    if (!check_stride(tid)) { ok = false; break; }
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

        // If the group's first op is W8A8 and its primary input is a graph input,
        // record the activation quantization parameters so the executor can
        // dequantize int8 activations to fp16 before dispatch.
        if (!group.node_ids.empty()) {
            const GraphNode& first_node = graph.node(group.node_ids[0]);
            if (first_node.op == LIBANE_OP_MATMUL_W8A8 &&
                !first_node.weights.empty()) {
                // Blob: hdr(8) + int8[IC*OC] + pad + float32[OC] + float32 + int32
                const size_t kHdrBytes = 2 * sizeof(int32_t);
                const int32_t* hdr = reinterpret_cast<const int32_t*>(
                    first_node.weights.data());
                int OC = hdr[0];
                int IC = hdr[1];
                if (IC > 0 && OC > 0) {
                    size_t wbytes       = static_cast<size_t>(IC) * OC;
                    size_t scales_start = (kHdrBytes + wbytes + 3) & ~size_t(3);
                    size_t act_off = scales_start + static_cast<size_t>(OC) * sizeof(float);
                    if (first_node.weights.size() >= act_off + sizeof(float) + sizeof(int32_t)) {
                        QuantParams qp;
                        std::memcpy(&qp.scale,
                                    first_node.weights.data() + act_off, sizeof(float));
                        std::memcpy(&qp.zero_point,
                                    first_node.weights.data() + act_off + 4, sizeof(int32_t));
                        // Register quant params for the primary input tensor
                        TensorId primary_in = group.inputs[0];
                        // Only register if this is a graph input (not intermediate)
                        bool is_graph_input = false;
                        for (TensorId gid : plan.graph_input_ids)
                            if (gid == primary_in) { is_graph_input = true; break; }
                        if (is_graph_input)
                            cg->quant_params_[primary_in] = qp;
                    }
                }
            }
        }

        const auto& out_shape = graph.tensor(group.output).shape;
        int eff_seq = out_shape.height * out_shape.seq;
        cg->ane_bufs_[group.output] = global_buffer_pool().acquire_tensor_padded(
            out_shape.channels, eff_seq, cg->io_alloc_bytes_);
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

// File-scope thread_local so GraphCompiler::thread_mil_backend() can
// return a reference to the same instance the default-routed compile()
// overload uses.  Each thread has its own cache by design — the cross-
// thread aggregation problem is a later phase.
static thread_local MilBackend      g_thread_mil;
static thread_local HwxBackend      g_thread_hwx(&g_thread_mil);
static thread_local EspressoBackend g_thread_espresso;
static thread_local RoutingBackend  g_thread_router(
    {&g_thread_hwx, &g_thread_espresso, &g_thread_mil});

std::unique_ptr<CompiledGraph> GraphCompiler::compile(const AneGraph& graph) {
    // MilBackend owns the thread-local URL-reconnect cache; HwxBackend
    // borrows it so both paths populate and hit the same cache.
    return compile(graph, g_thread_router);
}

MilBackend& GraphCompiler::thread_mil_backend() {
    return g_thread_mil;
}

/* ── Binary serialization helpers ────────────────────────────────────────── */

static constexpr uint32_t kMagic   = 0x414E4547u;  // "ANEG"
static constexpr uint32_t kVersion = 1u;

static void w32(std::ostream& os, uint32_t v) {
    os.write(reinterpret_cast<const char*>(&v), 4);
}
static void w64(std::ostream& os, uint64_t v) {
    os.write(reinterpret_cast<const char*>(&v), 8);
}
static void wstr(std::ostream& os, const std::string& s) {
    w32(os, static_cast<uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}
static void wbytes(std::ostream& os, const void* data, size_t n) {
    w64(os, static_cast<uint64_t>(n));
    os.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

static uint32_t r32(std::istream& is) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 4);
    return v;
}
static uint64_t r64(std::istream& is) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 8);
    return v;
}
static std::string rstr(std::istream& is) {
    uint32_t len = r32(is);
    std::string s(len, '\0');
    if (len) is.read(&s[0], len);
    return s;
}
static std::vector<uint8_t> rbytes(std::istream& is) {
    uint64_t len = r64(is);
    std::vector<uint8_t> v(len);
    if (len) is.read(reinterpret_cast<char*>(v.data()),
                     static_cast<std::streamsize>(len));
    return v;
}

/* ── CompiledGraph::save ─────────────────────────────────────────────────── */

bool CompiledGraph::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;

    w32(f, kMagic);
    w32(f, kVersion);

    // Graph I/O IDs
    w32(f, static_cast<uint32_t>(graph_input_ids_.size()));
    for (TensorId id : graph_input_ids_)  w32(f, id);
    w32(f, static_cast<uint32_t>(graph_output_ids_.size()));
    for (TensorId id : graph_output_ids_) w32(f, id);
    w64(f, static_cast<uint64_t>(io_alloc_bytes_));

    // Tensor shapes
    w32(f, static_cast<uint32_t>(tensor_shapes_.size()));
    for (const auto& [tid, sh] : tensor_shapes_) {
        w32(f, tid);
        w32(f, static_cast<uint32_t>(sh.channels));
        w32(f, static_cast<uint32_t>(sh.height));
        w32(f, static_cast<uint32_t>(sh.seq));
    }

    // Quant params
    w32(f, static_cast<uint32_t>(quant_params_.size()));
    for (const auto& [tid, qp] : quant_params_) {
        w32(f, tid);
        uint32_t scale_bits;
        std::memcpy(&scale_bits, &qp.scale, 4);
        w32(f, scale_bits);
        w32(f, static_cast<uint32_t>(qp.zero_point));
    }

    // Groups
    w32(f, static_cast<uint32_t>(groups_.size()));
    for (const auto& grp : groups_) {
        w32(f, static_cast<uint32_t>(grp.node_ids.size()));
        for (uint32_t nid : grp.node_ids) w32(f, nid);

        w32(f, static_cast<uint32_t>(grp.inputs.size()));
        for (TensorId id : grp.inputs) w32(f, id);

        w32(f, grp.output);

        // Serialize program
        runtime::SerializedProgram sp;
        if (!grp.program || !runtime::ane_serialize_program(grp.program, sp))
            return false;

        wstr(f, sp.debug_name);
        wstr(f, sp.mil_text);

        w32(f, static_cast<uint32_t>(sp.weights.size()));
        for (const auto& we : sp.weights) {
            wstr(f, we.filename);
            wbytes(f, we.data.data(), we.data.size());
        }

        wbytes(f, sp.hwx_bytes.data(), sp.hwx_bytes.size());
        wstr(f, sp.hwx_rel_path);

        w32(f, static_cast<uint32_t>(sp.input_param_names.size()));
        for (const auto& s : sp.input_param_names) wstr(f, s);

        w32(f, static_cast<uint32_t>(sp.output_var_names.size()));
        for (const auto& s : sp.output_var_names) wstr(f, s);
    }

    return f.good();
}

/* ── GraphCompiler::load ─────────────────────────────────────────────────── */

std::unique_ptr<CompiledGraph> GraphCompiler::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;

    if (r32(f) != kMagic || r32(f) != kVersion) return nullptr;

    if (runtime::initialize() == runtime::AneState::Fallback)
        return nullptr;

    auto cg = std::make_unique<CompiledGraph>();

    // Graph I/O IDs
    uint32_t n_in = r32(f);
    cg->graph_input_ids_.resize(n_in);
    for (auto& id : cg->graph_input_ids_) id = r32(f);

    uint32_t n_out = r32(f);
    cg->graph_output_ids_.resize(n_out);
    for (auto& id : cg->graph_output_ids_) id = r32(f);

    cg->io_alloc_bytes_ = static_cast<size_t>(r64(f));

    // Tensor shapes
    uint32_t n_shapes = r32(f);
    for (uint32_t i = 0; i < n_shapes; ++i) {
        TensorId tid = r32(f);
        mil::TensorShape sh;
        sh.batch    = 1;
        sh.channels = static_cast<int>(r32(f));
        sh.height   = static_cast<int>(r32(f));
        sh.seq      = static_cast<int>(r32(f));
        cg->tensor_shapes_[tid] = sh;
        cg->tensor_bytes_[tid]  = sh.bytes();
    }

    // Quant params
    uint32_t n_qp = r32(f);
    for (uint32_t i = 0; i < n_qp; ++i) {
        TensorId tid = r32(f);
        uint32_t scale_bits = r32(f);
        QuantParams qp;
        std::memcpy(&qp.scale, &scale_bits, 4);
        qp.zero_point = static_cast<int32_t>(r32(f));
        cg->quant_params_[tid] = qp;
    }

    // Groups
    uint32_t n_groups = r32(f);
    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        CompiledPlanGroup grp;

        uint32_t n_nodes = r32(f);
        for (uint32_t i = 0; i < n_nodes; ++i) grp.node_ids.push_back(r32(f));

        uint32_t n_inputs = r32(f);
        for (uint32_t i = 0; i < n_inputs; ++i) grp.inputs.push_back(r32(f));

        grp.output = r32(f);

        runtime::SerializedProgram sp;
        sp.debug_name = rstr(f);
        sp.mil_text   = rstr(f);

        uint32_t n_weights = r32(f);
        sp.weights.resize(n_weights);
        for (auto& we : sp.weights) {
            we.filename = rstr(f);
            we.data     = rbytes(f);
        }

        sp.hwx_bytes    = rbytes(f);
        sp.hwx_rel_path = rstr(f);

        uint32_t n_ipn = r32(f);
        for (uint32_t i = 0; i < n_ipn; ++i)
            sp.input_param_names.push_back(rstr(f));

        uint32_t n_ovn = r32(f);
        for (uint32_t i = 0; i < n_ovn; ++i)
            sp.output_var_names.push_back(rstr(f));

        if (!f.good()) return nullptr;

        grp.program = runtime::ane_restore_program(sp);
        if (!grp.program) return nullptr;

        // Pre-allocate the output ANE buffer
        const auto* out_shape = &cg->tensor_shapes_.at(grp.output);
        int eff_seq = out_shape->height * out_shape->seq;
        cg->ane_bufs_[grp.output] = global_buffer_pool().acquire_tensor_padded(
            out_shape->channels, eff_seq, cg->io_alloc_bytes_);

        cg->groups_.push_back(std::move(grp));
    }

    return f.good() ? std::move(cg) : nullptr;
}

} // namespace graph
} // namespace libane
