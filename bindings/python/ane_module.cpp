/**
 * Python bindings for libane v0.7.0.
 *
 * PyPI package: ane · Install: pip install ane
 * Requires: pybind11, numpy
 *
 * Single-op API:
 *   ane.available()               → bool
 *   ane.version()                 → str
 *   ane.matmul(A, B)              → np.ndarray (fp16)
 *   ane.matmul_f32(A, B)          → np.ndarray (fp32)
 *   ane.softmax(x)                → np.ndarray (fp16)
 *   ane.gelu(x)                   → np.ndarray (fp16)
 *   ane.set_backend(name)         → None
 *   ane.set_log_level(level)      → None
 *   ane.cache_flush()             → None
 *   ane.cache_size_bytes()        → int
 *
 * Graph API:
 *   g = ane.Graph()
 *   x = g.add_input("x", [1, 512, 1, 128])
 *   t = g.add_op(ane.MATMUL, [x], [1, 256, 1, 128], weights=W_np)
 *   g.mark_output(t)
 *   cg = g.compile()             → ane.CompiledGraph
 *   out = cg(x_np)               → np.ndarray (fp16)  [single-input]
 *   out = cg([a_np, b_np])       → list[np.ndarray]   [multi-input/output]
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <string>
#include "../../include/libane.h"

namespace py = pybind11;
using namespace pybind11::literals;

/* ── numpy dtype helpers ─────────────────────────────────────────────────── */

static void require_2d(const py::array& a, const char* name) {
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(name) +
            ": expected 2-D array, got ndim=" + std::to_string(a.ndim()));
}

/** Convert a Python list/tuple of 4 ints to libane_shape_t. */
static libane_shape_t to_shape(const std::vector<int>& dims) {
    if (dims.size() != 4)
        throw std::invalid_argument(
            "shape must be a list of 4 ints [batch, channels, height, seq]");
    libane_shape_t s;
    s.dims[0] = dims[0]; s.dims[1] = dims[1];
    s.dims[2] = dims[2]; s.dims[3] = dims[3];
    s.ndim = 4;
    return s;
}

/** Return a C-contiguous float16 copy of arr. */
static py::array ensure_f16(py::array arr) {
    py::module_ np = py::module_::import("numpy");
    return np.attr("ascontiguousarray")(
        np.attr("asarray")(arr, py::arg("dtype") = "float16"));
}

/* ── matmul (fp16) ───────────────────────────────────────────────────────── */

static py::array py_matmul(py::array A_in, py::array B_in) {
    require_2d(A_in, "A");
    require_2d(B_in, "B");

    py::module_ np = py::module_::import("numpy");
    py::array A = ensure_f16(A_in);
    py::array B = ensure_f16(B_in);
    auto abuf = A.request();
    auto bbuf = B.request();

    int M = static_cast<int>(abuf.shape[0]);
    int K = static_cast<int>(abuf.shape[1]);
    int Kb= static_cast<int>(bbuf.shape[0]);
    int N = static_cast<int>(bbuf.shape[1]);
    if (K != Kb) throw std::invalid_argument("A.shape[1] must equal B.shape[0]");

    // Core C API accepts/returns standard row-major matrices.
    // Layout bridging to ANE [1,C,1,S] is handled inside libane_matmul_f16().
    py::array C = np.attr("empty")(py::make_tuple(M, N), "dtype"_a="float16");
    auto cbuf = C.request();

    libane_status_t st = libane_matmul_f16(
        static_cast<const libane_f16_t*>(abuf.ptr),
        static_cast<const libane_f16_t*>(bbuf.ptr),
        static_cast<libane_f16_t*>(cbuf.ptr),
        M, K, N);
    if (st != LIBANE_OK)
        throw std::runtime_error(std::string("matmul failed: ") + libane_last_error());

    return C;
}

/* ── matmul (fp32) ───────────────────────────────────────────────────────── */

static py::array_t<float> py_matmul_f32(py::array_t<float> A, py::array_t<float> B) {
    require_2d(A, "A");
    require_2d(B, "B");
    auto abuf = A.request();
    auto bbuf = B.request();
    int M = static_cast<int>(abuf.shape[0]);
    int K = static_cast<int>(abuf.shape[1]);
    int Kb= static_cast<int>(bbuf.shape[0]);
    int N = static_cast<int>(bbuf.shape[1]);
    if (K != Kb) throw std::invalid_argument("A.shape[1] must equal B.shape[0]");

    auto C = py::array_t<float>({M, N});
    auto cbuf = C.request();
    libane_status_t st = libane_matmul_f32(
        static_cast<const float*>(abuf.ptr),
        static_cast<const float*>(bbuf.ptr),
        static_cast<float*>(cbuf.ptr),
        M, K, N);
    if (st != LIBANE_OK)
        throw std::runtime_error(std::string("matmul_f32 failed: ") + libane_last_error());
    return C;
}

/* ── softmax ─────────────────────────────────────────────────────────────── */

static py::array py_softmax(py::array x) {
    auto xbuf = x.request();
    int S    = static_cast<int>(xbuf.shape[xbuf.ndim - 1]);
    int rows = 1;
    for (int d = 0; d < xbuf.ndim - 1; ++d) rows *= static_cast<int>(xbuf.shape[d]);

    py::module_ np = py::module_::import("numpy");
    if (S % 8 != 0 || S > 65536 || rows > 16384) {
        py::object mx = np.attr("max")(x, "axis"_a=-1, "keepdims"_a=true);
        py::object e  = np.attr("exp")(np.attr("subtract")(x, mx));
        py::object s  = np.attr("sum")(e, "axis"_a=-1, "keepdims"_a=true);
        return py::array(np.attr("divide")(e, s));
    }
    libane_shape_t shape; shape.dims[0]=1; shape.dims[1]=rows;
                          shape.dims[2]=1; shape.dims[3]=S; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    if (!h) throw std::runtime_error(std::string("softmax compile: ") + libane_last_error());

    py::array x16 = np.attr("ascontiguousarray")(np.attr("asarray")(x, "dtype"_a="float16"));
    auto x16b = x16.request();
    py::array out = np.attr("empty")(py::make_tuple(rows, S), "dtype"_a="float16");
    auto outb = out.request();
    auto st = libane_execute(h, x16b.ptr, outb.ptr, shape);
    libane_release(h);
    if (st != LIBANE_OK)
        throw std::runtime_error(std::string("softmax execute: ") + libane_last_error());
    return py::array(out.attr("reshape")(x.attr("shape")));
}

/* ── gelu ────────────────────────────────────────────────────────────────── */

static py::array py_gelu(py::array x) {
    auto xbuf = x.request();
    int numel = 1;
    for (int d = 0; d < xbuf.ndim; ++d) numel *= static_cast<int>(xbuf.shape[d]);

    py::module_ np = py::module_::import("numpy");
    if (numel % 8 != 0 || numel > 65536) {
        py::object xf  = np.attr("asarray")(x, "dtype"_a="float32");
        py::object xf3 = np.attr("multiply")(xf, np.attr("multiply")(xf, xf));
        py::object inn = np.attr("add")(xf, np.attr("multiply")(py::float_(0.044715), xf3));
        py::object t   = np.attr("tanh")(np.attr("multiply")(py::float_(0.7978845608), inn));
        py::object res = np.attr("multiply")(py::float_(0.5),
                             np.attr("multiply")(xf, np.attr("add")(py::float_(1.0), t)));
        return py::array(np.attr("asarray")(res, "dtype"_a=x.dtype())
                             .attr("reshape")(x.attr("shape")));
    }
    libane_shape_t shape; shape.dims[0]=1; shape.dims[1]=1;
                          shape.dims[2]=1; shape.dims[3]=numel; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    if (!h) throw std::runtime_error(std::string("gelu compile: ") + libane_last_error());

    py::array x16 = np.attr("ascontiguousarray")(
        np.attr("asarray")(x, "dtype"_a="float16").attr("reshape")(py::make_tuple(numel)));
    auto x16b = x16.request();
    py::array out = np.attr("empty")(py::make_tuple(numel), "dtype"_a="float16");
    auto outb = out.request();
    auto st = libane_execute(h, x16b.ptr, outb.ptr, shape);
    libane_release(h);
    if (st != LIBANE_OK)
        throw std::runtime_error(std::string("gelu execute: ") + libane_last_error());
    return py::array(out.attr("reshape")(x.attr("shape")));
}

/* ── CompiledMil class ───────────────────────────────────────────────────── */

class PyMilProgram {
public:
    explicit PyMilProgram(libane_mil_handle_t h) : h_(h) {}
    ~PyMilProgram() { libane_mil_release(h_); }

    PyMilProgram(const PyMilProgram&)            = delete;
    PyMilProgram& operator=(const PyMilProgram&) = delete;

    /**
     * Execute the compiled MIL program.
     *
     * @param inputs_raw   List of np.ndarray inputs (converted to fp16 internally).
     *                     Must match the MIL function's parameter count and be in
     *                     alphabetical order of MIL parameter names (Orion constraint #13).
     * @param out_numel    Number of fp16 elements expected for each output.
     * @return             List of np.float16 arrays, one per output.
     */
    py::list run(const std::vector<py::array>& inputs_raw,
                 const std::vector<size_t>&    out_numel) {
        py::module_ np = py::module_::import("numpy");

        // Convert inputs to contiguous fp16
        std::vector<py::array>       in_f16;
        std::vector<py::buffer_info> in_bufs;
        std::vector<const void*>     in_ptrs;
        std::vector<size_t>          in_sizes;
        in_f16.reserve(inputs_raw.size());
        for (auto& a : inputs_raw) {
            in_f16.push_back(np.attr("ascontiguousarray")(
                np.attr("asarray")(a, "dtype"_a="float16")));
            in_bufs.push_back(in_f16.back().request());
            in_ptrs.push_back(in_bufs.back().ptr);
            in_sizes.push_back(
                static_cast<size_t>(in_bufs.back().size) * in_bufs.back().itemsize);
        }

        // Allocate outputs
        std::vector<py::array> out_arrays;
        std::vector<void*>     out_ptrs;
        std::vector<size_t>    out_sizes;
        out_arrays.reserve(out_numel.size());
        for (size_t n : out_numel) {
            out_arrays.push_back(
                np.attr("empty")(py::make_tuple(n), "dtype"_a="float16"));
            auto ob = out_arrays.back().request();
            out_ptrs.push_back(ob.ptr);
            out_sizes.push_back(n * 2);  // fp16: 2 bytes per element
        }

        libane_status_t st = libane_mil_execute(
            h_,
            in_ptrs.empty()  ? nullptr : in_ptrs.data(),
            in_sizes.empty() ? nullptr : in_sizes.data(),
            in_ptrs.size(),
            out_ptrs.empty()  ? nullptr : out_ptrs.data(),
            out_sizes.empty() ? nullptr : out_sizes.data(),
            out_ptrs.size());

        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("mil execute failed: ") + libane_last_error());

        py::list result;
        for (auto& a : out_arrays)
            result.append(a);
        return result;
    }

private:
    libane_mil_handle_t h_;
};

static PyMilProgram* py_compile_mil(const std::string& mil_text) {
    auto h = libane_mil_compile(mil_text.c_str(), nullptr, nullptr, nullptr, 0);
    if (!h)
        throw std::runtime_error(
            std::string("mil compile failed: ") + libane_last_error());
    return new PyMilProgram(h);
}

static PyMilProgram* py_compile_mil_with_weights(const std::string& mil_text,
                                                   py::dict           weights_dict) {
    py::module_ np = py::module_::import("numpy");

    std::vector<std::string> names;
    std::vector<py::array>   arrays;
    std::vector<py::buffer_info> bufs;

    for (auto item : weights_dict) {
        names.push_back(item.first.cast<std::string>());
        arrays.push_back(np.attr("ascontiguousarray")(
            np.attr("asarray")(item.second, "dtype"_a="float16")));
        bufs.push_back(arrays.back().request());
    }

    std::vector<const char*> name_ptrs;
    std::vector<const void*> data_ptrs;
    std::vector<size_t>      data_sizes;
    name_ptrs.reserve(names.size());
    data_ptrs.reserve(bufs.size());
    data_sizes.reserve(bufs.size());
    for (size_t i = 0; i < names.size(); ++i) {
        name_ptrs.push_back(names[i].c_str());
        data_ptrs.push_back(bufs[i].ptr);
        data_sizes.push_back(static_cast<size_t>(bufs[i].size) * bufs[i].itemsize);
    }

    auto h = libane_mil_compile(
        mil_text.c_str(),
        name_ptrs.empty() ? nullptr : name_ptrs.data(),
        data_ptrs.empty() ? nullptr : data_ptrs.data(),
        data_sizes.empty() ? nullptr : data_sizes.data(),
        names.size());

    if (!h)
        throw std::runtime_error(
            std::string("mil compile failed: ") + libane_last_error());
    return new PyMilProgram(h);
}

/* ── Graph class ─────────────────────────────────────────────────────────── */

class PyGraph {
public:
    PyGraph() : g_(libane_graph_create()) {
        if (!g_) throw std::runtime_error("libane_graph_create failed (OOM)");
    }
    ~PyGraph() { libane_graph_release(g_); }

    PyGraph(const PyGraph&)            = delete;
    PyGraph& operator=(const PyGraph&) = delete;

    uint32_t add_input(const std::string& name, const std::vector<int>& shape) {
        libane_shape_t s = to_shape(shape);
        uint32_t id = libane_graph_add_input(g_, name.c_str(), s);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_input failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_op(int op_int,
                    const std::vector<uint32_t>& inputs,
                    const std::vector<int>& output_shape,
                    py::object weights_obj) {
        libane_op_t op = static_cast<libane_op_t>(op_int);
        libane_shape_t out_s = to_shape(output_shape);
        const void* wptr = nullptr;
        size_t wlen = 0;
        py::array w_arr;

        if (!weights_obj.is_none()) {
            py::module_ np = py::module_::import("numpy");
            w_arr = np.attr("ascontiguousarray")(
                np.attr("asarray")(weights_obj, "dtype"_a="float16"));
            auto wb = w_arr.request();
            wptr = wb.ptr;
            wlen = static_cast<size_t>(wb.size) * wb.itemsize;
        }

        uint32_t id = libane_graph_add_op(g_, op,
                                           inputs.empty() ? nullptr : inputs.data(),
                                           inputs.size(),
                                           out_s, wptr, wlen);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_op failed: ") + libane_last_error());
        return id;
    }

    void mark_output(uint32_t tensor_id, const std::string& name = "") {
        libane_status_t st = libane_graph_mark_output(
            g_, tensor_id, name.empty() ? nullptr : name.c_str());
        if (st != LIBANE_OK)
            throw std::invalid_argument(
                std::string("mark_output failed: ") + libane_last_error());
    }

    libane_graph_t handle() const { return g_; }

private:
    libane_graph_t g_;
};

/* ── CompiledGraph class ─────────────────────────────────────────────────── */

class PyCompiledGraph {
public:
    explicit PyCompiledGraph(libane_compiled_graph_t cg) : cg_(cg) {}
    ~PyCompiledGraph() { libane_compiled_graph_release(cg_); }

    PyCompiledGraph(const PyCompiledGraph&)            = delete;
    PyCompiledGraph& operator=(const PyCompiledGraph&) = delete;

    /**
     * Execute the compiled graph.
     *
     * inputs: a single np.ndarray (for single-input graphs) or a list of
     *         np.ndarray (one per graph input, in add_input order).
     * Returns: a single np.float16 array for single-output graphs, or a list
     *          of np.float16 arrays for multi-output graphs.
     */
    py::object call(py::object inputs_obj) {
        py::module_ np = py::module_::import("numpy");

        // Normalise inputs to a list of contiguous f16 arrays
        std::vector<py::array> in_arrays;
        if (py::isinstance<py::list>(inputs_obj) ||
            py::isinstance<py::tuple>(inputs_obj)) {
            for (auto item : inputs_obj)
                in_arrays.push_back(np.attr("ascontiguousarray")(
                    np.attr("asarray")(item, "dtype"_a="float16")));
        } else {
            in_arrays.push_back(np.attr("ascontiguousarray")(
                np.attr("asarray")(inputs_obj, "dtype"_a="float16")));
        }

        std::vector<const void*> in_ptrs;
        std::vector<size_t>      in_bytes;
        std::vector<py::buffer_info> in_bufs;
        for (auto& a : in_arrays) {
            in_bufs.push_back(a.request());
            in_ptrs.push_back(in_bufs.back().ptr);
            in_bytes.push_back(static_cast<size_t>(in_bufs.back().size) *
                                in_bufs.back().itemsize);
        }

        // Get output count from the compiled graph via a dry-run query.
        // We use libane_graph_execute which reports the expected output count
        // via LIBANE_ERR_INVALID_ARG on mismatch, so first probe with 0 outputs
        // to extract the graph's output count.
        //
        // Simpler: just try calling with increasing output counts until success,
        // but that's silly. Instead we allocate outputs guided by the input
        // count (users of multi-output graphs should pass a list).
        //
        // For the common single-output case assume 1 output; for multi-output
        // the user must call execute() with explicit out_shapes.
        size_t n_out = out_shapes_.empty() ? 1 : out_shapes_.size();

        std::vector<py::array> out_arrays;
        std::vector<void*>   out_ptrs;
        std::vector<size_t>  out_bytes_vec;

        if (!out_shapes_.empty()) {
            for (auto& shape : out_shapes_) {
                size_t numel = 1;
                for (auto d : shape) numel *= d;
                py::list dims;
                for (auto d : shape) dims.append(d);
                out_arrays.emplace_back(np.attr("empty")(dims, "dtype"_a="float16"));
                auto ob = out_arrays.back().request();
                out_ptrs.push_back(ob.ptr);
                out_bytes_vec.push_back(numel * 2);
            }
        } else {
            // Single output, size derived from input (same shape assumed)
            size_t numel = in_bytes.empty() ? 0 : in_bytes[0] / 2;
            out_arrays.emplace_back(
                np.attr("empty")(py::make_tuple(numel), "dtype"_a="float16"));
            auto ob = out_arrays.back().request();
            out_ptrs.push_back(ob.ptr);
            out_bytes_vec.push_back(numel * 2);
        }

        libane_status_t st = libane_graph_execute(
            cg_,
            in_ptrs.data(), in_bytes.data(), in_ptrs.size(),
            out_ptrs.data(), out_bytes_vec.data(), out_ptrs.size());

        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("graph execute failed: ") + libane_last_error());

        if (n_out == 1) {
            return out_arrays[0];
        }
        py::list result;
        for (auto& a : out_arrays)
            result.append(a);
        return result;
    }

    /**
     * Set expected output shapes for multi-output graphs or when the output
     * shape differs from the input shape.
     * shapes: list of shape lists, e.g. [[1,256,1,128], [1,128,1,128]]
     */
    void set_output_shapes(const std::vector<std::vector<int>>& shapes) {
        out_shapes_ = shapes;
    }

private:
    libane_compiled_graph_t           cg_;
    std::vector<std::vector<int>>     out_shapes_;
};

/* ── Graph.compile() ─────────────────────────────────────────────────────── */

static PyCompiledGraph* py_compile(PyGraph& g) {
    libane_compiled_graph_t cg = libane_graph_compile(g.handle());
    if (!cg)
        throw std::runtime_error(
            std::string("graph compile failed: ") + libane_last_error());
    return new PyCompiledGraph(cg);
}

/* ── Module ──────────────────────────────────────────────────────────────── */

PYBIND11_MODULE(ane, m) {
    m.doc() = R"(
ane — Apple Neural Engine Python bindings (libane v0.7.0)
Amirani Labs

ANE-accelerated ML operations with automatic CPU fallback.
Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
Not for App Store submission.
)";

    libane_set_log_level(LIBANE_LOG_ERROR);

    /* ── Utility ──────────────────────────────────────────────────────── */
    m.def("available", []() { return libane_available() != 0; },
          "True if the Apple Neural Engine is accessible.");
    m.def("version",   []() { return std::string(libane_version()); },
          "libane version string.");
    m.def("last_error",[]() { return std::string(libane_last_error()); },
          "Last error message.");
    m.def("set_backend", [](py::object b) {
        if (b.is_none()) libane_set_backend(nullptr);
        else libane_set_backend(py::str(b).cast<std::string>().c_str());
    }, py::arg("backend"),
    "Force backend: 'ane', 'cpu', or None for auto-detect.");
    m.def("set_log_level", [](int l) {
        libane_set_log_level(static_cast<libane_log_level_t>(l));
    }, py::arg("level"), "0=SILENT 1=ERROR 2=WARN 3=INFO 4=DEBUG.");
    m.def("cache_flush",      &libane_cache_flush);
    m.def("cache_size_bytes", &libane_cache_size_bytes);

    /* ── Single-op ────────────────────────────────────────────────────── */
    m.def("matmul",    &py_matmul,    py::arg("A"), py::arg("B"),
          "ANE fp16 matmul: C = A @ B.  Falls back to BLAS.");
    m.def("matmul_f32",&py_matmul_f32,py::arg("A"), py::arg("B"),
          "ANE fp32 matmul (fp32→fp16→fp32 internally).");
    m.def("softmax",   &py_softmax,   py::arg("x"),
          "ANE softmax over last dimension. Falls back to numpy.");
    m.def("gelu",      &py_gelu,      py::arg("x"),
          "ANE GELU (tanh approximation). Falls back to numpy.");

    /* ── Raw MIL probe API ───────────────────────────────────────────── */
    py::class_<PyMilProgram>(m, "CompiledMil", R"(
Compiled raw MIL program.  Returned by compile_mil() and compile_mil_with_weights().

Use the ane.probe module for a higher-level interface.
)")
        .def("run", &PyMilProgram::run,
             py::arg("inputs"), py::arg("output_sizes"),
             "Execute the compiled MIL program.\n\n"
             "Args:\n"
             "    inputs: list of np.ndarray (converted to fp16 internally);\n"
             "            must be in alphabetical order of MIL parameter names.\n"
             "    output_sizes: list of int — number of fp16 elements per output.\n\n"
             "Returns:\n"
             "    list of np.float16 arrays, one per output.");

    m.def("compile_mil", &py_compile_mil,
          py::arg("mil_text"),
          py::return_value_policy::take_ownership,
          "Compile a raw MIL program (no external weights).\n\n"
          "Args:\n"
          "    mil_text: UTF-8 MIL source including buildInfo header.\n\n"
          "Returns:\n"
          "    CompiledMil ready for .run().\n\n"
          "Raises:\n"
          "    RuntimeError if ANE is unavailable or compilation fails.");

    m.def("compile_mil_with_weights", &py_compile_mil_with_weights,
          py::arg("mil_text"), py::arg("weights"),
          py::return_value_policy::take_ownership,
          "Compile a raw MIL program with external weight files.\n\n"
          "Args:\n"
          "    mil_text: UTF-8 MIL source.\n"
          "    weights:  dict mapping filename -> fp16 np.ndarray.\n"
          "              Filenames must match file() references in the MIL text.\n\n"
          "Returns:\n"
          "    CompiledMil ready for .run().\n\n"
          "Raises:\n"
          "    RuntimeError if ANE is unavailable or compilation fails.");

    /* ── Graph API ────────────────────────────────────────────────────── */
    py::class_<PyGraph>(m, "Graph", R"(
Mutable graph builder.

Example::

    g = ane.Graph()
    x = g.add_input("x", [1, 512, 1, 128])
    t = g.add_op(ane.MATMUL, [x], [1, 256, 1, 128], weights=W)
    t = g.add_op(ane.GELU,   [t], [1, 256, 1, 128])
    g.mark_output(t)
    cg = g.compile()
)")
        .def(py::init<>())
        .def("add_input", &PyGraph::add_input,
             py::arg("name"), py::arg("shape"),
             "Declare a graph input. shape = [1, C, 1, S]. Returns tensor ID.")
        .def("add_op", &PyGraph::add_op,
             py::arg("op"), py::arg("inputs"), py::arg("output_shape"),
             py::arg("weights") = py::none(),
             "Add an operation. inputs = list of tensor IDs. Returns output tensor ID.")
        .def("mark_output", &PyGraph::mark_output,
             py::arg("tensor_id"), py::arg("name") = "",
             "Mark tensor_id as a graph output.")
        .def("compile", &py_compile,
             py::return_value_policy::take_ownership,
             "Compile the graph. Returns CompiledGraph. Raises on failure.");

    py::class_<PyCompiledGraph>(m, "CompiledGraph", R"(
Compiled ANE graph. Call with numpy arrays to run inference.

Single-input / single-output::

    out = cg(x)           # x: np.float16, shape matches graph input

Multi-input::

    out = cg([a, b])      # list of np.float16 arrays

For multi-output graphs, call set_output_shapes() first::

    cg.set_output_shapes([[1, 256, 1, 128], [1, 128, 1, 128]])
    a, b = cg(x)
)")
        .def("__call__", &PyCompiledGraph::call, py::arg("inputs"),
             "Run a forward pass.")
        .def("set_output_shapes", &PyCompiledGraph::set_output_shapes,
             py::arg("shapes"),
             "Set expected output shapes (list of shape lists) for multi-output graphs.");

    /* ── Op constants ─────────────────────────────────────────────────── */
    m.attr("MATMUL")    = static_cast<int>(LIBANE_OP_MATMUL);
    m.attr("LAYER_NORM")= static_cast<int>(LIBANE_OP_LAYER_NORM);
    m.attr("LAYERNORM") = static_cast<int>(LIBANE_OP_LAYERNORM);
    m.attr("GELU")      = static_cast<int>(LIBANE_OP_GELU);
    m.attr("SOFTMAX")   = static_cast<int>(LIBANE_OP_SOFTMAX);
    m.attr("ADD")       = static_cast<int>(LIBANE_OP_ADD);
    m.attr("MUL")       = static_cast<int>(LIBANE_OP_MUL);
    m.attr("TRANSPOSE") = static_cast<int>(LIBANE_OP_TRANSPOSE);
    m.attr("SILU")      = static_cast<int>(LIBANE_OP_SILU);
    m.attr("RMSNORM")   = static_cast<int>(LIBANE_OP_RMSNORM);
    m.attr("AVG_POOL")  = static_cast<int>(LIBANE_OP_AVG_POOL);
    m.attr("MAX_POOL")  = static_cast<int>(LIBANE_OP_MAX_POOL);
    m.attr("LOGICAL_AND") = static_cast<int>(LIBANE_OP_LOGICAL_AND);
    m.attr("LOGICAL_OR") = static_cast<int>(LIBANE_OP_LOGICAL_OR);
    m.attr("LOGICAL_XOR") = static_cast<int>(LIBANE_OP_LOGICAL_XOR);
    m.attr("REDUCE_PROD") = static_cast<int>(LIBANE_OP_REDUCE_PROD);

    /* ── Log level constants ──────────────────────────────────────────── */
    m.attr("LOG_SILENT") = static_cast<int>(LIBANE_LOG_SILENT);
    m.attr("LOG_ERROR")  = static_cast<int>(LIBANE_LOG_ERROR);
    m.attr("LOG_WARN")   = static_cast<int>(LIBANE_LOG_WARN);
    m.attr("LOG_INFO")   = static_cast<int>(LIBANE_LOG_INFO);
    m.attr("LOG_DEBUG")  = static_cast<int>(LIBANE_LOG_DEBUG);

    m.attr("__version__") = LIBANE_VERSION;
}
