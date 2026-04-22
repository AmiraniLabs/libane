/**
 * Python bindings for libane v0.9.0.
 *
 * PyPI package: ane · Install: pip install ane
 * Requires: pybind11, numpy
 *
 * Utility:
 *   ane.available()               → bool
 *   ane.version()                 → str
 *   ane.device_info()             → dict (architecture, core_count, num_anes, available)
 *   ane.shape_limits()            → dict (max_seq, max_channels, seq_alignment)
 *   ane.set_backend(name)         → None
 *   ane.set_log_level(level)      → None
 *   ane.cache_flush()             → None
 *   ane.cache_size_bytes()        → int
 *
 * Single-op convenience:
 *   ane.matmul(A, B)              → np.ndarray (fp16)
 *   ane.matmul_f32(A, B)         → np.ndarray (fp32)
 *   ane.softmax(x)               → np.ndarray (fp16)
 *   ane.gelu(x)                  → np.ndarray (fp16)
 *
 * Compiled single-op handle (CompiledOp):
 *   h = ane.compile(op, shape, weights=None)  → CompiledOp
 *   h.execute(x)                              → np.ndarray (fp16)
 *   h.execute2(x0, x1)                        → np.ndarray (fp16)
 *   h.delta_reload()                           → None
 *   ane.compile_batch([(op, shape, weights), ...]) → list[CompiledOp | None]
 *
 * Graph API:
 *   g = ane.Graph()
 *   x = g.add_input("x", [1, 512, 1, 128])
 *   t = g.add_op(ane.MATMUL, [x], [1, 256, 1, 128], weights=W_np)
 *   t = g.add_pwl_activation(x, [1,C,1,S], x_min, x_max, samples_np)
 *   g.mark_output(t)
 *   cg = g.compile()             → ane.CompiledGraph
 *   out = cg(x_np)               → np.ndarray (fp16)  [single-input]
 *   out = cg([a_np, b_np])       → list[np.ndarray]   [multi-input/output]
 *
 * MIL API:
 *   prog = ane.compile_mil(mil_text)                    → CompiledMil
 *   prog = ane.compile_mil_with_weights(mil, weights)   → CompiledMil
 *   prog.run(inputs, output_sizes)                      → list[np.ndarray]
 *   prog.sram_spill                                     → bool
 *   prog.run_stats(inputs, output_sizes)                → (list[np.ndarray], dict)
 *
 * Tensor utilities (pure numpy, no ANE execution):
 *   ane.chunk(x, n, shape)           → list[np.ndarray]  split C-axis into n equal parts
 *   ane.split(x, sizes, shape)       → list[np.ndarray]  split C-axis by sizes
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

static py::array_t<float> py_matmul_f32(py::array A_in, py::array B_in) {
    require_2d(A_in, "A");
    require_2d(B_in, "B");

    py::module_ np = py::module_::import("numpy");
    py::array A = np.attr("ascontiguousarray")(
        np.attr("asarray")(A_in, "dtype"_a="float32"));
    py::array B = np.attr("ascontiguousarray")(
        np.attr("asarray")(B_in, "dtype"_a="float32"));

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

/* ── device_info / shape_limits ─────────────────────────────────────────── */

static py::dict py_device_info() {
    libane_device_info_t d{};
    libane_device_info(&d);
    py::dict out;
    out["architecture"] = std::string(d.architecture);
    out["core_count"]   = d.core_count;
    out["num_anes"]     = d.num_anes;
    out["available"]    = d.available != 0;
    return out;
}

static py::dict py_shape_limits() {
    libane_shape_limits_t lim = libane_get_shape_limits();
    py::dict out;
    out["max_seq"]       = lim.max_seq;
    out["max_channels"]  = lim.max_channels;
    out["seq_alignment"] = lim.seq_alignment;
    return out;
}

/* ── CompiledOp class (single-op handle) ────────────────────────────────── */

class PyCompiledOp {
public:
    explicit PyCompiledOp(libane_handle_t h) : h_(h) {}
    ~PyCompiledOp() { if (h_) libane_release(h_); }

    PyCompiledOp(const PyCompiledOp&)            = delete;
    PyCompiledOp& operator=(const PyCompiledOp&) = delete;

    py::array execute(py::array x_in, const std::vector<int>& shape) {
        py::module_ np = py::module_::import("numpy");
        py::array x = np.attr("ascontiguousarray")(
            np.attr("asarray")(x_in, "dtype"_a="float16"));
        auto xb = x.request();
        libane_shape_t s = to_shape(shape);
        size_t numel = (size_t)s.dims[0]*s.dims[1]*s.dims[2]*s.dims[3];
        py::array out = np.attr("empty")(py::make_tuple(numel), "dtype"_a="float16");
        auto ob = out.request();
        libane_status_t st = libane_execute(h_, xb.ptr, ob.ptr, s);
        if (st != LIBANE_OK)
            throw std::runtime_error(std::string("execute failed: ") + libane_last_error());
        return out;
    }

    py::array execute2(py::array x0_in, py::array x1_in, const std::vector<int>& shape) {
        py::module_ np = py::module_::import("numpy");
        py::array x0 = np.attr("ascontiguousarray")(
            np.attr("asarray")(x0_in, "dtype"_a="float16"));
        py::array x1 = np.attr("ascontiguousarray")(
            np.attr("asarray")(x1_in, "dtype"_a="float16"));
        auto x0b = x0.request();
        auto x1b = x1.request();
        libane_shape_t s = to_shape(shape);
        size_t numel = (size_t)s.dims[0]*s.dims[1]*s.dims[2]*s.dims[3];
        py::array out = np.attr("empty")(py::make_tuple(numel), "dtype"_a="float16");
        auto ob = out.request();
        libane_status_t st = libane_execute2(h_, x0b.ptr, x1b.ptr, ob.ptr, s);
        if (st != LIBANE_OK)
            throw std::runtime_error(std::string("execute2 failed: ") + libane_last_error());
        return out;
    }

    void delta_reload() {
        libane_status_t st = libane_delta_reload(h_);
        if (st != LIBANE_OK)
            throw std::runtime_error(std::string("delta_reload failed: ") + libane_last_error());
    }

private:
    libane_handle_t h_;
};

static PyCompiledOp* py_compile_op(int op_int,
                                    const std::vector<int>& shape,
                                    py::object weights_obj) {
    py::module_ np = py::module_::import("numpy");
    libane_shape_t s = to_shape(shape);
    const void* wptr = nullptr;
    size_t wlen = 0;
    py::array w_arr;
    if (!weights_obj.is_none()) {
        w_arr = np.attr("ascontiguousarray")(
            np.attr("asarray")(weights_obj, "dtype"_a="float16"));
        auto wb = w_arr.request();
        wptr = wb.ptr;
        wlen = static_cast<size_t>(wb.size) * wb.itemsize;
    }
    libane_handle_t h = libane_compile(static_cast<libane_op_t>(op_int), s, wptr, wlen);
    if (!h)
        throw std::runtime_error(std::string("compile failed: ") + libane_last_error());
    return new PyCompiledOp(h);
}

static py::list py_compile_batch(py::list requests) {
    py::module_ np = py::module_::import("numpy");

    // Build C-side arrays — keep Python objects alive
    std::vector<libane_compile_request_t> reqs;
    std::vector<py::array>  w_arrays;
    std::vector<py::buffer_info> w_bufs;

    reqs.reserve(requests.size());
    w_arrays.reserve(requests.size());
    w_bufs.reserve(requests.size());

    for (auto item : requests) {
        py::tuple t = item.cast<py::tuple>();
        int op_int      = t[0].cast<int>();
        auto shape      = t[1].cast<std::vector<int>>();
        py::object wobj = t[2];

        libane_compile_request_t r{};
        r.op    = static_cast<libane_op_t>(op_int);
        r.shape = to_shape(shape);

        if (!wobj.is_none()) {
            w_arrays.push_back(np.attr("ascontiguousarray")(
                np.attr("asarray")(wobj, "dtype"_a="float16")));
            w_bufs.push_back(w_arrays.back().request());
            r.weights     = w_bufs.back().ptr;
            r.weights_len = static_cast<size_t>(w_bufs.back().size) *
                            w_bufs.back().itemsize;
        } else {
            w_arrays.emplace_back();
            w_bufs.emplace_back();
        }
        reqs.push_back(r);
    }

    std::vector<libane_handle_t> handles(reqs.size(), nullptr);
    libane_compile_batch(reqs.data(), reqs.size(), handles.data());

    py::list out;
    for (auto h : handles) {
        if (h) out.append(py::cast(new PyCompiledOp(h),
                                    py::return_value_policy::take_ownership));
        else   out.append(py::none());
    }
    return out;
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
    bool sram_spill() const {
        return libane_mil_sram_spill(h_) != 0;
    }

    std::pair<py::list, py::dict> run_stats(const std::vector<py::array>& inputs_raw,
                                             const std::vector<size_t>&    out_numel) {
        py::module_ np = py::module_::import("numpy");

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
            in_sizes.push_back(static_cast<size_t>(in_bufs.back().size) *
                                in_bufs.back().itemsize);
        }

        std::vector<py::array> out_arrays;
        std::vector<void*>     out_ptrs;
        std::vector<size_t>    out_sizes;
        out_arrays.reserve(out_numel.size());
        for (size_t n : out_numel) {
            out_arrays.push_back(
                np.attr("empty")(py::make_tuple(n), "dtype"_a="float16"));
            auto ob = out_arrays.back().request();
            out_ptrs.push_back(ob.ptr);
            out_sizes.push_back(n * 2);
        }

        libane_perf_stats_t stats{};
        libane_status_t st = libane_mil_execute_stats(
            h_,
            in_ptrs.empty()  ? nullptr : in_ptrs.data(),
            in_sizes.empty() ? nullptr : in_sizes.data(),
            in_ptrs.size(),
            out_ptrs.empty()  ? nullptr : out_ptrs.data(),
            out_sizes.empty() ? nullptr : out_sizes.data(),
            out_ptrs.size(),
            &stats);

        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("mil execute_stats failed: ") + libane_last_error());

        py::list outputs;
        for (auto& a : out_arrays) outputs.append(a);

        py::dict s;
        s["ane_bw_utilization"] = stats.ane_bw_utilization;
        s["avg_bw_state"]       = stats.avg_bw_state;
        s["peak_bw_state"]      = stats.peak_bw_state;
        s["ane_energy_units"]   = stats.ane_energy_units;
        s["throttle_ns"]        = stats.throttle_ns;
        s["available"]          = stats.available != 0;
        return {outputs, s};
    }

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

    void save(const std::string& path) {
        auto st = libane_mil_save(h_, path.c_str());
        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("mil_save failed: ") + libane_last_error());
    }

    static PyMilProgram* load(const std::string& path) {
        auto h = libane_mil_load(path.c_str());
        if (!h)
            throw std::runtime_error(
                std::string("mil_load failed: ") + libane_last_error());
        return new PyMilProgram(h);
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

/* ── chunk / split (pure numpy, no ANE execution) ───────────────────────── */

/**
 * chunk(x, n, shape) — split a flat [C*S] fp16 ANE tensor into n equal parts
 * along the channel (C) axis.  Returns n flat fp16 arrays each of size (C/n)*S.
 *
 * shape must be [1, C, 1, S] (standard ANE layout).
 */
static py::list py_chunk(py::array x_in, int n, const std::vector<int>& shape) {
    if (shape.size() != 4)
        throw std::invalid_argument("shape must be a 4-element list [1, C, 1, S]");
    int C = shape[1];
    int S = shape[3];
    if (n <= 0)
        throw std::invalid_argument("n must be > 0");
    if (C % n != 0)
        throw std::invalid_argument(
            "C=" + std::to_string(C) + " is not divisible by n=" + std::to_string(n));

    py::module_ np = py::module_::import("numpy");
    // Ensure f16, then view as [C, S] — channel-major layout is correct.
    py::array x = np.attr("asarray")(x_in, "dtype"_a="float16");
    py::object mat = x.attr("reshape")(py::make_tuple(C, S));

    int chunk_size = C / n;
    py::list result;
    for (int i = 0; i < n; ++i) {
        py::object chunk = mat.attr("__getitem__")(
            py::slice(i * chunk_size, (i + 1) * chunk_size, 1));
        result.append(np.attr("ascontiguousarray")(chunk).attr("ravel")());
    }
    return result;
}

/**
 * split(x, sizes, shape) — split a flat [C*S] fp16 ANE tensor along the
 * channel (C) axis by the given sizes.
 *
 * sizes: int  → each chunk has that many channels (like PyTorch split with int)
 *        list → each element is the channel count for that chunk
 *
 * Returns a list of flat fp16 arrays.
 */
static py::list py_split(py::array x_in, py::object sizes_obj,
                          const std::vector<int>& shape) {
    if (shape.size() != 4)
        throw std::invalid_argument("shape must be a 4-element list [1, C, 1, S]");
    int C = shape[1];
    int S = shape[3];

    py::module_ np = py::module_::import("numpy");
    py::array x = np.attr("asarray")(x_in, "dtype"_a="float16");
    py::object mat = x.attr("reshape")(py::make_tuple(C, S));

    py::list result;

    if (py::isinstance<py::int_>(sizes_obj)) {
        int sz = sizes_obj.cast<int>();
        if (sz <= 0)
            throw std::invalid_argument("split size must be > 0");
        for (int off = 0; off < C; off += sz) {
            int end = std::min(off + sz, C);
            py::object chunk = mat.attr("__getitem__")(
                py::slice(off, end, 1));
            result.append(np.attr("ascontiguousarray")(chunk).attr("ravel")());
        }
    } else {
        auto sizes = sizes_obj.cast<std::vector<int>>();
        int off = 0;
        for (int sz : sizes) {
            if (sz <= 0)
                throw std::invalid_argument("each split size must be > 0");
            if (off + sz > C)
                throw std::invalid_argument(
                    "split sizes sum exceeds C=" + std::to_string(C));
            py::object chunk = mat.attr("__getitem__")(
                py::slice(off, off + sz, 1));
            result.append(np.attr("ascontiguousarray")(chunk).attr("ravel")());
            off += sz;
        }
    }
    return result;
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
            // SLICE/PAD weights are int32; CLIP weights are float32; others fp16.
            if (op == LIBANE_OP_SLICE || op == LIBANE_OP_PAD) {
                w_arr = np.attr("ascontiguousarray")(
                    np.attr("asarray")(weights_obj, "dtype"_a="int32"));
            } else if (op == LIBANE_OP_CLIP) {
                w_arr = np.attr("ascontiguousarray")(
                    np.attr("asarray")(weights_obj, "dtype"_a="float32"));
            } else {
                w_arr = np.attr("ascontiguousarray")(
                    np.attr("asarray")(weights_obj, "dtype"_a="float16"));
            }
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

    uint32_t add_pwl_activation(uint32_t input_id,
                                const std::vector<int>& output_shape,
                                float x_min, float x_max,
                                py::array samples_in) {
        py::module_ np = py::module_::import("numpy");
        py::array s = np.attr("ascontiguousarray")(
            np.attr("asarray")(samples_in, "dtype"_a="float32"));
        auto sb = s.request();
        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_pwl_activation(
            g_, input_id, out_s, x_min, x_max,
            static_cast<const float*>(sb.ptr),
            static_cast<uint32_t>(sb.size));
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_pwl_activation failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_dynamic_matmul(uint32_t x_id, uint32_t w_id,
                                const std::vector<int>& output_shape) {
        // Matrix format: X=[1,1,K,M]  W=[1,1,N,K]  output=[1,1,N,M] — no weights.
        std::vector<uint32_t> inputs = {x_id, w_id};
        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_op(g_, LIBANE_OP_DYNAMIC_MATMUL,
                                           inputs.data(), inputs.size(),
                                           out_s, nullptr, 0);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_dynamic_matmul failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_sdpa(uint32_t q_id, uint32_t k_id, uint32_t v_id,
                      const std::vector<int>& output_shape,
                      int mask_id = -1) {
        // Matrix format: Q/K/V=[1,H,S,D]  output=[1,H,S,D] — no weights.
        // H/S/D are inferred from the input shapes.
        std::vector<uint32_t> inputs = {q_id, k_id, v_id};
        if (mask_id >= 0)
            inputs.push_back(static_cast<uint32_t>(mask_id));
        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_op(g_, LIBANE_OP_SDPA,
                                           inputs.data(), inputs.size(),
                                           out_s, nullptr, 0);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_sdpa failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_sdpa_gqa(uint32_t q_id, uint32_t k_id, uint32_t v_id,
                          int mask_id = -1) {
        // Grouped Query Attention: Q=[1,H_q,S,D], K/V=[1,H_kv,S,D].
        // Output shape mirrors Q — derived from Q tensor in the C API.
        uint32_t mid = (mask_id >= 0) ? static_cast<uint32_t>(mask_id)
                                       : LIBANE_INVALID_TENSOR_ID;
        uint32_t id = libane_graph_add_sdpa_gqa(g_, q_id, k_id, v_id, mid);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_sdpa_gqa failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_matmul_w8a16(uint32_t       input_id,
                              const std::vector<int>& output_shape,
                              py::array weights_in,
                              py::array scales_in,
                              int IC, int OC) {
        // Ensure weights arrive as a contiguous int8 array.
        py::module_ np = py::module_::import("numpy");
        py::array W = np.attr("ascontiguousarray")(
            np.attr("asarray")(weights_in, "dtype"_a="int8"));
        // Ensure scales arrive as a contiguous float32 array.
        py::array sc = np.attr("ascontiguousarray")(
            np.attr("asarray")(scales_in, "dtype"_a="float32"));

        auto wb = W.request();
        auto sb = sc.request();

        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_matmul_w8a16(
            g_, input_id, out_s,
            static_cast<const int8_t*>(wb.ptr),
            static_cast<const float*>(sb.ptr),
            IC, OC);
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_matmul_w8a16 failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_matmul_w8a8(uint32_t       input_id,
                              const std::vector<int>& output_shape,
                              py::array weights_in,
                              py::array scales_in,
                              int IC, int OC,
                              float act_scale, int act_zero_point) {
        py::module_ np = py::module_::import("numpy");
        py::array W = np.attr("ascontiguousarray")(
            np.attr("asarray")(weights_in, "dtype"_a="int8"));
        py::array sc = np.attr("ascontiguousarray")(
            np.attr("asarray")(scales_in, "dtype"_a="float32"));
        auto wb = W.request();
        auto sb = sc.request();
        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_matmul_w8a8(
            g_, input_id, out_s,
            static_cast<const int8_t*>(wb.ptr),
            static_cast<const float*>(sb.ptr),
            IC, OC, act_scale, static_cast<int32_t>(act_zero_point));
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_matmul_w8a8 failed: ") + libane_last_error());
        return id;
    }

    uint32_t add_conv2d(uint32_t       input_id,
                        const std::vector<int>& output_shape,
                        int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_top, int pad_left,
                        int pad_bottom, int pad_right,
                        int dilation_h, int dilation_w,
                        int groups,
                        py::array kernel_in) {
        // Ensure the kernel arrives as a contiguous fp16 (float16) array.
        py::module_ np = py::module_::import("numpy");
        py::array k = np.attr("ascontiguousarray")(
            np.attr("asarray")(kernel_in, "dtype"_a="float16"));
        auto kb = k.request();
        libane_shape_t out_s = to_shape(output_shape);
        uint32_t id = libane_graph_add_conv2d(
            g_, input_id, out_s,
            kH, kW,
            stride_h, stride_w,
            pad_top, pad_left, pad_bottom, pad_right,
            dilation_h, dilation_w,
            groups,
            kb.ptr, static_cast<size_t>(kb.size) * 2);  // ×2: bytes per fp16
        if (id == LIBANE_INVALID_TENSOR_ID)
            throw std::invalid_argument(
                std::string("add_conv2d failed: ") + libane_last_error());
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

/* ── PyKVCache ───────────────────────────────────────────────────────────── */

class PyKVCache {
public:
    PyKVCache(int num_heads, int head_dim, int max_seq) {
        cache_ = libane_kv_cache_create(num_heads, head_dim, max_seq);
        if (!cache_)
            throw std::invalid_argument(
                std::string("KVCache: ") + libane_last_error());
        num_heads_ = num_heads;
        head_dim_  = head_dim;
        max_seq_   = max_seq;
    }
    ~PyKVCache() { libane_kv_cache_release(cache_); }

    PyKVCache(const PyKVCache&) = delete;
    PyKVCache& operator=(const PyKVCache&) = delete;

    int update(py::array k_arr, py::array v_arr) {
        py::module_ np = py::module_::import("numpy");
        auto K = np.attr("ascontiguousarray")(
            np.attr("asarray")(k_arr, "dtype"_a="float16"));
        auto V = np.attr("ascontiguousarray")(
            np.attr("asarray")(v_arr, "dtype"_a="float16"));
        auto kb = K.cast<py::array>().request();
        auto vb = V.cast<py::array>().request();
        size_t expected = static_cast<size_t>(num_heads_) * head_dim_;
        if (static_cast<size_t>(kb.size) != expected)
            throw std::invalid_argument(
                "KVCache.update: k must have " + std::to_string(expected) +
                " elements (num_heads × head_dim)");
        if (static_cast<size_t>(vb.size) != expected)
            throw std::invalid_argument(
                "KVCache.update: v must have " + std::to_string(expected) +
                " elements (num_heads × head_dim)");
        int r = libane_kv_cache_update(cache_, kb.ptr, vb.ptr);
        if (r < 0)
            throw std::runtime_error(
                std::string("KVCache.update: ") + libane_last_error());
        return r;
    }

    // Return numpy array that is a view over the internal K buffer.
    // Shape: [num_heads, max_seq, head_dim]
    py::array k_array() const {
        const void* ptr = libane_kv_cache_k(cache_);
        py::module_ np = py::module_::import("numpy");
        return np.attr("frombuffer")(
            py::bytes(static_cast<const char*>(ptr),
                      static_cast<size_t>(num_heads_) * max_seq_ * head_dim_ * 2),
            "dtype"_a="float16")
            .attr("reshape")(py::make_tuple(1, num_heads_, max_seq_, head_dim_));
    }

    py::array v_array() const {
        const void* ptr = libane_kv_cache_v(cache_);
        py::module_ np = py::module_::import("numpy");
        return np.attr("frombuffer")(
            py::bytes(static_cast<const char*>(ptr),
                      static_cast<size_t>(num_heads_) * max_seq_ * head_dim_ * 2),
            "dtype"_a="float16")
            .attr("reshape")(py::make_tuple(1, num_heads_, max_seq_, head_dim_));
    }

    int position() const { return libane_kv_cache_position(cache_); }
    int num_heads() const { return num_heads_; }
    int head_dim()  const { return head_dim_; }
    int max_seq()   const { return max_seq_; }

    void reset() { libane_kv_cache_reset(cache_); }

    libane_kv_cache_t raw() const { return cache_; }

private:
    libane_kv_cache_t cache_;
    int num_heads_, head_dim_, max_seq_;
};

/* ─────────────────────────────────────────────────────────────────────────── */

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

    /**
     * Save this compiled graph to a file.
     *
     * Writes MIL text, weight blobs, and compiled HWX binaries.
     * Reload with CompiledGraph.load(path) — skips recompilation (~8.5×
     * faster cold start).
     *
     * Raises RuntimeError on failure.
     */
    void save(const std::string& path) {
        libane_status_t st = libane_compiled_graph_save(cg_, path.c_str());
        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("save failed: ") + libane_last_error());
    }

    /**
     * Reload all ANE programs in this compiled graph into SRAM without
     * recompiling (~8.5× faster than compile()).
     *
     * Use after a host suspend/resume cycle or ANE context reset, or as part
     * of a LoRA weight hot-swap workflow (mutate the underlying AneProgram
     * weight blobs, then call delta_reload() to push updated weights).
     *
     * Raises RuntimeError on failure.
     */
    void delta_reload() {
        libane_status_t st = libane_compiled_graph_delta_reload(cg_);
        if (st != LIBANE_OK)
            throw std::runtime_error(
                std::string("delta_reload failed: ") + libane_last_error());
    }

    /**
     * Load a compiled graph from a file produced by save().
     * Restores programs via loadWithQoS: (no recompilation) for fast cold start.
     * Raises RuntimeError on failure.
     */
    static PyCompiledGraph* load(const std::string& path) {
        libane_compiled_graph_t cg = libane_compiled_graph_load(path.c_str());
        if (!cg)
            throw std::runtime_error(
                std::string("load failed: ") + libane_last_error());
        return new PyCompiledGraph(cg);
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
ane — Apple Neural Engine Python bindings (libane v0.9.0)
Amirani Labs

Run ML graphs directly on the Apple Neural Engine from Python.
Uses AppleNeuralEngine.framework via dlopen — private API, intentional.
)";

    libane_set_log_level(LIBANE_LOG_ERROR);

    /* ── Utility ──────────────────────────────────────────────────────── */
    m.def("available",    []() { return libane_available() != 0; },
          "True if the Apple Neural Engine is accessible.");
    m.def("version",      []() { return std::string(libane_version()); },
          "libane version string.");
    m.def("last_error",   []() { return std::string(libane_last_error()); },
          "Last error message.");
    m.def("device_info",  &py_device_info,
          "ANE hardware info: architecture, core_count, num_anes, available.");
    m.def("shape_limits", &py_shape_limits,
          "ANE tensor shape limits: max_seq, max_channels, seq_alignment.");
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

    /* ── CompiledOp ───────────────────────────────────────────────────── */
    py::class_<PyCompiledOp>(m, "CompiledOp")
        .def("execute",      &PyCompiledOp::execute,
             py::arg("x"), py::arg("shape"),
             "Execute with one input. shape=[B,C,H,S].")
        .def("execute2",     &PyCompiledOp::execute2,
             py::arg("x0"), py::arg("x1"), py::arg("shape"),
             "Execute with two inputs (alphabetical MIL parameter order).")
        .def("delta_reload", &PyCompiledOp::delta_reload,
             "Reload compiled program into ANE SRAM without recompiling (~8.5× faster than compile).");

    m.def("compile",
          &py_compile_op,
          py::arg("op"), py::arg("shape"), py::arg("weights") = py::none(),
          py::return_value_policy::take_ownership,
          "Compile a single op. Returns CompiledOp. op = ane.MATMUL etc.");

    m.def("compile_batch",
          &py_compile_batch,
          py::arg("requests"),
          R"(Compile multiple ops in one call.

Args:
    requests: list of (op, shape, weights_or_None) tuples.

Returns:
    list of CompiledOp | None — None for any that failed.
    Check ane.last_error() on partial failure.
)");

    /* ── Single-op ────────────────────────────────────────────────────── */
    m.def("matmul",    &py_matmul,    py::arg("A"), py::arg("B"),
          "ANE fp16 matmul: C = A @ B.");
    m.def("matmul_f32",&py_matmul_f32,py::arg("A"), py::arg("B"),
          "ANE fp32 matmul (fp32→fp16→fp32 internally).");
    m.def("softmax",   &py_softmax,   py::arg("x"),
          "ANE softmax over last dimension.");
    m.def("gelu",      &py_gelu,      py::arg("x"),
          "ANE GELU (tanh approximation).");

    /* ── Quantization utilities ──────────────────────────────────────────── */
    m.def("quantize_i8",
          [](py::array x_in, float scale, int zero_point) -> py::array {
              py::module_ np = py::module_::import("numpy");
              py::array x = np.attr("ascontiguousarray")(
                  np.attr("asarray")(x_in, "dtype"_a="float16"));
              auto xb = x.request();
              size_t n = static_cast<size_t>(xb.size);
              py::array_t<int8_t> out(n);
              auto ob = out.request();
              libane_quantize_i8(xb.ptr,
                                  static_cast<int8_t*>(ob.ptr),
                                  n, scale, static_cast<int32_t>(zero_point));
              return out.reshape(x.attr("shape"));
          },
          py::arg("x"),
          py::arg("scale"),
          py::arg("zero_point") = 0,
          R"(Quantize a float16 array to int8.

out[i] = clamp(round(x[i] / scale) + zero_point, -128, 127)

Use this to prepare activations before passing to a W8A8 compiled graph.

Args:
    x:           np.float16 array (any shape).
    scale:       Quantization scale (float > 0).
    zero_point:  Quantization zero-point (int, default 0).

Returns:
    np.int8 array with the same shape as x.
)");

    /* ── Tensor utilities ─────────────────────────────────────────────── */
    m.def("chunk", &py_chunk,
          py::arg("x"), py::arg("n"), py::arg("shape"),
          R"(Split a flat ANE tensor along the channel (C) axis into n equal chunks.

Args:
    x:     Flat fp16 array of C*S elements (output of CompiledGraph.__call__).
    n:     Number of equal chunks. C must be divisible by n.
    shape: Full 4-D shape [1, C, 1, S] of the tensor.

Returns:
    List of n contiguous fp16 arrays, each of size (C//n)*S.

Example::

    # QKV split: projection output is [1, 3*D, 1, S]
    qkv = cg(x)
    q, k, v = ane.chunk(qkv, 3, [1, 3*D, 1, S])
)");

    m.def("split", &py_split,
          py::arg("x"), py::arg("sizes"), py::arg("shape"),
          R"(Split a flat ANE tensor along the channel (C) axis by the given sizes.

Args:
    x:     Flat fp16 array of C*S elements.
    sizes: int — each chunk has that many channels (like torch.split with int);
           or list[int] — explicit channel count per chunk.
    shape: Full 4-D shape [1, C, 1, S].

Returns:
    List of contiguous fp16 arrays, one per chunk.

Example::

    # Unequal split: first 256 channels, remaining 128
    a, b = ane.split(out, [256, 128], [1, 384, 1, S])
)");

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
        .def("add_pwl_activation", &PyGraph::add_pwl_activation,
             py::arg("input_id"), py::arg("output_shape"),
             py::arg("x_min"), py::arg("x_max"), py::arg("samples"),
             R"(Add a piecewise-linear custom activation.

Args:
    input_id:     Tensor ID of the input.
    output_shape: [1, C, 1, S] — must match input shape.
    x_min, x_max: Domain of the approximation.
    samples:      np.float32 array of length n_samples (n_samples-1 linear segments).
                  Values are fn(linspace(x_min, x_max, n_samples)).

Returns:
    Output tensor ID.
)")
        .def("add_dynamic_matmul", &PyGraph::add_dynamic_matmul,
             py::arg("x_id"), py::arg("w_id"), py::arg("output_shape"),
             R"(Dynamic matmul Y = X @ W^T (both inputs are runtime tensors).

Uses matrix tensor format (height > 1, C = 1).  K/N/M are inferred from shapes.

Args:
    x_id:         Tensor ID of X, shape [1, 1, K, M]  (height=K inner, seq=M cols).
    w_id:         Tensor ID of W, shape [1, 1, N, K]  (height=N rows,  seq=K inner).
    output_shape: [1, 1, N, M].

Returns:
    Output tensor ID.
)")
        .def("add_sdpa", &PyGraph::add_sdpa,
             py::arg("q_id"), py::arg("k_id"), py::arg("v_id"),
             py::arg("output_shape"),
             py::arg("mask_id") = -1,
             R"(Scaled dot-product attention: out = softmax(Q @ K^T / sqrt(D)) @ V

Uses matrix tensor format (height > 1).  H/S/D are inferred from input shapes.

Args:
    q_id, k_id, v_id: Tensor IDs of Q/K/V, each shape [1, H, S, D].
    output_shape:     [1, H, S, D].
    mask_id:          Tensor ID of additive mask [1, 1, S, S], or -1 for none.

Returns:
    Output tensor ID.
)")
        .def("add_sdpa_gqa", &PyGraph::add_sdpa_gqa,
             py::arg("q_id"), py::arg("k_id"), py::arg("v_id"),
             py::arg("mask_id") = -1,
             R"(Grouped Query Attention SDPA: tiles K/V from H_kv to H_q heads.

Like add_sdpa() but Q has num_q_heads heads while K and V each have
num_kv_heads heads (num_q_heads must be divisible by num_kv_heads).
Output shape is derived from Q — no need to pass it explicitly.

Args:
    q_id:    Tensor ID of Q, shape [1, num_q_heads,  S, D].
    k_id:    Tensor ID of K, shape [1, num_kv_heads, S, D].
    v_id:    Tensor ID of V, shape [1, num_kv_heads, S, D].
    mask_id: Tensor ID of additive mask [1, 1, S, S], or -1 for none.

Returns:
    Output tensor ID, shape [1, num_q_heads, S, D].

Example (Llama-3.1 8B: 32 query heads, 8 KV heads)::

    H_q, H_kv, S, D = 32, 8, 512, 128
    Q = g.add_input("Q", [1, H_q,  S, D])
    K = g.add_input("K", [1, H_kv, S, D])
    V = g.add_input("V", [1, H_kv, S, D])
    out = g.add_sdpa_gqa(Q, K, V)   # output: [1, H_q, S, D]
    g.mark_output(out)
    cg = g.compile()
)")
        .def("add_matmul_w8a16", &PyGraph::add_matmul_w8a16,
             py::arg("input_id"),
             py::arg("output_shape"),
             py::arg("weights"),
             py::arg("scales"),
             py::arg("IC"),
             py::arg("OC"),
             R"(W8A16 quantized matrix multiply fused into the graph.

Dequantizes int8 weights with per-channel float32 scales at compile time,
then compiles as a standard conv1x1 matmul. Fuses with downstream elementwise
ops (GELU, SILU, ADD …) at zero extra cost — identical fusion behaviour to
add_op(ane.MATMUL, …).

Args:
    input_id:     Tensor ID of the activation input [1, IC, 1, S].
    output_shape: [1, OC, 1, S].
    weights:      np.int8 array, shape [IC, OC] row-major.
                  (Transpose of PyTorch's linear weight [OC, IC].)
    scales:       np.float32 array, shape [OC]. Per-output-channel scale.
                  Symmetric quantization: zero-point is implicitly 0.
    IC:           Input channel count.
    OC:           Output channel count.

Returns:
    Output tensor ID (uint32).

Example::

    # Quantize an fp16 weight matrix (PyTorch convention [OC, IC])
    W_fp16 = model.linear.weight.numpy().astype(np.float16)   # [OC, IC]
    scales = np.max(np.abs(W_fp16), axis=1) / 127.0           # [OC]
    W_int8 = np.round(W_fp16 / scales[:, None]).astype(np.int8)
    W_int8_IC_OC = W_int8.T.copy()  # transpose to [IC, OC]

    out = g.add_matmul_w8a16(x, [1, OC, 1, S],
                              weights=W_int8_IC_OC,
                              scales=scales.astype(np.float32),
                              IC=IC, OC=OC)
)")
        .def("add_matmul_w8a8", &PyGraph::add_matmul_w8a8,
             py::arg("input_id"),
             py::arg("output_shape"),
             py::arg("weights"),
             py::arg("scales"),
             py::arg("IC"),
             py::arg("OC"),
             py::arg("act_scale"),
             py::arg("act_zero_point") = 0,
             R"(W8A8 quantized matrix multiply — int8 weights AND int8 activations.

Both sides quantized to int8; dequantization is transparent:
  - Weights dequantized at compile time (same as W8A16).
  - Activations dequantized at execute time before IOSurface transfer.

At execute time, pass int8 numpy arrays as graph inputs (not float16).

Args:
    input_id:        Tensor ID of the int8 activation input [1, IC, 1, S].
    output_shape:    [1, OC, 1, S].
    weights:         np.int8 array, shape [IC, OC] row-major.
    scales:          np.float32 array, shape [OC]. Per-channel weight scales.
    IC:              Input channel count.
    OC:              Output channel count.
    act_scale:       Per-tensor activation scale (float > 0).
    act_zero_point:  Per-tensor activation zero-point (int, default 0).

Returns:
    Output tensor ID (uint32).

Example::

    import numpy as np
    import ane

    # Quantize weights
    W_fp16   = model.linear.weight.numpy()   # [OC, IC]
    w_scales = np.max(np.abs(W_fp16), axis=1) / 127.0
    W_int8   = (W_fp16 / w_scales[:, None]).round().clip(-128,127).astype(np.int8)
    W_IC_OC  = W_int8.T.copy()              # [IC, OC]

    # Activation quantization params (calibrated separately)
    act_scale = 0.02
    act_zp    = 0

    g = ane.Graph()
    x = g.add_input("x", [1, IC, 1, S])
    y = g.add_matmul_w8a8(x, [1, OC, 1, S],
                           weights=W_IC_OC, scales=w_scales.astype(np.float32),
                           IC=IC, OC=OC,
                           act_scale=act_scale, act_zero_point=act_zp)
    g.mark_output(y)
    cg = g.compile()

    # At inference: quantize activations before passing
    x_fp16 = activations.astype(np.float16)
    x_int8 = ane.quantize_i8(x_fp16, scale=act_scale, zero_point=act_zp)
    out = cg(x_int8)
)")
        .def("add_conv2d", &PyGraph::add_conv2d,
             py::arg("input_id"),
             py::arg("output_shape"),
             py::arg("kH"), py::arg("kW"),
             py::arg("stride_h") = 1, py::arg("stride_w") = 1,
             py::arg("pad_top") = 0, py::arg("pad_left") = 0,
             py::arg("pad_bottom") = 0, py::arg("pad_right") = 0,
             py::arg("dilation_h") = 1, py::arg("dilation_w") = 1,
             py::arg("groups") = 1,
             py::arg("kernel"),
             R"(Add a 2D convolution op to the graph.

Conv image tensors use shape [1, C, H, W] with H > 1 and W a multiple of 32.

Output shape formula:
    H_out = (H_in + pad_top + pad_bottom - dilation_h*(kH-1) - 1) // stride_h + 1
    W_out = (W_in + pad_left + pad_right - dilation_w*(kW-1) - 1) // stride_w + 1

Args:
    input_id:     Tensor ID of the input [1, IC, H_in, W_in].
    output_shape: [1, OC, H_out, W_out].
    kH, kW:       Kernel spatial size.
    stride_h/w:   Stride (default 1).
    pad_top/left/bottom/right: Explicit padding in pixels (default 0).
    dilation_h/w: Dilation (default 1).
    groups:       Group count for group/depthwise conv (default 1).
                  IC % groups must equal 0.
    kernel:       np.float16 array, shape [OC, IC//groups, kH, kW], row-major.
                  NO transpose required (unlike MATMUL).

Returns:
    Output tensor ID.

Example (pointwise 1×1 conv, IC=4 → OC=8)::

    W = np.zeros((8, 4, 1, 1), dtype=np.float16)
    # fill W with desired weights ...
    out = g.add_conv2d(x, [1, 8, H, W_seq],
                       kH=1, kW=1,
                       kernel=W)

Example (depthwise 3×3, SAME padding, IC=OC=C)::

    W = np.zeros((C, 1, 3, 3), dtype=np.float16)
    out = g.add_conv2d(x, [1, C, H, W_seq],
                       kH=3, kW=3,
                       pad_top=1, pad_left=1,
                       pad_bottom=1, pad_right=1,
                       groups=C,
                       kernel=W)
)")
        .def("mark_output", &PyGraph::mark_output,
             py::arg("tensor_id"), py::arg("name") = "",
             "Mark tensor_id as a graph output.")
        .def("compile", &py_compile,
             py::return_value_policy::take_ownership,
             "Compile the graph. Returns CompiledGraph. Raises on failure.");

    py::class_<PyKVCache>(m, "KVCache", R"(
CPU-side KV cache for autoregressive decode with SDPA/SDPA_GQA graphs.

Holds zero-initialised fp16 buffers for K and V at a fixed maximum sequence
length.  Append one token at a time with :meth:`update`, then pass
:attr:`k` and :attr:`v` (full buffers) as K/V inputs to your compiled graph.

Example (Llama-3.1 8B, GQA: 8 KV heads, D=128, max 512 tokens)::

    cache = ane.KVCache(num_heads=8, head_dim=128, max_seq=512)
    # at each decode step:
    pos = cache.update(k_token, v_token)  # k/v_token: float16 [8, 128]
    out = cg([q_vec, cache.k, cache.v])   # pass full buffers to graph
)")
        .def(py::init<int, int, int>(),
             py::arg("num_heads"), py::arg("head_dim"), py::arg("max_seq"),
             R"(Create a KV cache.

Args:
    num_heads: Number of KV heads.
    head_dim:  Head dimension D.
    max_seq:   Maximum sequence length.
)")
        .def("update", &PyKVCache::update,
             py::arg("k"), py::arg("v"),
             R"(Append one token's K and V to the cache.

k and v must be float16 arrays with num_heads × head_dim elements, shaped
arbitrarily but contiguously (reshaped internally if needed).

Returns the new position (1-based fill count) on success.
Raises RuntimeError if the cache is full.
)")
        .def_property_readonly("k", &PyKVCache::k_array,
             "Full K buffer as float16 numpy array, shape [1, num_heads, max_seq, head_dim].")
        .def_property_readonly("v", &PyKVCache::v_array,
             "Full V buffer as float16 numpy array, shape [1, num_heads, max_seq, head_dim].")
        .def_property_readonly("position", &PyKVCache::position,
             "Number of valid token positions written so far.")
        .def_property_readonly("num_heads", &PyKVCache::num_heads,
             "Number of KV heads.")
        .def_property_readonly("head_dim",  &PyKVCache::head_dim,
             "Head dimension D.")
        .def_property_readonly("max_seq",   &PyKVCache::max_seq,
             "Maximum sequence length.")
        .def("reset", &PyKVCache::reset,
             "Reset position to 0 and zero the K/V buffers.");

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
             "Set expected output shapes (list of shape lists) for multi-output graphs.")
        .def("save", &PyCompiledGraph::save, py::arg("path"),
             "Save compiled graph to file for fast cold-start restoration.")
        .def_static("load", &PyCompiledGraph::load, py::arg("path"),
                    py::return_value_policy::take_ownership,
                    "Load a compiled graph saved with save(). Skips recompilation.")
        .def("delta_reload", &PyCompiledGraph::delta_reload, R"(
Reload all ANE programs in this compiled graph into SRAM without recompiling.

~8.5× faster than recompiling. Use after host suspend/resume or ANE context
reset. Also enables LoRA-style weight hot-swap: mutate underlying AneProgram
weight blobs, then call delta_reload() to push updates to the accelerator.

Example::

    cg = g.compile()
    cg(x)                 # first inference

    # ... host suspended / resumed ...
    cg.delta_reload()     # restore programs to SRAM
    cg(x)                 # inference works again

Raises RuntimeError on failure.
)");

    /* ── CompiledMil / compile_mil ───────────────────────────────────── */
    py::class_<PyMilProgram>(m, "CompiledMil", R"(
Compiled raw MIL program. Call .run() to execute.

Example — probe relu::

    import ane
    import numpy as np

    mil = '''
    program(1.3)
    [buildInfo = dict<string, string>({"coremlc-component-MIL", "3510.2.1"},
     {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""},
     {"coremltools-version", "9.0"})]
    {
        func main<ios18>(tensor<fp16, [1,32,1,32]> a_input0) {
            tensor<fp16, [1,32,1,32]> z_output0 =
                relu(x=a_input0)[name=string("z_output0")];
        } -> (z_output0);
    }
    '''
    prog  = ane.compile_mil(mil)
    x     = np.random.randn(32, 32).astype(np.float16)
    (out,) = prog.run([x], [32 * 32])

Use ane.probe for a higher-level interface.
)")
        .def("run", &PyMilProgram::run,
             py::arg("inputs"), py::arg("output_sizes"),
             R"(Execute the MIL program.

Args:
    inputs:       List of numpy arrays (any dtype; converted to float16).
    output_sizes: List of output sizes in fp16 ELEMENTS (not bytes).

Returns:
    List of np.float16 ndarrays, one per output.

Raises:
    RuntimeError on ANE dispatch failure.
)")
        .def("run_stats", &PyMilProgram::run_stats,
             py::arg("inputs"), py::arg("output_sizes"),
             R"(Execute and return IOReport hardware counters.

Returns:
    (outputs, stats) where outputs is a list of np.float16 arrays and
    stats is a dict with keys: ane_bw_utilization (float, 0–1),
    avg_bw_state (float), peak_bw_state (int), ane_energy_units (int),
    throttle_ns (int), available (bool).
)")
        .def_property_readonly("sram_spill", &PyMilProgram::sram_spill,
             "True if the compiled program requires DRAM-backed intermediate buffers "
             "(SRAM spill). Always False for single-op programs.")
        .def("save", &PyMilProgram::save,
             py::arg("path"),
             R"(Save the compiled MIL program to a file for fast cold-start restoration.

Writes MIL source, weight blobs, and compiled HWX to a binary file (magic
"ANEM", version 1).  On reload via CompiledMil.load() only loadWithQoS: is
called — ~8.5× faster than full recompilation.

Args:
    path: Destination file path.  Created or overwritten.

Raises:
    RuntimeError on serialization or I/O failure.
)")
        .def_static("load", &PyMilProgram::load,
             py::arg("path"),
             py::return_value_policy::take_ownership,
             R"(Load a compiled MIL program saved by :meth:`save`.

Args:
    path: File path produced by :meth:`save`.

Returns:
    CompiledMil handle ready for .run().

Raises:
    RuntimeError on format error, I/O error, or ANE failure.
)");

    m.def("compile_mil", &py_compile_mil,
          py::arg("mil_text"),
          py::return_value_policy::take_ownership,
          R"(Compile a raw MIL program (no external weights).

Args:
    mil_text: UTF-8 MIL source text.

Returns:
    CompiledMil handle ready for .run().

Raises:
    RuntimeError if ANE is unavailable or compilation fails.
    Use ane.last_error() for details.
)");

    m.def("compile_mil_with_weights", &py_compile_mil_with_weights,
          py::arg("mil_text"), py::arg("weights"),
          py::return_value_policy::take_ownership,
          R"(Compile a raw MIL program with external weight files.

Args:
    mil_text: UTF-8 MIL source text.
    weights:  Dict mapping filename (str) → np.float16 array.
              Filenames must match the file() references in the MIL.

Returns:
    CompiledMil handle ready for .run().

Raises:
    RuntimeError if ANE is unavailable or compilation fails.
)");

    /* ── Op constants ─────────────────────────────────────────────────── */
    m.attr("MATMUL")    = static_cast<int>(LIBANE_OP_MATMUL);
    m.attr("LAYER_NORM")= static_cast<int>(LIBANE_OP_LAYER_NORM);
    m.attr("LAYERNORM") = static_cast<int>(LIBANE_OP_LAYERNORM);
    m.attr("GELU")      = static_cast<int>(LIBANE_OP_GELU);
    m.attr("SOFTMAX")   = static_cast<int>(LIBANE_OP_SOFTMAX);
    m.attr("ADD")       = static_cast<int>(LIBANE_OP_ADD);
    m.attr("MUL")       = static_cast<int>(LIBANE_OP_MUL);
    m.attr("SUB")       = static_cast<int>(LIBANE_OP_SUB);
    m.attr("REAL_DIV")  = static_cast<int>(LIBANE_OP_REAL_DIV);
    m.attr("SQRT")      = static_cast<int>(LIBANE_OP_SQRT);
    m.attr("LOG")       = static_cast<int>(LIBANE_OP_LOG);
    m.attr("RSQRT")     = static_cast<int>(LIBANE_OP_RSQRT);
    m.attr("TRANSPOSE") = static_cast<int>(LIBANE_OP_TRANSPOSE);
    m.attr("RESHAPE")   = static_cast<int>(LIBANE_OP_RESHAPE);
    m.attr("CONCAT")    = static_cast<int>(LIBANE_OP_CONCAT);
    m.attr("SLICE_BY_INDEX") = static_cast<int>(LIBANE_OP_SLICE_BY_INDEX);
    m.attr("REDUCE_SUM") = static_cast<int>(LIBANE_OP_REDUCE_SUM);
    m.attr("REDUCE_MEAN") = static_cast<int>(LIBANE_OP_REDUCE_MEAN);
    m.attr("REDUCE_MAX") = static_cast<int>(LIBANE_OP_REDUCE_MAX);
    m.attr("SILU")      = static_cast<int>(LIBANE_OP_SILU);
    m.attr("RMSNORM")   = static_cast<int>(LIBANE_OP_RMSNORM);
    m.attr("AVG_POOL")  = static_cast<int>(LIBANE_OP_AVG_POOL);
    m.attr("MAX_POOL")  = static_cast<int>(LIBANE_OP_MAX_POOL);
    m.attr("LOGICAL_AND") = static_cast<int>(LIBANE_OP_LOGICAL_AND);
    m.attr("LOGICAL_OR") = static_cast<int>(LIBANE_OP_LOGICAL_OR);
    m.attr("LOGICAL_XOR") = static_cast<int>(LIBANE_OP_LOGICAL_XOR);
    m.attr("REDUCE_PROD") = static_cast<int>(LIBANE_OP_REDUCE_PROD);
    m.attr("SCATTER")   = static_cast<int>(LIBANE_OP_SCATTER);
    m.attr("GATHER")    = static_cast<int>(LIBANE_OP_GATHER);
    m.attr("SCATTER_ND") = static_cast<int>(LIBANE_OP_SCATTER_ND);
    m.attr("SCATTER_ALONG_AXIS") = static_cast<int>(LIBANE_OP_SCATTER_ALONG_AXIS);
    m.attr("NEG")       = static_cast<int>(LIBANE_OP_NEG);
    m.attr("MOD")       = static_cast<int>(LIBANE_OP_MOD);
    m.attr("SINH")      = static_cast<int>(LIBANE_OP_SINH);
    m.attr("COSH")      = static_cast<int>(LIBANE_OP_COSH);
    m.attr("TAN")       = static_cast<int>(LIBANE_OP_TAN);
    m.attr("ASIN")      = static_cast<int>(LIBANE_OP_ASIN);
    m.attr("ACOS")      = static_cast<int>(LIBANE_OP_ACOS);
    m.attr("SELECT")    = static_cast<int>(LIBANE_OP_SELECT);
    m.attr("RELU")      = static_cast<int>(LIBANE_OP_RELU);
    m.attr("TANH")      = static_cast<int>(LIBANE_OP_TANH);
    m.attr("SIGMOID")   = static_cast<int>(LIBANE_OP_SIGMOID);
    m.attr("HARDSWISH") = static_cast<int>(LIBANE_OP_HARDSWISH);
    m.attr("LEAKY_RELU")= static_cast<int>(LIBANE_OP_LEAKY_RELU);
    m.attr("ELU")       = static_cast<int>(LIBANE_OP_ELU);
    m.attr("PIXEL_SHUFFLE") = static_cast<int>(LIBANE_OP_PIXEL_SHUFFLE);
    m.attr("CAST")      = static_cast<int>(LIBANE_OP_CAST);
    m.attr("CONV2D")    = static_cast<int>(LIBANE_OP_CONV2D);
    m.attr("PWL_ACTIVATION") = static_cast<int>(LIBANE_OP_PWL_ACTIVATION);
    m.attr("SLICE")     = static_cast<int>(LIBANE_OP_SLICE);
    m.attr("CLIP")            = static_cast<int>(LIBANE_OP_CLIP);
    m.attr("PAD")             = static_cast<int>(LIBANE_OP_PAD);
    m.attr("DYNAMIC_MATMUL")  = static_cast<int>(LIBANE_OP_DYNAMIC_MATMUL);
    m.attr("SDPA")            = static_cast<int>(LIBANE_OP_SDPA);
    m.attr("SDPA_GQA")        = static_cast<int>(LIBANE_OP_SDPA_GQA);
    m.attr("MATMUL_W8A8")     = static_cast<int>(LIBANE_OP_MATMUL_W8A8);
    m.attr("EXP")             = static_cast<int>(LIBANE_OP_EXP);
    m.attr("SIN")             = static_cast<int>(LIBANE_OP_SIN);
    m.attr("COS")             = static_cast<int>(LIBANE_OP_COS);
    m.attr("ABS")             = static_cast<int>(LIBANE_OP_ABS);
    m.attr("POW")             = static_cast<int>(LIBANE_OP_POW);
    m.attr("CEIL")            = static_cast<int>(LIBANE_OP_CEIL);
    m.attr("FLOOR")           = static_cast<int>(LIBANE_OP_FLOOR);
    m.attr("ROUND")           = static_cast<int>(LIBANE_OP_ROUND);
    m.attr("SIGN")            = static_cast<int>(LIBANE_OP_SIGN);

    /* ── Log level constants ──────────────────────────────────────────── */
    m.attr("LOG_SILENT") = static_cast<int>(LIBANE_LOG_SILENT);
    m.attr("LOG_ERROR")  = static_cast<int>(LIBANE_LOG_ERROR);
    m.attr("LOG_WARN")   = static_cast<int>(LIBANE_LOG_WARN);
    m.attr("LOG_INFO")   = static_cast<int>(LIBANE_LOG_INFO);
    m.attr("LOG_DEBUG")  = static_cast<int>(LIBANE_LOG_DEBUG);

    m.attr("__version__") = LIBANE_VERSION;

    /* ── from_torch Torch-FX tracer ─────────────────────────────────────── */
    // Injected as pure Python so it can use Python's string formatting,
    // comprehensions, and exception handling naturally.
    // Requires: torch (optional dep), numpy.
    py::exec(R"PY(
import operator as _op
import sys as _sys

def from_torch(module, example_input):
    """
    Trace a ``torch.nn.Module`` and compile it to an :class:`CompiledGraph`.

    Uses ``torch.fx.symbolic_trace`` to capture the computation graph, then
    ``ShapeProp`` to infer tensor shapes at each node, and finally maps each
    node to the corresponding libane op.

    **Supported operations** (everything else raises ``NotImplementedError``):

    +--------------------------------------+------------------+
    | PyTorch pattern                      | libane op        |
    +======================================+==================+
    | ``nn.Linear`` (bias=False)           | ``MATMUL``       |
    | ``nn.LayerNorm``                     | ``LAYERNORM``    |
    | ``F.gelu`` / ``F.silu`` / ``F.relu`` | ``GELU/SILU/RELU``|
    | ``F.softmax``                        | ``SOFTMAX``      |
    | ``a + b`` / ``a * b``               | ``ADD / MUL``    |
    | ``a @ b`` / ``torch.matmul``         | ``DYNAMIC_MATMUL``|
    | ``F.scaled_dot_product_attention``   | ``SDPA / SDPA_GQA``|
    +--------------------------------------+------------------+

    Tensor shapes must already use libane's tensor convention:
    ``[1, C, 1, S]`` for activations, or ``[1, H, S, D]`` for matrix tensors.

    Args:
        module:        ``torch.nn.Module`` to trace.
        example_input: A single ``torch.Tensor`` or a list/tuple of tensors
                       matching the module's positional inputs.  Used for
                       shape propagation — data values are not consumed.

    Returns:
        :class:`CompiledGraph` ready for repeated inference.

    Raises:
        ImportError:         if ``torch`` is not installed.
        NotImplementedError: for unsupported op patterns.
        RuntimeError:        if ANE compilation fails.

    Example — bias-free FFN block::

        import torch, torch.nn as nn, ane, numpy as np

        class FFN(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.w1 = nn.Linear(d, h, bias=False)
                self.w2 = nn.Linear(h, d, bias=False)
            def forward(self, x):
                return self.w2(torch.nn.functional.gelu(self.w1(x)))

        ffn = FFN(512, 2048).half().eval()
        x_ex = torch.zeros(1, 512, 1, 128, dtype=torch.float16)  # libane shape
        cg = ane.from_torch(ffn, x_ex)
        out = cg(x_ex.numpy())
    """
    try:
        import torch
        import torch.fx as fx
        from torch.fx.passes.shape_prop import ShapeProp
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "ane.from_torch requires PyTorch. Install it with: "
            "pip install torch"
        ) from exc
    import numpy as np
    import ane as _ane

    # ── 1. Symbolic trace ─────────────────────────────────────────────────
    traced = fx.symbolic_trace(module)

    # ── 2. Shape propagation ──────────────────────────────────────────────
    if isinstance(example_input, torch.Tensor):
        example_inputs = (example_input,)
    else:
        example_inputs = tuple(example_input)
    ShapeProp(traced).propagate(*example_inputs)

    # ── 3. Build libane graph ─────────────────────────────────────────────
    g = _ane.Graph()
    node_to_id = {}  # fx node name → libane tensor ID

    def _meta_shape(node):
        """Return the tensor shape stored by ShapeProp, as a list of ints."""
        meta = node.meta.get("tensor_meta")
        if meta is None:
            raise NotImplementedError(
                f"from_torch: ShapeProp did not infer shape for node "
                f"'{node.name}' ({node.op}). Make sure example_input covers "
                f"all live tensor paths.")
        return list(meta.shape)

    def _to_libane_shape(ts):
        """Coerce a torch shape to a 4-element libane shape list."""
        s = list(ts)
        if len(s) == 4:
            return s
        if len(s) == 3:
            return [s[0], s[1], s[2], 1]   # treat as [batch,H,S,1] — rare
        if len(s) == 2:
            return [1, s[0], 1, s[1]]       # standard [1,C,1,S]
        raise NotImplementedError(
            f"from_torch: cannot convert shape {s} (rank {len(s)}) to "
            f"libane format. Expected rank 2, 3, or 4.")

    def _node_id(arg):
        """Resolve an FX node argument to a libane tensor ID."""
        if isinstance(arg, fx.Node):
            if arg.name not in node_to_id:
                raise NotImplementedError(
                    f"from_torch: tensor '{arg.name}' not yet in libane graph "
                    f"(topological ordering problem?).")
            return node_to_id[arg.name]
        raise NotImplementedError(
            f"from_torch: non-node argument {arg!r} is not supported. "
            f"All operands must be live tensors (no Python scalars as inputs).")

    # The set of F.* functions that map to unary elementwise ops
    _UNARY_MAP = {
        F.gelu:     _ane.GELU,
        F.silu:     _ane.SILU,
        F.relu:     _ane.RELU,
        F.tanh:     _ane.TANH,
        F.sigmoid:  _ane.SIGMOID,
        F.softmax:  _ane.SOFTMAX,
    }
    # Binary +, * via operator module or torch
    _ADD_FNS = {torch.add, _op.add, _op.iadd}
    _MUL_FNS = {torch.mul, _op.mul, _op.imul}
    _MM_FNS  = {torch.matmul, torch.mm, _op.matmul}

    for node in traced.graph.nodes:

        # ── placeholder ────────────────────────────────────────────────────
        if node.op == "placeholder":
            shape = _to_libane_shape(_meta_shape(node))
            tid = g.add_input(node.name, shape)
            node_to_id[node.name] = tid

        # ── call_module ────────────────────────────────────────────────────
        elif node.op == "call_module":
            submod = traced.get_submodule(node.target)
            out_shape = _to_libane_shape(_meta_shape(node))

            if isinstance(submod, (torch.nn.Linear,)):
                if submod.bias is not None:
                    raise NotImplementedError(
                        f"from_torch: nn.Linear with bias is not supported "
                        f"(node '{node.name}'). Set bias=False or subtract "
                        f"the bias via a separate ADD node.")
                inp_id = _node_id(node.args[0])
                # Weight: PyTorch stores [OC, IC]; libane expects [IC, OC].
                W_fp16 = submod.weight.detach().to(torch.float16).numpy()  # [OC, IC]
                W_IC_OC = np.ascontiguousarray(W_fp16.T)  # [IC, OC]
                tid = g.add_op(_ane.MATMUL, [inp_id], out_shape, weights=W_IC_OC)

            elif isinstance(submod, torch.nn.LayerNorm):
                inp_id = _node_id(node.args[0])
                gamma = submod.weight.detach().to(torch.float16).numpy()
                beta  = submod.bias.detach().to(torch.float16).numpy()
                # libane LAYERNORM blob: gamma[C] + beta[C] (fp16, C×4 bytes total)
                weights = np.concatenate([gamma, beta])  # fp16 [2C]
                tid = g.add_op(_ane.LAYERNORM, [inp_id], out_shape, weights=weights)

            elif isinstance(submod, torch.nn.RMSNorm):
                inp_id = _node_id(node.args[0])
                weights = submod.weight.detach().to(torch.float16).numpy()
                tid = g.add_op(_ane.RMSNORM, [inp_id], out_shape, weights=weights)

            else:
                raise NotImplementedError(
                    f"from_torch: unsupported module type "
                    f"'{type(submod).__name__}' at node '{node.name}'. "
                    f"Supported: Linear (bias=False), LayerNorm, RMSNorm.")
            node_to_id[node.name] = tid

        # ── call_function ──────────────────────────────────────────────────
        elif node.op == "call_function":
            out_shape = _to_libane_shape(_meta_shape(node))
            fn = node.target

            if fn in _UNARY_MAP:
                tid = g.add_op(_UNARY_MAP[fn], [_node_id(node.args[0])], out_shape)

            elif fn in _ADD_FNS:
                tid = g.add_op(_ane.ADD, [_node_id(node.args[0]),
                                           _node_id(node.args[1])], out_shape)

            elif fn in _MUL_FNS:
                tid = g.add_op(_ane.MUL, [_node_id(node.args[0]),
                                           _node_id(node.args[1])], out_shape)

            elif fn in _MM_FNS:
                tid = g.add_op(_ane.DYNAMIC_MATMUL,
                               [_node_id(node.args[0]),
                                _node_id(node.args[1])],
                               out_shape)

            elif fn is F.scaled_dot_product_attention:
                q_id = _node_id(node.args[0])
                k_id = _node_id(node.args[1])
                v_id = _node_id(node.args[2])
                q_sh = _meta_shape(node.args[0])
                k_sh = _meta_shape(node.args[1])
                # GQA: Q and K differ in head count
                if len(q_sh) >= 2 and q_sh[-3] != k_sh[-3]:
                    tid = g.add_sdpa_gqa(q_id, k_id, v_id)
                else:
                    mask_arg = node.args[3] if len(node.args) > 3 else None
                    mask_id = _node_id(mask_arg) if isinstance(mask_arg, fx.Node) else -1
                    tid = g.add_sdpa(q_id, k_id, v_id, out_shape, mask_id)

            else:
                raise NotImplementedError(
                    f"from_torch: unsupported function '{getattr(fn, '__name__', fn)}' "
                    f"at node '{node.name}'. "
                    f"Supported: gelu, silu, relu, tanh, sigmoid, softmax, "
                    f"add, mul, matmul, scaled_dot_product_attention.")
            node_to_id[node.name] = tid

        # ── call_method ────────────────────────────────────────────────────
        elif node.op == "call_method":
            out_shape = _to_libane_shape(_meta_shape(node))
            method = node.target
            self_id = _node_id(node.args[0])

            if method in ("__add__", "add"):
                other_id = _node_id(node.args[1])
                tid = g.add_op(_ane.ADD, [self_id, other_id], out_shape)
            elif method in ("__mul__", "mul"):
                other_id = _node_id(node.args[1])
                tid = g.add_op(_ane.MUL, [self_id, other_id], out_shape)
            elif method in ("__matmul__", "matmul"):
                other_id = _node_id(node.args[1])
                tid = g.add_op(_ane.DYNAMIC_MATMUL, [self_id, other_id], out_shape)
            else:
                raise NotImplementedError(
                    f"from_torch: unsupported method '{method}' at node "
                    f"'{node.name}'. Supported: __add__, add, __mul__, mul, "
                    f"__matmul__, matmul.")
            node_to_id[node.name] = tid

        # ── output ─────────────────────────────────────────────────────────
        elif node.op == "output":
            args = node.args[0]
            if isinstance(args, (list, tuple)):
                for a in args:
                    if isinstance(a, fx.Node):
                        g.mark_output(node_to_id[a.name])
            elif isinstance(args, fx.Node):
                g.mark_output(node_to_id[args.name])
            # else: None output (unusual) — no output marking

        # ── get_attr ────────────────────────────────────────────────────────
        elif node.op == "get_attr":
            raise NotImplementedError(
                f"from_torch: raw parameter access via get_attr "
                f"('{node.target}') is not supported. Access weights through "
                f"an nn.Module submodule (call_module) instead.")

        # ── unknown ─────────────────────────────────────────────────────────
        else:
            raise NotImplementedError(
                f"from_torch: unknown FX node op '{node.op}' at node "
                f"'{node.name}'.")

    # ── 4. Compile ─────────────────────────────────────────────────────────
    cg = g.compile()
    if cg is None:
        raise RuntimeError(
            "from_torch: libane graph compilation failed. "
            "Check ane.last_error() for details.")
    return cg
)PY", m.attr("__dict__").cast<py::dict>());
}
