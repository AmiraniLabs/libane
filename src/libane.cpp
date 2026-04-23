/**
 * libane C API implementation.
 *
 * Ties together:
 *  - ANE runtime wrapper (ane_runtime)
 *  - MIL builder (mil_builder)
 *  - Compile cache (compile_cache)
 *  - IOSurface buffer manager (buffer_manager)
 *  - CPU fallback (fallback)
 */
#include "libane_internal.hpp"
#include "graph/mil_backend.hpp"

#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <fstream>

/* ── Global state ────────────────────────────────────────────────────────── */

static thread_local char tl_last_error[512] = "";
static std::atomic<libane_log_level_t> g_log_level{LIBANE_LOG_ERROR};
static std::atomic<int> g_force_backend{0}; // 0 = auto, 1 = ane, 2 = cpu

// Global compile cache (singleton)
static libane::CompileCache& cache() {
    static libane::CompileCache c;
    return c;
}

/* ── Logging ─────────────────────────────────────────────────────────────── */

static void libane_log(libane_log_level_t level, const char* fmt, ...) {
    if (level > g_log_level.load()) return;
    const char* prefix[] = { "", "[libane:ERROR]", "[libane:WARN]",
                              "[libane:INFO]",  "[libane:DEBUG]" };
    fprintf(stderr, "%s ", prefix[level]);
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void set_error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(tl_last_error, sizeof(tl_last_error), fmt, ap);
    va_end(ap);
    libane_log(LIBANE_LOG_ERROR, "%s", tl_last_error);
}

/* ── Backend selection ───────────────────────────────────────────────────── */

static bool use_ane() {
    int b = g_force_backend.load();
    if (b == 2) return false; // forced CPU
    if (b == 1) return libane::runtime::state() == libane::runtime::AneState::Available;
    return libane::runtime::state() == libane::runtime::AneState::Available;
}

/* ── Shape conversion helpers ────────────────────────────────────────────── */

static libane::mil::TensorShape to_mil_shape(const libane_shape_t& s) {
    libane::mil::TensorShape ms;
    ms.batch    = (s.ndim >= 1) ? s.dims[0] : 1;
    ms.channels = (s.ndim >= 2) ? s.dims[1] : 1;
    ms.height   = (s.ndim >= 3) ? s.dims[2] : 1;
    ms.seq      = (s.ndim >= 4) ? s.dims[3] : 8;
    return ms;
}

/**
 * Pack row-major matrix A[M,K] into ANE logical [K,M] channel-major layout.
 * ANE linear index is [channel * S + seq] => [k * M + m].
 */
static void pack_matmul_input_for_ane(const libane_f16_t* A_row_major,
                                      libane_f16_t*       A_ane,
                                      int M, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            // A_ane[k, m] = A_row_major[m, k]
            A_ane[static_cast<size_t>(k) * M + m] =
                A_row_major[static_cast<size_t>(m) * K + k];
        }
    }
}

/**
 * Unpack ANE logical output [N,M] channel-major into row-major C[M,N].
 * ANE linear index is [n * M + m].
 */
static void unpack_matmul_output_from_ane(const libane_f16_t* Y_ane,
                                          libane_f16_t*       C_row_major,
                                          int M, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            // C_row_major[m, n] = Y_ane[n, m]
            C_row_major[static_cast<size_t>(m) * N + n] =
                Y_ane[static_cast<size_t>(n) * M + m];
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

extern "C" {

/* ── Status ──────────────────────────────────────────────────────────────── */

int libane_available(void) {
    libane::runtime::initialize();
    return libane::runtime::state() == libane::runtime::AneState::Available ? 1 : 0;
}

const char* libane_version(void) {
    return LIBANE_VERSION;
}

const char* libane_last_error(void) {
    return tl_last_error;
}

void libane_set_log_level(libane_log_level_t level) {
    g_log_level = level;
}

void libane_set_backend(const char* backend) {
    if (!backend) {
        g_force_backend = 0;
    } else if (std::strcmp(backend, "ane") == 0) {
        g_force_backend = 1;
        libane::runtime::initialize();
    } else if (std::strcmp(backend, "cpu") == 0) {
        g_force_backend = 2;
    } else {
        set_error("Unknown backend '%s', use 'ane', 'cpu', or NULL", backend);
    }
}

void libane_cache_flush(void) {
    cache().flush();
}

size_t libane_cache_size_bytes(void) {
    return cache().size_bytes();
}

/* ── Compile budget ──────────────────────────────────────────────────────── */

int libane_compile_count(void) {
    libane::runtime::initialize();
    return libane::runtime::ane_compile_count();
}

int libane_compile_slots_remaining(void) {
    libane::runtime::initialize();
    return libane::runtime::ane_compile_slots_remaining();
}

libane_status_t libane_cache_stats(libane_cache_stats_t* out) {
    if (!out) { set_error("libane_cache_stats: out is NULL"); return LIBANE_ERR_INVALID_ARG; }
    libane::runtime::initialize();
    auto s = libane::graph::GraphCompiler::thread_mil_backend().cache_stats();
    out->entries       = s.entries;
    out->capacity      = s.capacity;
    out->bytes         = s.bytes;
    out->hits          = s.hits;
    out->misses        = s.misses;
    out->cold_compiles = s.cold_compiles;
    out->evictions     = s.evictions;
    out->lru_evictions = s.lru_evictions;
    return LIBANE_OK;
}

void libane_cache_clear(void) {
    libane::runtime::initialize();
    libane::graph::GraphCompiler::thread_mil_backend().cache_clear();
}

size_t libane_cache_prune(size_t max_entries) {
    libane::runtime::initialize();
    return libane::graph::GraphCompiler::thread_mil_backend().cache_prune(max_entries);
}

void libane_cache_set_capacity(size_t capacity) {
    libane::runtime::initialize();
    libane::graph::GraphCompiler::thread_mil_backend().set_cache_capacity(capacity);
}

/* ── Device introspection ────────────────────────────────────────────────── */

libane_status_t libane_device_info(libane_device_info_t* out) {
    if (!out) {
        set_error("libane_device_info: null output pointer");
        return LIBANE_ERR_INVALID_ARG;
    }
    libane::runtime::initialize();
    auto di = libane::runtime::device_info();
    std::memcpy(out->architecture, di.architecture, sizeof(out->architecture));
    out->core_count = di.core_count;
    out->num_anes   = di.num_anes;
    out->available  = di.available ? 1 : 0;
    return LIBANE_OK;
}

libane_shape_limits_t libane_get_shape_limits(void) {
    libane::runtime::initialize();
    auto di = libane::runtime::device_info();

    libane_shape_limits_t lim;
    lim.seq_alignment = 32;  // constant: ANE activation-output stride constraint

    // Chip-adaptive limits.  Conservative universally-safe values are used
    // when device info is unavailable or the architecture is unrecognised.
    // h16g (M4 family) has demonstrated stable operation at 2× the M3 limits.
    if (di.available && di.architecture[0] != '\0') {
        // h16g = M4 — relaxed limits confirmed via external research
        if (std::strncmp(di.architecture, "h16", 3) == 0) {
            lim.max_seq      = 131072;
            lim.max_channels = 16384;
            return lim;
        }
    }
    // Conservative default (safe for h15g / M3 and earlier)
    lim.max_seq      = 65536;
    lim.max_channels = 16384;
    return lim;
}

/* ── Compile ─────────────────────────────────────────────────────────────── */

libane_handle_t libane_compile(libane_op_t op,
                               libane_shape_t shape,
                               const void* weights,
                               size_t weights_len) {
    // Ensure runtime is initialized
    libane::runtime::initialize();

    // Build cache key
    libane::mil::WeightBlob weight_blob;
    if (weights && weights_len > 0)
        weight_blob = libane::mil::WeightBlob::from_fp16(weights, weights_len);

    libane::CacheKey key;
    key.op          = op;
    key.shape       = shape;
    key.weight_hash = weight_blob.hash;

    // Check cache
    auto cached = cache().get(key);
    if (cached) {
        auto* h = new libane_program_s{};
        h->entry = cached;
        h->op    = op;
        h->shape = shape;
        libane_log(LIBANE_LOG_DEBUG, "cache hit op=%d", (int)op);
        return h;
    }

    // Validate shape
    libane::mil::TensorShape ms;
    try {
        ms = to_mil_shape(shape);
        ms.validate();
    } catch (const std::exception& e) {
        set_error("Invalid shape: %s", e.what());
        return nullptr;
    }

    // Build MIL program (only needed for ANE path)
    libane::mil::MilProgram mil_prog;
    bool try_ane = use_ane();

    if (try_ane) {
        try {
            switch (op) {
            case LIBANE_OP_MATMUL:
                // libane_compile doesn't have separate IC/OC; use conv1x1 with
                // shape as [1, IC=C, 1, S]. Weight must be square for this path.
                mil_prog = libane::mil::MilBuilder::matmul_conv1x1(
                    ms.channels, ms.channels, ms.seq);
                break;
            case LIBANE_OP_SOFTMAX:
                mil_prog = libane::mil::MilBuilder::softmax(ms.channels, ms.seq);
                break;
            case LIBANE_OP_AVG_POOL:
                mil_prog = libane::mil::MilBuilder::avg_pool(ms.channels, ms.seq);
                break;
            case LIBANE_OP_MAX_POOL:
                mil_prog = libane::mil::MilBuilder::max_pool(ms.channels, ms.seq);
                break;
            case LIBANE_OP_GELU:
                mil_prog = libane::mil::MilBuilder::gelu(ms.channels, ms.seq);
                break;
            case LIBANE_OP_ADD:
                mil_prog = libane::mil::MilBuilder::add(ms.channels, ms.seq);
                break;
            case LIBANE_OP_MUL:
                mil_prog = libane::mil::MilBuilder::mul(ms.channels, ms.seq);
                break;
            case LIBANE_OP_LOGICAL_AND:
                mil_prog = libane::mil::MilBuilder::logical_and(ms.channels, ms.seq);
                break;
            case LIBANE_OP_LOGICAL_OR:
                mil_prog = libane::mil::MilBuilder::logical_or(ms.channels, ms.seq);
                break;
            case LIBANE_OP_LOGICAL_XOR:
                mil_prog = libane::mil::MilBuilder::logical_xor(ms.channels, ms.seq);
                break;
            case LIBANE_OP_REDUCE_PROD:
                set_error("REDUCE_PROD not supported via libane_compile — use graph API with output shape [1,1,1,S]");
                return nullptr;
            case LIBANE_OP_SCATTER:
                set_error("SCATTER not supported via libane_compile — use graph API with static mask weights");
                return nullptr;
            case LIBANE_OP_SCATTER_ND:
                set_error("SCATTER_ND not supported via libane_compile — use graph API with static mask weights");
                return nullptr;
            case LIBANE_OP_SCATTER_ALONG_AXIS:
                set_error("SCATTER_ALONG_AXIS not supported via libane_compile — use graph API with static mask weights");
                return nullptr;
            case LIBANE_OP_GATHER:
                set_error("GATHER not supported via libane_compile — use graph API (static or dynamic mask)");
                return nullptr;
            case LIBANE_OP_NEG:
                set_error("NEG not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_MOD:
                set_error("MOD not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_SINH:
                set_error("SINH not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_COSH:
                set_error("COSH not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_TAN:
                set_error("TAN not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_ASIN:
                set_error("ASIN not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_ACOS:
                set_error("ACOS not supported via libane_compile — use graph API");
                return nullptr;
            case LIBANE_OP_LAYER_NORM:
            case LIBANE_OP_LAYERNORM:
                mil_prog = libane::mil::MilBuilder::layernorm(ms.channels, ms.seq);
                break;
            case LIBANE_OP_TRANSPOSE:
                mil_prog = libane::mil::MilBuilder::transpose_cssc(ms.channels, ms.seq);
                break;
            case LIBANE_OP_SILU:
                mil_prog = libane::mil::MilBuilder::silu(ms.channels, ms.seq);
                break;
            case LIBANE_OP_RMSNORM:
                mil_prog = libane::mil::MilBuilder::rmsnorm(ms.channels, ms.seq);
                break;
            case LIBANE_OP_CONV2D:
                set_error("CONV2D not supported via libane_compile — use libane_matmul_f16 for projections");
                return nullptr;
            case LIBANE_OP_CAST:
                set_error("CAST not supported via libane_compile");
                return nullptr;
            case LIBANE_OP_RESHAPE:
                set_error("RESHAPE not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_CONCAT:
                set_error("CONCAT not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_SLICE_BY_INDEX:
                set_error("SLICE_BY_INDEX not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_REDUCE_SUM:
                set_error("REDUCE_SUM not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_REDUCE_MEAN:
                set_error("REDUCE_MEAN not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_REDUCE_MAX:
                set_error("REDUCE_MAX not supported via libane_compile (use graph API)");
                return nullptr;
            case LIBANE_OP_SELECT:
                set_error("SELECT not supported via libane_compile — use graph API with 3 inputs (condition, x, y)");
                return nullptr;
            default:
                set_error("op %d not supported", (int)op);
                return nullptr;
            }
        } catch (const std::exception& e) {
            set_error("MIL build failed: %s", e.what());
            return nullptr;
        }
    } else {
        // ANE unavailable — return nullptr (no CPU fallback for compile path)
        set_error("ANE unavailable: %s", libane::runtime::fallback_reason());
        return nullptr;
    }

    // Attempt ANE compilation
    libane::runtime::AneProgram* ane_prog = nullptr;
    if (!mil_prog.text.empty()) {
        std::vector<libane::runtime::WeightEntry> wentries;
        if (!mil_prog.weight_name.empty() && !weight_blob.data.empty())
            wentries.push_back({mil_prog.weight_name, weight_blob.data});

        ane_prog = libane::runtime::ane_compile(
            mil_prog.text,
            wentries,
            "libane_op_" + std::to_string((int)op));
        if (!ane_prog) {
            set_error("ANE compile failed: %s", libane::runtime::ane_last_error());
            return nullptr;
        }
    }

    // Store in cache
    auto entry = std::make_unique<libane::CacheEntry>();
    entry->key        = key;
    entry->is_ane     = (ane_prog != nullptr);
    entry->size_bytes = weights_len + mil_prog.text.size() + 4096; // approx

    if (ane_prog) {
        entry->backend_handle = ane_prog;
        entry->release_fn = [](void* p) {
            libane::runtime::ane_unload(static_cast<libane::runtime::AneProgram*>(p));
        };
    }

    auto stored = cache().put(std::move(entry));

    auto* h = new libane_program_s{};
    h->entry = stored;
    h->op    = op;
    h->shape = shape;
    libane_log(LIBANE_LOG_DEBUG,
               "compiled op=%d backend=ANE", (int)op);
    return h;
}

/**
 * Compile multiple operations in a single call.
 * Each request is compiled independently; if one fails, others may still succeed.
 */
libane_status_t libane_compile_batch(const libane_compile_request_t* requests,
                                     size_t num_requests,
                                     libane_handle_t* out_handles) {
    if (!requests || !out_handles || num_requests == 0) {
        set_error("libane_compile_batch: null argument or num_requests == 0");
        return LIBANE_ERR_INVALID_ARG;
    }

    bool all_success = true;
    for (size_t i = 0; i < num_requests; ++i) {
        libane_handle_t h = libane_compile(
            requests[i].op,
            requests[i].shape,
            requests[i].weights,
            requests[i].weights_len);
        out_handles[i] = h;
        if (!h) {
            all_success = false;
            libane_log(LIBANE_LOG_WARN,
                      "libane_compile_batch: request %zu failed: %s",
                      i, libane_last_error());
        }
    }

    if (all_success) {
        libane_log(LIBANE_LOG_DEBUG, "compile_batch: all %zu requests succeeded", num_requests);
        return LIBANE_OK;
    } else {
        libane_log(LIBANE_LOG_WARN, "compile_batch: %zu requests completed with some failures", num_requests);
        return LIBANE_ERR_COMPILE_FAILED;
    }
}

/* ── Execute ─────────────────────────────────────────────────────────────── */

libane_status_t libane_execute(libane_handle_t h,
                               const void* input,
                               void* output,
                               libane_shape_t shape) {
    if (!h || !input || !output) {
        set_error("libane_execute: null argument");
        return LIBANE_ERR_INVALID_ARG;
    }

    auto& entry = *h->entry;
    libane::mil::TensorShape ms = to_mil_shape(shape);
    size_t numel = static_cast<size_t>(ms.batch) * ms.channels * ms.height * ms.seq;

    // ANE path
    if (entry.is_ane && entry.backend_handle) {
#ifdef __APPLE__
        auto* prog = static_cast<libane::runtime::AneProgram*>(entry.backend_handle);
        auto& pool = libane::global_buffer_pool();

        PooledBuffer in_buf(pool.acquire_with_data(input, numel * 2), pool);
        PooledBuffer out_buf(pool.acquire(numel * 2), pool);

        bool ok = libane::runtime::ane_execute(prog,
                                               in_buf->iosurface(),
                                               out_buf->iosurface());
        if (ok) {
            out_buf->copy_to(output, numel * 2);
            return LIBANE_OK;
        }

        set_error("ANE execute failed: %s", libane::runtime::ane_last_error());
        return LIBANE_ERR_EXECUTE_FAILED;
#endif
    }

    set_error("libane_execute: no ANE program available");
    return LIBANE_ERR_UNAVAILABLE;
}

/* ── Execute2 ────────────────────────────────────────────────────────────── */

libane_status_t libane_execute2(libane_handle_t h,
                                const void* input0,
                                const void* input1,
                                void* output,
                                libane_shape_t shape) {
    if (!h || !input0 || !input1 || !output) {
        set_error("libane_execute2: null argument");
        return LIBANE_ERR_INVALID_ARG;
    }

    auto& entry = *h->entry;
    libane::mil::TensorShape ms = to_mil_shape(shape);
    size_t numel = static_cast<size_t>(ms.batch) * ms.channels * ms.height * ms.seq;
    size_t bytes = numel * 2;

    if (entry.is_ane && entry.backend_handle) {
#ifdef __APPLE__
        auto* prog = static_cast<libane::runtime::AneProgram*>(entry.backend_handle);
        auto& pool = libane::global_buffer_pool();

        PooledBuffer in0_buf(pool.acquire_with_data(input0, bytes), pool);
        PooledBuffer in1_buf(pool.acquire_with_data(input1, bytes), pool);
        PooledBuffer out_buf(pool.acquire(bytes), pool);

        bool ok = libane::runtime::ane_execute_multi(
            prog,
            {in0_buf->iosurface(), in1_buf->iosurface()},
            {out_buf->iosurface()});

        if (ok) {
            out_buf->copy_to(output, bytes);
            return LIBANE_OK;
        }

        set_error("ANE execute2 failed: %s", libane::runtime::ane_last_error());
        return LIBANE_ERR_EXECUTE_FAILED;
#endif
    }

    set_error("libane_execute2: no ANE program available");
    return LIBANE_ERR_UNAVAILABLE;
}

/* ── Delta reload ────────────────────────────────────────────────────────── */

libane_status_t libane_delta_reload(libane_handle_t h) {
    if (!h) {
        set_error("libane_delta_reload: null handle");
        return LIBANE_ERR_INVALID_ARG;
    }
    auto& entry = *h->entry;
    if (!entry.is_ane || !entry.backend_handle) {
        set_error("libane_delta_reload: handle is not ANE-compiled");
        return LIBANE_ERR_UNAVAILABLE;
    }

    auto* prog = static_cast<libane::runtime::AneProgram*>(entry.backend_handle);
    bool ok = libane::runtime::ane_delta_reload(prog);
    if (!ok) {
        set_error("libane_delta_reload failed: %s", libane::runtime::ane_last_error());
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

/* ── SRAM-only unload ────────────────────────────────────────────────────── */

void libane_unload_sram(libane_handle_t h) {
    if (!h) return;
    auto& entry = *h->entry;
    if (!entry.is_ane || !entry.backend_handle) return;
    auto* prog = static_cast<libane::runtime::AneProgram*>(entry.backend_handle);
    libane::runtime::ane_unload_sram(prog);
}

/* ── Release ─────────────────────────────────────────────────────────────── */

void libane_release(libane_handle_t h) {
    if (!h) return;
    // Unpin this program from the global cache so release can deterministically
    // drive ANE teardown once the final handle reference is dropped.
    cache().erase(h->entry->key);
    delete h;
}

void libane_end_job(void) {
    // Explicit lifecycle boundary: callers can mark end-of-batch/end-of-job to
    // force release of any idle cached ANE programs.
    cache().flush();
}

/* ── Convenience matmul ──────────────────────────────────────────────────── */

libane_status_t libane_matmul_f16(const libane_f16_t* A,
                                  const libane_f16_t* B,
                                  libane_f16_t* C,
                                  int M, int K, int N) {
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) {
        set_error("libane_matmul_f16: invalid arguments");
        return LIBANE_ERR_INVALID_ARG;
    }

    libane::runtime::initialize();

    // Try ANE if available and shapes meet constraints
    bool shapes_ok = (M % 8 == 0) && (N % 8 == 0) && (K % 8 == 0)
                     && (M <= 65536) && (N <= 65536) && (K <= 16384);

    if (use_ane() && shapes_ok) {
        // Build weight blob (B matrix)
        auto weight_blob = libane::mil::WeightBlob::from_fp16(B,
                               static_cast<size_t>(K) * N * 2);

        // Build shape descriptors
        // Cache key for this exact matmul
        libane_shape_t cs;
        cs.dims[0] = 1; cs.dims[1] = N; cs.dims[2] = 1; cs.dims[3] = M; cs.ndim = 4;

        libane::CacheKey key;
        key.op          = LIBANE_OP_MATMUL;
        key.shape       = cs;
        key.weight_hash = weight_blob.hash;

        auto cached = cache().get(key);
        libane::runtime::AneProgram* prog = nullptr;
        std::shared_ptr<libane::CacheEntry> entry_ptr;

        if (cached && cached->is_ane && cached->backend_handle) {
            prog = static_cast<libane::runtime::AneProgram*>(cached->backend_handle);
            entry_ptr = cached;
        } else if (!cached) {
            try {
                // Build MIL text program: conv1x1 maps [1,K,1,M] → [1,N,1,M]
                auto mil_prog = libane::mil::MilBuilder::matmul_conv1x1(K, N, M);

                // Build weight blob from B (fp16 [K,N]) transposed to [N,K] for conv1x1
                // Promote to fp32 for the transpose, then back to fp16 in blob
                std::vector<float> B_f32(static_cast<size_t>(K) * N);
                const auto* B_h = reinterpret_cast<const uint16_t*>(B);
                for (int i = 0; i < K * N; ++i) {
                    uint16_t h = B_h[i];
                    uint32_t s = (h & 0x8000) << 16;
                    uint32_t e = ((h >> 10) & 0x1F);
                    uint32_t m = (h & 0x3FF);
                    uint32_t fb;
                    if (e == 0)       fb = s | (m << 13);
                    else if (e == 31) fb = s | 0x7F800000 | (m << 13);
                    else              fb = s | ((e + 112) << 23) | (m << 13);
                    std::memcpy(&B_f32[i], &fb, 4);
                }
                // from_fp32 with transpose=true: [K,N] → stores as [N,K] (OC,IC layout)
                auto wblob = libane::mil::WeightBlob::from_fp32(B_f32.data(), K, N, true);

                std::vector<libane::runtime::WeightEntry> wentries;
                wentries.push_back({mil_prog.weight_name, wblob.data});

                auto* ane_prog = libane::runtime::ane_compile(
                    mil_prog.text, wentries, "matmul_f16");

                auto e = std::make_unique<libane::CacheEntry>();
                e->key        = key;
                e->is_ane     = (ane_prog != nullptr);
                e->size_bytes = static_cast<size_t>(K) * N * 2 + mil_prog.text.size() + 4096;
                if (ane_prog) {
                    e->backend_handle = ane_prog;
                    e->release_fn = [](void* p) {
                        libane::runtime::ane_unload(
                            static_cast<libane::runtime::AneProgram*>(p));
                    };
                    prog = ane_prog;
                }
                entry_ptr = cache().put(std::move(e));
            } catch (const std::exception& ex) {
                libane_log(LIBANE_LOG_DEBUG, "matmul_f16 ANE compile failed: %s", ex.what());
            }
        }

        if (prog) {
#ifdef __APPLE__
            size_t in_bytes  = static_cast<size_t>(M) * K * 2;
            size_t out_bytes = static_cast<size_t>(M) * N * 2;

            // Bridge row-major API tensors <-> ANE [1,C,1,S] channel-major buffers.
            std::vector<libane_f16_t> A_ane(static_cast<size_t>(M) * K);
            std::vector<libane_f16_t> Y_ane(static_cast<size_t>(M) * N);
            pack_matmul_input_for_ane(A, A_ane.data(), M, K);

            auto& pool = libane::global_buffer_pool();
            PooledBuffer in_buf(pool.acquire_with_data(A_ane.data(), in_bytes), pool);
            PooledBuffer out_buf(pool.acquire(out_bytes), pool);

            bool ok = libane::runtime::ane_execute(prog,
                                                    in_buf->iosurface(),
                                                    out_buf->iosurface());
            if (ok) {
                out_buf->copy_to(Y_ane.data(), out_bytes);
                unpack_matmul_output_from_ane(Y_ane.data(), C, M, N);
                return LIBANE_OK;
            }
#endif
        }
    }

    // CPU fallback
    libane_log(LIBANE_LOG_DEBUG, "matmul_f16 [%d×%d × %d×%d] via CPU fallback", M, K, K, N);
    // fp16_t from libane_internal.hpp
    libane::fallback::matmul_f16(
        reinterpret_cast<const fp16_t*>(A),
        reinterpret_cast<const fp16_t*>(B),
        reinterpret_cast<fp16_t*>(C),
        M, K, N);
    return LIBANE_OK;
}

libane_status_t libane_matmul_f32(const float* A, const float* B, float* C,
                                  int M, int K, int N) {
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) {
        set_error("libane_matmul_f32: invalid arguments");
        return LIBANE_ERR_INVALID_ARG;
    }

    libane::runtime::initialize();

    bool shapes_ok = (M % 8 == 0) && (N % 8 == 0) && (K % 8 == 0)
                     && (M <= 65536) && (N <= 65536) && (K <= 16384);

    if (use_ane() && shapes_ok) {
        // Cast to fp16 and dispatch via ANE
        std::vector<fp16_t> A16(static_cast<size_t>(M) * K);
        std::vector<fp16_t> B16(static_cast<size_t>(K) * N);
        std::vector<fp16_t> C16(static_cast<size_t>(M) * N);

        libane::fallback::cast_f32_to_f16(A, A16.data(), A16.size());
        libane::fallback::cast_f32_to_f16(B, B16.data(), B16.size());

        libane_status_t st = libane_matmul_f16(
            reinterpret_cast<const libane_f16_t*>(A16.data()),
            reinterpret_cast<const libane_f16_t*>(B16.data()),
            reinterpret_cast<libane_f16_t*>(C16.data()),
            M, K, N);

        if (st == LIBANE_OK) {
            libane::fallback::cast_f16_to_f32(C16.data(), C, C16.size());
            return LIBANE_OK;
        }
    }

    // Direct fp32 BLAS
    libane::fallback::matmul_f32(A, B, C, M, K, N);
    return LIBANE_OK;
}

/* ── Graph API ───────────────────────────────────────────────────────────── */

libane_graph_t libane_graph_create(void) {
    auto* g = new (std::nothrow) libane_graph_s{};
    if (!g) set_error("libane_graph_create: out of memory");
    return g;
}

void libane_graph_release(libane_graph_t g) {
    delete g;
}

uint32_t libane_graph_add_input(libane_graph_t g,
                                 const char*    name,
                                 libane_shape_t shape) {
    if (!g) {
        set_error("libane_graph_add_input: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    try {
        libane::mil::TensorShape ms = to_mil_shape(shape);
        return g->graph.add_input(name ? name : "", ms);
    } catch (const std::exception& e) {
        set_error("libane_graph_add_input: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

uint32_t libane_graph_add_op(libane_graph_t  g,
                              libane_op_t     op,
                              const uint32_t* input_ids,
                              size_t          num_inputs,
                              libane_shape_t  output_shape,
                              const void*     weights,
                              size_t          weights_len) {
    if (!g) {
        set_error("libane_graph_add_op: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (num_inputs > 0 && !input_ids) {
        set_error("libane_graph_add_op: null input_ids");
        return LIBANE_INVALID_TENSOR_ID;
    }
    try {
        std::vector<libane::graph::TensorId> ids(input_ids, input_ids + num_inputs);
        libane::mil::TensorShape ms = to_mil_shape(output_shape);
        return g->graph.add_op(op, std::move(ids), ms, weights, weights_len);
    } catch (const std::exception& e) {
        set_error("libane_graph_add_op: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

uint32_t libane_graph_add_pwl_activation(libane_graph_t g,
                                          uint32_t       input_id,
                                          libane_shape_t output_shape,
                                          float          x_min,
                                          float          x_max,
                                          const float*   samples,
                                          uint32_t       n_samples) {
    if (!g) {
        set_error("libane_graph_add_pwl_activation: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (!samples || n_samples < 2) {
        set_error("libane_graph_add_pwl_activation: need at least 2 sample points");
        return LIBANE_INVALID_TENSOR_ID;
    }
    // Encode: [x_min, x_max, samples[0..n_samples-1]] as float32 array
    std::vector<float> packed;
    packed.reserve(2 + n_samples);
    packed.push_back(x_min);
    packed.push_back(x_max);
    for (uint32_t i = 0; i < n_samples; ++i)
        packed.push_back(samples[i]);
    try {
        libane::mil::TensorShape ms = to_mil_shape(output_shape);
        return g->graph.add_op(LIBANE_OP_PWL_ACTIVATION, {input_id}, ms,
                               packed.data(), packed.size() * sizeof(float));
    } catch (const std::exception& e) {
        set_error("libane_graph_add_pwl_activation: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

uint32_t libane_graph_add_conv2d(libane_graph_t g,
                                  uint32_t       input_id,
                                  libane_shape_t output_shape,
                                  int            kH,
                                  int            kW,
                                  int            stride_h,
                                  int            stride_w,
                                  int            pad_top,
                                  int            pad_left,
                                  int            pad_bottom,
                                  int            pad_right,
                                  int            dilation_h,
                                  int            dilation_w,
                                  int            groups,
                                  const void*    kernel,
                                  size_t         kernel_bytes) {
    if (!g) {
        set_error("libane_graph_add_conv2d: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    // Basic parameter sanity checks (full semantic validation happens in
    // graph_validator during compile — these guards catch obvious misuse early).
    if (kH <= 0 || kW <= 0) {
        set_error("libane_graph_add_conv2d: kH and kW must be >= 1");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (stride_h <= 0 || stride_w <= 0) {
        set_error("libane_graph_add_conv2d: strides must be >= 1");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (dilation_h <= 0 || dilation_w <= 0) {
        set_error("libane_graph_add_conv2d: dilations must be >= 1");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (groups <= 0) {
        set_error("libane_graph_add_conv2d: groups must be >= 1");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (pad_top < 0 || pad_left < 0 || pad_bottom < 0 || pad_right < 0) {
        set_error("libane_graph_add_conv2d: padding values must be >= 0");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (!kernel || kernel_bytes == 0) {
        set_error("libane_graph_add_conv2d: kernel must be non-null and non-empty");
        return LIBANE_INVALID_TENSOR_ID;
    }

    // Pack: int32[11] header + raw fp16 kernel bytes.
    // Header layout mirrors LIBANE_OP_CONV2D blob spec in libane.h:
    //   {kH, kW, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right,
    //    dilation_h, dilation_w, groups}
    constexpr size_t kNumParams  = 11;
    constexpr size_t kParamBytes = kNumParams * sizeof(int32_t);

    std::vector<uint8_t> blob;
    blob.reserve(kParamBytes + kernel_bytes);
    blob.resize(kParamBytes);

    int32_t params[kNumParams] = {
        kH, kW, stride_h, stride_w,
        pad_top, pad_left, pad_bottom, pad_right,
        dilation_h, dilation_w, groups
    };
    std::memcpy(blob.data(), params, kParamBytes);

    // Append raw fp16 kernel
    const auto* kptr = static_cast<const uint8_t*>(kernel);
    blob.insert(blob.end(), kptr, kptr + kernel_bytes);

    try {
        libane::mil::TensorShape ms = to_mil_shape(output_shape);
        return g->graph.add_op(LIBANE_OP_CONV2D, {input_id}, ms,
                               blob.data(), blob.size());
    } catch (const std::exception& e) {
        set_error("libane_graph_add_conv2d: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

uint32_t libane_graph_add_matmul_w8a16(libane_graph_t g,
                                        uint32_t       input_id,
                                        libane_shape_t output_shape,
                                        const int8_t*  weights,
                                        const float*   scales,
                                        int            IC,
                                        int            OC) {
    if (!g) {
        set_error("libane_graph_add_matmul_w8a16: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (!weights || !scales) {
        set_error("libane_graph_add_matmul_w8a16: null weights or scales");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (IC <= 0 || OC <= 0) {
        set_error("libane_graph_add_matmul_w8a16: IC and OC must be positive");
        return LIBANE_INVALID_TENSOR_ID;
    }

    // Pack blob: int32[2]={OC,IC} + int8[IC×OC] + pad-to-4 + float32[OC]
    const size_t kHdrBytes    = 2 * sizeof(int32_t);
    const size_t wbytes       = static_cast<size_t>(IC) * OC;
    const size_t scales_start = (kHdrBytes + wbytes + 3) & ~size_t(3);
    const size_t total_bytes  = scales_start + static_cast<size_t>(OC) * sizeof(float);

    std::vector<uint8_t> blob(total_bytes, 0);
    int32_t hdr[2] = { OC, IC };
    std::memcpy(blob.data(),              hdr,     kHdrBytes);
    std::memcpy(blob.data() + kHdrBytes,  weights, wbytes);
    std::memcpy(blob.data() + scales_start, scales,
                static_cast<size_t>(OC) * sizeof(float));

    try {
        libane::mil::TensorShape ms = to_mil_shape(output_shape);
        return g->graph.add_op(LIBANE_OP_MATMUL_W8A16, {input_id}, ms,
                               blob.data(), blob.size());
    } catch (const std::exception& e) {
        set_error("libane_graph_add_matmul_w8a16: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

uint32_t libane_graph_add_matmul_w8a8(libane_graph_t g,
                                       uint32_t       input_id,
                                       libane_shape_t output_shape,
                                       const int8_t*  weights,
                                       const float*   scales,
                                       int            IC,
                                       int            OC,
                                       float          act_scale,
                                       int32_t        act_zero_point) {
    if (!g) {
        set_error("libane_graph_add_matmul_w8a8: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (!weights || !scales) {
        set_error("libane_graph_add_matmul_w8a8: null weights or scales");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (IC <= 0 || OC <= 0) {
        set_error("libane_graph_add_matmul_w8a8: IC and OC must be positive");
        return LIBANE_INVALID_TENSOR_ID;
    }
    if (act_scale <= 0.0f) {
        set_error("libane_graph_add_matmul_w8a8: act_scale must be positive");
        return LIBANE_INVALID_TENSOR_ID;
    }

    // Blob format: int32[2]={OC,IC} + int8[IC×OC] + pad-to-4
    //              + float32[OC] (weight scales) + float32[1] (act_scale) + int32[1] (act_zp)
    const size_t kHdrBytes    = 2 * sizeof(int32_t);
    const size_t wbytes       = static_cast<size_t>(IC) * OC;
    const size_t scales_start = (kHdrBytes + wbytes + 3) & ~size_t(3);
    const size_t act_off      = scales_start + static_cast<size_t>(OC) * sizeof(float);
    const size_t total_bytes  = act_off + sizeof(float) + sizeof(int32_t);

    std::vector<uint8_t> blob(total_bytes, 0);
    int32_t hdr[2] = { OC, IC };
    std::memcpy(blob.data(),                hdr,      kHdrBytes);
    std::memcpy(blob.data() + kHdrBytes,    weights,  wbytes);
    std::memcpy(blob.data() + scales_start, scales,   static_cast<size_t>(OC) * sizeof(float));
    std::memcpy(blob.data() + act_off,      &act_scale,      sizeof(float));
    std::memcpy(blob.data() + act_off + 4,  &act_zero_point, sizeof(int32_t));

    try {
        libane::mil::TensorShape ms = to_mil_shape(output_shape);
        return g->graph.add_op(LIBANE_OP_MATMUL_W8A8, {input_id}, ms,
                               blob.data(), blob.size());
    } catch (const std::exception& e) {
        set_error("libane_graph_add_matmul_w8a8: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

void libane_quantize_i8(const void* in_fp16,
                         int8_t*     out_i8,
                         size_t      n,
                         float       act_scale,
                         int32_t     act_zero_point) {
    if (!in_fp16 || !out_i8 || n == 0 || act_scale <= 0.0f) return;
    const auto* src = static_cast<const fp16_t*>(in_fp16);
    for (size_t i = 0; i < n; ++i) {
        float f;
        libane::fallback::cast_f16_to_f32(src + i, &f, 1);
        float q  = f / act_scale + static_cast<float>(act_zero_point);
        int   qi = static_cast<int>(std::round(q));
        if (qi < -128) qi = -128;
        if (qi >  127) qi =  127;
        out_i8[i] = static_cast<int8_t>(qi);
    }
}

/* ── KV Cache ────────────────────────────────────────────────────────────── */

libane_kv_cache_t libane_kv_cache_create(int num_heads, int head_dim, int max_seq) {
    if (num_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
        set_error("libane_kv_cache_create: all dimensions must be > 0");
        return nullptr;
    }
    auto* c = new (std::nothrow) libane_kv_cache_s;
    if (!c) {
        set_error("libane_kv_cache_create: out of memory");
        return nullptr;
    }
    c->num_heads = num_heads;
    c->head_dim  = head_dim;
    c->max_seq   = max_seq;
    c->pos       = 0;
    try {
        c->k_buf.assign(c->total_elems(), 0);
        c->v_buf.assign(c->total_elems(), 0);
    } catch (const std::bad_alloc&) {
        delete c;
        set_error("libane_kv_cache_create: out of memory for buffers");
        return nullptr;
    }
    return c;
}

int libane_kv_cache_update(libane_kv_cache_t cache,
                            const void*       new_k,
                            const void*       new_v) {
    if (!cache || !new_k || !new_v) {
        set_error("libane_kv_cache_update: null argument");
        return -1;
    }
    if (cache->pos >= cache->max_seq) {
        set_error("libane_kv_cache_update: cache full (pos=%d, max_seq=%d)",
                  cache->pos, cache->max_seq);
        return -1;
    }
    // Each token's K/V slice is laid out as [num_heads, head_dim].
    // We scatter-write to [head][pos][dim] in the buffer.
    const auto* k_src = static_cast<const uint16_t*>(new_k);
    const auto* v_src = static_cast<const uint16_t*>(new_v);
    const int pos      = cache->pos;
    const int H        = cache->num_heads;
    const int D        = cache->head_dim;
    const int max_seq  = cache->max_seq;
    for (int h = 0; h < H; ++h) {
        // Buffer layout: [h * max_seq * D + pos * D .. + D)
        size_t buf_off = static_cast<size_t>(h) * max_seq * D + pos * D;
        std::memcpy(cache->k_buf.data() + buf_off, k_src + h * D, D * sizeof(uint16_t));
        std::memcpy(cache->v_buf.data() + buf_off, v_src + h * D, D * sizeof(uint16_t));
    }
    cache->pos = pos + 1;
    return cache->pos;
}

const void* libane_kv_cache_k(libane_kv_cache_t cache) {
    if (!cache) return nullptr;
    return cache->k_buf.data();
}

const void* libane_kv_cache_v(libane_kv_cache_t cache) {
    if (!cache) return nullptr;
    return cache->v_buf.data();
}

int libane_kv_cache_position(libane_kv_cache_t cache) {
    if (!cache) return -1;
    return cache->pos;
}

void libane_kv_cache_reset(libane_kv_cache_t cache) {
    if (!cache) return;
    cache->pos = 0;
    std::fill(cache->k_buf.begin(), cache->k_buf.end(), uint16_t(0));
    std::fill(cache->v_buf.begin(), cache->v_buf.end(), uint16_t(0));
}

void libane_kv_cache_release(libane_kv_cache_t cache) {
    delete cache;
}

/* ─────────────────────────────────────────────────────────────────────────── */

uint32_t libane_graph_add_sdpa_gqa(libane_graph_t g,
                                    uint32_t       Q_id,
                                    uint32_t       K_id,
                                    uint32_t       V_id,
                                    uint32_t       mask_id) {
    if (!g) {
        set_error("libane_graph_add_sdpa_gqa: null graph");
        return LIBANE_INVALID_TENSOR_ID;
    }
    try {
        // Build inputs list; append mask only if provided.
        std::vector<uint32_t> inputs = {Q_id, K_id, V_id};
        if (mask_id != LIBANE_INVALID_TENSOR_ID)
            inputs.push_back(mask_id);

        // Output shape mirrors Q shape.
        const auto& q_shape = g->graph.tensor(Q_id).shape;
        return g->graph.add_op(LIBANE_OP_SDPA_GQA, inputs, q_shape);
    } catch (const std::exception& e) {
        set_error("libane_graph_add_sdpa_gqa: %s", e.what());
        return LIBANE_INVALID_TENSOR_ID;
    }
}

libane_status_t libane_graph_mark_output(libane_graph_t g,
                                          uint32_t       tensor_id,
                                          const char*    name) {
    if (!g) {
        set_error("libane_graph_mark_output: null graph");
        return LIBANE_ERR_INVALID_ARG;
    }
    try {
        g->graph.mark_output(tensor_id, name ? name : "");
        return LIBANE_OK;
    } catch (const std::exception& e) {
        set_error("libane_graph_mark_output: %s", e.what());
        return LIBANE_ERR_INVALID_ARG;
    }
}

libane_compiled_graph_t libane_graph_compile(libane_graph_t g) {
    if (!g) {
        set_error("libane_graph_compile: null graph");
        return nullptr;
    }
    auto cg = libane::graph::GraphCompiler::compile(g->graph);
    if (!cg) {
        set_error("libane_graph_compile: compilation failed");
        return nullptr;
    }
    auto* h = new (std::nothrow) libane_compiled_graph_s{std::move(cg)};
    if (!h) {
        set_error("libane_graph_compile: out of memory");
        return nullptr;
    }
    return h;
}

void libane_compiled_graph_release(libane_compiled_graph_t cg) {
    delete cg;
}

libane_status_t libane_compiled_graph_save(libane_compiled_graph_t cg,
                                            const char* path) {
    if (!cg || !cg->cg) {
        set_error("libane_compiled_graph_save: null compiled graph");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (!path) {
        set_error("libane_compiled_graph_save: null path");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (!cg->cg->save(path)) {
        set_error("libane_compiled_graph_save: save failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

libane_compiled_graph_t libane_compiled_graph_load(const char* path) {
    if (!path) {
        set_error("libane_compiled_graph_load: null path");
        return nullptr;
    }
    auto cg = libane::graph::GraphCompiler::load(path);
    if (!cg) {
        set_error("libane_compiled_graph_load: load failed");
        return nullptr;
    }
    auto* h = new (std::nothrow) libane_compiled_graph_s{std::move(cg)};
    if (!h) {
        set_error("libane_compiled_graph_load: out of memory");
        return nullptr;
    }
    return h;
}

libane_status_t libane_compiled_graph_delta_reload(libane_compiled_graph_t cg) {
    if (!cg || !cg->cg) {
        set_error("libane_compiled_graph_delta_reload: null compiled graph");
        return LIBANE_ERR_INVALID_ARG;
    }
    bool ok = libane::graph::GraphExecutor::delta_reload(*cg->cg);
    if (!ok) {
        const char* rt = libane::runtime::ane_last_error();
        if (rt && rt[0] != '\0')
            set_error("libane_compiled_graph_delta_reload: reload failed: %s", rt);
        else
            set_error("libane_compiled_graph_delta_reload: reload failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

libane_status_t libane_graph_execute(libane_compiled_graph_t cg,
                                      const void**            input_ptrs,
                                      const size_t*           input_bytes,
                                      size_t                  num_inputs,
                                      void**                  output_ptrs,
                                      const size_t*           output_bytes,
                                      size_t                  num_outputs) {
    if (!cg || !cg->cg) {
        set_error("libane_graph_execute: null compiled graph");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_inputs > 0 && (!input_ptrs || !input_bytes)) {
        set_error("libane_graph_execute: null input arrays");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_outputs > 0 && (!output_ptrs || !output_bytes)) {
        set_error("libane_graph_execute: null output arrays");
        return LIBANE_ERR_INVALID_ARG;
    }

    std::vector<const void*> in_ptrs(input_ptrs, input_ptrs + num_inputs);
    std::vector<size_t>      in_bytes(input_bytes, input_bytes + num_inputs);
    std::vector<void*>       out_ptrs(output_ptrs, output_ptrs + num_outputs);
    std::vector<size_t>      out_bytes(output_bytes, output_bytes + num_outputs);

    bool ok = libane::graph::GraphExecutor::execute(
        *cg->cg, in_ptrs, in_bytes, out_ptrs, out_bytes);

    if (!ok) {
        const char* rt = libane::runtime::ane_last_error();
        if (rt && rt[0] != '\0')
            set_error("libane_graph_execute: execution failed: %s", rt);
        else
            set_error("libane_graph_execute: execution failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

/* ── Raw MIL probe API ───────────────────────────────────────────────────── */

libane_mil_handle_t libane_mil_compile(const char*   mil_text,
                                        const char**  weight_names,
                                        const void**  weight_data,
                                        const size_t* weight_sizes,
                                        size_t        num_weights) {
    libane::runtime::initialize();
    if (!mil_text) {
        set_error("libane_mil_compile: null mil_text");
        return nullptr;
    }
    if (num_weights > 0 && (!weight_names || !weight_data || !weight_sizes)) {
        set_error("libane_mil_compile: null weight arrays with num_weights > 0");
        return nullptr;
    }

    std::vector<libane::runtime::WeightEntry> entries;
    entries.reserve(num_weights);
    for (size_t i = 0; i < num_weights; ++i) {
        auto blob = libane::mil::WeightBlob::from_fp16(weight_data[i], weight_sizes[i]);
        entries.push_back({weight_names[i], std::move(blob.data)});
    }

    auto* prog = libane::runtime::ane_compile(mil_text, entries, "mil_probe");
    if (!prog) {
        set_error("libane_mil_compile: %s", libane::runtime::ane_last_error());
        return nullptr;
    }

    auto* h = new (std::nothrow) libane_mil_program_s{prog};
    if (!h) {
        libane::runtime::ane_unload(prog);
        set_error("libane_mil_compile: out of memory");
        return nullptr;
    }
    return h;
}

libane_status_t libane_mil_execute(libane_mil_handle_t h,
                                    const void**  in_data,
                                    const size_t* in_sizes,
                                    size_t        num_inputs,
                                    void**        out_data,
                                    const size_t* out_sizes,
                                    size_t        num_outputs) {
    if (!h || !h->prog) {
        set_error("libane_mil_execute: null handle");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_inputs  > 0 && (!in_data  || !in_sizes)) {
        set_error("libane_mil_execute: null input arrays");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_outputs > 0 && (!out_data || !out_sizes)) {
        set_error("libane_mil_execute: null output arrays");
        return LIBANE_ERR_INVALID_ARG;
    }

    // Orion constraints #2 and #18: all input IOSurfaces must share one alloc
    // size; same for outputs.  Minimum is 49KB (constraint #4 — also enforced
    // independently inside BufferPool::allocate()).
    static constexpr size_t kMinIOS = 49152;
    size_t max_in  = kMinIOS;
    size_t max_out = kMinIOS;
    for (size_t i = 0; i < num_inputs;  ++i) max_in  = std::max(max_in,  in_sizes[i]);
    for (size_t i = 0; i < num_outputs; ++i) max_out = std::max(max_out, out_sizes[i]);

    auto& pool = libane::global_buffer_pool();

    std::vector<std::unique_ptr<libane::AneBuffer>> in_bufs, out_bufs;
    std::vector<IOSurfaceRef> in_ios, out_ios;

    for (size_t i = 0; i < num_inputs; ++i) {
        auto buf = pool.acquire(max_in);
        buf->copy_from(in_data[i], in_sizes[i]);
        in_ios.push_back(buf->iosurface());
        in_bufs.push_back(std::move(buf));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto buf = pool.acquire(max_out);
        out_ios.push_back(buf->iosurface());
        out_bufs.push_back(std::move(buf));
    }

    bool ok = libane::runtime::ane_execute_multi(h->prog, in_ios, out_ios);

    if (ok) {
        for (size_t i = 0; i < num_outputs; ++i)
            out_bufs[i]->copy_to(out_data[i], out_sizes[i]);
    }

    for (auto& b : in_bufs)  pool.release(std::move(b));
    for (auto& b : out_bufs) pool.release(std::move(b));

    if (!ok) {
        const char* rt = libane::runtime::ane_last_error();
        if (rt && rt[0] != '\0')
            set_error("libane_mil_execute: %s", rt);
        else
            set_error("libane_mil_execute: execution failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

libane_status_t libane_mil_execute_stats(libane_mil_handle_t h,
                                          const void**         in_data,
                                          const size_t*        in_sizes,
                                          size_t               num_inputs,
                                          void**               out_data,
                                          const size_t*        out_sizes,
                                          size_t               num_outputs,
                                          libane_perf_stats_t* stats_out) {
    if (!h || !h->prog) {
        set_error("libane_mil_execute_stats: null handle");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_inputs  > 0 && (!in_data  || !in_sizes)) {
        set_error("libane_mil_execute_stats: null input arrays");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_outputs > 0 && (!out_data || !out_sizes)) {
        set_error("libane_mil_execute_stats: null output arrays");
        return LIBANE_ERR_INVALID_ARG;
    }

    libane::runtime::AnePerfStats rt_stats;

    static constexpr size_t kMinIOS = 49152;
    size_t max_in  = kMinIOS;
    size_t max_out = kMinIOS;
    for (size_t i = 0; i < num_inputs;  ++i) max_in  = std::max(max_in,  in_sizes[i]);
    for (size_t i = 0; i < num_outputs; ++i) max_out = std::max(max_out, out_sizes[i]);

    auto& pool = libane::global_buffer_pool();

    std::vector<std::unique_ptr<libane::AneBuffer>> in_bufs, out_bufs;
    std::vector<IOSurfaceRef> in_ios, out_ios;

    for (size_t i = 0; i < num_inputs; ++i) {
        auto buf = pool.acquire(max_in);
        buf->copy_from(in_data[i], in_sizes[i]);
        in_ios.push_back(buf->iosurface());
        in_bufs.push_back(std::move(buf));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto buf = pool.acquire(max_out);
        out_ios.push_back(buf->iosurface());
        out_bufs.push_back(std::move(buf));
    }

    bool ok = libane::runtime::ane_execute_multi(
        h->prog, in_ios, out_ios,
        stats_out ? &rt_stats : nullptr);

    if (ok) {
        for (size_t i = 0; i < num_outputs; ++i)
            out_bufs[i]->copy_to(out_data[i], out_sizes[i]);
        if (stats_out) {
            stats_out->ane_bw_utilization = rt_stats.ane_bw_utilization;
            stats_out->avg_bw_state       = rt_stats.avg_bw_state;
            stats_out->peak_bw_state      = rt_stats.peak_bw_state;
            stats_out->ane_energy_units   = rt_stats.ane_energy_units;
            stats_out->throttle_ns        = rt_stats.throttle_ns;
            stats_out->available          = rt_stats.available;
        }
    }

    for (auto& b : in_bufs)  pool.release(std::move(b));
    for (auto& b : out_bufs) pool.release(std::move(b));

    if (!ok) {
        const char* rt = libane::runtime::ane_last_error();
        if (rt && rt[0] != '\0')
            set_error("libane_mil_execute_stats: %s", rt);
        else
            set_error("libane_mil_execute_stats: execution failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

int libane_mil_sram_spill(libane_mil_handle_t h) {
    if (!h || !h->prog) return -1;
    return h->prog->sram_spill ? 1 : 0;
}

void libane_mil_release(libane_mil_handle_t h) {
    if (!h) return;
    if (h->prog) libane::runtime::ane_unload(h->prog);
    delete h;
}

/* ── MIL program save / load ─────────────────────────────────────────────── */

// Binary file helpers (local to this translation unit)
namespace {
static constexpr uint32_t kMilMagic   = 0x414E454Du; // "ANEM"
static constexpr uint32_t kMilVersion = 1u;

static bool mil_w32(std::ofstream& f, uint32_t v) {
    return static_cast<bool>(f.write(reinterpret_cast<const char*>(&v), 4));
}
static bool mil_wstr(std::ofstream& f, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    return mil_w32(f, len) && f.write(s.data(), len);
}
static bool mil_wbytes(std::ofstream& f, const std::vector<uint8_t>& b) {
    uint32_t len = static_cast<uint32_t>(b.size());
    return mil_w32(f, len) && (b.empty() || f.write(
        reinterpret_cast<const char*>(b.data()), len));
}

static bool mil_r32(std::ifstream& f, uint32_t& v) {
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), 4));
}
static bool mil_rstr(std::ifstream& f, std::string& s) {
    uint32_t len;
    if (!mil_r32(f, len)) return false;
    s.resize(len);
    if (len == 0) return true;
    return static_cast<bool>(f.read(s.data(), len));
}
static bool mil_rbytes(std::ifstream& f, std::vector<uint8_t>& b) {
    uint32_t len;
    if (!mil_r32(f, len)) return false;
    b.resize(len);
    if (len == 0) return true;
    return static_cast<bool>(f.read(reinterpret_cast<char*>(b.data()), len));
}
} // anonymous namespace

libane_status_t libane_mil_save(libane_mil_handle_t h, const char* path) {
    if (!h || !path) {
        set_error("libane_mil_save: null argument");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (!h->prog) {
        set_error("libane_mil_save: handle has no compiled program");
        return LIBANE_ERR_INVALID_ARG;
    }

    libane::runtime::SerializedProgram sp;
    if (!libane::runtime::ane_serialize_program(h->prog, sp)) {
        set_error("libane_mil_save: ane_serialize_program failed");
        return LIBANE_ERR_EXECUTE_FAILED;
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        set_error("libane_mil_save: cannot open '%s' for writing", path);
        return LIBANE_ERR_EXECUTE_FAILED;
    }

    // Header
    mil_w32(f, kMilMagic);
    mil_w32(f, kMilVersion);

    // MIL text + metadata
    mil_wstr(f, sp.mil_text);
    mil_wstr(f, sp.hwx_rel_path);
    mil_wstr(f, sp.debug_name);
    mil_wbytes(f, sp.hwx_bytes);

    // Weight entries
    mil_w32(f, static_cast<uint32_t>(sp.weights.size()));
    for (const auto& we : sp.weights) {
        mil_wstr(f, we.filename);
        std::vector<uint8_t> data(we.data.begin(), we.data.end());
        mil_wbytes(f, data);
    }

    // I/O param names
    mil_w32(f, static_cast<uint32_t>(sp.input_param_names.size()));
    for (const auto& n : sp.input_param_names)
        mil_wstr(f, n);
    mil_w32(f, static_cast<uint32_t>(sp.output_var_names.size()));
    for (const auto& n : sp.output_var_names)
        mil_wstr(f, n);

    if (!f) {
        set_error("libane_mil_save: write error on '%s'", path);
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

libane_mil_handle_t libane_mil_load(const char* path) {
    if (!path) {
        set_error("libane_mil_load: null path");
        return nullptr;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        set_error("libane_mil_load: cannot open '%s'", path);
        return nullptr;
    }

    uint32_t magic = 0, version = 0;
    if (!mil_r32(f, magic) || magic != kMilMagic) {
        set_error("libane_mil_load: not an ANEM file (bad magic)");
        return nullptr;
    }
    if (!mil_r32(f, version) || version != kMilVersion) {
        set_error("libane_mil_load: unsupported ANEM version %u", version);
        return nullptr;
    }

    libane::runtime::SerializedProgram sp;
    if (!mil_rstr(f, sp.mil_text)    ||
        !mil_rstr(f, sp.hwx_rel_path)||
        !mil_rstr(f, sp.debug_name)  ||
        !mil_rbytes(f, sp.hwx_bytes)) {
        set_error("libane_mil_load: truncated header in '%s'", path);
        return nullptr;
    }

    uint32_t nweights = 0;
    if (!mil_r32(f, nweights)) {
        set_error("libane_mil_load: truncated weight count in '%s'", path);
        return nullptr;
    }
    sp.weights.resize(nweights);
    for (auto& we : sp.weights) {
        std::vector<uint8_t> data;
        if (!mil_rstr(f, we.filename) || !mil_rbytes(f, data)) {
            set_error("libane_mil_load: truncated weight data in '%s'", path);
            return nullptr;
        }
        we.data.assign(data.begin(), data.end());
    }

    uint32_t nin = 0, nout = 0;
    if (!mil_r32(f, nin)) return nullptr;
    sp.input_param_names.resize(nin);
    for (auto& n : sp.input_param_names)
        if (!mil_rstr(f, n)) return nullptr;
    if (!mil_r32(f, nout)) return nullptr;
    sp.output_var_names.resize(nout);
    for (auto& n : sp.output_var_names)
        if (!mil_rstr(f, n)) return nullptr;

    auto* prog = libane::runtime::ane_restore_program(sp);
    if (!prog) {
        set_error("libane_mil_load: ane_restore_program failed");
        return nullptr;
    }

    auto* h = new (std::nothrow) libane_mil_program_s{prog};
    if (!h) {
        libane::runtime::ane_unload(prog);
        set_error("libane_mil_load: out of memory");
        return nullptr;
    }
    return h;
}

} // extern "C"
