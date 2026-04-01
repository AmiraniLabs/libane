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

#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>

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
            case LIBANE_OP_GELU:
                mil_prog = libane::mil::MilBuilder::gelu(ms.channels, ms.seq);
                break;
            case LIBANE_OP_ADD:
                mil_prog = libane::mil::MilBuilder::add(ms.channels, ms.seq);
                break;
            case LIBANE_OP_MUL:
                mil_prog = libane::mil::MilBuilder::mul(ms.channels, ms.seq);
                break;
            case LIBANE_OP_SUB:
                mil_prog = libane::mil::MilBuilder::sub(ms.channels, ms.seq);
                break;
            case LIBANE_OP_REAL_DIV:
                mil_prog = libane::mil::MilBuilder::real_div(ms.channels, ms.seq);
                break;
            case LIBANE_OP_SQRT:
                mil_prog = libane::mil::MilBuilder::sqrt(ms.channels, ms.seq);
                break;
            case LIBANE_OP_LOG:
                mil_prog = libane::mil::MilBuilder::log(ms.channels, ms.seq);
                break;
            case LIBANE_OP_RSQRT:
                mil_prog = libane::mil::MilBuilder::rsqrt(ms.channels, ms.seq);
                break;
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

libane_status_t libane_delta_reload(libane_handle_t h,
                                     const void* new_weights,
                                     size_t weights_len) {
    if (!h || !new_weights || weights_len == 0) {
        set_error("libane_delta_reload: null or empty argument");
        return LIBANE_ERR_INVALID_ARG;
    }
    auto& entry = *h->entry;
    if (!entry.is_ane || !entry.backend_handle) {
        set_error("libane_delta_reload: handle is not ANE-compiled");
        return LIBANE_ERR_UNAVAILABLE;
    }

    auto* prog = static_cast<libane::runtime::AneProgram*>(entry.backend_handle);
    if (prog->weights.empty()) {
        set_error("libane_delta_reload: program has no stored weight entries (weight-free ops cannot be reloaded)");
        return LIBANE_ERR_INVALID_ARG;
    }

    // Build new weight blob from raw fp16 input
    auto new_blob = libane::mil::WeightBlob::from_fp16(new_weights, weights_len);

    // Build WeightEntry using the filename from the original compile
    std::vector<libane::runtime::WeightEntry> new_entries;
    new_entries.push_back({prog->weights[0].filename, new_blob.data});

    // Update cache key with new weight hash
    entry.key.weight_hash = new_blob.hash;

    bool ok = libane::runtime::ane_delta_reload(prog, new_entries);
    if (!ok) {
        set_error("libane_delta_reload failed: %s", libane::runtime::ane_last_error());
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
}

/* ── Release ─────────────────────────────────────────────────────────────── */

void libane_release(libane_handle_t h) {
    delete h; // shared_ptr to CacheEntry is released; cache still holds its copy
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
    if (!mil_text) {
        set_error("libane_mil_compile: null mil_text");
        return nullptr;
    }
    libane::runtime::initialize();
    if (!use_ane()) {
        set_error("libane_mil_compile: ANE unavailable: %s",
                  libane::runtime::fallback_reason());
        return nullptr;
    }

    std::vector<libane::runtime::WeightEntry> wentries;
    wentries.reserve(num_weights);
    for (size_t i = 0; i < num_weights; ++i) {
        if (!weight_names || !weight_names[i] || !weight_data || !weight_data[i]) {
            set_error("libane_mil_compile: null weight entry at index %zu", i);
            return nullptr;
        }
        auto blob = libane::mil::WeightBlob::from_fp16(weight_data[i], weight_sizes[i]);
        wentries.push_back({weight_names[i], blob.data});
    }

    auto* prog = libane::runtime::ane_compile(mil_text, wentries, "mil_probe");
    if (!prog) {
        set_error("libane_mil_compile: %s", libane::runtime::ane_last_error());
        return nullptr;
    }

    auto* h = new (std::nothrow) libane_mil_program_s{};
    if (!h) {
        libane::runtime::ane_unload(prog);
        set_error("libane_mil_compile: out of memory");
        return nullptr;
    }
    h->prog = prog;
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
    if (num_inputs > 0 && (!in_data || !in_sizes)) {
        set_error("libane_mil_execute: null input arrays");
        return LIBANE_ERR_INVALID_ARG;
    }
    if (num_outputs > 0 && (!out_data || !out_sizes)) {
        set_error("libane_mil_execute: null output arrays");
        return LIBANE_ERR_INVALID_ARG;
    }

#ifdef __APPLE__
    // Satisfy ANE IOSurface uniform-alloc constraints (#2, #4, #18).
    // All input IOSurfaces share one alloc size; all outputs share another.
    // Both are rounded up to the 49 KB ANE minimum.
    static constexpr size_t kMinIOS = 49152;
    size_t max_in  = kMinIOS;
    size_t max_out = kMinIOS;
    for (size_t i = 0; i < num_inputs;  ++i) max_in  = std::max(max_in,  in_sizes[i]);
    for (size_t i = 0; i < num_outputs; ++i) max_out = std::max(max_out, out_sizes[i]);

    auto& pool = libane::global_buffer_pool();

    std::vector<std::unique_ptr<libane::AneBuffer>> in_bufs, out_bufs;
    in_bufs.reserve(num_inputs);
    out_bufs.reserve(num_outputs);
    std::vector<IOSurfaceRef> in_ios, out_ios;
    in_ios.reserve(num_inputs);
    out_ios.reserve(num_outputs);

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
        set_error("libane_mil_execute: %s", libane::runtime::ane_last_error());
        return LIBANE_ERR_EXECUTE_FAILED;
    }
    return LIBANE_OK;
#else
    (void)in_data; (void)in_sizes; (void)out_data; (void)out_sizes;
    set_error("libane_mil_execute: ANE not available on this platform");
    return LIBANE_ERR_UNAVAILABLE;
#endif
}

void libane_mil_release(libane_mil_handle_t h) {
    if (!h) return;
    if (h->prog) libane::runtime::ane_unload(h->prog);
    delete h;
}

} // extern "C"
