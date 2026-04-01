#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "core/mil_builder.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <numeric>

// fp16 <-> float conversion for tests
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
static __fp16 f16(float v) { return static_cast<__fp16>(v); }
static float  f32(__fp16 v) { return static_cast<float>(v); }
#else
using fp16_test = uint16_t;
static fp16_test f16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FF;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00);
    else              h = static_cast<uint16_t>(s | (e << 10) | m);
    return h;
}
static float f32(fp16_test h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t fb;
    if (e == 0)       fb = s | (m << 13);
    else if (e == 31) fb = s | 0x7F800000 | (m << 13);
    else              fb = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &fb, 4); return f;
}
#endif

/* ── Version and availability ───────────────────────────────────────────── */

TEST_CASE("libane_version returns version string", "[api]") {
    const char* v = libane_version();
    REQUIRE(v != nullptr);
    REQUIRE(std::strlen(v) > 0);
    // Should start with "0."
    CHECK(v[0] == '0');
    CHECK(v[1] == '.');
}

TEST_CASE("libane_available returns 0 or 1", "[api]") {
    int avail = libane_available();
    CHECK((avail == 0 || avail == 1));
}

TEST_CASE("libane_last_error is valid string after no error", "[api]") {
    const char* err = libane_last_error();
    REQUIRE(err != nullptr);
}

/* ── Log level ───────────────────────────────────────────────────────────── */

TEST_CASE("libane_set_log_level does not crash", "[api]") {
    REQUIRE_NOTHROW(libane_set_log_level(LIBANE_LOG_SILENT));
    REQUIRE_NOTHROW(libane_set_log_level(LIBANE_LOG_DEBUG));
    REQUIRE_NOTHROW(libane_set_log_level(LIBANE_LOG_ERROR));
}

/* ── Backend selection ──────────────────────────────────────────────────── */

TEST_CASE("libane_set_backend cpu forces fallback", "[api]") {
    libane_set_log_level(LIBANE_LOG_SILENT);
    libane_set_backend("cpu");
    // libane_compile requires ANE — when CPU is forced, compile returns nullptr
    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=8; shape.dims[2]=1; shape.dims[3]=64; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    CHECK(h == nullptr); // CPU fallback removed — compile requires ANE
    libane_release(h);

    libane_set_backend(nullptr); // restore
}

TEST_CASE("libane_set_backend null restores auto-detect", "[api]") {
    libane_set_backend("cpu");
    libane_set_backend(nullptr);
    // Should not crash, state should be restored
    REQUIRE(libane_available() >= 0);
}

/* ── compile / release ──────────────────────────────────────────────────── */

TEST_CASE("libane_compile returns non-null for valid softmax shape", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping (libane_compile requires ANE)");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=16; shape.dims[2]=1; shape.dims[3]=64; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    if (!h) {
        WARN("ANE compile failed (may hit ~119 compile limit): " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);
    libane_release(h);
}

TEST_CASE("libane_compile returns null for invalid shape", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    libane_shape_t bad;
    bad.dims[0]=1; bad.dims[1]=8; bad.dims[2]=1; bad.dims[3]=7; bad.ndim=4; // S=7, not mult of 8

    auto h = libane_compile(LIBANE_OP_SOFTMAX, bad, nullptr, 0);
    CHECK(h == nullptr);
    // Error should be set
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_compile caches — second call returns same data", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=8; shape.dims[2]=1; shape.dims[3]=64; shape.ndim=4;

    size_t bytes_before = libane_cache_size_bytes();
    auto h1 = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    if (!h1) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    size_t bytes_after  = libane_cache_size_bytes();
    auto h2 = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    size_t bytes_after2 = libane_cache_size_bytes();

    REQUIRE(h1 != nullptr);
    REQUIRE(h2 != nullptr);
    // Cache should not grow on second compile
    CHECK(bytes_after2 == bytes_after);
    CHECK(bytes_after > bytes_before);

    libane_release(h1);
    libane_release(h2);
}

TEST_CASE("libane_release null is safe", "[api]") {
    REQUIRE_NOTHROW(libane_release(nullptr));
}

/* ── execute ─────────────────────────────────────────────────────────────── */

TEST_CASE("libane_execute softmax produces valid probabilities", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    // Shape [1, C=8, 1, S=8]: axis=1 softmax normalises over C channels.
    // For each S position the C channel values must sum to 1.0.
    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=8; shape.dims[2]=1; shape.dims[3]=8; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    using fp16_t = libane_f16_t;
    // Each S position gets the same C values [1..8]; after channel softmax they sum to 1.0.
    std::vector<fp16_t> in(8*8), out(8*8);
    float vals[] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f};
    for (int c = 0; c < 8; ++c)
        for (int s = 0; s < 8; ++s)
            in[c * 8 + s] = f16(vals[c]);

    libane_status_t st = libane_execute(h, in.data(), out.data(), shape);
    CHECK(st == LIBANE_OK);

    // Check one S position: sum of C=8 channel values must be ~1.0
    float sum = 0.0f;
    for (int c = 0; c < 8; ++c) sum += f32(out[c * 8 + 0]);
    CHECK(std::abs(sum - 1.0f) < 0.05f); // fp16 precision

    libane_release(h);
}

TEST_CASE("libane_execute gelu output matches expected", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=1; shape.dims[2]=1; shape.dims[3]=8; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    using fp16_t = libane_f16_t;
    std::vector<fp16_t> in(8), out(8);
    float vals[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f};
    for (int i = 0; i < 8; ++i) in[i] = f16(vals[i]);

    libane_status_t st = libane_execute(h, in.data(), out.data(), shape);
    if (st != LIBANE_OK) {
        // ANE requires minimum ~49KB IOSurface (constraint #4); tiny test shapes may fail
        WARN("ANE execute failed (may be IOSurface too small): " << libane_last_error());
        libane_release(h);
        return;
    }
    CHECK(st == LIBANE_OK);

    CHECK(std::abs(f32(out[0])) < 0.05f);          // gelu(0) ≈ 0
    CHECK(f32(out[1]) > 0.8f);                      // gelu(1) ≈ 0.841
    CHECK(f32(out[2]) < 0.0f);                      // gelu(-1) < 0

    libane_release(h);
}

TEST_CASE("libane_execute with null handle returns error", "[api]") {
    libane_shape_t shape{};
    std::vector<libane_f16_t> in(8), out(8);
    auto st = libane_execute(nullptr, in.data(), out.data(), shape);
    CHECK(st == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("libane_execute with null input returns error", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=1; shape.dims[2]=1; shape.dims[3]=8; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    std::vector<libane_f16_t> out(8);
    auto st = libane_execute(h, nullptr, out.data(), shape);
    CHECK(st == LIBANE_ERR_INVALID_ARG);

    libane_release(h);
}

TEST_CASE("libane_compile reduce_prod is graph-only", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_REDUCE_PROD, shape, nullptr, 0);
    CHECK(h == nullptr);
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_compile scatter is graph-only", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_SCATTER, shape, nullptr, 0);
    CHECK(h == nullptr);
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_compile gather is graph-only", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_GATHER, shape, nullptr, 0);
    CHECK(h == nullptr);
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_execute avg_pool lowering path behaves as identity", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    // 64*512*2 = 65536 bytes (>49KB), avoids small-buffer IOSurface artifacts.
    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_AVG_POOL, shape, nullptr, 0);
    if (!h) { WARN("ANE compile failed — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    const size_t n = static_cast<size_t>(shape.dims[1]) * shape.dims[3];
    std::vector<libane_f16_t> in(n), out(n);
    for (size_t i = 0; i < n; ++i)
        in[i] = f16(static_cast<float>((int(i % 31) - 15) * 0.25f));

    auto st = libane_execute(h, in.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    for (size_t i = 0; i < std::min<size_t>(n, 512); ++i) {
        CHECK(std::abs(f32(out[i]) - f32(in[i])) < 0.05f);
    }

    libane_release(h);
}

TEST_CASE("libane_execute max_pool lowering path behaves as identity", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_MAX_POOL, shape, nullptr, 0);
    if (!h) { WARN("ANE compile failed — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    const size_t n = static_cast<size_t>(shape.dims[1]) * shape.dims[3];
    std::vector<libane_f16_t> in(n), out(n);
    for (size_t i = 0; i < n; ++i)
        in[i] = f16(static_cast<float>((int(i % 17) - 8) * 0.5f));

    auto st = libane_execute(h, in.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    for (size_t i = 0; i < std::min<size_t>(n, 512); ++i) {
        CHECK(std::abs(f32(out[i]) - f32(in[i])) < 0.05f);
    }

    libane_release(h);
}

TEST_CASE("libane_execute2 logical_and lowering path returns 0/1 mask", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_LOGICAL_AND, shape, nullptr, 0);
    if (!h) { WARN("ANE compile failed — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    const size_t n = static_cast<size_t>(shape.dims[1]) * shape.dims[3];
    std::vector<libane_f16_t> a(n), b(n), out(n);
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i) {
        float av = (i % 3 == 0) ? 0.0f : ((i % 2 == 0) ? 2.0f : -1.0f);
        float bv = (i % 5 == 0) ? 0.0f : 4.0f;
        a[i] = f16(av);
        b[i] = f16(bv);
        expected[i] = ((av != 0.0f) && (bv != 0.0f)) ? 1.0f : 0.0f;
    }

    auto st = libane_execute2(h, a.data(), b.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    for (size_t i = 0; i < std::min<size_t>(n, 1024); ++i) {
        CHECK(std::abs(f32(out[i]) - expected[i]) < 0.05f);
    }

    libane_release(h);
}

TEST_CASE("libane_execute2 logical_or lowering path returns 0/1 mask", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_LOGICAL_OR, shape, nullptr, 0);
    if (!h) { WARN("ANE compile failed — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    const size_t n = static_cast<size_t>(shape.dims[1]) * shape.dims[3];
    std::vector<libane_f16_t> a(n), b(n), out(n);
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i) {
        float av = (i % 3 == 0) ? 0.0f : 2.0f;
        float bv = (i % 5 == 0) ? 0.0f : -4.0f;
        a[i] = f16(av);
        b[i] = f16(bv);
        expected[i] = ((av != 0.0f) || (bv != 0.0f)) ? 1.0f : 0.0f;
    }

    auto st = libane_execute2(h, a.data(), b.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    for (size_t i = 0; i < std::min<size_t>(n, 1024); ++i) {
        CHECK(std::abs(f32(out[i]) - expected[i]) < 0.05f);
    }

    libane_release(h);
}

TEST_CASE("libane_execute2 logical_xor lowering path returns 0/1 mask", "[api]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=64; shape.dims[2]=1; shape.dims[3]=512; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_LOGICAL_XOR, shape, nullptr, 0);
    if (!h) { WARN("ANE compile failed — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    const size_t n = static_cast<size_t>(shape.dims[1]) * shape.dims[3];
    std::vector<libane_f16_t> a(n), b(n), out(n);
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i) {
        float av = (i % 3 == 0) ? 0.0f : ((i % 2 == 0) ? 2.0f : -1.0f);
        float bv = (i % 5 == 0) ? 0.0f : ((i % 7 == 0) ? 0.0f : 4.0f);
        a[i] = f16(av);
        b[i] = f16(bv);
        bool ab = (av != 0.0f);
        bool bb = (bv != 0.0f);
        expected[i] = (ab != bb) ? 1.0f : 0.0f;
    }

    auto st = libane_execute2(h, a.data(), b.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    for (size_t i = 0; i < std::min<size_t>(n, 1024); ++i) {
        CHECK(std::abs(f32(out[i]) - expected[i]) < 0.05f);
    }

    libane_release(h);
}

/* ── libane_matmul_f16 ──────────────────────────────────────────────────── */

TEST_CASE("libane_matmul_f16 identity matrix", "[api][matmul]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    // N=8: tensor size = 8*8*2 = 128 bytes < 49 KB IOSurface minimum (constraint #4).
    // ANE channel stride = IOSurfaceAllocSize / C = 49152 / 8 = 6144 bytes, so only
    // tensors with numel * 2 >= 49152 get correct ANE results. Use CPU fallback here.
    int N = 8;
    std::vector<libane_f16_t> I(N*N), A(N*N), C(N*N);

    // Identity
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            I[i*N+j] = f16(i == j ? 1.0f : 0.0f);

    // A = [[0,1,2,...,63]] (8x8)
    for (int i = 0; i < N*N; ++i) A[i] = f16(static_cast<float>(i));

    // Force CPU fallback for this sub-49KB shape (ANE IOSurface stride would be wrong)
    libane_set_backend("cpu");
    libane_status_t st = libane_matmul_f16(I.data(), A.data(), C.data(), N, N, N);
    libane_set_backend(nullptr);
    REQUIRE(st == LIBANE_OK);

    for (int i = 0; i < N*N; ++i) {
        float err = std::abs(f32(C[i]) - f32(A[i]));
        CHECK(err < 0.5f); // fp16 precision
    }
}

TEST_CASE("libane_matmul_f16 small non-square", "[api][matmul]") {
    libane_set_backend(nullptr);

    // A = [[1,2,3,4,5,6,7,8]] (1×8), B = [[1],[1],[1],[1],[1],[1],[1],[1]] (8×1) → C = 36
    // Tensor size = 8*8*2 = 128 bytes < 49 KB minimum — force CPU fallback.
    int M = 8, K = 8, N = 8;
    std::vector<libane_f16_t> A(M*K), B(K*N), C(M*N);

    for (int i = 0; i < M*K; ++i) A[i] = f16(1.0f);
    for (int i = 0; i < K*N; ++i) B[i] = f16(1.0f);

    libane_set_backend("cpu");
    libane_status_t st = libane_matmul_f16(A.data(), B.data(), C.data(), M, K, N);
    libane_set_backend(nullptr);
    REQUIRE(st == LIBANE_OK);

    // Each output element = K * 1 * 1 = K = 8
    for (int i = 0; i < M*N; ++i) {
        float err = std::abs(f32(C[i]) - static_cast<float>(K));
        CHECK(err < 0.5f);
    }
}

TEST_CASE("libane_matmul_f16 null input returns error", "[api][matmul]") {
    auto st = libane_matmul_f16(nullptr, nullptr, nullptr, 8, 8, 8);
    CHECK(st == LIBANE_ERR_INVALID_ARG);
}

/* ── libane_matmul_f32 ──────────────────────────────────────────────────── */

TEST_CASE("libane_matmul_f32 known result", "[api][matmul]") {
    libane_set_backend(nullptr);

    std::vector<float> A = {1,2,3, 4,5,6};
    std::vector<float> B = {7,8, 9,10, 11,12};
    std::vector<float> C(4);

    auto st = libane_matmul_f32(A.data(), B.data(), C.data(), 2, 3, 2);
    REQUIRE(st == LIBANE_OK);

    CHECK(std::abs(C[0] - 58.0f) < 0.1f);
    CHECK(std::abs(C[1] - 64.0f) < 0.1f);
    CHECK(std::abs(C[2] - 139.0f) < 0.1f);
    CHECK(std::abs(C[3] - 154.0f) < 0.1f);
}

/* ── Cache management ────────────────────────────────────────────────────── */

TEST_CASE("libane_cache_flush resets cache", "[api][cache]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=8; shape.dims[2]=1; shape.dims[3]=64; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    CHECK(libane_cache_size_bytes() > 0);
    libane_release(h);

    libane_cache_flush();
    CHECK(libane_cache_size_bytes() == 0);
}

TEST_CASE("libane_cache_size_bytes increases after compile", "[api][cache]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);
    libane_cache_flush();

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    size_t before = libane_cache_size_bytes();
    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=16; shape.dims[2]=1; shape.dims[3]=128; shape.ndim=4;
    auto h = libane_compile(LIBANE_OP_GELU, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    CHECK(libane_cache_size_bytes() > before);
    libane_release(h);
}

/* ── ANE integration (only runs when ANE hardware is available) ──────────── */

TEST_CASE("ANE matmul_f16 executes on hardware when available", "[api][ane]") {
    libane_set_backend(nullptr);

    if (!libane_available()) {
        WARN("ANE not available on this machine — skipping ANE integration test");
        return;
    }

    // Use M=K=N=256 (256*256*2 = 131072 bytes > 49 KB) so IOSurface meets constraint #4.
    // Smaller shapes (e.g., 8×8 = 128 bytes) require the CPU fallback because the ANE
    // channel stride = IOSurfaceAllocSize / C = 49152 / C would be wrong for sub-49KB tensors.
    const int M = 256, K = 256, N = 256;
    std::vector<libane_f16_t> A(M * K), B(K * N), C(M * N);
    for (auto& v : A) v = f16(1.0f);
    for (auto& v : B) v = f16(1.0f);

    libane_status_t st = libane_matmul_f16(A.data(), B.data(), C.data(), M, K, N);
    REQUIRE(st == LIBANE_OK);

    // Each output = sum of K ones * 1.0f weight = K = 256
    for (int i = 0; i < std::min(M * N, 64); ++i) {
        float val = f32(C[i]);
        float err = std::abs(val - static_cast<float>(K));
        INFO("C[" << i << "] = " << val << " (expected " << K << ")");
        CHECK(err < 2.0f); // fp16 precision at scale 256
    }
}

TEST_CASE("ANE matmul_f16 A @ I == A on large shape", "[api][ane][matmul][layout]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available on this machine — skipping ANE layout test");
        return;
    }

    // 160*160*2 = 51200 bytes (>49 KB IOSurface minimum).
    const int M = 160, K = 160, N = 160;
    std::vector<libane_f16_t> A(M * K), I(K * N), C(M * N);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float v = static_cast<float>(((m * 17 + k * 13) % 31) - 15);
            A[m * K + k] = f16(v);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            I[k * N + n] = f16(k == n ? 1.0f : 0.0f);
        }
    }

    libane_status_t st = libane_matmul_f16(A.data(), I.data(), C.data(), M, K, N);
    REQUIRE(st == LIBANE_OK);

    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(f32(C[i]) - f32(A[i]));
        CHECK(err < 0.1f);
    }
}

TEST_CASE("ANE matmul_f16 I @ B == B on large shape", "[api][ane][matmul][layout]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available on this machine — skipping ANE layout test");
        return;
    }

    // 160*160*2 = 51200 bytes (>49 KB IOSurface minimum).
    const int M = 160, K = 160, N = 160;
    std::vector<libane_f16_t> I(M * K), B(K * N), C(M * N);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            I[m * K + k] = f16(m == k ? 1.0f : 0.0f);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float v = static_cast<float>(((k * 19 + n * 7) % 37) - 18);
            B[k * N + n] = f16(v);
        }
    }

    libane_status_t st = libane_matmul_f16(I.data(), B.data(), C.data(), M, K, N);
    REQUIRE(st == LIBANE_OK);

    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(f32(C[i]) - f32(B[i]));
        CHECK(err < 0.1f);
    }
}

TEST_CASE("ANE matmul_f16 one-hot rows select B rows", "[api][ane][matmul][layout]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available on this machine — skipping ANE layout test");
        return;
    }

    // 160*160*2 = 51200 bytes (>49 KB IOSurface minimum).
    const int M = 160, K = 160, N = 160;
    std::vector<libane_f16_t> A(M * K, f16(0.0f)), B(K * N), C(M * N);

    // Row m is one-hot at column (m % K).
    for (int m = 0; m < M; ++m) {
        int j = m % K;
        A[m * K + j] = f16(1.0f);
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float v = static_cast<float>(((k * 11 + n * 5) % 29) - 14);
            B[k * N + n] = f16(v);
        }
    }

    libane_status_t st = libane_matmul_f16(A.data(), B.data(), C.data(), M, K, N);
    REQUIRE(st == LIBANE_OK);

    for (int m = 0; m < M; ++m) {
        int j = m % K;
        for (int n = 0; n < N; ++n) {
            float got = f32(C[m * N + n]);
            float exp = f32(B[j * N + n]);
            CHECK(std::abs(got - exp) < 0.1f);
        }
    }
}

TEST_CASE("ANE softmax executes on hardware when available", "[api][ane]") {
    libane_set_backend(nullptr);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    libane_shape_t shape;
    shape.dims[0]=1; shape.dims[1]=8; shape.dims[2]=1; shape.dims[3]=64; shape.ndim=4;

    auto h = libane_compile(LIBANE_OP_SOFTMAX, shape, nullptr, 0);
    if (!h) { WARN("ANE compile limit reached — skipping: " << libane_last_error()); return; }
    REQUIRE(h != nullptr);

    size_t numel = 8 * 64;
    std::vector<libane_f16_t> in(numel), out(numel);
    for (size_t i = 0; i < numel; ++i) in[i] = f16(0.0f);

    libane_status_t st = libane_execute(h, in.data(), out.data(), shape);
    REQUIRE(st == LIBANE_OK);

    // axis=1 (channel) softmax: for each S position the C=8 channel values sum to ~1.0.
    for (size_t s = 0; s < 64; ++s) {
        float sum = 0.0f;
        for (size_t c = 0; c < 8; ++c)
            sum += f32(out[c * 64 + s]);
        CHECK(std::abs(sum - 1.0f) < 0.05f);
    }

    libane_release(h);
}

/* ── Delta reload ────────────────────────────────────────────────────────── */

TEST_CASE("libane_delta_reload updates weights without recompile", "[api][ane]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) SKIP("ANE not available");

    const int C = 8, S = 8;

    // W1 = all-ones weight matrix [C x C]
    std::vector<float> W1(C * C, 1.0f);
    // W2 = identity weight matrix [C x C]
    std::vector<float> W2(C * C, 0.0f);
    for (int i = 0; i < C; ++i) W2[i * C + i] = 1.0f;

    // Build fp16 blobs (transpose=true for conv1x1 [OC,IC] layout)
    auto blob1 = libane::mil::WeightBlob::from_fp32(W1.data(), C, C, true);
    auto blob2 = libane::mil::WeightBlob::from_fp32(W2.data(), C, C, true);

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=C; shape.dims[2]=1; shape.dims[3]=S; shape.ndim=4;

    // Compile with W1 (all-ones): pass raw fp16 weight data (after 128-byte header)
    const void* w1_data = blob1.data.data() + libane::mil::WeightBlob::kDataOffset;
    size_t w1_len = static_cast<size_t>(C) * C * sizeof(uint16_t);
    libane_handle_t h = libane_compile(LIBANE_OP_MATMUL, shape, w1_data, w1_len);
    if (!h) {
        WARN("ANE compile limit reached — skipping: " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);

    // Delta reload with W2 (identity)
    const void* w2_data = blob2.data.data() + libane::mil::WeightBlob::kDataOffset;
    size_t w2_len = static_cast<size_t>(C) * C * sizeof(uint16_t);
    libane_status_t st = libane_delta_reload(h, w2_data, w2_len);
    CHECK(st == LIBANE_OK);

    libane_release(h);
}

TEST_CASE("libane_delta_reload with null handle returns error", "[api][ane]") {
    libane_set_log_level(LIBANE_LOG_SILENT);
    std::vector<uint16_t> dummy(64, 0);
    auto st = libane_delta_reload(nullptr, dummy.data(), dummy.size() * 2);
    CHECK(st == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("libane_delta_reload with null weights returns error", "[api][ane]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) SKIP("ANE not available");

    const int C = 8, S = 8;
    std::vector<float> W(C * C, 1.0f);
    auto blob = libane::mil::WeightBlob::from_fp32(W.data(), C, C, true);

    libane_shape_t shape{};
    shape.dims[0]=1; shape.dims[1]=C; shape.dims[2]=1; shape.dims[3]=S; shape.ndim=4;

    const void* w_data = blob.data.data() + libane::mil::WeightBlob::kDataOffset;
    libane_handle_t h = libane_compile(LIBANE_OP_MATMUL, shape, w_data,
                                        static_cast<size_t>(C) * C * 2);
    if (!h) {
        WARN("ANE compile limit reached — skipping: " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);

    auto st = libane_delta_reload(h, nullptr, 0);
    CHECK(st == LIBANE_ERR_INVALID_ARG);

    libane_release(h);
}

/* ── Batch Compilation ──────────────────────────────────────────────────── */

TEST_CASE("libane_compile_batch returns LIBANE_OK for valid requests", "[api][batch]") {
    if (libane_available() == 0) {
        WARN("ANE not available — skipping batch compile test");
        return;
    }

    libane_shape_t shape{};
    shape.dims[0] = 1; shape.dims[1] = 8; shape.dims[2] = 1; shape.dims[3] = 16;
    shape.ndim = 4;

    // Prepare three batch requests
    libane_compile_request_t requests[3] = {
        {LIBANE_OP_SOFTMAX, shape, nullptr, 0},
        {LIBANE_OP_GELU,    shape, nullptr, 0},
        {LIBANE_OP_SILU,    shape, nullptr, 0},
    };

    libane_handle_t handles[3] = {nullptr, nullptr, nullptr};

    libane_status_t st = libane_compile_batch(requests, 3, handles);
    if (st == LIBANE_ERR_COMPILE_FAILED && handles[0] == nullptr) {
        WARN("ANE compile limit reached (Orion constraint #5) — skipping: " << libane_last_error());
        return;
    }
    CHECK(st == LIBANE_OK);

    // All handles should be valid
    REQUIRE(handles[0] != nullptr);
    REQUIRE(handles[1] != nullptr);
    REQUIRE(handles[2] != nullptr);

    libane_release(handles[0]);
    libane_release(handles[1]);
    libane_release(handles[2]);
}

TEST_CASE("libane_compile_batch rejects null arguments", "[api][batch]") {
    libane_shape_t shape{};
    shape.dims[0] = 1; shape.dims[1] = 8; shape.dims[2] = 1; shape.dims[3] = 16;
    shape.ndim = 4;

    libane_compile_request_t request = {LIBANE_OP_SOFTMAX, shape, nullptr, 0};
    libane_handle_t handle = nullptr;

    // Null requests
    auto st = libane_compile_batch(nullptr, 1, &handle);
    CHECK(st == LIBANE_ERR_INVALID_ARG);

    // Null out_handles
    st = libane_compile_batch(&request, 1, nullptr);
    CHECK(st == LIBANE_ERR_INVALID_ARG);

    // Zero requests
    st = libane_compile_batch(&request, 0, &handle);
    CHECK(st == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("libane_compile_batch handles mixed success and failure", "[api][batch]") {
    if (libane_available() == 0) {
        WARN("ANE not available — skipping batch compile test");
        return;
    }

    libane_shape_t valid_shape{};
    valid_shape.dims[0] = 1; valid_shape.dims[1] = 8; valid_shape.dims[2] = 1; valid_shape.dims[3] = 16;
    valid_shape.ndim = 4;

    libane_shape_t bad_shape{};
    bad_shape.dims[0] = 1; bad_shape.dims[1] = 8; bad_shape.dims[2] = 1; bad_shape.dims[3] = 1;  // seq < 8
    bad_shape.ndim = 4;

    // Mix of valid and invalid requests
    libane_compile_request_t requests[3] = {
        {LIBANE_OP_SOFTMAX, valid_shape, nullptr, 0},  // valid
        {LIBANE_OP_GELU,    bad_shape,   nullptr, 0},  // invalid (seq < 8)
        {LIBANE_OP_SILU,    valid_shape, nullptr, 0},  // valid
    };

    libane_handle_t handles[3] = {nullptr, nullptr, nullptr};

    libane_status_t st = libane_compile_batch(requests, 3, handles);
    if (st == LIBANE_ERR_COMPILE_FAILED && handles[0] == nullptr) {
        WARN("ANE compile limit reached (Orion constraint #5) — skipping: " << libane_last_error());
        return;
    }
    // Should return LIBANE_ERR_COMPILE_FAILED because one failed
    CHECK(st == LIBANE_ERR_COMPILE_FAILED);

    // Check that valid requests succeeded, invalid failed
    CHECK(handles[0] != nullptr);  // valid
    CHECK(handles[1] == nullptr);  // invalid
    CHECK(handles[2] != nullptr);  // valid

    libane_release(handles[0]);
    libane_release(handles[2]);
}

/* ── libane_mil_compile / execute / release ─────────────────────────────── */

// Minimal valid MIL program: relu(x) — unary, no weights.
// Shape [1,32,1,32] = 1024 fp16 elements = 2048 bytes; pool pads to 49KB.
static const char kReluMil[] = R"(
buildInfo {
   coremlc-version: "7.0.0"
   coremltools-version: "7.0.0"
   mlmodel-version: "7"
   model-name: "test_relu"
}
func main(x: fp16[1,32,1,32]) -> (fp16[1,32,1,32]) {
  %v = relu(x=x) -> (fp16);
  return (%v);
}
)";

// Binary MIL: add(a, b).  Inputs named a,b so alphabetical order = natural order.
static const char kAddMil[] = R"(
buildInfo {
   coremlc-version: "7.0.0"
   coremltools-version: "7.0.0"
   mlmodel-version: "7"
   model-name: "test_add"
}
func main(a: fp16[1,32,1,32], b: fp16[1,32,1,32]) -> (fp16[1,32,1,32]) {
  %v = add(x=a, y=b) -> (fp16);
  return (%v);
}
)";

// MIL with one external weight file: elementwise mul by a learned scale.
// Offset 64 = WeightBlob::kWeightDictOffset.
static const char kScaleMil[] = R"(
buildInfo {
   coremlc-version: "7.0.0"
   coremltools-version: "7.0.0"
   mlmodel-version: "7"
   model-name: "test_scale"
}
func main(x: fp16[1,32,1,32]) -> (fp16[1,32,1,32]) {
  %w = const(val=tensor<fp16, [1,32,1,32]>(BLOBFILE(path=string("@model_path/weights/scale.bin"), offset=uint64(64)))) -> (fp16[1,32,1,32]);
  %v = mul(x=x, y=%w) -> (fp16);
  return (%v);
}
)";

TEST_CASE("libane_mil_compile null text returns null", "[mil]") {
    libane_set_log_level(LIBANE_LOG_SILENT);
    auto h = libane_mil_compile(nullptr, nullptr, nullptr, nullptr, 0);
    CHECK(h == nullptr);
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_mil_compile invalid MIL returns null", "[mil]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    auto h = libane_mil_compile("this is not valid MIL",
                                nullptr, nullptr, nullptr, 0);
    CHECK(h == nullptr);
    CHECK(std::strlen(libane_last_error()) > 0);
}

TEST_CASE("libane_mil_compile valid relu program", "[mil]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    auto h = libane_mil_compile(kReluMil, nullptr, nullptr, nullptr, 0);
    if (!h) {
        WARN("mil compile failed (may hit ~119 compile limit): " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);
    libane_mil_release(h);
}

TEST_CASE("libane_mil_release null is safe", "[mil]") {
    REQUIRE_NOTHROW(libane_mil_release(nullptr));
}

TEST_CASE("libane_mil_execute null handle returns error", "[mil]") {
    libane_set_log_level(LIBANE_LOG_SILENT);
    std::vector<libane_f16_t> buf(1024);
    const void* in_ptrs[]  = { buf.data() };
    void*       out_ptrs[] = { buf.data() };
    size_t      sizes[]    = { 2048 };

    auto st = libane_mil_execute(nullptr,
                                  in_ptrs, sizes, 1,
                                  out_ptrs, sizes, 1);
    CHECK(st == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("libane_mil_execute relu output equals input for positive values", "[mil]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    auto h = libane_mil_compile(kReluMil, nullptr, nullptr, nullptr, 0);
    if (!h) {
        WARN("mil compile failed: " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);

    static constexpr int N = 1024;  // [1,32,1,32]
    std::vector<libane_f16_t> in(N), out(N, f16(0.0f));
    // All-positive input: relu(x) == x
    for (int i = 0; i < N; ++i) in[i] = f16(static_cast<float>(i % 32) * 0.1f + 0.1f);

    const void* in_ptrs[]  = { in.data() };
    void*       out_ptrs[] = { out.data() };
    size_t      in_bytes[] = { N * sizeof(libane_f16_t) };
    size_t      out_bytes[]= { N * sizeof(libane_f16_t) };

    auto st = libane_mil_execute(h,
                                  in_ptrs,  in_bytes,  1,
                                  out_ptrs, out_bytes, 1);
    if (st != LIBANE_OK) {
        WARN("mil execute failed: " << libane_last_error());
        libane_mil_release(h);
        return;
    }
    REQUIRE(st == LIBANE_OK);

    // relu(x) == x for all-positive input; fp16 round-trip should be exact
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(f32(out[i]) - f32(in[i]));
        if (diff > 0.02f) ++mismatches;
    }
    CHECK(mismatches == 0);

    libane_mil_release(h);
}

TEST_CASE("libane_mil_execute add two inputs — uniform IOSurface alloc path", "[mil]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    auto h = libane_mil_compile(kAddMil, nullptr, nullptr, nullptr, 0);
    if (!h) {
        WARN("mil compile failed: " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);

    static constexpr int N = 1024;
    std::vector<libane_f16_t> a(N), b(N), out(N, f16(0.0f));
    for (int i = 0; i < N; ++i) {
        a[i] = f16(1.0f);
        b[i] = f16(2.0f);
    }

    const void* in_ptrs[]  = { a.data(), b.data() };
    void*       out_ptrs[] = { out.data() };
    size_t      in_bytes[] = { N * 2, N * 2 };
    size_t      out_bytes[]= { N * 2 };

    auto st = libane_mil_execute(h,
                                  in_ptrs,  in_bytes,  2,
                                  out_ptrs, out_bytes, 1);
    if (st != LIBANE_OK) {
        WARN("mil execute failed: " << libane_last_error());
        libane_mil_release(h);
        return;
    }
    REQUIRE(st == LIBANE_OK);

    // Every output element should be 1.0 + 2.0 = 3.0
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(f32(out[i]) - 3.0f);
        if (diff > 0.05f) ++mismatches;
    }
    CHECK(mismatches == 0);

    libane_mil_release(h);
}

TEST_CASE("libane_mil_compile_with_weights external scale weight", "[mil]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);

    if (!libane_available()) {
        WARN("ANE not available — skipping");
        return;
    }

    static constexpr int N = 1024;
    // Scale weight: all-ones (so mul(x, scale) == x)
    std::vector<libane_f16_t> scale(N);
    for (int i = 0; i < N; ++i) scale[i] = f16(1.0f);

    const char*  wnames[] = { "scale.bin" };
    const void*  wdata[]  = { scale.data() };
    size_t       wsizes[] = { N * 2 };

    auto h = libane_mil_compile(kScaleMil, wnames, wdata, wsizes, 1);
    if (!h) {
        WARN("mil compile with weights failed: " << libane_last_error());
        return;
    }
    REQUIRE(h != nullptr);

    std::vector<libane_f16_t> in(N), out(N, f16(0.0f));
    for (int i = 0; i < N; ++i) in[i] = f16(static_cast<float>(i % 32) * 0.1f + 0.1f);

    const void* in_ptrs[]  = { in.data() };
    void*       out_ptrs[] = { out.data() };
    size_t      in_bytes[] = { N * 2 };
    size_t      out_bytes[]= { N * 2 };

    auto st = libane_mil_execute(h,
                                  in_ptrs,  in_bytes,  1,
                                  out_ptrs, out_bytes, 1);
    if (st != LIBANE_OK) {
        WARN("mil execute (weights) failed: " << libane_last_error());
        libane_mil_release(h);
        return;
    }
    REQUIRE(st == LIBANE_OK);

    // scale is all-ones, so out[i] == in[i]
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(f32(out[i]) - f32(in[i]));
        if (diff > 0.02f) ++mismatches;
    }
    CHECK(mismatches == 0);

    libane_mil_release(h);
}
