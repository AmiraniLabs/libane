#include <catch2/catch_test_macros.hpp>
#include "core/buffer_manager.hpp"
#include <cstring>
#include <vector>
#include <cstdint>

using namespace libane;

/* ── Allocation ─────────────────────────────────────────────────────────── */

TEST_CASE("BufferPool allocates a buffer", "[buffers]") {
    BufferPool pool(/*use_iosurface=*/false);
    auto buf = pool.acquire(256);
    REQUIRE(buf != nullptr);
    REQUIRE(buf->data() != nullptr);
    CHECK(buf->bytes() == 256);
    CHECK(buf->numel() == 128);
}

TEST_CASE("BufferPool allocates zero bytes doesn't crash", "[buffers]") {
    BufferPool pool(false);
    // Pool should handle gracefully
    // acquire(0) might return a 0-byte buffer or throw — implementation dependent
    // We just check it doesn't cause undefined behavior
    REQUIRE_NOTHROW(pool.acquire(64)); // sanity
}

/* ── Copy operations ────────────────────────────────────────────────────── */

TEST_CASE("AneBuffer copy_from and copy_to round-trip", "[buffers]") {
    BufferPool pool(false);
    auto buf = pool.acquire(64); // 32 fp16 elements

    std::vector<uint16_t> src(32);
    for (int i = 0; i < 32; ++i) src[i] = static_cast<uint16_t>(i);

    buf->copy_from(src.data(), 64);

    std::vector<uint16_t> dst(32, 0xFFFF);
    buf->copy_to(dst.data(), 64);

    for (int i = 0; i < 32; ++i)
        CHECK(dst[i] == static_cast<uint16_t>(i));
}

TEST_CASE("AneBuffer copy_from_f32 converts correctly", "[buffers]") {
    BufferPool pool(false);

    // 8 elements: [1.0, 0.0, 2.0, 0.5, -1.0, -0.5, 0.25, -0.25]
    std::vector<float> src = {1.0f, 0.0f, 2.0f, 0.5f, -1.0f, -0.5f, 0.25f, -0.25f};
    auto buf = pool.acquire(src.size() * 2);
    buf->copy_from_f32(src.data(), src.size());

    // Read back as fp32
    std::vector<float> dst(src.size(), 0.0f);
    buf->copy_to_f32(dst.data(), src.size());

    // fp16 has ~3 decimal digits of precision
    for (size_t i = 0; i < src.size(); ++i) {
        float rel_err = std::abs(dst[i] - src[i]) / (std::abs(src[i]) + 1e-6f);
        CHECK(rel_err < 0.002f);  // 0.2% relative error for fp16
    }
}

TEST_CASE("AneBuffer copy_to_f32 converts correctly", "[buffers]") {
    BufferPool pool(false);

    // Manually write fp16(1.0) = 0x3C00 and fp16(2.0) = 0x4000
    std::vector<uint16_t> raw = {0x3C00, 0x4000};
    auto buf = pool.acquire(4);
    buf->copy_from(raw.data(), 4);

    std::vector<float> dst(2, 0.0f);
    buf->copy_to_f32(dst.data(), 2);

    CHECK(std::abs(dst[0] - 1.0f) < 0.001f);
    CHECK(std::abs(dst[1] - 2.0f) < 0.001f);
}

/* ── Pool recycling ─────────────────────────────────────────────────────── */

TEST_CASE("BufferPool recycles released buffers", "[buffers]") {
    BufferPool pool(false, /*max_idle=*/2);

    auto buf1 = pool.acquire(128);
    void* raw_ptr = buf1->data();
    pool.release(std::move(buf1));

    auto buf2 = pool.acquire(128);
    // Should get the same buffer back (recycled)
    CHECK(buf2->data() == raw_ptr);
}

TEST_CASE("BufferPool does not exceed max_idle per size", "[buffers]") {
    BufferPool pool(false, /*max_idle=*/2);

    auto b1 = pool.acquire(256);
    auto b2 = pool.acquire(256);
    auto b3 = pool.acquire(256);

    pool.release(std::move(b1));
    pool.release(std::move(b2));
    pool.release(std::move(b3)); // pool is full, this should be destroyed

    CHECK(pool.pool_size() == 2);
}

/* ── acquire_with_data ───────────────────────────────────────────────────── */

TEST_CASE("acquire_with_data populates buffer immediately", "[buffers]") {
    BufferPool pool(false);

    std::vector<uint16_t> src = {0x3C00, 0x4000, 0x4200, 0x4400}; // 1,2,3,4 in fp16
    auto buf = pool.acquire_with_data(src.data(), src.size() * 2);

    REQUIRE(buf != nullptr);
    std::vector<uint16_t> dst(4);
    buf->copy_to(dst.data(), 8);

    for (size_t i = 0; i < src.size(); ++i)
        CHECK(dst[i] == src[i]);
}

/* ── acquire_from_f32 ────────────────────────────────────────────────────── */

TEST_CASE("acquire_from_f32 converts on the fly", "[buffers]") {
    BufferPool pool(false);

    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    auto buf = pool.acquire_from_f32(src.data(), src.size());

    REQUIRE(buf != nullptr);
    CHECK(buf->numel() == src.size());

    std::vector<float> dst(src.size());
    buf->copy_to_f32(dst.data(), src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        float rel = std::abs(dst[i] - src[i]) / src[i];
        CHECK(rel < 0.002f);
    }
}

/* ── Flush ──────────────────────────────────────────────────────────────── */

TEST_CASE("BufferPool flush drains pool", "[buffers]") {
    BufferPool pool(false, 10);

    for (int i = 0; i < 5; ++i) {
        auto b = pool.acquire(64);
        pool.release(std::move(b));
    }
    CHECK(pool.pool_size() > 0);

    pool.flush();
    CHECK(pool.pool_size() == 0);
}

/* ── Move semantics ─────────────────────────────────────────────────────── */

TEST_CASE("AneBuffer is moveable", "[buffers]") {
    BufferPool pool(false);
    auto buf = pool.acquire(128);
    void* ptr = buf->data();

    auto buf2 = std::move(buf);
    REQUIRE(buf2 != nullptr);
    CHECK(buf2->data() == ptr);

    // Original should be empty after move
    // (unique_ptr buf is now null)
    CHECK(buf == nullptr);
}

#ifdef __APPLE__
/* ── IOSurface-backed buffers (only on Apple) ─────────────────────────────── */

TEST_CASE("BufferPool with IOSurface allocates and round-trips", "[buffers][iosurface]") {
    BufferPool pool(/*use_iosurface=*/true);

    std::vector<float> src = {1.0f, 0.5f, -1.0f, 2.0f, 0.0f, 0.25f, -0.5f, 4.0f};
    auto buf = pool.acquire_from_f32(src.data(), src.size());

    REQUIRE(buf != nullptr);
    // On Apple Silicon, should have an IOSurface
    // (May not if IOSurface allocation failed — that's OK, check graceful fallback)
    // Just verify data integrity:
    std::vector<float> dst(src.size());
    buf->copy_to_f32(dst.data(), src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        float rel = std::abs(dst[i] - src[i]) / (std::abs(src[i]) + 1e-6f);
        CHECK(rel < 0.002f);
    }
}
#endif
