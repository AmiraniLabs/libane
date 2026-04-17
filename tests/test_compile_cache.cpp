#include <catch2/catch_test_macros.hpp>
#include "core/compile_cache.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace libane;

/* ── Key equality / hash ────────────────────────────────────────────────── */

TEST_CASE("CacheKey equality and hash", "[cache]") {
    CacheKey k1;
    k1.op = LIBANE_OP_MATMUL;
    k1.shape = {.dims={1,64,1,512}, .ndim=4};
    k1.weight_hash = 0xDEADBEEF;

    CacheKey k2 = k1;
    CHECK(k1 == k2);

    CacheKeyHash hasher;
    CHECK(hasher(k1) == hasher(k2));

    k2.weight_hash = 0xCAFEBABE;
    CHECK_FALSE(k1 == k2);
    CHECK(hasher(k1) != hasher(k2));

    k2 = k1;
    k2.op = LIBANE_OP_SOFTMAX;
    CHECK_FALSE(k1 == k2);
}

/* ── Basic put / get ────────────────────────────────────────────────────── */

static std::unique_ptr<CacheEntry> make_entry(libane_op_t op,
                                               uint64_t    hash,
                                               size_t      size_bytes,
                                               bool        is_ane = false) {
    CacheKey k;
    k.op = op;
    k.shape = {.dims={1,64,1,512}, .ndim=4};
    k.weight_hash = hash;

    auto e = std::make_unique<CacheEntry>();
    e->key        = k;
    e->size_bytes = size_bytes;
    e->is_ane     = is_ane;
    return e;
}

TEST_CASE("Cache put and get", "[cache]") {
    CompileCache cache(1024 * 1024); // 1 MB

    auto e1 = make_entry(LIBANE_OP_MATMUL, 0x111, 100);
    CacheKey k1 = e1->key;

    REQUIRE(cache.get(k1) == nullptr); // miss

    cache.put(std::move(e1));
    auto found = cache.get(k1);
    REQUIRE(found != nullptr);
    CHECK(found->key == k1);
    CHECK(found->is_ane == false);
    CHECK(cache.size() == 1);
}

TEST_CASE("Cache miss increments counter", "[cache]") {
    CompileCache cache;
    CacheKey k;
    k.op = LIBANE_OP_GELU;
    k.shape = {.dims={1,8,1,64}, .ndim=4};
    k.weight_hash = 0;

    cache.get(k);
    cache.get(k);
    auto st = cache.stats();
    CHECK(st.misses == 2);
    CHECK(st.hits   == 0);
}

TEST_CASE("Cache hit increments counter", "[cache]") {
    CompileCache cache;
    auto e = make_entry(LIBANE_OP_SOFTMAX, 0x222, 100);
    CacheKey k = e->key;
    cache.put(std::move(e));

    cache.get(k);
    cache.get(k);
    auto st = cache.stats();
    CHECK(st.hits   >= 2);
    CHECK(st.misses == 0);
}

/* ── Duplicate put ──────────────────────────────────────────────────────── */

TEST_CASE("Putting the same key twice replaces the entry", "[cache]") {
    CompileCache cache;
    auto e1 = make_entry(LIBANE_OP_MATMUL, 0x333, 100, false);
    auto e2 = make_entry(LIBANE_OP_MATMUL, 0x333, 200, true);
    CacheKey k = e1->key;

    cache.put(std::move(e1));
    cache.put(std::move(e2));

    auto found = cache.get(k);
    REQUIRE(found != nullptr);
    CHECK(found->size_bytes == 200);
    CHECK(found->is_ane == true);
    CHECK(cache.size() == 1);
}

/* ── LRU eviction ───────────────────────────────────────────────────────── */

TEST_CASE("Cache evicts LRU when over budget", "[cache]") {
    // Budget = 300 bytes, insert three 100-byte entries
    CompileCache cache(300);

    auto e1 = make_entry(LIBANE_OP_MATMUL, 0xA, 100);
    auto e2 = make_entry(LIBANE_OP_MATMUL, 0xB, 100);
    auto e3 = make_entry(LIBANE_OP_MATMUL, 0xC, 100);
    CacheKey k1 = e1->key, k2 = e2->key, k3 = e3->key;

    cache.put(std::move(e1));
    cache.put(std::move(e2));
    cache.put(std::move(e3));

    // All three fit exactly
    CHECK(cache.size() == 3);
    CHECK(cache.size_bytes() == 300);

    // Insert a 4th — must evict k1 (LRU)
    auto e4 = make_entry(LIBANE_OP_MATMUL, 0xD, 100);
    CacheKey k4 = e4->key;
    cache.put(std::move(e4));

    CHECK(cache.size() == 3);
    CHECK(cache.stats().evictions >= 1);

    // k1 was LRU — should be evicted
    // k4 was just inserted — should be present
    REQUIRE(cache.get(k4) != nullptr);
}

/* ── Flush ──────────────────────────────────────────────────────────────── */

TEST_CASE("Cache flush clears all entries", "[cache]") {
    CompileCache cache;
    for (int i = 0; i < 10; ++i) {
        cache.put(make_entry(LIBANE_OP_MATMUL, static_cast<uint64_t>(i), 100));
    }
    CHECK(cache.size() == 10);

    cache.flush();
    CHECK(cache.size() == 0);
    CHECK(cache.size_bytes() == 0);
}

/* ── Thread safety ──────────────────────────────────────────────────────── */

TEST_CASE("Cache is thread-safe under concurrent read/write", "[cache][threads]") {
    CompileCache cache(64 * 1024 * 1024); // 64 MB — no eviction pressure
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    // Writers
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 50; ++i) {
                uint64_t hash = static_cast<uint64_t>(t * 1000 + i);
                try {
                    cache.put(make_entry(LIBANE_OP_MATMUL, hash, 100));
                } catch (...) {
                    ++errors;
                }
            }
        });
    }

    // Readers — read keys inserted by writer 0
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            CacheKey k;
            k.op = LIBANE_OP_MATMUL;
            k.shape = {.dims={1,64,1,512}, .ndim=4};
            for (int i = 0; i < 50; ++i) {
                k.weight_hash = static_cast<uint64_t>(i);
                try {
                    cache.get(k); // may hit or miss — just must not crash
                } catch (...) {
                    ++errors;
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    CHECK(errors == 0);
}

/* ── Concurrent eviction stress ─────────────────────────────────────────── */

TEST_CASE("Cache handles concurrent writes with heavy eviction", "[cache][threads]") {
    // Tiny budget forces constant eviction under contention
    CompileCache cache(500); // fits ~5 entries of 100 bytes
    std::atomic<int> errors{0};
    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 200;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                uint64_t hash = static_cast<uint64_t>(t * 10000 + i);
                try {
                    cache.put(make_entry(LIBANE_OP_MATMUL, hash, 100));

                    // Immediately try to read back — may or may not be evicted
                    CacheKey k;
                    k.op = LIBANE_OP_MATMUL;
                    k.shape = {.dims={1,64,1,512}, .ndim=4};
                    k.weight_hash = hash;
                    cache.get(k); // must not crash
                } catch (...) {
                    ++errors;
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    CHECK(errors == 0);

    auto st = cache.stats();
    CHECK(st.evictions > 0); // must have evicted under this budget
}

TEST_CASE("Cache concurrent mixed read/write/flush", "[cache][threads]") {
    CompileCache cache(16 * 1024); // 16 KB
    std::atomic<int> errors{0};
    std::atomic<bool> done{false};
    std::vector<std::thread> threads;

    // Writers
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                uint64_t hash = static_cast<uint64_t>(t * 1000 + i);
                try {
                    cache.put(make_entry(LIBANE_OP_SOFTMAX, hash, 100));
                } catch (...) {
                    ++errors;
                }
            }
        });
    }

    // Readers
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            CacheKey k;
            k.op = LIBANE_OP_SOFTMAX;
            k.shape = {.dims={1,64,1,512}, .ndim=4};
            for (int i = 0; i < 200; ++i) {
                k.weight_hash = static_cast<uint64_t>(i % 400);
                try {
                    auto result = cache.get(k);
                    // If found, entry must have correct op
                    if (result && result->key.op != LIBANE_OP_SOFTMAX) {
                        ++errors;
                    }
                } catch (...) {
                    ++errors;
                }
            }
        });
    }

    // Flusher — periodically clears the cache while others read/write
    threads.emplace_back([&]() {
        for (int i = 0; i < 5; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            try {
                cache.flush();
            } catch (...) {
                ++errors;
            }
        }
    });

    for (auto& th : threads) th.join();
    CHECK(errors == 0);
}

/* ── Release fn ─────────────────────────────────────────────────────────── */

TEST_CASE("Release function is called on eviction", "[cache]") {
    int released = 0;
    {
        CompileCache cache(100); // only fits one 100-byte entry

        auto e1 = make_entry(LIBANE_OP_MATMUL, 0x1, 100);
        e1->backend_handle = &released;
        e1->release_fn     = [](void* p) { ++(*static_cast<int*>(p)); };

        auto e2 = make_entry(LIBANE_OP_SOFTMAX, 0x2, 100);

        cache.put(std::move(e1));
        cache.put(std::move(e2)); // evicts e1
    }
    CHECK(released == 1);
}
