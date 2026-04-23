/**
 * test_cache_observability.cpp — libane_cache_stats / libane_cache_clear
 *
 * Verifies that the warm-path cache exposes accurate counters and footprint,
 * and that clear() empties entries while preserving cumulative counters
 * (so callers can reason about activity across clear cycles).
 *
 * The C API reports stats for the *calling thread's* MilBackend.  The test
 * drives the production dispatch path via libane_graph_compile() so the
 * thread-local cache is the one being exercised.
 *
 * Tags: [cache][observability]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include <cstring>
#include <vector>

static bool ane_ready() {
    if (!libane_available()) return false;
    return true;
}

// Helper: compile a small matmul graph via the C API and return a handle.
// Graph shape is arbitrary but deterministic — same inputs produce the same
// hex_id so the cache hits on the second call.
static libane_compiled_graph_t build_and_compile_matmul(int IC, int OC, int SP) {
    libane_graph_t g = libane_graph_create();
    if (!g) return nullptr;

    libane_shape_t in_shape  = {{1, IC, 1, SP}, 4};
    libane_shape_t out_shape = {{1, OC, 1, SP}, 4};

    uint32_t x = libane_graph_add_input(g, "x", in_shape);
    std::vector<uint16_t> w(static_cast<size_t>(IC) * OC, 0x3C00u);  // fp16 1.0
    uint32_t y = libane_graph_add_op(g, LIBANE_OP_MATMUL, &x, 1, out_shape,
                                      w.data(), w.size() * sizeof(uint16_t));
    libane_graph_mark_output(g, y, "y");

    libane_compiled_graph_t cg = libane_graph_compile(g);
    libane_graph_release(g);
    return cg;
}

TEST_CASE("libane_cache_stats: rejects NULL out pointer", "[cache][observability]") {
    CHECK(libane_cache_stats(nullptr) == LIBANE_ERR_INVALID_ARG);
}

TEST_CASE("libane_cache_stats: populated after a successful compile",
          "[cache][observability]") {
    if (!ane_ready()) { WARN("ANE unavailable — skipping"); return; }

    // Clear first so the test's entry counter starts from a known baseline.
    libane_cache_clear();

    libane_cache_stats_t before{};
    REQUIRE(libane_cache_stats(&before) == LIBANE_OK);
    CHECK(before.entries == 0);
    CHECK(before.bytes   == 0);
    // Counters survive clear — don't check hits/misses values, just that
    // stats returns them without error.

    auto* cg = build_and_compile_matmul(64, 64, 128);
    if (!cg) { WARN("Compile failed: " << libane_last_error()); return; }

    libane_cache_stats_t after{};
    REQUIRE(libane_cache_stats(&after) == LIBANE_OK);

    // One cold compile populated one entry.
    CHECK(after.entries       == before.entries + 1);
    CHECK(after.cold_compiles == before.cold_compiles + 1);
    // Footprint grew by at least the weight blob size (64*64*2 bytes).
    CHECK(after.bytes         >= before.bytes + 64 * 64 * 2);
    // First compile was a miss (cache was empty at lookup time).
    CHECK(after.misses        == before.misses + 1);
    CHECK(after.hits          == before.hits);  // no hits yet

    libane_compiled_graph_release(cg);
}

TEST_CASE("libane_cache_stats: hits increment on warm recompile",
          "[cache][observability]") {
    if (!ane_ready()) { WARN("ANE unavailable — skipping"); return; }

    // Cold compile once to populate.
    auto* cg1 = build_and_compile_matmul(32, 32, 64);
    if (!cg1) { WARN("Cold compile failed: " << libane_last_error()); return; }

    libane_cache_stats_t before{};
    REQUIRE(libane_cache_stats(&before) == LIBANE_OK);

    // Warm compile — same graph shape → same hex_id → reconnect hit.
    auto* cg2 = build_and_compile_matmul(32, 32, 64);
    REQUIRE(cg2);

    libane_cache_stats_t after{};
    REQUIRE(libane_cache_stats(&after) == LIBANE_OK);

    CHECK(after.hits          == before.hits + 1);
    CHECK(after.misses        == before.misses);       // no new miss
    CHECK(after.cold_compiles == before.cold_compiles); // no new cold
    CHECK(after.entries       == before.entries);       // same entry reused

    libane_compiled_graph_release(cg2);
    libane_compiled_graph_release(cg1);
}

TEST_CASE("libane_cache_clear: empties entries, preserves counters",
          "[cache][observability]") {
    if (!ane_ready()) { WARN("ANE unavailable — skipping"); return; }

    auto* cg = build_and_compile_matmul(16, 16, 32);
    if (!cg) { WARN("Compile failed: " << libane_last_error()); return; }

    libane_cache_stats_t before_clear{};
    REQUIRE(libane_cache_stats(&before_clear) == LIBANE_OK);
    CHECK(before_clear.entries >  0);
    CHECK(before_clear.bytes   >  0);
    const uint64_t cumulative_cold = before_clear.cold_compiles;
    const uint64_t cumulative_miss = before_clear.misses;

    libane_cache_clear();

    libane_cache_stats_t after_clear{};
    REQUIRE(libane_cache_stats(&after_clear) == LIBANE_OK);

    // Entries and bytes go to zero.
    CHECK(after_clear.entries == 0);
    CHECK(after_clear.bytes   == 0);

    // Counters are preserved across clear — cumulative activity is visible.
    CHECK(after_clear.cold_compiles == cumulative_cold);
    CHECK(after_clear.misses        == cumulative_miss);

    libane_compiled_graph_release(cg);
}

// ── Weight-sharing invariant ──────────────────────────────────────────────────
//
// After the shared_ptr<vector<WeightEntry>> refactor (gap #1), a live program
// and its cache entry should point at the same weight buffer — not two copies.
// Test: compile, snapshot cache bytes, release the program, snapshot again.
// If the cache copied weights, bytes stay flat (program-held bytes freed
// elsewhere, cache bytes unchanged).  If the cache shares weights via
// refcount, bytes also stay flat because cache holds the only remaining
// strong reference.  Either way, the cache's reported footprint is stable.
//
// The more diagnostic check: compile the SAME graph twice.  First compile is
// a cold populate (bytes grow by weight-size).  Second compile should hit
// reconnect without growing bytes at all — cache reused the same entry and
// did not allocate a second copy.

TEST_CASE("libane_cache_stats: warm reconnect does not double-allocate weights",
          "[cache][observability][weights]") {
    if (!ane_ready()) { WARN("ANE unavailable — skipping"); return; }

    // Fresh cache so weight-footprint math is simple.
    libane_cache_clear();

    // First compile — populates cache with a new entry.
    auto* cg1 = build_and_compile_matmul(128, 128, 64);
    if (!cg1) { WARN("Cold compile failed: " << libane_last_error()); return; }

    libane_cache_stats_t after_cold{};
    REQUIRE(libane_cache_stats(&after_cold) == LIBANE_OK);
    REQUIRE(after_cold.entries == 1);
    const size_t bytes_one_entry = after_cold.bytes;
    // 128×128 fp16 = 32KB weight blob; entry should account for at least that.
    CHECK(bytes_one_entry >= 128 * 128 * 2);

    // Second compile — same shape → same hex_id → cache hit.
    auto* cg2 = build_and_compile_matmul(128, 128, 64);
    REQUIRE(cg2);

    libane_cache_stats_t after_warm{};
    REQUIRE(libane_cache_stats(&after_warm) == LIBANE_OK);

    // Entries count unchanged (same hex_id reused).
    CHECK(after_warm.entries == 1);
    // Bytes unchanged — no second copy of weights was stashed anywhere
    // visible to the cache accounting.
    CHECK(after_warm.bytes   == bytes_one_entry);
    CHECK(after_warm.hits    == after_cold.hits + 1);

    libane_compiled_graph_release(cg2);
    libane_compiled_graph_release(cg1);
}
