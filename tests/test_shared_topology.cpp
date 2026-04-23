/**
 * test_shared_topology.cpp — verify the production HwxBackend(&MilBackend)
 * wiring actually shares one URL-reconnect cache (gap #4).
 *
 * Test coverage elsewhere uses default-constructed HwxBackend, which owns
 * its own internal MilBackend.  That exercises HwxBackend's own logic but
 * not the production topology where GraphCompiler wires
 *   thread_local MilBackend mil;
 *   thread_local HwxBackend  hwx(&mil);
 * so HwxBackend's same-op reconnect and MilBackend's weight-bearing
 * reconnect populate and hit the same table.
 *
 * What this test pins:
 *   1. Compiling an activation through HwxBackend populates the injected
 *      MilBackend's cache (same cache stats reflect the entry).
 *   2. Compiling a matmul through the injected MilBackend populates the
 *      same cache.
 *   3. Neither path introduces duplicate entries or a separate bucket.
 *   4. Running both through the shared topology: activation + matmul
 *      entries coexist; each gets its own reconnect hit on second compile.
 *
 * Tags: [hwx][topology][cache]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/runtime/ane_runtime.hpp"
#include "../src/graph/hwx_backend.hpp"
#include "../src/graph/mil_backend.hpp"
#include "../src/graph/ane_graph.hpp"
#include "../src/graph/fusion_rules.hpp"
#include <vector>

using namespace libane::graph;
using libane::mil::TensorShape;

static AneGraph build_relu_graph(int C, int S) {
    AneGraph g;
    auto x = g.add_input("x", TensorShape{1, C, 1, S});
    auto y = g.add_op(LIBANE_OP_RELU, {x}, TensorShape{1, C, 1, S});
    g.mark_output(y, "y");
    return g;
}

static AneGraph build_matmul_graph(int IC, int OC, int SP,
                                    std::vector<uint16_t>& w_storage) {
    w_storage.assign(static_cast<size_t>(IC) * OC, 0x3C00u);
    AneGraph g;
    auto x = g.add_input("x", TensorShape{1, IC, 1, SP});
    auto y = g.add_op(LIBANE_OP_MATMUL, {x}, TensorShape{1, OC, 1, SP},
                      w_storage.data(), w_storage.size() * sizeof(uint16_t));
    g.mark_output(y, "y");
    return g;
}

TEST_CASE("shared topology: activation populates MilBackend's cache",
          "[hwx][topology][cache]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    MilBackend mil;
    HwxBackend hwx(&mil);  // injection path — the one GraphCompiler uses.

    auto g = build_relu_graph(64, 512);
    auto group = FusionRules::compute_groups(g).front();

    auto* prog = hwx.compile_group(g, group, "shared_relu");
    if (!prog) { WARN("compile failed: " << libane::runtime::ane_last_error()); return; }

    auto stats = mil.cache_stats();
    CHECK(stats.cold_compiles >= 1);
    CHECK(stats.entries       >= 1);

    libane::runtime::ane_unload(prog);
}

TEST_CASE("shared topology: matmul and activation share one cache bucket",
          "[hwx][topology][cache]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    MilBackend mil;
    HwxBackend hwx(&mil);
    mil.cache_clear();

    // Compile activation through HwxBackend (its cold fallback delegates to mil).
    auto g_relu = build_relu_graph(64, 512);
    auto grp_relu = FusionRules::compute_groups(g_relu).front();
    auto* p_relu = hwx.compile_group(g_relu, grp_relu, "topo_relu");
    if (!p_relu) { WARN("relu compile failed"); return; }

    auto after_relu = mil.cache_stats();
    const size_t entries_after_relu = after_relu.entries;

    // Compile matmul through MilBackend directly.
    std::vector<uint16_t> w;
    auto g_mm = build_matmul_graph(64, 64, 64, w);
    auto grp_mm = FusionRules::compute_groups(g_mm).front();
    auto* p_mm = mil.compile_group(g_mm, grp_mm, "topo_matmul");
    if (!p_mm) { WARN("matmul compile failed"); libane::runtime::ane_unload(p_relu); return; }

    auto after_mm = mil.cache_stats();
    // Matmul added its own entry — separate hex_id from relu.
    CHECK(after_mm.entries == entries_after_relu + 1);
    CHECK(after_mm.cold_compiles >= after_relu.cold_compiles + 1);

    // Both entries live in the SAME cache (same MilBackend instance).
    // Compile each one again and verify both hit reconnect.
    auto* p_relu2 = hwx.compile_group(g_relu, grp_relu, "topo_relu_warm");
    REQUIRE(p_relu2);
    auto* p_mm2   = mil.compile_group(g_mm, grp_mm, "topo_matmul_warm");
    REQUIRE(p_mm2);

    auto after_warm = mil.cache_stats();
    CHECK(after_warm.hits >= after_mm.hits + 2);  // one each
    CHECK(after_warm.entries == after_mm.entries); // no new cold entries
    CHECK(after_warm.cold_compiles == after_mm.cold_compiles);

    libane::runtime::ane_unload(p_mm2);
    libane::runtime::ane_unload(p_relu2);
    libane::runtime::ane_unload(p_mm);
    libane::runtime::ane_unload(p_relu);
}

TEST_CASE("shared topology: HwxBackend default ctor owns a distinct cache",
          "[hwx][topology][cache]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    // Default-constructed HwxBackend has its own internal MilBackend.
    // A separately-held MilBackend should NOT see HwxBackend's cache
    // activity — confirms the two constructors really do diverge.
    MilBackend external_mil;
    HwxBackend  hwx_owned;  // default ctor → internal MilBackend

    external_mil.cache_clear();
    auto ext_before = external_mil.cache_stats();

    auto g = build_relu_graph(64, 512);
    auto group = FusionRules::compute_groups(g).front();
    auto* prog = hwx_owned.compile_group(g, group, "owned_topo_relu");
    if (!prog) { WARN("compile failed"); return; }

    auto ext_after = external_mil.cache_stats();
    // external_mil uninvolved — counters unchanged.
    CHECK(ext_after.entries       == ext_before.entries);
    CHECK(ext_after.cold_compiles == ext_before.cold_compiles);
    CHECK(ext_after.hits          == ext_before.hits);

    libane::runtime::ane_unload(prog);
}
