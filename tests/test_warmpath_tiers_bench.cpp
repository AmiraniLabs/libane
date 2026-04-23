/**
 * test_warmpath_tiers_bench.cpp — Phase 3: measure the three-tier cost model.
 *
 * Measures each tier of HwxBackend's compile-path cascade for both a
 * weight-bearing workload (matmul) and a weight-free workload (activation
 * ops).  Numbers feed back into the hwx_backend.cpp header comment and the
 * published cost model.
 *
 * Tier 1 — MilBackend::try_warm_reconnect (~0.7 ms)
 *   Exercised by compiling the same graph twice.  Second compile hits the
 *   hexID-keyed URL cache populated on the first.
 *
 * Tier 2 — HwxEmitter cross-op patch + ane_load_hwx (~20–40 ms)
 *   Exercised by compiling op-A (e.g. RELU) then op-B (e.g. TANH) at the
 *   same shape.  op-A's cold compile seeds HwxEmitter's shape template;
 *   op-B's compile hits the cross-op patch path.
 *
 * Tier 3 — MilBackend::compile_group full cold (~100+ ms on macOS 26)
 *   Exercised by each first-time compile of a new (mil_text, weights) pair.
 *
 * Prints measurements via WARN — invoke with `-s` to see output.  Timing
 * assertions are loose (generous upper bounds only) since CI jitter, thermal
 * state, and aned's own disk cache make tight bounds unreliable.
 *
 * Tags: [bench][warmpath][tiers]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"
#include "../src/graph/hwx_backend.hpp"
#include "../src/graph/mil_backend.hpp"
#include "../src/graph/ane_graph.hpp"
#include "../src/graph/fusion_rules.hpp"
#include <chrono>
#include <sstream>
#include <vector>

static double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double measure(const std::function<void()>& f) {
    double t0 = ms_now();
    f();
    return ms_now() - t0;
}

// ── Activation-op tier bench: RELU → TANH cross-op at same shape ──────────────

TEST_CASE("warmpath tiers — activation ops (relu, tanh, sigmoid)",
          "[bench][warmpath][tiers]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    const int C = 64, S = 512;
    libane::mil::TensorShape shape{1, C, 1, S};

    libane::graph::MilBackend mil;
    libane::graph::HwxBackend hwx(&mil);

    auto build = [&](libane_op_t op) {
        libane::graph::AneGraph g;
        auto x = g.add_input("x", shape);
        auto y = g.add_op(op, {x}, shape);
        g.mark_output(y, "y");
        return g;
    };
    auto groups_of = [&](const libane::graph::AneGraph& g) {
        return libane::graph::FusionRules::compute_groups(g);
    };

    // Tier 3 — cold compile of RELU (populates both MilBackend url_cache_
    // and HwxEmitter shape template).
    auto g_relu = build(LIBANE_OP_RELU);
    auto grp_relu = groups_of(g_relu).front();

    libane::runtime::AneProgram* relu_cold = nullptr;
    double t_relu_cold = measure([&](){
        relu_cold = hwx.compile_group(g_relu, grp_relu, "relu_cold");
    });
    if (!relu_cold) {
        WARN("relu cold failed: " << libane::runtime::ane_last_error());
        return;
    }
    WARN("Tier 3 (RELU cold compile):  " << t_relu_cold << " ms");

    // Tier 1 — compile RELU again.  Same hexID → try_warm_reconnect hit.
    libane::runtime::AneProgram* relu_warm = nullptr;
    double t_relu_warm = measure([&](){
        relu_warm = hwx.compile_group(g_relu, grp_relu, "relu_warm");
    });
    REQUIRE(relu_warm);
    WARN("Tier 1 (RELU warm reconnect): " << t_relu_warm << " ms");
    CHECK(t_relu_warm < t_relu_cold / 2.0);  // at least 2x speedup

    // Tier 2 — compile TANH at same shape.  HwxEmitter has RELU template;
    // cross-op patch emits TANH bytes, ane_load_hwx pre-stages them.
    auto g_tanh = build(LIBANE_OP_TANH);
    auto grp_tanh = groups_of(g_tanh).front();

    libane::runtime::AneProgram* tanh_xop = nullptr;
    double t_tanh_xop = measure([&](){
        tanh_xop = hwx.compile_group(g_tanh, grp_tanh, "tanh_crossop");
    });
    REQUIRE(tanh_xop);
    WARN("Tier 2 (TANH cross-op patch): " << t_tanh_xop << " ms");

    // Tier 1 — compile TANH again.  The cross-op path's ane_load_hwx
    // populated MilBackend's cache under TANH's hexID, so this should
    // hit try_warm_reconnect too.
    libane::runtime::AneProgram* tanh_warm = nullptr;
    double t_tanh_warm = measure([&](){
        tanh_warm = hwx.compile_group(g_tanh, grp_tanh, "tanh_warm");
    });
    REQUIRE(tanh_warm);
    WARN("Tier 1 (TANH warm reconnect): " << t_tanh_warm << " ms");

    // Summary
    std::ostringstream ss;
    ss << "\n────── Activation ops tier summary (C=" << C << ", S=" << S << ") ──────\n"
       << "  Tier 3 (cold)         RELU: " << t_relu_cold << " ms\n"
       << "  Tier 1 (reconnect)    RELU: " << t_relu_warm << " ms\n"
       << "  Tier 2 (cross-op)     TANH: " << t_tanh_xop  << " ms\n"
       << "  Tier 1 (reconnect)    TANH: " << t_tanh_warm << " ms\n";
    WARN(ss.str());

    libane::runtime::ane_unload(tanh_warm);
    libane::runtime::ane_unload(tanh_xop);
    libane::runtime::ane_unload(relu_warm);
    libane::runtime::ane_unload(relu_cold);
}

// ── Matmul tier bench (weight-bearing, HwxEmitter not applicable) ────────────

TEST_CASE("warmpath tiers — matmul (cold vs warm reconnect)",
          "[bench][warmpath][tiers]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    const int IC = 64, OC = 64, SP = 128;
    std::vector<uint16_t> w(static_cast<size_t>(IC) * OC, 0x3C00u);  // fp16 1.0

    libane::graph::MilBackend mil;
    // Matmul is weight-bearing → owned by MilBackend, not HwxBackend.
    // Exercise MilBackend directly.

    libane::graph::AneGraph g;
    auto x = g.add_input("x", libane::mil::TensorShape{1, IC, 1, SP});
    auto y = g.add_op(LIBANE_OP_MATMUL, {x},
                      libane::mil::TensorShape{1, OC, 1, SP},
                      w.data(), w.size() * sizeof(uint16_t));
    g.mark_output(y, "y");

    auto group = libane::graph::FusionRules::compute_groups(g).front();

    libane::runtime::AneProgram* cold = nullptr;
    double t_cold = measure([&](){
        cold = mil.compile_group(g, group, "matmul_cold");
    });
    if (!cold) {
        WARN("matmul cold failed: " << libane::runtime::ane_last_error());
        return;
    }
    WARN("Tier 3 (MATMUL cold compile):   " << t_cold << " ms");

    libane::runtime::AneProgram* warm = nullptr;
    double t_warm = measure([&](){
        warm = mil.compile_group(g, group, "matmul_warm");
    });
    REQUIRE(warm);
    WARN("Tier 1 (MATMUL warm reconnect): " << t_warm << " ms");

    CHECK(warm->hex_id == cold->hex_id);
    CHECK(t_warm < t_cold / 2.0);  // at least 2x speedup

    std::ostringstream ss;
    ss << "\n────── Matmul tier summary (IC=" << IC << ", OC=" << OC
       << ", SP=" << SP << ") ──────\n"
       << "  Tier 3 (cold)       MATMUL: " << t_cold << " ms\n"
       << "  Tier 1 (reconnect)  MATMUL: " << t_warm << " ms\n"
       << "  Speedup:                    " << (t_cold / t_warm) << "x\n";
    WARN(ss.str());

    libane::runtime::ane_unload(warm);
    libane::runtime::ane_unload(cold);
}
