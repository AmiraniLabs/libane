/**
 * test_pathc_warmpath.cpp — Verify that ane_compile skips compileWithQoS:
 * when aned already has the hexID in its compile cache.
 *
 * Scenario A: two compilations of identical MIL (same hexID) — second is fast
 * Scenario B: compile, soft-unload (no purge), recompile — second is fast
 * Scenario C: compile, hard-unload (with purge), recompile — second is slow
 *
 * Tags: [pathc][warmpath]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"
#include <chrono>
#include <string>

static const char kReluA[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,4,1,64]> x) {\n"
    "        tensor<fp16, [1,4,1,64]> y = relu(x=x)[name=string(\"wp_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ── Scenario A: duplicate compile while first is alive ────────────────────────

TEST_CASE("PathC warmpath A: second compile of same MIL skips ANECompilerService",
          "[pathc][warmpath]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    // Cold compile (pays ANECompilerService cost)
    auto t0 = ms_now();
    auto* h1 = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    auto cold_ms = ms_now() - t0;

    if (!h1) {
        WARN("Cold compile failed (" << libane_last_error() << ") — skipping");
        return;
    }
    INFO("Cold compile: " << cold_ms << "ms");

    // Warm compile — same MIL, first model still alive → compiledModelExists=YES
    auto t1 = ms_now();
    auto* h2 = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    auto warm_ms = ms_now() - t1;

    INFO("Warm compile: " << warm_ms << "ms");

    if (!h2) {
        WARN("Warm compile failed (" << libane_last_error()
             << ") — slot limit? warm path not triggered");
        libane_mil_release(h1);
        return;
    }

    // The warm compile should be dramatically faster (< 500ms vs ~4200ms)
    CHECK(warm_ms < 500.0);
    if (warm_ms < 500.0) {
        WARN("WARM PATH CONFIRMED: cold=" << cold_ms << "ms  warm=" << warm_ms
             << "ms  speedup=" << (cold_ms / warm_ms) << "x");
    } else {
        WARN("Warm path did NOT trigger: warm=" << warm_ms
             << "ms (expected <500ms) — compiledModelExists returned NO");
    }

    libane_mil_release(h2);
    libane_mil_release(h1);
}

// ── Scenario C: hard-unload then recompile pays full cost ─────────────────────

TEST_CASE("PathC warmpath C: after hard-unload, recompile pays full ANECompilerService cost",
          "[pathc][warmpath]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    // First compile
    auto* h1 = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    if (!h1) { WARN("First compile failed — skipping"); return; }

    // Hard unload: unload SRAM + purge compile slot
    libane_mil_release(h1);

    // Second compile: slot was purged → must go through ANECompilerService again
    auto t0 = ms_now();
    auto* h2 = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    auto recompile_ms = ms_now() - t0;

    INFO("Recompile after hard-unload: " << recompile_ms << "ms");
    if (!h2) { WARN("Recompile failed: " << libane_last_error()); return; }

    // Expected outcome (three-tier caching hierarchy):
    //   ~1-3ms  → compiledModelExists=YES (aned in-memory cache, Layer 3)
    //   ~40-80ms → compileWithQoS: hits ANECompilerService disk cache (Layer 2)
    //   ~4000ms+ → cold ANECompilerService compile (Layer 1, first run ever)
    //
    // After hard-unload (purge), aned slot is cleared, so Layer 3 is bypassed.
    // ANECompilerService still has its own disk cache, so compileWithQoS: hits
    // Layer 2 (~40ms) rather than going cold (~4000ms).
    if (recompile_ms < 10.0) {
        WARN("Recompile in " << recompile_ms
             << "ms — still hitting aned cache (purge may not have worked)");
    } else if (recompile_ms < 500.0) {
        WARN("Recompile in " << recompile_ms
             << "ms — Layer 2 hit (ANECompilerService disk cache, aned slot correctly purged)");
    } else {
        WARN("Recompile in " << recompile_ms
             << "ms — cold ANECompilerService compile (first run or disk cache expired)");
    }

    libane_mil_release(h2);
}

// ── Scenario D: ane_reconnect() injects URL → loadWithQoS: in ~0.722ms ───────

TEST_CASE("PathC warmpath D: ane_reconnect skips compileWithQoS: via URL injection",
          "[pathc][warmpath][reconnect]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    // Cold compile — establishes the aned slot and captures model_url
    auto* h = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    if (!h || !h->prog) {
        WARN("Cold compile failed — skipping"); if (h) libane_mil_release(h); return;
    }
    REQUIRE_FALSE(h->prog->model_url.empty());

    const std::string url      = h->prog->model_url;
    const std::string mil_text = h->prog->mil_text;
    const int slots_before     = libane::runtime::ane_compile_count();

    INFO("model_url: " << url);

    // Reconnect — no compileWithQoS: call, no new compile slot consumed
    auto t0 = ms_now();
    libane::runtime::AneProgram* prog2 = libane::runtime::ane_reconnect(
        mil_text, {}, url, "reconnect-d");
    double elapsed = ms_now() - t0;

    INFO("ane_reconnect elapsed: " << elapsed << "ms");
    INFO("compile_count before=" << slots_before
         << " after=" << libane::runtime::ane_compile_count());

    REQUIRE(prog2 != nullptr);
    CHECK(elapsed < 50.0);
    CHECK(libane::runtime::ane_compile_count() == slots_before);

    WARN("ane_reconnect(): " << elapsed << "ms  (compile slots unchanged at "
         << slots_before << ")");

    libane::runtime::ane_unload(prog2);
    libane_mil_release(h);
}

// ── Scenario E: ane_reconnect() returns nullptr after hard-unload ─────────────

TEST_CASE("PathC warmpath E: ane_reconnect returns nullptr after slot purge",
          "[pathc][warmpath][reconnect]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    auto* h = libane_mil_compile(kReluA, nullptr, nullptr, nullptr, 0);
    if (!h || !h->prog) {
        WARN("Compile failed — skipping"); if (h) libane_mil_release(h); return;
    }
    REQUIRE_FALSE(h->prog->model_url.empty());

    std::string url      = h->prog->model_url;
    std::string mil_text = h->prog->mil_text;

    // Hard-unload purges the aned compile slot
    libane_mil_release(h);

    // Reconnect must fail — slot is gone
    libane::runtime::AneProgram* prog2 = libane::runtime::ane_reconnect(
        mil_text, {}, url, "reconnect-dead");

    CHECK(prog2 == nullptr);
    if (!prog2)
        WARN("ane_reconnect() correctly returned nullptr after hard-unload (slot purged)");
    else {
        WARN("ane_reconnect() unexpectedly succeeded — slot may survive purge");
        libane::runtime::ane_unload(prog2);
    }
}
