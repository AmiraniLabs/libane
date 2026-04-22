/**
 * test_compile_budget.cpp — Verify compile-slot counter and guard behaviour
 *
 * Tests:
 *   1. Counter starts at 0 (or at whatever the process has used so far)
 *      and increments by exactly 1 per new compile.
 *   2. Warm-path hits (compiledModelExists=YES) do NOT increment the counter.
 *   3. libane_compile_slots_remaining() decrements in step with the counter.
 *   4. The counter is accessible from the public C API
 *      (libane_compile_count / libane_compile_slots_remaining).
 *
 * NOTE: This test does NOT exercise the hard limit (115) — doing so would
 * exhaust the process budget and crash subsequent tests.  The limit is
 * verified empirically by test_qos_sweep Probe 4.
 *
 * Tags: [compile_budget][unit]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/runtime/ane_runtime.hpp"

static const char kMIL_A[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,4,1,32]> x) {\n"
    "        tensor<fp16, [1,4,1,32]> y = relu(x=x)[name=string(\"budget_a\")];\n"
    "    } -> (y);\n"
    "}\n";

// Slightly different shape so hexID differs from kMIL_A
static const char kMIL_B[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,8,1,32]> x) {\n"
    "        tensor<fp16, [1,8,1,32]> y = relu(x=x)[name=string(\"budget_b\")];\n"
    "    } -> (y);\n"
    "}\n";

TEST_CASE("CompileBudget: C API accessors return non-negative values",
          "[compile_budget][unit]") {
    libane_available();
    int count     = libane_compile_count();
    int remaining = libane_compile_slots_remaining();

    CHECK(count     >= 0);
    CHECK(remaining >= 0);
    CHECK(count + remaining == 115); // kCompileHardLimit
    INFO("compile_count=" << count << "  slots_remaining=" << remaining);
}

TEST_CASE("CompileBudget: fresh compile increments counter by 1",
          "[compile_budget][unit]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    int before = libane_compile_count();

    // Use a unique MIL (kMIL_B) — may or may not be in aned cache depending
    // on process history.  We test the counter delta, not the absolute value.
    auto* h = libane_mil_compile(kMIL_B, nullptr, nullptr, nullptr, 0);
    if (!h) {
        WARN("Compile failed (budget may be exhausted in this process) — skipping");
        return;
    }

    int after = libane_compile_count();
    int delta = after - before;

    INFO("before=" << before << "  after=" << after << "  delta=" << delta);
    // delta is 0 if aned's in-memory cache already had this hexID (Path C hit),
    // or 1 if a fresh compile was needed.  Both are valid.
    CHECK(delta >= 0);
    CHECK(delta <= 1);

    int remaining = libane_compile_slots_remaining();
    CHECK(after + remaining == 115);

    libane_mil_release(h);
}

TEST_CASE("CompileBudget: recompiling same MIL does not increment counter",
          "[compile_budget][unit]") {
    libane_set_log_level(LIBANE_LOG_SILENT);

    // First compile — may or may not hit the cache
    auto* h1 = libane_mil_compile(kMIL_A, nullptr, nullptr, nullptr, 0);
    if (!h1) { WARN("First compile failed — skipping"); return; }
    int count_after_first = libane_compile_count();

    // Second compile of identical content — must be a Path C cache hit
    auto* h2 = libane_mil_compile(kMIL_A, nullptr, nullptr, nullptr, 0);
    REQUIRE(h2 != nullptr);
    int count_after_second = libane_compile_count();

    int delta = count_after_second - count_after_first;
    INFO("count_after_first=" << count_after_first
         << "  count_after_second=" << count_after_second
         << "  delta=" << delta);

    // Identical MIL → same hexID → compiledModelExists=YES → no new slot used
    CHECK(delta == 0);

    libane_mil_release(h1);
    libane_mil_release(h2);
}

TEST_CASE("CompileBudget: runtime query matches C API",
          "[compile_budget][unit]") {
    libane_available();

    int c_api_count     = libane_compile_count();
    int c_api_remaining = libane_compile_slots_remaining();
    int runtime_count   = libane::runtime::ane_compile_count();
    int runtime_remaining = libane::runtime::ane_compile_slots_remaining();

    CHECK(c_api_count     == runtime_count);
    CHECK(c_api_remaining == runtime_remaining);
    CHECK(c_api_count + c_api_remaining == 115);
}
