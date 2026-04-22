/**
 * test_mil_backend_warmpath.cpp — Phase 1 regression for MilBackend's
 * hexID-keyed reconnect cache.
 *
 * Compiling the same graph twice through MilBackend should hit
 * try_warm_reconnect on the second call, completing in reconnect-path
 * time (~1-3 ms) rather than paying the full cold compile (~100+ ms).
 *
 * The graph uses MATMUL so we exercise a weight-bearing path — HwxBackend
 * (activation ops) would not otherwise need MilBackend's warm path, so
 * matmul is the honest test of the new primitive.
 *
 * Tags: [pathc][warmpath][milbackend]
 */

#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"
#include "../src/graph/mil_backend.hpp"
#include "../src/graph/ane_graph.hpp"
#include "../src/graph/fusion_rules.hpp"
#include <chrono>
#include <vector>

static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ── Primitive: ane_compute_hex_id determinism + identity ──────────────────────

TEST_CASE("ane_compute_hex_id: identical inputs produce identical hexID",
          "[pathc][warmpath][milbackend]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    static const char kMIL[] =
        "program(1.3)\n[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
        "        tensor<fp16, [1,16,1,64]> y = relu(x=x)"
        "[name=string(\"hexid_test\")];\n    } -> (y);\n}\n";

    std::string h1 = libane::runtime::ane_compute_hex_id(kMIL, {});
    std::string h2 = libane::runtime::ane_compute_hex_id(kMIL, {});

    CHECK_FALSE(h1.empty());
    CHECK(h1 == h2);
}

TEST_CASE("ane_compute_hex_id: distinct MIL text produces distinct hexID",
          "[pathc][warmpath][milbackend]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    static const char kReluMIL[] =
        "program(1.3)\n[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
        "        tensor<fp16, [1,16,1,64]> y = relu(x=x)"
        "[name=string(\"distinct_relu\")];\n    } -> (y);\n}\n";
    static const char kTanhMIL[] =
        "program(1.3)\n[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
        "        tensor<fp16, [1,16,1,64]> y = tanh(x=x)"
        "[name=string(\"distinct_tanh\")];\n    } -> (y);\n}\n";

    std::string h_relu = libane::runtime::ane_compute_hex_id(kReluMIL, {});
    std::string h_tanh = libane::runtime::ane_compute_hex_id(kTanhMIL, {});

    CHECK_FALSE(h_relu.empty());
    CHECK_FALSE(h_tanh.empty());
    CHECK(h_relu != h_tanh);
}

// ── AneProgram::hex_id populated after cold compile ───────────────────────────

TEST_CASE("ane_compile: AneProgram::hex_id matches ane_compute_hex_id",
          "[pathc][warmpath][milbackend]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    static const char kMIL[] =
        "program(1.3)\n[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
        "        tensor<fp16, [1,16,1,64]> y = relu(x=x)"
        "[name=string(\"hexid_populate\")];\n    } -> (y);\n}\n";

    auto* prog = libane::runtime::ane_compile(kMIL, {}, "hexid_populate");
    if (!prog) { WARN("Compile failed (may hit slot limit): "
                      << libane::runtime::ane_last_error()); return; }

    CHECK_FALSE(prog->hex_id.empty());
    CHECK(prog->hex_id == libane::runtime::ane_compute_hex_id(kMIL, {}));

    libane::runtime::ane_unload(prog);
}

// ── MilBackend::try_warm_reconnect ────────────────────────────────────────────

TEST_CASE("MilBackend::try_warm_reconnect: cache miss returns nullptr",
          "[pathc][warmpath][milbackend]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    libane::graph::MilBackend backend;

    // Fresh backend, never compiled — cache is empty.
    static const char kMIL[] =
        "program(1.3)\n[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
        "        tensor<fp16, [1,16,1,64]> y = relu(x=x)"
        "[name=string(\"miss\")];\n    } -> (y);\n}\n";

    auto* prog = backend.try_warm_reconnect(kMIL, {}, "miss");
    CHECK(prog == nullptr);
}

// ── End-to-end: MilBackend compile_group hits reconnect on 2nd call ──────────

static libane::graph::AneGraph build_matmul_graph(int IC, int OC, int SP,
                                                    std::vector<uint16_t>& weights_storage) {
    // Weight layout expected by MilBackend's matmul path: fp16 [IC, OC].
    weights_storage.assign(static_cast<size_t>(IC) * OC, 0x3C00u /* fp16 1.0 */);

    libane::graph::AneGraph g;
    libane::mil::TensorShape in_shape{1, IC, 1, SP};
    libane::mil::TensorShape out_shape{1, OC, 1, SP};

    auto x = g.add_input("x", in_shape);
    auto y = g.add_op(LIBANE_OP_MATMUL, {x}, out_shape,
                     weights_storage.data(),
                     weights_storage.size() * sizeof(uint16_t));
    g.mark_output(y, "y");
    return g;
}

TEST_CASE("MilBackend::compile_group: second compile of same matmul hits reconnect",
          "[pathc][warmpath][milbackend]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

    const int IC = 16, OC = 16, SP = 64;
    std::vector<uint16_t> w_storage;
    auto g = build_matmul_graph(IC, OC, SP, w_storage);

    libane::graph::MilBackend backend;

    // Cold compile
    auto groups = libane::graph::FusionRules::compute_groups(g);
    REQUIRE_FALSE(groups.empty());
    const auto& group = groups.front();

    double t0 = ms_now();
    auto* cold = backend.compile_group(g, group, "matmul_cold");
    double cold_ms = ms_now() - t0;
    if (!cold) { WARN("Cold compile failed: "
                      << libane::runtime::ane_last_error()); return; }
    REQUIRE_FALSE(cold->hex_id.empty());
    WARN("cold=" << cold_ms << "ms  hex_id=" << cold->hex_id.substr(0, 16) << "...");

    // Second compile — should hit try_warm_reconnect.  Cold was still alive
    // (not unloaded), so aned's slot is guaranteed present.
    double t1 = ms_now();
    auto* warm = backend.compile_group(g, group, "matmul_warm");
    double warm_ms = ms_now() - t1;
    REQUIRE(warm);
    WARN("warm=" << warm_ms << "ms");

    // Same hexID — cache keyed on it correctly
    CHECK(warm->hex_id == cold->hex_id);

    // Warm path dramatically faster than cold.  Cold on macOS 26 is ~100+ ms;
    // reconnect is ~0.7-5 ms.  Allow generous headroom for CI jitter.
    CHECK(warm_ms < 50.0);
    if (cold_ms > 20.0) {
        CHECK(warm_ms < cold_ms / 3.0);  // at least 3x speedup
    }

    libane::runtime::ane_unload(warm);
    libane::runtime::ane_unload(cold);
}
