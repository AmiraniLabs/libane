/**
 * test_ane_load_hwx.mm — regression tests for ane_load_hwx on macOS 26+.
 *
 * On macOS 26+, aned withholds the compiled HWX binary from the client-visible
 * model_dir.  ane_load_hwx works around this by:
 *   1. Creating an _ANEInMemoryModel from the target-op MIL text (fixes hexID +
 *      localModelPath = TempDir/{hexID} deterministically).
 *   2. Pre-staging model.mil + model.hwx (patched bytes) at localModelPath.
 *   3. Calling compileWithQoS: — aned's compileAsNeeded path uses the
 *      pre-staged binary rather than running ANECCompile().
 *   4. loadWithQoS: to get a kernel handle.
 *
 * aned validates that the MIL and HWX compute the same operation.  The HwxEmitter
 * cross-op patch (RELU template + target-op config words) passes this check:
 * the binary structurally differs from a fresh target-op compile but produces
 * the same output, which is what aned verifies.
 *
 * Mechanism discovered via XPC-14..XPC-18 investigation (see
 * research/xpc-investigation/NOTES.md).  These tests pin the current behavior.
 *
 * Tags: [hwx][ane_load_hwx]
 */
#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"
#include "../src/graph/hwx_emitter.hpp"
#include "../src/graph/hwx_inline_capture.hpp"
#include "../src/core/mil_builder.hpp"

#import <Foundation/Foundation.h>
#include <chrono>
#include <string>
#include <vector>

static std::string write_mil_dir_(const char* subdir, const char* mil_text) {
    NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
                     [NSString stringWithUTF8String:subdir]];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
        withIntermediateDirectories:YES attributes:nil error:nil];
    NSString* mil = [dir stringByAppendingPathComponent:@"model.mil"];
    [[NSData dataWithBytes:mil_text length:strlen(mil_text)]
        writeToFile:mil atomically:YES];
    return [dir UTF8String];
}

// ── Pre-staged HWX bytes + matching MIL → aned uses the staged binary ─────────

TEST_CASE("ane_load_hwx: pre-staged matching HWX loads without recompile",
          "[hwx][ane_load_hwx]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

        const int C = 64, S = 512;
        const size_t N = (size_t)C * S;

        static const char kTanhMIL[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"load_hwx_tanh\")];\n    } -> (y);\n}\n";

        std::string tanh_src = write_mil_dir_("load_hwx_tanh_src", kTanhMIL);
        auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);
        REQUIRE_FALSE(tanh_hwx.empty());

        const std::string in_var  = "x";
        const std::string out_var = "y";
        const libane::mil::TensorShape shape{1, C, 1, S};
        std::string tanh_mil = libane::mil::MilBuilder::build_fused(
            in_var, shape,
            {libane::mil::MilBuilder::tanh_fragment(C, S, in_var, out_var)}).text;

        auto* prog = libane::runtime::ane_load_hwx(
            tanh_hwx, tanh_mil, C, S, in_var, out_var, "tanh_baseline");
        REQUIRE(prog);

        std::vector<uint16_t> in_d(N, 0x2E66u);  // fp16(0.1)
        auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(
            in_d.data(), C, S);
        auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
        REQUIRE(in_b);
        REQUIRE(out_b);

        bool ex = libane::runtime::ane_execute(
            prog, in_b->iosurface(), out_b->iosurface());
        REQUIRE(ex);

        std::vector<uint16_t> od(N);
        out_b->copy_to(od.data(), N * sizeof(uint16_t));
        CHECK(od[0] == 0x2E5Du);  // TANH fingerprint

        libane::runtime::ane_unload(prog);
    }
}

// ── Cross-op patch (RELU template + TANH op config) loads and runs TANH ───────

TEST_CASE("ane_load_hwx: cross-op patched HWX executes target op",
          "[hwx][ane_load_hwx]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }

        const int C = 64, S = 512;
        const size_t N = (size_t)C * S;

        static const char kReluMIL[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"load_hwx_relu\")];\n    } -> (y);\n}\n";
        static const char kTanhMIL[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"load_hwx_xop_tanh\")];\n    } -> (y);\n}\n";

        std::string relu_src = write_mil_dir_("load_hwx_xop_relu_src", kReluMIL);
        std::string tanh_src = write_mil_dir_("load_hwx_xop_tanh_src", kTanhMIL);
        auto relu_hwx = libane::graph::hwx_capture_inline(relu_src);
        auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);
        REQUIRE_FALSE(relu_hwx.empty());
        REQUIRE_FALSE(tanh_hwx.empty());

        libane::graph::HwxEmitter em;
        em.capture_from_bytes(relu_hwx, C, S, LIBANE_OP_RELU);
        em.capture_from_bytes(tanh_hwx, C, S, LIBANE_OP_TANH);
        REQUIRE(em.can_emit(C, S, LIBANE_OP_TANH));
        auto patched = em.emit(C, S, LIBANE_OP_TANH);
        REQUIRE_FALSE(patched.empty());

        const std::string in_var  = "x";
        const std::string out_var = "y";
        const libane::mil::TensorShape shape{1, C, 1, S};
        std::string tanh_mil = libane::mil::MilBuilder::build_fused(
            in_var, shape,
            {libane::mil::MilBuilder::tanh_fragment(C, S, in_var, out_var)}).text;

        auto t0 = std::chrono::steady_clock::now();
        auto* prog = libane::runtime::ane_load_hwx(
            patched, tanh_mil, C, S, in_var, out_var, "tanh_patched");
        auto t1 = std::chrono::steady_clock::now();
        double load_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
        INFO("ane_load_hwx time=" << load_ms << "ms");
        REQUIRE(prog);

        std::vector<uint16_t> in_d(N, 0x2E66u);
        auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(
            in_d.data(), C, S);
        auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
        REQUIRE(in_b);
        REQUIRE(out_b);

        bool ex = libane::runtime::ane_execute(
            prog, in_b->iosurface(), out_b->iosurface());
        REQUIRE(ex);

        std::vector<uint16_t> od(N);
        out_b->copy_to(od.data(), N * sizeof(uint16_t));
        CHECK(od[0] == 0x2E5Du);  // TANH fingerprint — patched binary ran TANH

        libane::runtime::ane_unload(prog);
    }
}

// ── Argument validation ───────────────────────────────────────────────────────

TEST_CASE("ane_load_hwx: rejects empty hwx_bytes", "[hwx][ane_load_hwx]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }
    auto* prog = libane::runtime::ane_load_hwx(
        {}, "stub", 4, 16, "x", "y", "empty_hwx");
    CHECK(prog == nullptr);
}

TEST_CASE("ane_load_hwx: rejects empty mil_text", "[hwx][ane_load_hwx]") {
    if (!libane_available()) { WARN("ANE unavailable — skipping"); return; }
    std::vector<uint8_t> stub(64, 0xFF);
    auto* prog = libane::runtime::ane_load_hwx(
        stub, "", 4, 16, "x", "y", "empty_mil");
    CHECK(prog == nullptr);
}

#else
TEST_CASE("ane_load_hwx: Apple-only", "[hwx]") { WARN("Apple-only"); }
#endif
