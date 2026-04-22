/**
 * test_qos_sweep.mm — Measure ANE latency across all 6 QoS levels
 *
 * Background: libane uses kQoS = 21 (QOS_CLASS_DEFAULT) for every
 * compileWithQoS:, loadWithQoS:, and evaluateWithQoS: call, based on
 * Orion's "proven value".  ANE-Training reports QoS Background (9) is
 * 42% faster than Default for ANE execution.  This probe measures all
 * six levels and reports median latency for each.
 *
 * QoS values (Apple qos_class_t):
 *   QOS_CLASS_USER_INTERACTIVE = 0x21 = 33
 *   QOS_CLASS_USER_INITIATED   = 0x19 = 25
 *   QOS_CLASS_DEFAULT          = 0x15 = 21   ← current kQoS
 *   QOS_CLASS_UTILITY          = 0x11 = 17
 *   QOS_CLASS_BACKGROUND       = 0x09 =  9   ← claimed 42% faster
 *   QOS_CLASS_UNSPECIFIED      = 0x00 =  0
 *
 * Measurements:
 *   Probe 1 — evaluateWithQoS: latency (100 warm iterations, median)
 *             Tests the hot inference path for each QoS level.
 *   Probe 2 — processRequest: latency (100 warm iterations, median)
 *             The faster Tier-0 path also takes a qos parameter.
 *   Probe 3 — loadWithQoS: latency (median of 5 unload/reload cycles)
 *             Tests whether QoS affects SRAM load time.
 *   Probe 4 — compile latency (one compile per level after purge)
 *             Tests whether QoS affects ANECompilerService round-trip.
 *
 * Tags: [qos][perf][probe]
 */

#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"
#include "../src/core/buffer_manager.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <IOSurface/IOSurface.h>

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// ── helpers ──────────────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// MIL: relu [1,16,1,64] — small enough to be fast, large enough to measure
static const char kMIL[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,16,1,64]> x) {\n"
    "        tensor<fp16, [1,16,1,64]> y = relu(x=x)[name=string(\"qos_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

struct QosLevel {
    unsigned int value;
    const char*  name;
};

static const QosLevel kLevels[] = {
    { 33, "USER_INTERACTIVE (33)" },
    { 25, "USER_INITIATED   (25)" },
    { 21, "DEFAULT          (21)" },   // current libane default
    { 17, "UTILITY          (17)" },
    {  9, "BACKGROUND        (9)" },   // claimed faster
    {  0, "UNSPECIFIED       (0)" },
};
static const int kNLevels = 6;

// ── Probe 1: evaluateWithQoS: latency ────────────────────────────────────────

TEST_CASE("QoS 1: evaluateWithQoS: latency across all levels", "[qos][perf]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Compile once — reuse the same model for all QoS levels
        auto* h = libane_mil_compile(kMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        REQUIRE(h->prog != nullptr);
        REQUIRE(h->prog->objc_model != nullptr);

        id model = (id)h->prog->objc_model;

        // Build a persistent request using freshly allocated IOSurface buffers.
        // [1,16,1,64] fp16 — same shape as kMIL.
        // io_alloc_bytes() from the compiled graph gives the padded alloc size.
        Class cls_surf_obj = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cls_request  = NSClassFromString(@"_ANERequest");
        SEL s_obj_surf = sel_registerName("objectWithIOSurface:");
        SEL s_req      = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:");
        SEL s_evaluate = sel_registerName("evaluateWithQoS:options:request:error:");

        static constexpr int kCh = 16, kSeq = 64;
        auto in_buf  = libane::global_buffer_pool().acquire_tensor(kCh, kSeq);
        auto out_buf = libane::global_buffer_pool().acquire_tensor(kCh, kSeq);
        if (!in_buf || !out_buf) {
            WARN("Buffer pool allocation failed — skipping evaluate probe");
            libane_mil_release(h);
            return;
        }

        IOSurfaceRef in_surf  = in_buf->iosurface();
        IOSurfaceRef out_surf = out_buf->iosurface();
        if (!in_surf || !out_surf) {
            WARN("IOSurface handles are null — skipping evaluate probe");
            libane_mil_release(h);
            return;
        }

        typedef id (*ObjFn)(Class, SEL, IOSurfaceRef);
        id in_obj  = ((ObjFn)objc_msgSend)(cls_surf_obj, s_obj_surf, in_surf);
        id out_obj = ((ObjFn)objc_msgSend)(cls_surf_obj, s_obj_surf, out_surf);
        if (!in_obj || !out_obj) {
            WARN("_ANEIOSurfaceObject creation failed — skipping evaluate probe");
            libane_mil_release(h);
            return;
        }

        typedef id (*ReqFn)(Class, SEL, NSArray*, NSArray*, NSArray*, NSArray*,
                            id, id, NSUInteger);
        id request = ((ReqFn)objc_msgSend)(cls_request, s_req,
            @[in_obj], @[@0], @[out_obj], @[@0], nil, nil, (NSUInteger)0);
        if (!request) {
            WARN("_ANERequest creation failed — skipping evaluate probe");
            libane_mil_release(h);
            return;
        }

        static constexpr int kWarmup = 10;
        static constexpr int kIter   = 100;

        WARN("=== evaluateWithQoS: latency (median of " << kIter << " iters) ===");

        double baseline_median = -1.0;

        for (int li = 0; li < kNLevels; ++li) {
            unsigned int qos = kLevels[li].value;

            // Warmup
            for (int i = 0; i < kWarmup; ++i) {
                NSError* err = nil;
                typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);
                ((EvalFn)objc_msgSend)(model, s_evaluate, qos, @{}, request, &err);
            }

            // Measure
            std::vector<double> samples;
            samples.reserve(kIter);
            bool any_error = false;
            for (int i = 0; i < kIter; ++i) {
                NSError* err = nil;
                typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);
                double t0 = now_ms();
                BOOL ok = ((EvalFn)objc_msgSend)(model, s_evaluate, qos, @{}, request, &err);
                double t1 = now_ms();
                if (!ok || err) { any_error = true; break; }
                samples.push_back(t1 - t0);
            }

            if (any_error) {
                WARN("  " << kLevels[li].name << " → ERROR (qos rejected?)");
                continue;
            }

            double med = median(samples);
            double mn  = mean(samples);
            double mn_val  = *std::min_element(samples.begin(), samples.end());
            double mx_val  = *std::max_element(samples.begin(), samples.end());

            if (qos == 21) baseline_median = med;

            double pct = (baseline_median > 0 && qos != 21)
                ? (baseline_median - med) / baseline_median * 100.0 : 0.0;

            char buf[256];
            if (qos == 21) {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.3fms  mean=%.3fms  [%.3f–%.3f]  ← baseline",
                    kLevels[li].name, med, mn, mn_val, mx_val);
            } else {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.3fms  mean=%.3fms  [%.3f–%.3f]  (%+.1f%% vs DEFAULT)",
                    kLevels[li].name, med, mn, mn_val, mx_val, pct);
            }
            WARN(buf);
        }

        libane_mil_release(h);
    }
}

// ── Probe 2: processRequest: latency ─────────────────────────────────────────

TEST_CASE("QoS 2: processRequest: latency across all levels", "[qos][perf]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        REQUIRE(h->prog != nullptr);

        auto* prog = h->prog;
        id inner_prog = (id)prog->objc_program; // _ANEProgramForEvaluation
        if (!inner_prog) {
            WARN("objc_program is null — processRequest: fast path not available, skipping");
            libane_mil_release(h);
            return;
        }

        // Build the 9-arg _ANERequest for processRequest:
        Class cls_surf_obj = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cls_request  = NSClassFromString(@"_ANERequest");
        SEL s_obj_surf = sel_registerName("objectWithIOSurface:");
        SEL s_req9     = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:");
        SEL s_proc     = sel_registerName(
            "processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:");

        static constexpr int kCh2 = 16, kSeq2 = 64;
        auto in_buf2  = libane::global_buffer_pool().acquire_tensor(kCh2, kSeq2);
        auto out_buf2 = libane::global_buffer_pool().acquire_tensor(kCh2, kSeq2);
        if (!in_buf2 || !out_buf2) {
            WARN("Buffer pool allocation failed — skipping processRequest probe");
            libane_mil_release(h);
            return;
        }
        IOSurfaceRef in_surf  = in_buf2->iosurface();
        IOSurfaceRef out_surf = out_buf2->iosurface();

        typedef id (*ObjFn)(Class, SEL, IOSurfaceRef);
        id in_obj  = ((ObjFn)objc_msgSend)(cls_surf_obj, s_obj_surf, in_surf);
        id out_obj = ((ObjFn)objc_msgSend)(cls_surf_obj, s_obj_surf, out_surf);

        typedef id (*ReqFn)(Class, SEL, NSArray*, NSArray*, NSArray*, NSArray*,
                            id, id, NSUInteger);
        id request = ((ReqFn)objc_msgSend)(cls_request, s_req9,
            @[in_obj], @[@0], @[out_obj], @[@0], nil, nil, (NSUInteger)0);
        if (!request) {
            WARN("_ANERequest creation failed — skipping processRequest probe");
            libane_mil_release(h);
            return;
        }

        // Get hex ID for processRequest:
        id model = (id)prog->objc_model;
        SEL s_hex = sel_registerName("hexStringIdentifier");
        NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(model, s_hex);

        static constexpr int kWarmup = 10;
        static constexpr int kIter   = 100;

        WARN("=== processRequest: latency (median of " << kIter << " iters) ===");

        double baseline_median = -1.0;

        for (int li = 0; li < kNLevels; ++li) {
            unsigned int qos = kLevels[li].value;

            // Warmup
            for (int i = 0; i < kWarmup; ++i) {
                NSError* err = nil; int rv = 0;
                typedef BOOL (*ProcFn)(id,SEL,id,id,unsigned int,
                                       NSUInteger,NSString*,id,int*,NSError**);
                ((ProcFn)objc_msgSend)(inner_prog, s_proc,
                    request, model, qos, (NSUInteger)0, hex_id, @{}, &rv, &err);
            }

            std::vector<double> samples;
            samples.reserve(kIter);
            bool any_error = false;
            for (int i = 0; i < kIter; ++i) {
                NSError* err = nil; int rv = 0;
                typedef BOOL (*ProcFn)(id,SEL,id,id,unsigned int,
                                       NSUInteger,NSString*,id,int*,NSError**);
                double t0 = now_ms();
                BOOL ok = ((ProcFn)objc_msgSend)(inner_prog, s_proc,
                    request, model, qos, (NSUInteger)0, hex_id, @{}, &rv, &err);
                double t1 = now_ms();
                if (!ok || err) { any_error = true; break; }
                samples.push_back(t1 - t0);
            }

            if (any_error) {
                WARN("  " << kLevels[li].name << " → ERROR");
                continue;
            }

            double med = median(samples);
            double mn  = mean(samples);
            double mn_val = *std::min_element(samples.begin(), samples.end());
            double mx_val = *std::max_element(samples.begin(), samples.end());

            if (qos == 21) baseline_median = med;
            double pct = (baseline_median > 0 && qos != 21)
                ? (baseline_median - med) / baseline_median * 100.0 : 0.0;

            char buf[256];
            if (qos == 21) {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.3fms  mean=%.3fms  [%.3f–%.3f]  ← baseline",
                    kLevels[li].name, med, mn, mn_val, mx_val);
            } else {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.3fms  mean=%.3fms  [%.3f–%.3f]  (%+.1f%% vs DEFAULT)",
                    kLevels[li].name, med, mn, mn_val, mx_val, pct);
            }
            WARN(buf);
        }

        libane_mil_release(h);
    }
}

// ── Probe 3: loadWithQoS: latency ────────────────────────────────────────────

TEST_CASE("QoS 3: loadWithQoS: latency across all levels", "[qos][perf]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Compile once — keep model alive (compiledModelExists = YES throughout)
        auto* h = libane_mil_compile(kMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        REQUIRE(h->prog != nullptr);

        id model = (id)h->prog->objc_model;

        SEL s_load   = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload = sel_registerName("unloadWithQoS:error:");

        static constexpr int kIter = 5; // load/unload is expensive, fewer reps

        WARN("=== loadWithQoS: latency (median of " << kIter << " unload+load cycles) ===");

        double baseline_median = -1.0;

        for (int li = 0; li < kNLevels; ++li) {
            unsigned int qos = kLevels[li].value;
            std::vector<double> samples;
            bool any_error = false;

            for (int i = 0; i < kIter; ++i) {
                // Unload (QoS 21 — we're only measuring load)
                NSError* uerr = nil;
                typedef BOOL (*UnloadFn)(id,SEL,unsigned int,NSError**);
                ((UnloadFn)objc_msgSend)(model, s_unload, 21u, &uerr);

                // Load with target QoS
                NSError* lerr = nil;
                typedef BOOL (*LoadFn)(id,SEL,unsigned int,id,NSError**);
                double t0 = now_ms();
                BOOL ok = ((LoadFn)objc_msgSend)(model, s_load, qos, @{}, &lerr);
                double t1 = now_ms();

                if (!ok || lerr) { any_error = true; break; }
                samples.push_back(t1 - t0);
            }

            if (any_error) {
                WARN("  " << kLevels[li].name << " → ERROR");
                // Try to restore loaded state
                NSError* rerr = nil;
                typedef BOOL (*LoadFn)(id,SEL,unsigned int,id,NSError**);
                ((LoadFn)objc_msgSend)(model, s_load, 21u, @{}, &rerr);
                continue;
            }

            double med = median(samples);
            if (qos == 21) baseline_median = med;
            double pct = (baseline_median > 0 && qos != 21)
                ? (baseline_median - med) / baseline_median * 100.0 : 0.0;

            char buf[256];
            if (qos == 21) {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.2fms  ← baseline",
                    kLevels[li].name, med);
            } else {
                snprintf(buf, sizeof(buf),
                    "  %s  median=%.2fms  (%+.1f%% vs DEFAULT)",
                    kLevels[li].name, med, pct);
            }
            WARN(buf);
        }

        libane_mil_release(h);
    }
}

// ── Probe 4: compile latency ──────────────────────────────────────────────────
// NOTE: Each compile consumes one slot from the ~119 limit.
// We use a unique MIL per level (tiny shape difference) to force a true compile.

TEST_CASE("QoS 4: compileWithQoS: latency across all levels", "[qos][perf]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Generate per-level unique MIL (different channel count so hexID differs)
        auto make_mil = [](int ch) -> std::string {
            return std::string(
                "program(1.3)\n"
                "[buildInfo = dict<string, string>({"
                "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, "
                "{\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}"
                "})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1,") +
                std::to_string(ch) + ",1,64]> x) {\n"
                "        tensor<fp16, [1," + std::to_string(ch) + ",1,64]> y"
                " = relu(x=x)[name=string(\"qos_c_relu\")];\n"
                "    } -> (y);\n"
                "}\n";
        };

        // Channel counts chosen to be distinct valid ANE shapes
        static const int kChannels[] = { 8, 12, 16, 20, 24, 28 };

        Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_model = NSClassFromString(@"_ANEInMemoryModel");
        SEL s_mil    = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inmem  = sel_registerName("inMemoryModelWithDescriptor:");
        SEL s_cme    = sel_registerName("compiledModelExists");
        SEL s_compile= sel_registerName("compileWithQoS:options:error:");
        SEL s_load   = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload = sel_registerName("unloadWithQoS:error:");
        SEL s_purge  = sel_registerName("purgeCompiledModel");

        WARN("=== compileWithQoS: latency (one compile per level, Layer 2 cache warm) ===");
        WARN("    (First compile per shape may be slower — Layer 1 cold)");

        double baseline_ms = -1.0;

        for (int li = 0; li < kNLevels; ++li) {
            unsigned int qos = kLevels[li].value;
            std::string mil = make_mil(kChannels[li]);
            NSData* mil_data = [NSData dataWithBytes:mil.data() length:mil.size()];

            typedef id (*DescFn)(Class,SEL,NSData*,NSDictionary*,id);
            typedef id (*ModelFn)(Class,SEL,id);

            id descriptor = ((DescFn)objc_msgSend)(cls_desc, s_mil, mil_data, @{}, nil);
            if (!descriptor) { WARN("  " << kLevels[li].name << " → descriptor nil"); continue; }

            id model = ((ModelFn)objc_msgSend)(cls_model, s_inmem, descriptor);
            if (!model) { WARN("  " << kLevels[li].name << " → model nil"); continue; }

            // Check if already compiled (Layer 2/3 cache)
            bool cached = false;
            if ([cls_model instancesRespondToSelector:s_cme])
                cached = (bool)((BOOL(*)(id,SEL))objc_msgSend)(model, s_cme);

            typedef BOOL (*QoSFn)(id,SEL,unsigned int,id,NSError**);
            NSError* cerr = nil;
            double t0 = now_ms();
            BOOL ok = ((QoSFn)objc_msgSend)(model, s_compile, qos, @{}, &cerr);
            double t1 = now_ms();
            double ms = t1 - t0;

            if (!ok || cerr) {
                WARN("  " << kLevels[li].name << " → COMPILE ERROR: "
                     << (cerr ? [[cerr localizedDescription] UTF8String] : "unknown"));
                [model release];
                continue;
            }

            // Load to confirm it works, then unload + purge
            NSError* lerr = nil;
            ((QoSFn)objc_msgSend)(model, s_load, 21u, @{}, &lerr);
            NSError* uerr = nil;
            typedef BOOL (*UnloadFn)(id,SEL,unsigned int,NSError**);
            ((UnloadFn)objc_msgSend)(model, s_unload, 21u, &uerr);
            // Purge slot so it doesn't count against the 119 limit permanently
            if ([model respondsToSelector:s_purge])
                ((void(*)(id,SEL))objc_msgSend)(model, s_purge);

            if (qos == 21) baseline_ms = ms;
            double pct = (baseline_ms > 0 && qos != 21)
                ? (baseline_ms - ms) / baseline_ms * 100.0 : 0.0;

            char buf[256];
            if (qos == 21) {
                snprintf(buf, sizeof(buf),
                    "  %s  %.1fms  (cache=%s)  ← baseline",
                    kLevels[li].name, ms, cached ? "YES" : "NO");
            } else {
                snprintf(buf, sizeof(buf),
                    "  %s  %.1fms  (cache=%s)  (%+.1f%% vs DEFAULT)",
                    kLevels[li].name, ms, cached ? "YES" : "NO", pct);
            }
            WARN(buf);
            [model release];
        }
    }
}

#else
TEST_CASE("QoS sweep: Apple-only", "[qos]") { WARN("Apple-only"); }
#endif
