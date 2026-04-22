/**
 * probe_qos.m — Which QoS level gives lowest ANE execution latency on M3 Pro?
 *
 * Claim from external ane.h:
 *   ANE_QOS_BACKGROUND = 9  "Fastest on M3 Pro! Best for training."
 *   (Counter-intuitive — most code uses QOS_CLASS_DEFAULT = 21.)
 *
 * Test protocol:
 *   Compile one model (identity conv1x1 [1,16,1,8]).
 *   Run 30 executions at each QoS level in the list below.
 *   Measure wall-clock time per evaluateWithQoS: call.
 *   Report: min, median, mean, max per QoS level.
 *
 * If QoS=9 is genuinely fastest, libane's hardcoded kQoS=21 should change.
 * If QoS=21 is fastest or they're equal, leave it alone.
 *
 * QoS values probed (from external ane.h + GCD headers):
 *   0  = QOS_CLASS_UNSPECIFIED / REALTIME
 *   9  = QOS_CLASS_BACKGROUND
 *   17 = QOS_CLASS_UTILITY
 *   21 = QOS_CLASS_DEFAULT  (libane current)
 *   25 = QOS_CLASS_USER_INITIATED
 *   33 = QOS_CLASS_USER_INTERACTIVE
 *
 * Build:
 *   clang -x objective-c++ -std=c++17 -fobjc-arc -framework Foundation \
 *         -framework IOSurface -lc++ \
 *         -o probe_qos probe_qos.m && ./probe_qos
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

static const char* kMIL =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "    {\"coremlc-component-MIL\", \"3510.2.1\"},"
    "    {\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1, 16, 1, 8]> x) {\n"
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
    "        tensor<fp16, [16,16,1,1]> W = const()[name=string(\"W\"),\n"
    "            val=tensor<fp16, [16,16,1,1]>(\n"
    "                BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),\n"
    "                         offset=uint64(64)))];\n"
    "        tensor<fp16, [1,16,1,8]> y = conv(\n"
    "            dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st,\n"
    "            weight=W, x=x)[name=string(\"y\")];\n"
    "    } -> (y);\n"
    "}\n";

static constexpr int kRuns = 30;
static constexpr int kWarmup = 5;

// ── Helpers ────────────────────────────────────────────────────────────────

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static IOSurfaceRef make_ios(size_t bytes) {
    size_t aligned = ((bytes + 63) & ~63UL);
    if (aligned < 49152) aligned = 49152;
    NSDictionary* props = @{
        (id)kIOSurfaceWidth:           @(aligned),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfacePixelFormat:     @0,
    };
    return IOSurfaceCreate((CFDictionaryRef)props);
}

static NSData* make_weight_blob(void) {
    uint16_t w[16 * 16] = {};
    uint16_t one_fp16 = 0x3C00;
    for (int i = 0; i < 16; ++i) w[i * 16 + i] = one_fp16;
    size_t weight_bytes = sizeof(w);
    size_t blob_len = 128 + weight_bytes;
    uint8_t* blob = (uint8_t*)calloc(1, blob_len);
    blob[0] = 0x01; blob[4] = 0x02;
    uint32_t magic = 0xDEADBEEF;
    memcpy(blob + 64, &magic, 4);
    blob[68] = 0x01;
    uint32_t sz = (uint32_t)weight_bytes;
    memcpy(blob + 72, &sz, 4);
    uint32_t data_offset = 128;
    memcpy(blob + 80, &data_offset, 4);  // ← required: data start offset
    memcpy(blob + 128, w, weight_bytes);
    NSData* d = [NSData dataWithBytes:blob length:blob_len];
    free(blob);
    return d;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(void) {
    @autoreleasepool {
        void* fh = dlopen(kFrameworkPath, RTLD_NOW | RTLD_LOCAL);
        if (!fh) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        Class cls_IOS   = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cls_Req   = NSClassFromString(@"_ANERequest");
        if (!cls_Desc || !cls_Model || !cls_IOS || !cls_Req) {
            fprintf(stderr, "[-] Required classes not found\n"); return 1;
        }

        // ── Compile once at QoS=21 ─────────────────────────────────────────
        printf("=== probe_qos ===\n");
        printf("Compiling identity conv1x1 [1,16,1,8] at QoS=21...\n");

        NSData* mil_data = [NSData dataWithBytes:kMIL length:strlen(kMIL)];
        NSData* wdata    = make_weight_blob();

        NSString* wkey = @"@model_path/weights/weight.bin";
        NSDictionary* weights_dict = @{ wkey: @{@"offset": @0, @"data": wdata} };

        SEL sel_desc  = sel_registerName("modelWithMILText:weights:optionsPlist:");
        typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
        id descriptor = ((DescFn)objc_msgSend)(cls_Desc, sel_desc, mil_data, weights_dict, nil);
        if (!descriptor) { fprintf(stderr, "[-] Descriptor failed\n"); return 1; }

        SEL sel_mm = sel_registerName("inMemoryModelWithDescriptor:");
        typedef id (*ModelFn)(Class, SEL, id);
        id model = ((ModelFn)objc_msgSend)(cls_Model, sel_mm, descriptor);
        if (!model) { fprintf(stderr, "[-] Model failed\n"); return 1; }

        SEL sel_hexID = sel_registerName("hexStringIdentifier");
        NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(model, sel_hexID);
        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* model_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id];
        NSString* wdir = [model_dir stringByAppendingPathComponent:@"weights"];
        [fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil];
        [mil_data writeToFile:[model_dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"] atomically:YES];

        NSError* err = nil;
        typedef BOOL (*QoSFn)(id, SEL, unsigned int, id, NSError**);
        if (!((QoSFn)objc_msgSend)(model, sel_registerName("compileWithQoS:options:error:"),
                                    21, @{}, &err) || err) {
            fprintf(stderr, "[-] Compile failed\n"); return 1;
        }
        err = nil;
        if (!((QoSFn)objc_msgSend)(model, sel_registerName("loadWithQoS:options:error:"),
                                    21, @{}, &err) || err) {
            fprintf(stderr, "[-] Load failed\n"); return 1;
        }
        printf("[+] Compiled and loaded.\n\n");

        // ── Setup IOSurfaces ───────────────────────────────────────────────
        IOSurfaceRef in_ios  = make_ios(16 * 8 * 2);
        IOSurfaceRef out_ios = make_ios(16 * 8 * 2);

        SEL sel_iosobj = sel_registerName("objectWithIOSurface:");
        typedef id (*SurfFn)(Class, SEL, IOSurfaceRef);
        id in_obj  = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, in_ios);
        id out_obj = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, out_ios);

        SEL sel_req = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:");
        typedef id (*ReqFn)(Class, SEL, NSArray*, NSArray*, NSArray*, NSArray*,
                            id, id, NSUInteger);

        SEL sel_eval = sel_registerName("evaluateWithQoS:options:request:error:");
        typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);

        // ── QoS levels to probe ────────────────────────────────────────────
        struct { unsigned int qos; const char* name; } levels[] = {
            { 0,  "REALTIME/UNSPECIFIED" },
            { 9,  "BACKGROUND (claimed fastest)" },
            { 17, "UTILITY" },
            { 21, "DEFAULT (libane current)" },
            { 25, "USER_INITIATED" },
            { 33, "USER_INTERACTIVE" },
        };
        constexpr int kNumLevels = sizeof(levels) / sizeof(levels[0]);

        printf("Running %d warmup + %d timed executions per QoS level.\n\n", kWarmup, kRuns);
        printf("%-8s  %-28s  %8s  %8s  %8s  %8s  %s\n",
               "QoS", "Class", "min_ms", "median", "mean", "max_ms", "ok/total");
        printf("%-8s  %-28s  %8s  %8s  %8s  %8s  %s\n",
               "--------", "----------------------------", "--------", "--------",
               "--------", "--------", "--------");

        unsigned int best_qos = 21;
        double best_median = 1e9;

        for (int li = 0; li < kNumLevels; ++li) {
            unsigned int qos = levels[li].qos;
            std::vector<double> times;
            int ok_count = 0;

            for (int r = 0; r < kWarmup + kRuns; ++r) {
                @autoreleasepool {
                    id request = ((ReqFn)objc_msgSend)(
                        cls_Req, sel_req,
                        @[in_obj], @[@0], @[out_obj], @[@0],
                        nil, nil, (NSUInteger)0);
                    if (!request) continue;

                    err = nil;
                    double t0 = now_ms();
                    BOOL ok = ((EvalFn)objc_msgSend)(model, sel_eval, qos, @{}, request, &err);
                    double elapsed = now_ms() - t0;

                    if (r >= kWarmup) {  // skip warmup in stats
                        times.push_back(elapsed);
                        if (ok && !err) ok_count++;
                    }
                }
            }

            if (times.empty()) {
                printf("%-8u  %-28s  (no successful runs)\n", qos, levels[li].name);
                continue;
            }

            double med = median(times);
            double mn  = mean(times);
            double mn_val = *std::min_element(times.begin(), times.end());
            double mx_val = *std::max_element(times.begin(), times.end());

            printf("%-8u  %-28s  %8.3f  %8.3f  %8.3f  %8.3f  %d/%d\n",
                   qos, levels[li].name, mn_val, med, mn, mx_val,
                   ok_count, (int)times.size());
            fflush(stdout);

            if (med < best_median) {
                best_median = med;
                best_qos    = qos;
            }
        }

        // ── Verdict ────────────────────────────────────────────────────────
        printf("\n=== VERDICT ===\n");
        printf("  Fastest QoS (by median): %u\n", best_qos);
        if (best_qos == 9) {
            printf("  [+] CONFIRMED: QoS=9 (BACKGROUND) is fastest on this chip.\n");
            printf("      libane kQoS=21 should be changed to 9 for execute path.\n");
            printf("      (Keep compile at 21 unless compile is also faster at 9.)\n");
        } else if (best_qos == 21) {
            printf("  [+] QoS=21 (DEFAULT) is fastest — libane is already optimal.\n");
            printf("      External ane.h claim about QoS=9 does not apply here.\n");
        } else {
            printf("  [?] Unexpected winner (QoS=%u). Inspect the table above.\n", best_qos);
        }

        CFRelease(in_ios);
        CFRelease(out_ios);
        printf("\nDone.\n");
        return 0;
    }
}
