/**
 * probe_compile_budget.m — Does ANE silently fail after ~119 compilations?
 *
 * Claim from external ane.h:
 *   "ANE silently fails after ~119 compilations per process.
 *    Use ane_compile_count() to monitor, restart process before hitting limit."
 *
 * Test protocol:
 *   Same model (identity conv1x1 [1,16,1,8]) compiled in a loop.
 *   Before each iteration the compiled output directory is deleted so the
 *   framework cannot serve from its E5 FlatBuffer cache — every call to
 *   compileWithQoS: must do real work.
 *
 *   After each compile+load, one execution is run and output is validated.
 *   "Silent failure" = compile/load succeed but execution produces wrong values.
 *
 * Run up to ANE_PROBE_MAX (125) iterations. Stop early on 3 consecutive failures.
 *
 * WARNING: Each compile of even a tiny model takes 1–5 s on first run.
 *          125 iterations = ~3–10 minutes. Progress is printed every iteration.
 *
 * Build:
 *   clang -x objective-c++ -std=c++17 -fobjc-arc -framework Foundation \
 *         -framework IOSurface -lc++ \
 *         -o probe_compile_budget probe_compile_budget.m && ./probe_compile_budget
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

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
static constexpr unsigned int kQoS = 21;
static constexpr int ANE_PROBE_MAX = 125;

static constexpr uint16_t kFP16_1 = 0x3C00;  // 1.0
static constexpr uint16_t kFP16_0 = 0x0000;  // 0.0

// ── Model: identity conv1x1 [1,16,1,8] (known to compile on M3 Pro) ───────

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

// ── Timer ──────────────────────────────────────────────────────────────────

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// ── Weight blob ────────────────────────────────────────────────────────────

static NSData* make_weight_blob(void) {
    uint16_t w[16 * 16] = {};
    for (int i = 0; i < 16; ++i) w[i * 16 + i] = kFP16_1;
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

// ── IOSurface helpers ──────────────────────────────────────────────────────

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

static void write_ios_fill(IOSurfaceRef ios, uint16_t val, int count) {
    IOSurfaceLock(ios, 0, nullptr);
    uint16_t* base = (uint16_t*)IOSurfaceGetBaseAddress(ios);
    for (int i = 0; i < count; ++i) base[i] = val;
    IOSurfaceUnlock(ios, 0, nullptr);
}

static float read_ios_first(IOSurfaceRef ios) {
    IOSurfaceLock(ios, kIOSurfaceLockReadOnly, nullptr);
    uint16_t h = ((const uint16_t*)IOSurfaceGetBaseAddress(ios))[0];
    IOSurfaceUnlock(ios, kIOSurfaceLockReadOnly, nullptr);
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t fb;
    if (e == 0)       fb = s | (m << 13);
    else if (e == 31) fb = s | 0x7F800000 | (m << 13);
    else              fb = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &fb, 4);
    return f;
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

        SEL sel_desc    = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL sel_mm      = sel_registerName("inMemoryModelWithDescriptor:");
        SEL sel_hexID   = sel_registerName("hexStringIdentifier");
        SEL sel_compile = sel_registerName("compileWithQoS:options:error:");
        SEL sel_load    = sel_registerName("loadWithQoS:options:error:");
        SEL sel_unload  = sel_registerName("unloadWithQoS:error:");
        SEL sel_eval    = sel_registerName("evaluateWithQoS:options:request:error:");
        SEL sel_iosobj  = sel_registerName("objectWithIOSurface:");
        SEL sel_req     = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:");

        NSData* mil_data = [NSData dataWithBytes:kMIL length:strlen(kMIL)];
        NSData* wdata    = make_weight_blob();

        IOSurfaceRef in_ios  = make_ios(16 * 8 * 2);
        IOSurfaceRef out_ios = make_ios(16 * 8 * 2);
        write_ios_fill(in_ios, kFP16_1, 16 * 8);

        printf("=== probe_compile_budget ===\n");
        printf("Testing claim: ANE silently fails after ~119 compilations per process.\n");
        printf("Model: identity conv1x1 [1,16,1,8]. Running %d iterations.\n", ANE_PROBE_MAX);
        printf("Each iteration deletes the compiled E5 to force a real compile.\n\n");
        printf("%-5s  %-11s  %-10s  %-8s  %-8s  %s\n",
               "iter", "compile_ms", "load_ms", "exec_ok", "out[0]", "status");
        printf("%-5s  %-11s  %-10s  %-8s  %-8s  %s\n",
               "-----", "-----------", "----------", "--------", "--------", "------");

        int first_compile_fail = -1;
        int first_load_fail    = -1;
        int first_exec_fail    = -1;
        int first_silent_fail  = -1;
        int consecutive_fails  = 0;

        // Capture the model_dir once (same hex ID each iteration, same model content)
        NSString* model_dir_global = nil;

        for (int i = 0; i < ANE_PROBE_MAX; ++i) {
            @autoreleasepool {
                NSString* wkey = @"@model_path/weights/weight.bin";
                NSDictionary* weights_dict = @{ wkey: @{@"offset": @0, @"data": wdata} };

                typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
                id descriptor = ((DescFn)objc_msgSend)(
                    cls_Desc, sel_desc, mil_data, weights_dict, nil);
                if (!descriptor) {
                    printf("%-5d  descriptor creation failed — stopping\n", i + 1);
                    if (first_compile_fail < 0) first_compile_fail = i + 1;
                    return 1;
                }

                typedef id (*ModelFn)(Class, SEL, id);
                id model = ((ModelFn)objc_msgSend)(cls_Model, sel_mm, descriptor);
                if (!model) {
                    printf("%-5d  model creation failed — stopping\n", i + 1);
                    if (first_compile_fail < 0) first_compile_fail = i + 1;
                    return 1;
                }

                NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(model, sel_hexID);
                NSFileManager* fm = [NSFileManager defaultManager];
                NSString* model_dir = [NSTemporaryDirectory()
                                       stringByAppendingPathComponent:hex_id];
                if (!model_dir_global) model_dir_global = model_dir;

                // Delete previously compiled output so we force a fresh compile.
                // Leaves the dir itself but removes any cached E5 FlatBuffer.
                NSArray* contents = [fm contentsOfDirectoryAtPath:model_dir error:nil];
                for (NSString* f in contents) {
                    if ([f hasSuffix:@".hwx"] || [f hasSuffix:@".mil"]) {
                        [fm removeItemAtPath:[model_dir stringByAppendingPathComponent:f]
                                       error:nil];
                    }
                }
                // Ensure directories exist and write fresh files
                NSString* wdir = [model_dir stringByAppendingPathComponent:@"weights"];
                [fm createDirectoryAtPath:wdir withIntermediateDirectories:YES
                               attributes:nil error:nil];
                [mil_data writeToFile:[model_dir stringByAppendingPathComponent:@"model.mil"]
                           atomically:YES];
                [wdata writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"]
                        atomically:YES];

                // Compile
                NSError* err = nil;
                typedef BOOL (*QoSFn)(id, SEL, unsigned int, id, NSError**);
                double t0 = now_ms();
                BOOL compiled = ((QoSFn)objc_msgSend)(model, sel_compile, kQoS, @{}, &err);
                double compile_ms = now_ms() - t0;

                if (!compiled || err) {
                    printf("%-5d  %-11.0f  %-10s  %-8s  %-8s  COMPILE_FAIL: %s\n",
                           i + 1, compile_ms, "-", "-", "-",
                           err ? [[err localizedDescription] UTF8String] : "unknown");
                    if (first_compile_fail < 0) first_compile_fail = i + 1;
                    if (++consecutive_fails >= 3) break;
                    continue;
                }

                // Load
                err = nil;
                double t1 = now_ms();
                BOOL loaded = ((QoSFn)objc_msgSend)(model, sel_load, kQoS, @{}, &err);
                double load_ms = now_ms() - t1;

                if (!loaded || err) {
                    printf("%-5d  %-11.0f  %-10.0f  %-8s  %-8s  LOAD_FAIL: %s\n",
                           i + 1, compile_ms, load_ms, "-", "-",
                           err ? [[err localizedDescription] UTF8String] : "unknown");
                    if (first_load_fail < 0) first_load_fail = i + 1;
                    if (++consecutive_fails >= 3) break;
                    continue;
                }

                // Execute
                write_ios_fill(out_ios, kFP16_0, 1);  // reset first element

                typedef id (*SurfFn)(Class, SEL, IOSurfaceRef);
                id in_obj  = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, in_ios);
                id out_obj = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, out_ios);
                typedef id (*ReqFn)(Class, SEL, NSArray*, NSArray*, NSArray*, NSArray*,
                                    id, id, NSUInteger);
                id request = ((ReqFn)objc_msgSend)(
                    cls_Req, sel_req,
                    @[in_obj], @[@0], @[out_obj], @[@0], nil, nil, (NSUInteger)0);

                BOOL exec_ok = NO;
                err = nil;
                if (request) {
                    typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);
                    exec_ok = ((EvalFn)objc_msgSend)(model, sel_eval, kQoS, @{}, request, &err);
                }

                float out0 = read_ios_first(out_ios);
                // Identity model: output should be non-zero when input is all 1.0
                // (exact value depends on weight semantics, but 0.0 = silent failure)
                bool seems_ok = (fabsf(out0) > 0.01f);

                const char* status = "OK";
                if (!exec_ok) {
                    status = "EXEC_FAIL";
                    if (first_exec_fail < 0) first_exec_fail = i + 1;
                    consecutive_fails++;
                } else if (!seems_ok) {
                    status = "SILENT_FAIL(zero)";
                    if (first_silent_fail < 0) first_silent_fail = i + 1;
                    consecutive_fails++;
                } else {
                    consecutive_fails = 0;
                }

                printf("%-5d  %-11.0f  %-10.0f  %-8s  %-8.3f  %s\n",
                       i + 1, compile_ms, load_ms,
                       exec_ok ? "YES" : "NO", out0, status);
                fflush(stdout);

                // Unload to free SRAM before next iteration
                err = nil;
                typedef BOOL (*UnloadFn)(id, SEL, unsigned int, NSError**);
                ((UnloadFn)objc_msgSend)(model, sel_unload, kQoS, &err);

                if (consecutive_fails >= 3) {
                    printf("\n[!] 3 consecutive failures — stopping early.\n");
                    break;
                }
            }
        }

        // ── Summary ────────────────────────────────────────────────────────
        printf("\n=== VERDICT ===\n");
        if (first_compile_fail > 0)
            printf("  Hard compile failure first at iteration: %d\n", first_compile_fail);
        else
            printf("  No compile failures in %d iterations\n", ANE_PROBE_MAX);

        if (first_load_fail > 0)
            printf("  Hard load failure first at iteration: %d\n", first_load_fail);
        else
            printf("  No load failures in %d iterations\n", ANE_PROBE_MAX);

        if (first_exec_fail > 0)
            printf("  Execution failure first at iteration: %d\n", first_exec_fail);
        else
            printf("  No execution failures in %d iterations\n", ANE_PROBE_MAX);

        if (first_silent_fail > 0)
            printf("  Silent wrong-output first at iteration: %d\n", first_silent_fail);
        else
            printf("  No silent failures in %d iterations\n", ANE_PROBE_MAX);

        int any_fail = (first_compile_fail > 0 ? first_compile_fail :
                        first_load_fail > 0    ? first_load_fail    :
                        first_exec_fail > 0    ? first_exec_fail    :
                        first_silent_fail > 0  ? first_silent_fail  : -1);
        if (any_fail < 0) {
            printf("\n[+] No failures in %d compilations.\n", ANE_PROBE_MAX);
            printf("    The 119-compile budget claim is not observed on this firmware.\n");
            printf("    Either it applies only to more complex models, or a higher macOS\n");
            printf("    version fixed/changed the behavior.\n");
        } else {
            printf("\n[-] First failure at iteration %d — compile budget confirmed.\n", any_fail);
            printf("    Add compile count tracking and restart process before limit.\n");
        }

        CFRelease(in_ios);
        CFRelease(out_ios);
        printf("\nDone.\n");
        return 0;
    }
}
