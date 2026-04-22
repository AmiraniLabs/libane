/**
 * probe_perf_stats.m — _ANEPerformanceStats interface probe
 *
 * Tests whether _ANEPerformanceStats can be created and populated after ANE
 * execution on the current firmware/entitlement context.  Determines the
 * exact method names for hardware counters and their units.
 *
 * Build:
 *   clang -fobjc-arc -fmodules -framework Foundation \
 *         -o probe_perf_stats probe_perf_stats.m && ./probe_perf_stats
 *
 * If all counters read 0.0 after execution the class exists but may require
 * an entitlement.  If the class is not found it is simply unavailable on
 * this firmware version.
 *
 * hwExecutionTime units:
 *   Apple NSTimeInterval convention = seconds.  The probe prints the raw
 *   value and its ms conversion so units can be confirmed empirically.
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

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

static constexpr unsigned int kQoS = 21;  // QOS_CLASS_DEFAULT

// ── Minimal MIL program (identity conv1x1 on [1,16,1,8]) ─────────────────
// Small enough to compile quickly; big enough to produce measurable HW time.

static const char* kMilText =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "    {\"coremlc-component-MIL\", \"3510.2.1\"},"
    "    {\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1, 16, 1, 8]> x) {\n"
    "        %w = const()[val = tensor<fp16, [16, 1, 1, 16]>(\n"
    "                     file(\"@model_path/weights/weight.bin\", offset=64))];\n"
    "        %out = conv(x = %x, weight = %w, bias = nothing,\n"
    "                    strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0],\n"
    "                    dilations=[1,1], groups=1);\n"
    "        return %out;\n"
    "    } -> (tensor<fp16, [1, 16, 1, 8]>);\n"
    "}\n";

// ── Helpers ────────────────────────────────────────────────────────────────

static void dump_methods(Class cls, const char* label) {
    printf("\n[%s instance methods]\n", label);
    unsigned int count = 0;
    Method* methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; ++i) {
        printf("  - %s  (enc: %s)\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);

    printf("[%s class methods]\n", label);
    methods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; ++i) {
        printf("  + %s  (enc: %s)\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);
}

static IOSurfaceRef make_ios(size_t bytes) {
    size_t aligned = ((bytes + 63) & ~63UL);
    if (aligned < 49152) aligned = 49152;
    NSDictionary* props = @{
        (id)kIOSurfaceWidth:       @(aligned),
        (id)kIOSurfaceHeight:      @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((CFDictionaryRef)props);
}

static void build_weight_blob(uint8_t* blob, size_t weight_bytes,
                               const uint16_t* fp16_data) {
    // 64-byte file header
    memset(blob, 0, 128);
    blob[0] = 0x01; blob[4] = 0x02;
    // 64-byte chunk header
    uint32_t magic = 0xDEADBEEF;
    memcpy(blob + 64, &magic, 4);
    blob[68] = 0x01;
    uint32_t sz = (uint32_t)weight_bytes;
    memcpy(blob + 72, &sz, 4);
    // Weight data
    memcpy(blob + 128, fp16_data, weight_bytes);
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(void) {
    @autoreleasepool {
        void* fh = dlopen(kFrameworkPath, RTLD_NOW | RTLD_LOCAL);
        if (!fh) {
            fprintf(stderr, "dlopen failed: %s\n", dlerror());
            return 1;
        }

        // ── Resolve _ANEPerformanceStats ────────────────────────────────────
        Class cls_PS = NSClassFromString(@"_ANEPerformanceStats");
        if (!cls_PS) {
            printf("_ANEPerformanceStats: NOT FOUND on this firmware\n");
            return 0;
        }
        printf("_ANEPerformanceStats: found\n\n");
        dump_methods(cls_PS, "_ANEPerformanceStats");
        printf("\n");

        // ── Create an instance to check initial state ──────────────────────
        // NOTE: [[alloc] init] returns nil — +new must be used.
        id ps_alloc = [[cls_PS alloc] init];
        printf("  [[alloc] init] → %s\n", ps_alloc ? "non-nil" : "nil (expected)");

        id ps = [cls_PS new];
        if (!ps) {
            printf("[-] [+new] also returned nil — class requires entitlements\n");
            printf("\nDone.\n"); return 0;
        }
        printf("[+] Created _ANEPerformanceStats via +new\n");

        // Probe counters.  hwExecutionTime has encoding 'Q' = uint64_t (nanoseconds).
        // Print raw uint64_t AND the double interpretation to confirm units.
        printf("\n--- Counter values (before execution — expect 0) ---\n");
        NSArray* candidates = @[
            @"hwExecutionTime",     // confirmed: Q = uint64_t nanoseconds
            @"gpuSubmitTime",
            @"scheduleTime",
            @"schedulingLatency",
            @"compilationTime",
            @"ANEExecutionTime",
            @"neuralEngineExecutionTime",
        ];
        // hwExecutionTime is uint64_t; read with uint64_t and double both to confirm
        SEL sel_hwt_pre = sel_registerName("hwExecutionTime");
        if ([ps respondsToSelector:sel_hwt_pre]) {
            uint64_t raw = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, sel_hwt_pre);
            printf("  hwExecutionTime (uint64_t) → %llu ns  (%.6f ms)\n",
                   (unsigned long long)raw, (double)raw / 1e6);
        }
        // Other candidates as doubles
        for (NSString* sname in candidates) {
            if ([sname isEqualToString:@"hwExecutionTime"]) continue;
            SEL s = sel_registerName([sname UTF8String]);
            if ([ps respondsToSelector:s]) {
                @try {
                    double val = ((double(*)(id,SEL))objc_msgSend)(ps, s);
                    printf("  %s → %.9f\n", [sname UTF8String], val);
                } @catch (NSException* e) {
                    printf("  %s → EXCEPTION: %s\n", [sname UTF8String],
                           [[e reason] UTF8String]);
                }
            }
        }

        // ── Compile a minimal model for test execution ─────────────────────
        printf("\n--- Compiling minimal identity conv1x1 [1,16,1,8] ---\n");

        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        Class cls_IOS   = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cls_Req   = NSClassFromString(@"_ANERequest");

        if (!cls_Desc || !cls_Model || !cls_IOS || !cls_Req) {
            printf("[-] Required classes not found — cannot run execution test\n");
            printf("\nDone.\n"); return 0;
        }

        {
            // Build identity weight matrix: 16×16 fp16 identity
            uint16_t w_fp16[16 * 16];
            memset(w_fp16, 0, sizeof(w_fp16));
            uint16_t one_fp16 = 0x3C00; // 1.0 in fp16
            for (int i = 0; i < 16; ++i) w_fp16[i * 16 + i] = one_fp16;

            size_t weight_bytes = sizeof(w_fp16);
            size_t blob_bytes = 128 + weight_bytes;
            uint8_t* blob = (uint8_t*)malloc(blob_bytes);
            build_weight_blob(blob, weight_bytes, w_fp16);

            NSData* mil_data = [NSData dataWithBytes:kMilText length:strlen(kMilText)];
            NSData* wdata    = [NSData dataWithBytes:blob length:blob_bytes];
            free(blob);

            NSString* wkey = @"@model_path/weights/weight.bin";
            NSDictionary* weights_dict = @{
                wkey: @{@"offset": @64, @"data": wdata}
            };

            SEL sel_desc = sel_registerName("modelWithMILText:weights:optionsPlist:");
            typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
            id descriptor = ((DescFn)objc_msgSend)(cls_Desc, sel_desc,
                                                    mil_data, weights_dict, nil);
            if (!descriptor) {
                printf("[-] Descriptor creation failed\n");
                printf("\nDone.\n"); return 0;
            }

            SEL sel_mm = sel_registerName("inMemoryModelWithDescriptor:");
            typedef id (*ModelFn)(Class, SEL, id);
            id model = ((ModelFn)objc_msgSend)(cls_Model, sel_mm, descriptor);
            if (!model) { printf("[-] Model creation failed\n"); printf("\nDone.\n"); return 0; }

            SEL sel_hexID   = sel_registerName("hexStringIdentifier");
            SEL sel_compile = sel_registerName("compileWithQoS:options:error:");
            SEL sel_load    = sel_registerName("loadWithQoS:options:error:");

            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(model, sel_hexID);
            NSString* model_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id];
            NSString* weights_dir = [model_dir stringByAppendingPathComponent:@"weights"];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:weights_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil_data writeToFile:[model_dir stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES];
            [wdata writeToFile:[weights_dir stringByAppendingPathComponent:@"weight.bin"]
                    atomically:YES];

            NSError* error = nil;
            typedef BOOL (*QoSFn)(id, SEL, unsigned int, id, NSError**);
            BOOL ok = ((QoSFn)objc_msgSend)(model, sel_compile, kQoS, @{}, &error);
            if (!ok || error) {
                printf("[-] Compile failed: %s\n",
                       error ? [[error localizedDescription] UTF8String] : "unknown");
                printf("\nDone.\n"); return 0;
            }
            error = nil;
            ok = ((QoSFn)objc_msgSend)(model, sel_load, kQoS, @{}, &error);
            if (!ok || error) {
                printf("[-] Load failed: %s\n",
                       error ? [[error localizedDescription] UTF8String] : "unknown");
                printf("\nDone.\n"); return 0;
            }
            printf("[+] Compiled and loaded\n");

            // ── Execute with _ANEPerformanceStats ──────────────────────────
            size_t ios_bytes = 16 * 8 * 2;  // [1,16,1,8] fp16
            IOSurfaceRef in_ios  = make_ios(ios_bytes);
            IOSurfaceRef out_ios = make_ios(ios_bytes);

            SEL sel_iosobj = sel_registerName("objectWithIOSurface:");
            typedef id (*SurfFn)(Class, SEL, IOSurfaceRef);
            id in_obj  = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, in_ios);
            id out_obj = ((SurfFn)objc_msgSend)(cls_IOS, sel_iosobj, out_ios);

            // Create a fresh _ANEPerformanceStats instance for this execution
            id ps_exec = [[cls_PS alloc] init];

            SEL sel_req = sel_registerName(
                "requestWithInputs:inputIndices:outputs:outputIndices:"
                "weightsBuffer:perfStats:procedureIndex:");
            typedef id (*ReqFn)(Class, SEL,
                                NSArray*, NSArray*, NSArray*, NSArray*,
                                id, id, NSUInteger);
            id request = ((ReqFn)objc_msgSend)(
                cls_Req, sel_req,
                @[in_obj], @[@0], @[out_obj], @[@0],
                nil, ps_exec, (NSUInteger)0);

            if (!request) {
                printf("[-] Request creation failed\n");
                printf("\nDone.\n"); return 0;
            }

            printf("[+] Executing with perfStats...\n");
            SEL sel_eval = sel_registerName("evaluateWithQoS:options:request:error:");
            typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);
            error = nil;
            ok = ((EvalFn)objc_msgSend)(model, sel_eval, kQoS, @{}, request, &error);

            if (!ok || error) {
                printf("[-] Execution failed: %s\n",
                       error ? [[error localizedDescription] UTF8String] : "unknown");
            } else {
                printf("[+] Execution succeeded\n");
            }

            // ── Read counters after execution ──────────────────────────────
            printf("\n--- Counter values (after execution) ---\n");
            SEL sel_hwt = sel_registerName("hwExecutionTime");
            if ([ps_exec respondsToSelector:sel_hwt]) {
                uint64_t raw = ((uint64_t(*)(id,SEL))objc_msgSend)(ps_exec, sel_hwt);
                printf("  hwExecutionTime (uint64_t) → %llu ns  (%.6f ms)\n",
                       (unsigned long long)raw, (double)raw / 1e6);
                if (raw == 0)
                    printf("  NOTE: 0 ns — may need entitlements or +new is wrong init\n");
            }
            // Other candidates
            for (NSString* sname in candidates) {
                if ([sname isEqualToString:@"hwExecutionTime"]) continue;
                SEL s = sel_registerName([sname UTF8String]);
                if ([ps_exec respondsToSelector:s]) {
                    @try {
                        double val = ((double(*)(id,SEL))objc_msgSend)(ps_exec, s);
                        printf("  %s → %.9f\n", [sname UTF8String], val);
                    } @catch (...) {}
                }
            }

            // Run 5 more times to confirm repeatability
            printf("\n--- 5 repeated executions (hwExecutionTime, uint64_t ns) ---\n");
            if ([ps_exec respondsToSelector:sel_hwt]) {
                for (int i = 0; i < 5; ++i) {
                    id ps_i = [cls_PS new];
                    if (!ps_i) { printf("  [+new] failed on run %d\n", i+1); break; }
                    id req_i = ((ReqFn)objc_msgSend)(
                        cls_Req, sel_req,
                        @[in_obj], @[@0], @[out_obj], @[@0],
                        nil, ps_i, (NSUInteger)0);
                    if (req_i) {
                        error = nil;
                        ((EvalFn)objc_msgSend)(model, sel_eval, kQoS, @{}, req_i, &error);
                        uint64_t t_ns = ((uint64_t(*)(id,SEL))objc_msgSend)(ps_i, sel_hwt);
                        printf("  run %d: %llu ns  (%.6f ms)%s\n",
                               i + 1, (unsigned long long)t_ns, (double)t_ns / 1e6,
                               t_ns == 0 ? "  ← zero (entitlement?)" : "  ✓");
                    }
                    
                }
            } else {
                printf("  hwExecutionTime not found on instance\n");
            }

            CFRelease(in_ios);
            CFRelease(out_ios);
        }

        printf("\nDone.\n");
        return 0;
    }
}
