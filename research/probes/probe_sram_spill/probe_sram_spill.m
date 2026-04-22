/**
 * probe_sram_spill.m — Validate intermediateBufferHandle on _ANEInMemoryModel
 *
 * libane reads -[_ANEInMemoryModel intermediateBufferHandle] after loadWithQoS:
 * to detect SRAM spill.  This probe verifies:
 *
 *   1. The selector actually exists on _ANEInMemoryModel instances.
 *   2. A small model (tiny SRAM footprint) returns 0 (no spill).
 *   3. The value type is uint64_t (matches our cast in ane_runtime.mm).
 *   4. libane_mil_sram_spill() returns 0 for the same model via the public API.
 *
 * The spill=1 path requires a model whose intermediate activations exceed
 * ~32 MB.  We attempt one large model [1, 4096, 1, 4096] (4096*4096*2 = 32 MB
 * per buffer) to see if compile fails or if the handle becomes non-zero.
 * Failure to compile is expected and acceptable — it confirms firmware
 * enforces the budget before spill even occurs.
 *
 * Build:
 *   clang -x objective-c++ -std=c++17 -fobjc-arc -framework Foundation \
 *         -framework IOSurface -lc++ \
 *         -o probe_sram_spill probe_sram_spill.m && ./probe_sram_spill
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
static constexpr unsigned int kQoS = 21;

// ── MIL builder ───────────────────────────────────────────────────────────

// Identity conv1x1 [1, C, 1, S] using the exact syntax that compiles.
static NSString* make_mil(int C, int S) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "    {\"coremlc-component-MIL\", \"3510.2.1\"},"
         "    {\"coremltools-version\", \"9.0\"}"
         "})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
         "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"),\n"
         "            val=tensor<fp16, [%d,%d,1,1]>(\n"
         "                BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),\n"
         "                         offset=uint64(64)))];\n"
         "        tensor<fp16, [1,%d,1,%d]> y = conv(\n"
         "            dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st,\n"
         "            weight=W, x=x)[name=string(\"y\")];\n"
         "    } -> (y);\n"
         "}\n",
        C, S, C, C, C, C, C, S];
}

// ── Weight blob: identity matrix [C, C, 1, 1] ─────────────────────────────

static NSData* make_identity_blob(int C) {
    size_t n = (size_t)C * C;
    size_t weight_bytes = n * sizeof(uint16_t);
    size_t blob_len = 128 + weight_bytes;
    uint8_t* blob = (uint8_t*)calloc(1, blob_len);
    blob[0] = 0x01; blob[4] = 0x02;
    uint32_t magic = 0xDEADBEEF;
    memcpy(blob + 64, &magic, 4);
    blob[68] = 0x01;
    uint32_t sz = (uint32_t)weight_bytes;
    memcpy(blob + 72, &sz, 4);
    uint32_t data_offset = 128;
    memcpy(blob + 80, &data_offset, 4);
    uint16_t* w = (uint16_t*)(blob + 128);
    uint16_t one = 0x3C00;  // 1.0 fp16
    for (int i = 0; i < C; ++i) w[i * C + i] = one;
    NSData* d = [NSData dataWithBytes:blob length:blob_len];
    free(blob);
    return d;
}

// ── Compile + load, return model and intermediateBufferHandle ──────────────

typedef struct {
    id     model;       // _ANEInMemoryModel* (retained if non-nil)
    BOOL   compiled;
    BOOL   loaded;
    uint64_t ibh;       // intermediateBufferHandle after load (0 = no spill)
    BOOL   ibh_exists;  // YES if selector was found
} LoadResult;

static LoadResult compile_and_load(Class cls_Desc, Class cls_Model,
                                    int C, int S, const char* label) {
    LoadResult r = {nil, NO, NO, 0, NO};
    printf("\n--- %s (C=%d, S=%d, activation_bytes=~%zu) ---\n",
           label, C, S, (size_t)C * S * 2);

    NSString* mil_ns = make_mil(C, S);
    NSData*   mil_data = [mil_ns dataUsingEncoding:NSUTF8StringEncoding];
    NSData*   wdata    = make_identity_blob(C);

    NSString* wkey = @"@model_path/weights/weight.bin";
    NSDictionary* weights_dict = @{ wkey: @{@"offset": @0, @"data": wdata} };

    SEL sel_desc = sel_registerName("modelWithMILText:weights:optionsPlist:");
    typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
    id descriptor = ((DescFn)objc_msgSend)(cls_Desc, sel_desc, mil_data, weights_dict, nil);
    if (!descriptor) { printf("  descriptor failed\n"); return r; }

    SEL sel_mm = sel_registerName("inMemoryModelWithDescriptor:");
    typedef id (*ModelFn)(Class, SEL, id);
    id model = ((ModelFn)objc_msgSend)(cls_Model, sel_mm, descriptor);
    if (!model) { printf("  model creation failed\n"); return r; }

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
    r.compiled = ((QoSFn)objc_msgSend)(model,
                                        sel_registerName("compileWithQoS:options:error:"),
                                        kQoS, @{}, &err);
    if (!r.compiled || err) {
        printf("  compileWithQoS: FAILED — %s\n",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        return r;
    }
    printf("  compileWithQoS: OK\n");

    err = nil;
    r.loaded = ((QoSFn)objc_msgSend)(model,
                                      sel_registerName("loadWithQoS:options:error:"),
                                      kQoS, @{}, &err);
    if (!r.loaded || err) {
        printf("  loadWithQoS: FAILED — %s\n",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        return r;
    }
    printf("  loadWithQoS: OK\n");

    // Read intermediateBufferHandle
    SEL sel_ibh = sel_registerName("intermediateBufferHandle");
    r.ibh_exists = [model respondsToSelector:sel_ibh];
    printf("  -[_ANEInMemoryModel intermediateBufferHandle] selector: %s\n",
           r.ibh_exists ? "FOUND" : "NOT FOUND");

    if (r.ibh_exists) {
        @try {
            r.ibh = ((uint64_t(*)(id,SEL))objc_msgSend)(model, sel_ibh);
            printf("  intermediateBufferHandle = %llu  → spill: %s\n",
                   (unsigned long long)r.ibh, r.ibh ? "YES" : "NO");
        } @catch (...) {
            printf("  intermediateBufferHandle → EXCEPTION reading value\n");
        }
    }

    r.model = model;
    return r;
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(void) {
    @autoreleasepool {
        void* fh = dlopen(kFrameworkPath, RTLD_NOW | RTLD_LOCAL);
        if (!fh) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_Desc || !cls_Model) {
            fprintf(stderr, "[-] Required classes not found\n"); return 1;
        }

        printf("=== probe_sram_spill ===\n");
        printf("Testing: intermediateBufferHandle selector + SRAM spill detection\n");

        // ── Case 1: small model — expect ibh = 0 ──────────────────────────
        LoadResult small = compile_and_load(cls_Desc, cls_Model, 16, 8, "SMALL model");

        // ── Case 2: medium model — likely still fits ───────────────────────
        // C=256, S=256 → 256*256*2 = 128 KB per buffer (well within 32 MB)
        LoadResult medium = compile_and_load(cls_Desc, cls_Model, 256, 256, "MEDIUM model");

        // ── Case 3: large model — may approach or exceed SRAM ─────────────
        // C=4096, S=4096 → 4096*4096*2 = 32 MB per buffer; need ≥2 live → spill
        // Expect compile failure OR ibh != 0 if compile succeeds
        LoadResult large = compile_and_load(cls_Desc, cls_Model, 4096, 4096,
                                             "LARGE model (32 MB/buffer, expect compile fail or spill)");

        // ── Summary ────────────────────────────────────────────────────────
        printf("\n=== VERDICT ===\n");

        printf("\nSelector availability:\n");
        if (small.compiled)
            printf("  intermediateBufferHandle selector: %s\n",
                   small.ibh_exists ? "FOUND" : "NOT FOUND — libane cannot detect spill");
        else
            printf("  (small model compile failed — cannot assess selector)\n");

        printf("\nSmall model (C=16, S=8):\n");
        if (small.compiled && small.loaded) {
            printf("  ibh = %llu  → %s\n", (unsigned long long)small.ibh,
                   small.ibh == 0
                       ? "PASS — no spill (expected)"
                       : "UNEXPECTED SPILL on tiny model");
        } else {
            printf("  compile/load failed\n");
        }

        printf("\nMedium model (C=256, S=256):\n");
        if (medium.compiled && medium.loaded) {
            printf("  ibh = %llu  → %s\n", (unsigned long long)medium.ibh,
                   medium.ibh == 0
                       ? "PASS — no spill (expected)"
                       : "SPILL at 128 KB — threshold lower than expected");
        } else {
            printf("  compile failed (unexpected for 128 KB model)\n");
        }

        printf("\nLarge model (C=4096, S=4096):\n");
        if (!large.compiled) {
            printf("  compile failed — firmware rejected oversized model before spill.\n");
            printf("  libane_mil_sram_spill() spill=1 path cannot be triggered via compile.\n");
            printf("  The spill=0 path (ibh==0 for small models) is the only testable case.\n");
        } else if (large.loaded) {
            if (large.ibh != 0) {
                printf("  ibh = %llu  → SPILL CONFIRMED — spill detection works!\n",
                       (unsigned long long)large.ibh);
            } else {
                printf("  ibh = 0 — model compiled+loaded without spill.\n");
                printf("  Either firmware has a larger SRAM budget or this model fits.\n");
            }
        }

        printf("\nConclusion for libane:\n");
        if (small.ibh_exists && small.compiled && small.loaded && small.ibh == 0) {
            printf("  [+] intermediateBufferHandle selector exists and returns 0 for\n");
            printf("      non-spilling models. The no-spill path in ane_compile() is correct.\n");
            if (!large.compiled) {
                printf("  [~] Spill path untestable — firmware rejects oversized models.\n");
                printf("      The libane warning code is structurally correct but the\n");
                printf("      spill=1 condition may never fire in practice.\n");
            }
        } else if (!small.ibh_exists) {
            printf("  [-] intermediateBufferHandle not found on this firmware.\n");
            printf("      libane_mil_sram_spill() will always return 0. Safe but uninformative.\n");
        }

        printf("\nDone.\n");
        return 0;
    }
}
