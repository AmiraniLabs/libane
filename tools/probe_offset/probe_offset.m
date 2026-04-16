/**
 * probe_offset.m
 *
 * Investigates _ANEIOSurfaceObject startOffset for tensor packing:
 * multiple tensors sharing one IOSurface at non-zero offsets.
 *
 * Key question: what offset values are legal, and do kernels correctly
 * read/write through offset AIO objects during processRequest:?
 *
 * Methodology:
 *   PART 1 — Alignment sweep:  offsets 0..max, find minimum legal alignment
 *   PART 2 — Functional test:  two tensors packed into one surface,
 *                               kernels A and B each reading/writing at offsets
 *   PART 3 — Benchmark:        separate surfaces vs packed surface,
 *                               measure allocation + eval overhead delta
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── helpers ────────────────────────────────────────────────────────────── */

static void *kANE;
static Class cls_AIO;   /* _ANEIOSurfaceObject */
static Class cls_AR;    /* _ANERequest         */
static Class cls_Desc;  /* _ANEInMemoryModelDescriptor */
static Class cls_Model; /* _ANEInMemoryModel   */

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *d = @{
        @"IOSurfaceWidth":          @((int)bytes),
        @"IOSurfaceHeight":         @(1),
        @"IOSurfaceBytesPerElement":@(1),
        @"IOSurfacePixelFormat":    @(0)
    };
    return IOSurfaceCreate((CFDictionaryRef)d);
}

/* 128-byte ANE weight header + fp16 identity kernel (CH×CH×1×1 conv).
 * Matches the exact header format confirmed working in probe_chaining. */
static NSData *make_identity_weight(int CH) {
    int n_elems = CH * CH;
    int ws = n_elems * 2;
    int total = 128 + ws;
    uint8_t *blob = (uint8_t *)calloc(total, 1);

    blob[0] = 1;                            /* version */
    blob[4] = 2;                            /* type    */
    blob[64] = 0xEF; blob[65] = 0xBE;      /* chunk magic 0xDEADBEEF LE */
    blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 1;                           /* chunk type */
    *(uint32_t *)(blob + 72) = (uint32_t)ws;
    *(uint32_t *)(blob + 80) = 128;         /* data offset */

    uint16_t *fp16 = (uint16_t *)(blob + 128);
    for (int i = 0; i < CH; i++) fp16[i * CH + i] = 0x3C00; /* fp16 1.0 */

    return [NSData dataWithBytesNoCopy:blob length:total freeWhenDone:YES];
}

/* MIL text for a CH×CH identity conv over [1,CH,1,SP] fp32 tensors.
 * Matches the program(1.3)/buildInfo/ios18 format confirmed working in probe_chaining. */
static NSString *make_mil(int CH, int SP) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,"
        "pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n",
        CH, SP,
        CH, SP,
        CH, CH, CH, CH,
        CH, SP,
        CH, SP];
}

static id compile_and_load(int CH, int SP, const char *tag) {
    NSData *mil_data = [make_mil(CH, SP) dataUsingEncoding:NSUTF8StringEncoding];
    NSData *wdata    = make_identity_weight(CH);
    NSDictionary *weights = @{
        @"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}
    };

    id desc = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
        cls_Desc,
        sel_registerName("modelWithMILText:weights:optionsPlist:"),
        mil_data, weights, nil);
    if (!desc) { printf("[%s] descriptor failed\n", tag); return nil; }

    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc);
    if (!model) { printf("[%s] model creation failed\n", tag); return nil; }

    NSString *hexID = ((NSString*(*)(id,SEL))objc_msgSend)(
        model, sel_registerName("hexStringIdentifier"));
    NSString *model_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexID];
    NSString *wdir      = [model_dir stringByAppendingPathComponent:@"weights"];
    NSFileManager *fm   = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil];
    [mil_data writeToFile:[model_dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata    writeToFile:[wdir       stringByAppendingPathComponent:@"weight.bin"] atomically:YES];

    NSError *err = nil;
    typedef BOOL (*QFn)(id,SEL,unsigned int,id,NSError**);
    BOOL ok = ((QFn)objc_msgSend)(model, sel_registerName("compileWithQoS:options:error:"), 21, @{}, &err);
    if (!ok) { printf("[%s] compile failed: %s\n", tag, err?[[err localizedDescription] UTF8String]:"?"); return nil; }
    err = nil;
    ok   = ((QFn)objc_msgSend)(model, sel_registerName("loadWithQoS:options:error:"), 21, @{}, &err);
    if (!ok) { printf("[%s] load failed: %s\n", tag, err?[[err localizedDescription] UTF8String]:"?"); return nil; }

    printf("[%s] compiled + loaded OK\n", tag);
    return model;
}

/* Build _ANERequest wrapping one input AIO and one output AIO */
static id build_request(id in_aio, id out_aio) {
    return ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
        cls_AR,
        sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:"
                         "weightsBuffer:perfStats:procedureIndex:"),
        @[in_aio], @[@0], @[out_aio], @[@0], nil, nil, (NSUInteger)0);
}

/* processRequest: dispatch via _ANEProgramForEvaluation */
static BOOL proc_req(id prog, id inner, id req, uint64_t sid) {
    uint32_t rv = 0; NSError *e = nil;
    return ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,uint32_t*,NSError**))objc_msgSend)(
        prog,
        sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:"),
        req, inner, (unsigned int)21, (uint64_t)0, sid, @{}, &rv, &e);
}

int main(void) {
    kANE = dlopen("/System/Library/PrivateFrameworks/"
                  "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!kANE) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    @autoreleasepool {
        cls_AIO   = NSClassFromString(@"_ANEIOSurfaceObject");
        cls_AR    = NSClassFromString(@"_ANERequest");
        cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        cls_Model = NSClassFromString(@"_ANEInMemoryModel");

        if (!cls_AIO || !cls_AR || !cls_Desc || !cls_Model) {
            fprintf(stderr, "class lookup failed\n"); return 1;
        }

        SEL sel_withOffset     = sel_registerName("objectWithIOSurface:startOffset:");
        SEL sel_getStartOffset = sel_registerName("startOffset");

        /* ═══════════════════════════════════════════════════════════════
           PART 1 — Alignment sweep
           Create _ANEIOSurfaceObject with increasing offsets to find the
           minimum legal alignment. Purely object creation — no ANE eval.
           Test offsets: 0, 1, 2, 4, 8, 16, 64, 128, 256, 512, 1K, 4K,
                         8K, 16K, 32K, 64K, 128K.
           For each: create the AIO, read back startOffset, see if creation
           succeeds or throws.
        ════════════════════════════════════════════════════════════════ */
        printf("═══════════════════════════════════════════════════════\n");
        printf("PART 1 — _ANEIOSurfaceObject startOffset alignment sweep\n");
        printf("═══════════════════════════════════════════════════════\n");

        /* One surface large enough for all tests: 1 MB */
        IOSurfaceRef sweep_surf = make_surface(1024 * 1024);

        size_t offsets[] = {
            0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
        };
        int n_offsets = (int)(sizeof(offsets)/sizeof(offsets[0]));

        size_t first_working_nonzero = 0;
        for (int i = 0; i < n_offsets; i++) {
            size_t off = offsets[i];
            id aio = nil;
            NSNumber *got_offset = nil;
            const char *result = "FAIL (nil)";

            @try {
                aio = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, sweep_surf, @(off));
                if (aio) {
                    got_offset = ((id(*)(id,SEL))objc_msgSend)(aio, sel_getStartOffset);
                    result = "OK";
                    if (off > 0 && first_working_nonzero == 0) first_working_nonzero = off;
                }
            } @catch (NSException *ex) {
                result = [[ex reason] UTF8String];
            }

            printf("  offset=%-8zu  → %-8s  readback=%s\n",
                   off, result,
                   got_offset ? [[got_offset stringValue] UTF8String] : "—");
        }

        printf("\n  First working non-zero offset: %zu bytes\n", first_working_nonzero);
        CFRelease(sweep_surf);

        /* ═══════════════════════════════════════════════════════════════
           PART 2 — Functional test: two tensors packed into one surface
           CH=64, SP=32 → 64×32×2 = 4096 bytes per fp16 tensor surface.
           We use float32 surfaces here (×4 = 16384 bytes each).

           Packed surface layout:
             [0 .. TSIZE-1]        → tensor A (input to kernel_A)
             [TSIZE .. 2*TSIZE-1]  → tensor B (output of kernel_A / input to kernel_B)
             [2*TSIZE .. 3*TSIZE-1]→ tensor C (output of kernel_B)

           Each AIO object points at packed_surf with a different startOffset.
           If this works:
             - kernel_A reads from offset=0, writes to offset=TSIZE
             - kernel_B reads from offset=TSIZE, writes to offset=2*TSIZE
             - Three tensors, one IOSurface allocation.
        ════════════════════════════════════════════════════════════════ */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 2 — Functional: 3 tensors packed into one surface\n");
        printf("═══════════════════════════════════════════════════════\n");

        int CH = 64, SP = 32;
        /* ANE surfaces are fp16 internally; we use float32 IOSurfaces as the
         * host-side buffers for writing/reading, matching existing libane convention.
         * Tensor byte size in float32: CH * SP * sizeof(float) */
        size_t TSIZE = (size_t)CH * SP * sizeof(float);  /* 8192 bytes */
        size_t PACKED_SIZE = TSIZE * 3;

        printf("  CH=%d  SP=%d  TSIZE=%zu bytes  packed=%zu bytes\n",
               CH, SP, TSIZE, PACKED_SIZE);

        id mdl_A = compile_and_load(CH, SP, "pack_A");
        id mdl_B = compile_and_load(CH, SP, "pack_B");

        if (!mdl_A || !mdl_B) {
            printf("  SKIP — model compile failed\n");
            goto done;
        }

        id p2_inner_A = nil, p2_inner_B = nil, p2_prog_A = nil, p2_prog_B = nil;
        @try { p2_inner_A = [mdl_A valueForKey:@"model"]; } @catch (...) {}
        @try { p2_inner_B = [mdl_B valueForKey:@"model"]; } @catch (...) {}
        @try { p2_prog_A  = [p2_inner_A valueForKey:@"program"]; } @catch (...) {}
        @try { p2_prog_B  = [p2_inner_B valueForKey:@"program"]; } @catch (...) {}

        if (!p2_prog_A || !p2_prog_B) {
            printf("  SKIP — _ANEProgramForEvaluation unavailable\n");
            goto done;
        }

        uint64_t p2_sid_A = ((uint64_t(*)(id,SEL))objc_msgSend)(
            p2_inner_A, sel_registerName("string_id"));
        uint64_t p2_sid_B = ((uint64_t(*)(id,SEL))objc_msgSend)(
            p2_inner_B, sel_registerName("string_id"));

        /* Allocate separate baseline surfaces (current libane approach) */
        IOSurfaceRef sep_in  = make_surface(TSIZE);
        IOSurfaceRef sep_mid = make_surface(TSIZE);
        IOSurfaceRef sep_out = make_surface(TSIZE);

        /* Allocate single packed surface */
        IOSurfaceRef packed = make_surface(PACKED_SIZE);

        /* Fill tensor A slot in packed surface and in sep_in */
        float fill_val = 1.5f;
        {
            /* sep_in */
            IOSurfaceLock(sep_in, 0, NULL);
            float *p = (float *)IOSurfaceGetBaseAddress(sep_in);
            for (int i = 0; i < CH * SP; i++) p[i] = fill_val + (float)i * 0.001f;
            IOSurfaceUnlock(sep_in, 0, NULL);

            /* packed[0..TSIZE-1] same data */
            IOSurfaceLock(packed, 0, NULL);
            float *pp = (float *)IOSurfaceGetBaseAddress(packed);
            for (int i = 0; i < CH * SP; i++) pp[i] = fill_val + (float)i * 0.001f;
            IOSurfaceUnlock(packed, 0, NULL);
        }

        /* ── Baseline: separate surfaces via evaluateWithQoS: ────────────── */
        {
            id a_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_in);
            id a_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_mid);
            id b_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_mid);
            id b_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_out);
            id req_A = build_request(a_in, a_out);
            id req_B = build_request(b_in, b_out);
            BOOL ok_A = proc_req(p2_prog_A, p2_inner_A, req_A, p2_sid_A);
            BOOL ok_B = proc_req(p2_prog_B, p2_inner_B, req_B, p2_sid_B);
            if (ok_A && ok_B) {
                IOSurfaceLock(sep_out, kIOSurfaceLockReadOnly, NULL);
                float *o = (float *)IOSurfaceGetBaseAddress(sep_out);
                printf("  Baseline (separate): A=%s B=%s  out[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                       ok_A?"OK":"FAIL", ok_B?"OK":"FAIL", o[0],o[1],o[2],o[3]);
                IOSurfaceUnlock(sep_out, kIOSurfaceLockReadOnly, NULL);
            } else {
                printf("  Baseline FAIL A=%s B=%s\n", ok_A?"OK":"FAIL", ok_B?"OK":"FAIL");
            }
        }

        /* ── Packed surface: AIO objects at offsets 0, TSIZE, 2*TSIZE ───── */
        /* Try progressively larger alignment multiples until create doesn't fail.
         * We know from PART 1 the first working offset; start from there.
         * For the functional test, we need offset=TSIZE to be a valid alignment.
         * If TSIZE is not aligned, round up to the next working boundary. */
        {
            printf("\n  [packed surface attempt]\n");
            printf("  Tensor slot offsets: A=0  B=%zu  C=%zu\n", TSIZE, TSIZE*2);

            id aio_A_in  = nil, aio_A_out = nil, aio_B_in = nil, aio_B_out = nil;
            BOOL created = NO;

            @try {
                aio_A_in  = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, packed, @(0));
                aio_A_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, packed, @(TSIZE));
                aio_B_in  = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, packed, @(TSIZE));
                aio_B_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, packed, @(TSIZE*2));
                if (aio_A_in && aio_A_out && aio_B_in && aio_B_out) created = YES;
                printf("  AIO creation: %s (offsets 0 / %zu / %zu)\n",
                       created ? "OK" : "FAIL (nil returned)", TSIZE, TSIZE*2);
            } @catch (NSException *ex) {
                printf("  AIO creation EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            if (created) {
                id req_pA = build_request(aio_A_in, aio_A_out);
                id req_pB = build_request(aio_B_in, aio_B_out);

                /* Run kernel A only first, then read all three slots to find
                 * where the output actually landed. This reveals whether the
                 * ANE honors the offset at eval time or ignores it. */
                BOOL ok_A = NO;
                @try { ok_A = proc_req(p2_prog_A, p2_inner_A, req_pA, p2_sid_A); } @catch (NSException *ex) {
                    printf("  processRequest A EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                printf("  processRequest A: %s\n", ok_A?"OK":"FAIL");

                if (ok_A) {
                    IOSurfaceLock(packed, kIOSurfaceLockReadOnly, NULL);
                    float *base = (float *)IOSurfaceGetBaseAddress(packed);
                    float *slot_A = base;                       /* offset 0       */
                    float *slot_B = base + TSIZE/sizeof(float); /* offset TSIZE   */
                    float *slot_C = base + TSIZE*2/sizeof(float);/* offset 2*TSIZE */
                    printf("  After kernel A:\n");
                    printf("    slot_A [0]  (input):   [%.4f, %.4f, %.4f, %.4f]\n",
                           slot_A[0], slot_A[1], slot_A[2], slot_A[3]);
                    printf("    slot_B [TSIZE] (A out): [%.4f, %.4f, %.4f, %.4f]  ← expect ≈ input if offset honored\n",
                           slot_B[0], slot_B[1], slot_B[2], slot_B[3]);
                    printf("    slot_C [2*TSIZE]:       [%.4f, %.4f, %.4f, %.4f]  ← expect 0 (not written yet)\n",
                           slot_C[0], slot_C[1], slot_C[2], slot_C[3]);

                    /* Check if data ended up at slot A (offset ignored — overwrote input)
                     * or at slot B (offset honored correctly) */
                    IOSurfaceLock(sep_mid, kIOSurfaceLockReadOnly, NULL);
                    float *ref_mid = (float *)IOSurfaceGetBaseAddress(sep_mid);
                    float diff_at_B = 0.0f, diff_at_A_overwrite = 0.0f;
                    for (int i = 0; i < CH * SP; i++) {
                        diff_at_B += fabsf(slot_B[i] - ref_mid[i]);
                        diff_at_A_overwrite += fabsf(slot_A[i]); /* expect 0 if A was overwritten */
                    }
                    IOSurfaceUnlock(sep_mid, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(packed, kIOSurfaceLockReadOnly, NULL);

                    printf("  Diagnosis: output_at_slot_B_diff=%.4f  slot_A_sum=%.4f\n",
                           diff_at_B, diff_at_A_overwrite);
                    if (diff_at_B < 1.0f) {
                        printf("  → Offset HONORED: kernel wrote to correct slot B\n");
                    } else {
                        printf("  → Offset IGNORED or wrong: kernel did not write to slot B\n");
                        printf("     (data may be at slot A overwriting input, or elsewhere)\n");
                    }
                }

                BOOL ok_B = NO;
                @try { ok_B = proc_req(p2_prog_B, p2_inner_B, req_pB, p2_sid_B); } @catch (NSException *ex) {
                    printf("  processRequest B EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                printf("  processRequest B: %s\n", ok_B?"OK":"FAIL");

                if (ok_A && ok_B) {
                    IOSurfaceLock(packed, kIOSurfaceLockReadOnly, NULL);
                    float *base = (float *)IOSurfaceGetBaseAddress(packed);
                    float *slot_C = base + TSIZE*2/sizeof(float);
                    printf("  Final slot_C [0..3]: [%.4f,%.4f,%.4f,%.4f]  (expect ≈ input)\n",
                           slot_C[0], slot_C[1], slot_C[2], slot_C[3]);
                    IOSurfaceLock(sep_out, kIOSurfaceLockReadOnly, NULL);
                    float *sep_C = (float *)IOSurfaceGetBaseAddress(sep_out);
                    float max_diff = 0.0f;
                    for (int i = 0; i < CH * SP; i++) {
                        float d = fabsf(slot_C[i] - sep_C[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    IOSurfaceUnlock(sep_out, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(packed, kIOSurfaceLockReadOnly, NULL);
                    printf("  Max diff vs baseline: %.6f  → %s\n", max_diff,
                           max_diff < 0.05f
                               ? "CORRECT — packed surface produces same result as separate surfaces"
                               : "MISMATCH — offset read/write incorrect");
                }
            }
        }

        /* ── Isolation test: startOffset on output only (separate input) ─── */
        /* If the problem was using the same IOSurface for both in and out,
         * this eliminates that confound. Uses sep_in (known-good, offset=0)
         * as input, and a 2×TSIZE "half_out" surface for output at offset TSIZE.
         * If output appears at TSIZE → startOffset works, same-surface was the bug.
         * If output appears at 0 (or nowhere) → ANE ignores startOffset entirely. */
        {
            printf("\n  [isolation] startOffset on output only (separate input surface)\n");
            IOSurfaceRef half_out = make_surface(TSIZE * 2);
            /* Zero both slots */
            IOSurfaceLock(half_out, 0, NULL);
            memset(IOSurfaceGetBaseAddress(half_out), 0, TSIZE * 2);
            IOSurfaceUnlock(half_out, 0, NULL);

            id iso_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), sep_in);
            id iso_out_offset = nil;
            @try {
                iso_out_offset = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, half_out, @(TSIZE));
            } @catch (NSException *ex) {
                printf("  AIO creation EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            if (iso_out_offset) {
                id iso_req = build_request(iso_in, iso_out_offset);
                BOOL iso_ok = NO;
                @try { iso_ok = proc_req(p2_prog_A, p2_inner_A, iso_req, p2_sid_A); } @catch (...) {}
                printf("  processRequest: %s\n", iso_ok ? "OK" : "FAIL");
                if (iso_ok) {
                    IOSurfaceLock(half_out, kIOSurfaceLockReadOnly, NULL);
                    float *ho = (float *)IOSurfaceGetBaseAddress(half_out);
                    float *slot0  = ho;                       /* offset 0     */
                    float *slotT  = ho + TSIZE/sizeof(float); /* offset TSIZE */
                    printf("  half_out slot0 [0..3]:  [%.4f, %.4f, %.4f, %.4f]  ← expect 0 if offset honored\n",
                           slot0[0], slot0[1], slot0[2], slot0[3]);
                    printf("  half_out slotT [0..3]:  [%.4f, %.4f, %.4f, %.4f]  ← expect ≈1.5 if offset honored\n",
                           slotT[0], slotT[1], slotT[2], slotT[3]);
                    float sum0 = 0, sumT = 0;
                    for (int i = 0; i < CH*SP; i++) { sum0 += fabsf(slot0[i]); sumT += fabsf(slotT[i]); }
                    IOSurfaceUnlock(half_out, kIOSurfaceLockReadOnly, NULL);
                    printf("  sum(|slot0|)=%.2f  sum(|slotT|)=%.2f\n", sum0, sumT);
                    if (sumT > sum0 && sumT > 100.0f) {
                        printf("  → startOffset HONORED for output: data at correct offset\n");
                    } else if (sum0 > sumT && sum0 > 100.0f) {
                        printf("  → startOffset IGNORED: ANE wrote to offset 0 regardless\n");
                    } else {
                        printf("  → output not found at either location (DMA to unmapped region?)\n");
                    }
                }
            }
            CFRelease(half_out);
        }

        /* ── If TSIZE alignment failed, probe the correct alignment ──────── */
        /* Try TSIZE rounded up to powers of 2: 16K, 32K, 64K */
        {
            size_t candidates[] = {16384, 32768, 65536, 131072};
            for (int ci = 0; ci < 4; ci++) {
                size_t aligned = candidates[ci];
                if (aligned <= TSIZE) continue; /* only try larger alignments */
                if (aligned * 3 > IOSurfaceGetAllocSize(packed)) continue;

                printf("\n  [retry with aligned offset = %zu (%.0f KB)]\n",
                       aligned, aligned/1024.0);
                id aio_test = nil;
                @try {
                    aio_test = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                        cls_AIO, sel_withOffset, packed, @(aligned));
                } @catch (...) {}
                if (!aio_test) { printf("  still fails at %zu\n", aligned); continue; }

                /* Rebuild packed surface at this alignment */
                size_t aligned_packed = aligned * 3;
                IOSurfaceRef p2 = make_surface(aligned_packed);
                IOSurfaceLock(p2, 0, NULL);
                float *pp2 = (float *)IOSurfaceGetBaseAddress(p2);
                for (int i = 0; i < CH * SP; i++) pp2[i] = fill_val + (float)i * 0.001f;
                IOSurfaceUnlock(p2, 0, NULL);

                id a2_in=nil, a2_out=nil, b2_in=nil, b2_out=nil;
                BOOL c2 = NO;
                @try {
                    a2_in  = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(cls_AIO, sel_withOffset, p2, @(0));
                    a2_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(cls_AIO, sel_withOffset, p2, @(aligned));
                    b2_in  = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(cls_AIO, sel_withOffset, p2, @(aligned));
                    b2_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(cls_AIO, sel_withOffset, p2, @(aligned*2));
                    c2 = (a2_in && a2_out && b2_in && b2_out);
                } @catch (...) {}

                if (!c2) { CFRelease(p2); continue; }

                BOOL ok_A=NO, ok_B=NO;
                @try { ok_A = proc_req(p2_prog_A, p2_inner_A, build_request(a2_in, a2_out), p2_sid_A); } @catch (...) {}
                @try { ok_B = proc_req(p2_prog_B, p2_inner_B, build_request(b2_in, b2_out), p2_sid_B); } @catch (...) {}
                printf("  eval: A=%s B=%s\n", ok_A?"OK":"FAIL", ok_B?"OK":"FAIL");

                if (ok_A && ok_B) {
                    IOSurfaceLock(p2, kIOSurfaceLockReadOnly, NULL);
                    float *base2 = (float *)IOSurfaceGetBaseAddress(p2);
                    float *slot2C = base2 + (aligned*2)/sizeof(float);
                    IOSurfaceLock(sep_out, kIOSurfaceLockReadOnly, NULL);
                    float *sep2  = (float *)IOSurfaceGetBaseAddress(sep_out);
                    float mx = 0.0f;
                    for (int i=0;i<CH*SP;i++){float dv=fabsf(slot2C[i]-sep2[i]);if(dv>mx)mx=dv;}
                    IOSurfaceUnlock(sep_out, kIOSurfaceLockReadOnly, NULL);
                    printf("  Max diff vs baseline: %.6f  → %s\n", mx,
                           mx < 0.05f ? "CORRECT — packing works at this alignment" : "MISMATCH");
                    IOSurfaceUnlock(p2, kIOSurfaceLockReadOnly, NULL);

                    if (mx < 0.05f) {
                        printf("\n  *** Minimum working alignment: %zu bytes (%.0f KB) ***\n",
                               aligned, aligned/1024.0);

                        /* ═══════════════════════════════════════════════════════
                           PART 3 — Benchmark: 3 separate surfaces vs 1 packed
                           N=50 evals of A+B chain, measure full round-trip
                           including surface creation overhead.
                        ════════════════════════════════════════════════════════ */
                        printf("\n═══════════════════════════════════════════════════════\n");
                        printf("PART 3 — Benchmark: separate surfaces vs packed surface\n");
                        printf("═══════════════════════════════════════════════════════\n");

                        int N = 50;
                        struct timespec t0, t1;
                        id req_a2 = build_request(a2_in, a2_out);
                        id req_b2 = build_request(b2_in, b2_out);

                        /* Warm up */
                        for (int w=0;w<3;w++){
                            proc_req(p2_prog_A, p2_inner_A, req_a2, p2_sid_A);
                            proc_req(p2_prog_B, p2_inner_B, req_b2, p2_sid_B);
                        }

                        /* Packed */
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        for (int i=0;i<N;i++){
                            proc_req(p2_prog_A, p2_inner_A, req_a2, p2_sid_A);
                            proc_req(p2_prog_B, p2_inner_B, req_b2, p2_sid_B);
                        }
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        double packed_ms = ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/(double)N/1e6;

                        /* Separate (reuse sep_in/sep_mid/sep_out) */
                        id s_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_in);
                        id s_mid = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_mid);
                        id s_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), sep_out);
                        id req_sA = build_request(s_in, s_mid);
                        id req_sB = build_request(s_mid, s_out);
                        for (int w=0;w<3;w++){
                            proc_req(p2_prog_A, p2_inner_A, req_sA, p2_sid_A);
                            proc_req(p2_prog_B, p2_inner_B, req_sB, p2_sid_B);
                        }
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        for (int i=0;i<N;i++){
                            proc_req(p2_prog_A, p2_inner_A, req_sA, p2_sid_A);
                            proc_req(p2_prog_B, p2_inner_B, req_sB, p2_sid_B);
                        }
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        double sep_ms = ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/(double)N/1e6;

                        printf("  Separate surfaces:  %.3f ms/pair\n", sep_ms);
                        printf("  Packed surface:     %.3f ms/pair\n", packed_ms);
                        printf("  Difference:         %.3f ms  (%.1f%%)\n",
                               sep_ms - packed_ms,
                               (sep_ms - packed_ms) / sep_ms * 100.0);
                        printf("\n  Alignment required: %zu bytes (%.0f KB)\n",
                               aligned, aligned/1024.0);
                        printf("  Packing ratio: 3 tensors × %zu bytes → %zu bytes total"
                               " (%.1fx vs separate)\n",
                               TSIZE, aligned_packed,
                               (double)aligned_packed / (TSIZE * 3));
                    }
                }
                CFRelease(p2);
                break;
            }
        }

        /* ── PART 3: _ANEProgramIOSurfacesMapper + startOffset ──────────── */
        /* Central question: does calling mapIOSurfacesWithRequest:cacheInference:
         * with an offset AIO register the offset as an IOMMU hint in the firmware,
         * such that a subsequent processRequest: honors the offset?
         *
         * Two sub-tests:
         *   [A] mapIOSurfaces with standard offset-0 AIO — establish baseline error
         *   [B] mapIOSurfaces with offset AIO — does the error change?
         *       If it fails at a DIFFERENT point, the firmware is seeing the offset.
         *   [C] If [B] succeeds: processRequest: with the same offset AIO.
         *       Does the output now land at the correct offset?
         *
         * The mapper is accessed via the _ANEInMemoryModel directly (same class
         * that evaluateWithQoS: dispatches through).
         */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 3 — mapIOSurfacesWithRequest: + startOffset\n");
        printf("═══════════════════════════════════════════════════════\n");

        {
            SEL sel_map = sel_registerName("mapIOSurfacesWithRequest:cacheInference:error:");

            /* [A] Baseline: map with offset-0 AIO (same as prior probe) */
            {
                printf("  [A] mapIOSurfaces with standard offset-0 AIOs\n");
                id m_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    cls_AIO, sel_registerName("objectWithIOSurface:"), sep_in);
                id m_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    cls_AIO, sel_registerName("objectWithIOSurface:"), sep_mid);
                id m_req = build_request(m_in, m_out);
                NSError *me = nil;
                BOOL mok = NO;
                @try {
                    mok = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                        mdl_A, sel_map, m_req, YES, &me);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                printf("  result: %s  err: %s\n",
                       mok ? "OK" : "FAIL",
                       me ? [[me description] UTF8String] : "nil");
            }

            /* [B] Map with offset AIO on both input and output */
            {
                printf("\n  [B] mapIOSurfaces with offset AIOs (offset=TSIZE)\n");
                IOSurfaceRef map_surf = make_surface(TSIZE * 2);

                /* Fill slot 0 with input data */
                IOSurfaceLock(map_surf, 0, NULL);
                float *ms_p = (float *)IOSurfaceGetBaseAddress(map_surf);
                for (int i = 0; i < CH * SP; i++) ms_p[i] = 1.5f + (float)i * 0.001f;
                IOSurfaceUnlock(map_surf, 0, NULL);

                /* input AIO at offset 0, output AIO at offset TSIZE */
                id mo_in = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, map_surf, @(0));
                id mo_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, map_surf, @(TSIZE));
                id mo_req = build_request(mo_in, mo_out);
                NSError *moe = nil;
                BOOL mbok = NO;
                @try {
                    mbok = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                        mdl_A, sel_map, mo_req, YES, &moe);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                printf("  result: %s  err: %s\n",
                       mbok ? "OK" : "FAIL",
                       moe ? [[moe description] UTF8String] : "nil");

                /* [C] If map succeeded, try processRequest: and check offset */
                if (mbok) {
                    printf("\n  [C] mapIOSurfaces succeeded — now processRequest: with same offset AIO\n");
                    BOOL prok = NO;
                    @try { prok = proc_req(p2_prog_A, p2_inner_A, mo_req, p2_sid_A); } @catch (...) {}
                    printf("  processRequest: %s\n", prok ? "OK" : "FAIL");
                    if (prok) {
                        IOSurfaceLock(map_surf, kIOSurfaceLockReadOnly, NULL);
                        float *ms = (float *)IOSurfaceGetBaseAddress(map_surf);
                        float *s0 = ms, *s1 = ms + TSIZE/sizeof(float);
                        float sum0 = 0, sum1 = 0;
                        for (int i = 0; i < CH*SP; i++) { sum0 += fabsf(s0[i]); sum1 += fabsf(s1[i]); }
                        printf("  slot0 [0..3]: [%.4f, %.4f, %.4f, %.4f]  sum=%.2f\n",
                               s0[0],s0[1],s0[2],s0[3], sum0);
                        printf("  slot1 [0..3]: [%.4f, %.4f, %.4f, %.4f]  sum=%.2f\n",
                               s1[0],s1[1],s1[2],s1[3], sum1);
                        IOSurfaceUnlock(map_surf, kIOSurfaceLockReadOnly, NULL);
                        if (sum1 > sum0 && sum1 > 100.0f)
                            printf("  → startOffset HONORED after mapIOSurfaces pre-registration!\n");
                        else if (sum0 > sum1 && sum0 > 100.0f)
                            printf("  → startOffset STILL IGNORED — mapper does not affect DMA addressing\n");
                        else
                            printf("  → output not at either slot (unexpected)\n");
                    }
                } else {
                    /* Compare error codes between [A] and [B] — if different, firmware saw the offset */
                    printf("  (map failed — compare error code with [A] to determine firmware visibility)\n");
                }

                CFRelease(map_surf);
            }

            /* [D] mapIOSurfaces with mismatched offset: in at TSIZE, out at 0
             * (offset on input only) — does the firmware validate this differently? */
            {
                printf("\n  [D] mapIOSurfaces: input at offset TSIZE, output at offset 0 (reversed)\n");
                IOSurfaceRef rev_surf = make_surface(TSIZE * 2);
                id rev_in = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, rev_surf, @(TSIZE));
                id rev_out = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_AIO, sel_withOffset, rev_surf, @(0));
                id rev_req = build_request(rev_in, rev_out);
                NSError *re = nil;
                BOOL rok = NO;
                @try {
                    rok = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                        mdl_A, sel_map, rev_req, YES, &re);
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                printf("  result: %s  err: %s\n",
                       rok ? "OK" : "FAIL",
                       re ? [[re description] UTF8String] : "nil");
                printf("  (if error differs from [A]/[B], firmware validated the offset)\n");
                CFRelease(rev_surf);
            }
        }

        CFRelease(sep_in);
        CFRelease(sep_mid);
        CFRelease(sep_out);
        CFRelease(packed);

done:
        printf("\nDone.\n");
    }
    return 0;
}
