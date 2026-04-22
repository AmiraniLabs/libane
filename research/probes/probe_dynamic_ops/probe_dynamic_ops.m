/**
 * probe_dynamic_ops.m
 *
 * Answers three open questions for libane 0.9.0:
 *
 *  Q1. Does the `matmul` MIL op accept two runtime tensor inputs?
 *      If yes, the packed-IOSurface dynamic matmul technique works.
 *
 *  Q2. Does `scaled_dot_product_attention` compile and execute correctly?
 *      If yes, dynamic SDPA is feasible as a single MIL op.
 *
 *  Q3. Why do activation-only graphs (relu, tanh) produce alternating zeros
 *      when S=16 but not when S=32?  What is the actual minimum S requirement
 *      for non-conv graphs?
 *
 * Build:
 *   clang -x objective-c -fobjc-arc -framework Foundation \
 *         -framework IOSurface -o probe_dynamic_ops probe_dynamic_ops.m \
 *         && ./probe_dynamic_ops
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

/* ── ANE private API ────────────────────────────────────────────────────── */

static Class g_Desc, g_Model, g_IOS, g_Req;
static unsigned int kQOS = 21;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    g_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_Model = NSClassFromString(@"_ANEInMemoryModel");
    g_IOS   = NSClassFromString(@"_ANEIOSurfaceObject");
    g_Req   = NSClassFromString(@"_ANERequest");
}

/* ── IOSurface helpers ───────────────────────────────────────────────────── */

static IOSurfaceRef make_surface(size_t bytes) {
    size_t sz = ((bytes + 63) & ~63UL);
    if (sz < 49152) sz = 49152;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:           @(sz),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfacePixelFormat:     @0,
    });
}

/* fp16 ↔ float */
static uint16_t f32_to_f16(float f) {
    uint32_t fb; memcpy(&fb, &f, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FF;
    if (e <= 0)  return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t fb;
    if (e == 0)       fb = s | (m << 13);
    else if (e == 31) fb = s | 0x7F800000u | (m << 13);
    else              fb = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &fb, 4); return f;
}

static void ios_write_f16(IOSurfaceRef ios, const float *vals, int n) {
    IOSurfaceLock(ios, 0, NULL);
    uint16_t *p = (uint16_t *)IOSurfaceGetBaseAddress(ios);
    for (int i = 0; i < n; i++) p[i] = f32_to_f16(vals[i]);
    IOSurfaceUnlock(ios, 0, NULL);
}
static void ios_read_f16(IOSurfaceRef ios, float *out, int n) {
    IOSurfaceLock(ios, kIOSurfaceLockReadOnly, NULL);
    const uint16_t *p = (const uint16_t *)IOSurfaceGetBaseAddress(ios);
    for (int i = 0; i < n; i++) out[i] = f16_to_f32(p[i]);
    IOSurfaceUnlock(ios, kIOSurfaceLockReadOnly, NULL);
}

/* ── Weight blob ────────────────────────────────────────────────────────── */

static NSData *make_weight_blob(const uint16_t *data, int n_elem) {
    size_t ws = (size_t)n_elem * 2;
    size_t total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)ws;
    *(uint32_t *)(buf + 80) = 128;
    memcpy(buf + 128, data, ws);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

/* ── Compile + run helpers ──────────────────────────────────────────────── */

typedef struct {
    id model;
    NSString *tmp_dir;
} Model;

static Model compile_mil(NSString *mil, NSDictionary *weights) {
    Model m = { nil, nil };
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
    id desc = ((DescFn)objc_msgSend)(g_Desc,
        sel_registerName("modelWithMILText:weights:optionsPlist:"),
        md, weights ?: @{}, nil);
    if (!desc) return m;

    typedef id (*ModelFn)(Class, SEL, id);
    id model = ((ModelFn)objc_msgSend)(g_Model,
        sel_registerName("inMemoryModelWithDescriptor:"), desc);
    if (!model) return m;

    NSString *hx = ((NSString*(*)(id,SEL))objc_msgSend)(
        model, sel_registerName("hexStringIdentifier"));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager]
        createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *key in weights) {
        NSString *rel = [key stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[key][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, sel_registerName("compileWithQoS:options:error:"), kQOS, @{}, &e);
    if (!ok) {
        printf("  [compile FAIL] %s\n",
            e ? [[e localizedDescription] UTF8String] : "unknown");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return m;
    }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, sel_registerName("loadWithQoS:options:error:"), kQOS, @{}, &e);
    if (!ok) {
        printf("  [load FAIL]\n");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return m;
    }
    m.model = model;
    m.tmp_dir = td;
    return m;
}

static void cleanup_model(Model *m) {
    if (!m->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        m->model, sel_registerName("unloadWithQoS:error:"), kQOS, &e);
    [[NSFileManager defaultManager] removeItemAtPath:m->tmp_dir error:nil];
    m->model = nil;
}

/* Run model with n_in input surfaces, n_out output surfaces.
   Returns YES on success. */
static BOOL run_model(Model *m,
                      IOSurfaceRef *ins, int n_in,
                      IOSurfaceRef *outs, int n_out) {
    NSMutableArray *inArr  = [NSMutableArray array];
    NSMutableArray *inIdx  = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array];
    NSMutableArray *outIdx = [NSMutableArray array];
    SEL sel_ios = sel_registerName("objectWithIOSurface:");
    for (int i = 0; i < n_in;  i++) {
        [inArr  addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOS, sel_ios, ins[i])];
        [inIdx  addObject:@(i)];
    }
    for (int i = 0; i < n_out; i++) {
        [outArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOS, sel_ios, outs[i])];
        [outIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Req,
        sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:"
                         "weightsBuffer:perfStats:procedureIndex:"),
        inArr, inIdx, outArr, outIdx, nil, nil, @0);
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        m->model, sel_registerName("evaluateWithQoS:options:request:error:"),
        kQOS, @{}, req, &e);
    if (!ok) printf("  [eval FAIL] %s\n",
        e ? [[e localizedDescription] UTF8String] : "unknown");
    return ok;
}

/* ════════════════════════════════════════════════════════════════════════
   Q3: S alignment diagnosis for activation-only graphs
   ════════════════════════════════════════════════════════════════════════ */

static void q3_activation_alignment(void) {
    printf("\n══════════════════════════════════════════\n");
    printf(" Q3: Activation-only S alignment (relu)\n");
    printf("══════════════════════════════════════════\n");
    printf("%-8s %-8s %-8s %-12s %s\n", "C", "S", "ok", "max_err", "channels_zero");

    int test_C[] = {16, 16, 16, 64, 64, 64, 64};
    int test_S[] = {16, 32, 48, 16, 32, 48, 64};
    int n = (int)(sizeof(test_C) / sizeof(test_C[0]));

    for (int ti = 0; ti < n; ti++) {
        int C = test_C[ti], S = test_S[ti];
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremltools-version\", \"9.0\"}})"
            "]\n{\n"
            "    func main<ios18>(tensor<fp16, [1,%d,1,%d]> x) {\n"
            "        tensor<fp16, [1,%d,1,%d]> y = relu(x=x)"
            "[name=string(\"relu\")];\n"
            "    } -> (y);\n}\n",
            C, S, C, S];

        Model m = compile_mil(mil, nil);
        if (!m.model) { printf("%-8d %-8d compile_fail\n", C, S); continue; }

        int n_elem = C * S;
        float *in_f  = (float *)malloc((size_t)n_elem * sizeof(float));
        float *out_f = (float *)malloc((size_t)n_elem * sizeof(float));
        for (int i = 0; i < n_elem; i++) in_f[i] = 1.0f + (i % 17) * 0.1f;

        IOSurfaceRef ios_in  = make_surface((size_t)n_elem * 2);
        IOSurfaceRef ios_out = make_surface((size_t)n_elem * 2);
        ios_write_f16(ios_in, in_f, n_elem);

        IOSurfaceRef ins[]  = { ios_in };
        IOSurfaceRef outs[] = { ios_out };
        BOOL ok = run_model(&m, ins, 1, outs, 1);
        ios_read_f16(ios_out, out_f, n_elem);

        float max_err = 0.0f;
        int zeros = 0;
        for (int i = 0; i < n_elem; i++) {
            float e = fabsf(out_f[i] - in_f[i]);
            if (e > max_err) max_err = e;
            if (out_f[i] == 0.0f) zeros++;
        }

        /* Find which channels are zero */
        int zero_chans = 0;
        for (int c = 0; c < C; c++) {
            int all_zero = 1;
            for (int s = 0; s < S; s++)
                if (out_f[c * S + s] != 0.0f) { all_zero = 0; break; }
            if (all_zero) zero_chans++;
        }

        printf("C=%-5d S=%-5d %-8s max_err=%-8.4f zero_elems=%d/%d zero_chans=%d/%d\n",
               C, S, ok ? "PASS" : "FAIL", max_err, zeros, n_elem, zero_chans, C);

        /* Extra: dump raw output bytes for C=16,S=16 to see layout */
        if (C == 16 && S == 16) {
            printf("  raw output[0..47] fp16 as f32: ");
            for (int i = 0; i < 48 && i < n_elem; i++)
                printf("%.2f ", out_f[i]);
            printf("\n");
            printf("  raw input [0..47] fp16 as f32: ");
            for (int i = 0; i < 48 && i < n_elem; i++)
                printf("%.2f ", in_f[i]);
            printf("\n");
        }

        CFRelease(ios_in); CFRelease(ios_out);
        free(in_f); free(out_f);
        cleanup_model(&m);
    }

    /* Extra sub-test: does a conv-then-relu work at S=16? */
    printf("\n  Sub-test: identity_conv → relu at S=16 (does conv fix S alignment?)\n");
    {
        int C = 16, S = 16;
        /* identity 1×1 conv weight */
        uint16_t *wdata = (uint16_t *)calloc((size_t)C * C, 2);
        for (int i = 0; i < C; i++) wdata[i * C + i] = 0x3C00; /* fp16 1.0 */
        NSData *wb = make_weight_blob(wdata, C * C);
        free(wdata);

        NSString *wkey = @"@model_path/weights/w.bin";
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremltools-version\", \"9.0\"}})"
            "]\n{\n"
            "    func main<ios18>(tensor<fp16, [1,%d,1,%d]> x) {\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()["
            "name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>"
            "(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n"
            "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<fp16, [1,%d,1,%d]> c = conv(dilations=dl, groups=1,"
            " pad=pd, pad_type=pt, strides=st, weight=W, x=x)[name=string(\"cv\")];\n"
            "        tensor<fp16, [1,%d,1,%d]> y = relu(x=c)[name=string(\"relu\")];\n"
            "    } -> (y);\n}\n",
            C, S, C, C, C, C, wkey, C, S, C, S];

        NSDictionary *weights = @{ wkey: @{@"offset":@0, @"data":wb} };
        Model m = compile_mil(mil, weights);
        if (!m.model) { printf("  compile FAIL\n"); }
        else {
            int n_elem = C * S;
            float *in_f  = (float *)malloc((size_t)n_elem * sizeof(float));
            float *out_f = (float *)malloc((size_t)n_elem * sizeof(float));
            for (int i = 0; i < n_elem; i++) in_f[i] = 1.0f + (i % 17) * 0.1f;
            IOSurfaceRef ios_in  = make_surface((size_t)n_elem * 2);
            IOSurfaceRef ios_out = make_surface((size_t)n_elem * 2);
            ios_write_f16(ios_in, in_f, n_elem);
            IOSurfaceRef ins[]  = { ios_in };
            IOSurfaceRef outs[] = { ios_out };
            BOOL ok = run_model(&m, ins, 1, outs, 1);
            ios_read_f16(ios_out, out_f, n_elem);
            float max_err = 0; int zeros = 0;
            for (int i = 0; i < n_elem; i++) {
                float e = fabsf(out_f[i] - in_f[i]);
                if (e > max_err) max_err = e;
                if (out_f[i] == 0.0f) zeros++;
            }
            printf("  C=%d S=%d conv→relu: %s max_err=%.4f zeros=%d/%d\n",
                   C, S, ok?"PASS":"FAIL", max_err, zeros, n_elem);
            CFRelease(ios_in); CFRelease(ios_out);
            free(in_f); free(out_f);
            cleanup_model(&m);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════
   Q1: matmul MIL op with two runtime tensor inputs
   ════════════════════════════════════════════════════════════════════════ */

static void q1_runtime_matmul(void) {
    printf("\n══════════════════════════════════════════\n");
    printf(" Q1: matmul MIL op — two runtime inputs\n");
    printf("══════════════════════════════════════════\n");

    /* Test A @ B where A=[1,1,M,K] B=[1,1,K,N] → [1,1,M,N]
     * Using M=K=N=32 (S=16 hits the stride-32 output bug; test both) */
    int M = 32, K = 32, N = 32;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremltools-version\", \"9.0\"}})"
        "]\n{\n"
        "    func main<ios18>(tensor<fp16, [1,1,%d,%d]> A,"
        " tensor<fp16, [1,1,%d,%d]> B) {\n"
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        "        tensor<fp16, [1,1,%d,%d]> C = matmul("
        "transpose_x=bF, transpose_y=bF, x=A, y=B)"
        "[name=string(\"mm\")];\n"
        "    } -> (C);\n}\n",
        M, K, K, N, M, N];

    printf("  MIL: A[1,1,%d,%d] @ B[1,1,%d,%d] → C[1,1,%d,%d]  (B=identity)\n",
           M, K, K, N, M, N);

    Model m = compile_mil(mil, nil);
    if (!m.model) { printf("  RESULT: compile failed — matmul op unsupported\n"); return; }
    printf("  compile: OK\n");

    int n_AB = M * K;
    int n_C  = M * N;
    float *A_f  = (float *)malloc((size_t)n_AB * sizeof(float));
    float *B_f  = (float *)malloc((size_t)n_AB * sizeof(float));
    float *C_f  = (float *)malloc((size_t)n_C  * sizeof(float));
    for (int i = 0; i < n_AB; i++) A_f[i] = (float)(i + 1) * 0.1f;
    memset(B_f, 0, (size_t)n_AB * sizeof(float));
    for (int i = 0; i < K; i++) B_f[i * N + i] = 1.0f;  /* identity */

    IOSurfaceRef ios_A = make_surface((size_t)n_AB * 2);
    IOSurfaceRef ios_B = make_surface((size_t)n_AB * 2);
    IOSurfaceRef ios_C = make_surface((size_t)n_C  * 2);
    ios_write_f16(ios_A, A_f, n_AB);
    ios_write_f16(ios_B, B_f, n_AB);

    IOSurfaceRef ins[]  = { ios_A, ios_B };
    IOSurfaceRef outs[] = { ios_C };
    BOOL ok = run_model(&m, ins, 2, outs, 1);
    ios_read_f16(ios_C, C_f, n_C);

    float max_err = 0.0f;
    for (int i = 0; i < n_C; i++) {
        float e = fabsf(C_f[i] - A_f[i]);
        if (e > max_err) max_err = e;
    }
    printf("  execute: %s  max_err=%.4f (identity B, expect C==A)\n",
           ok ? "OK" : "FAIL", max_err);
    printf("  RESULT: matmul-with-runtime-inputs = %s\n",
           (ok && max_err < 0.1f) ? "WORKS ✓" : "BROKEN ✗");

    CFRelease(ios_A); CFRelease(ios_B); CFRelease(ios_C);
    free(A_f); free(B_f); free(C_f);
    cleanup_model(&m);

    /* Also test M=K=N=16 to confirm stride is the only issue */
    printf("\n  Retry with M=K=N=16 (expect stride bug, max_err ≈ max(A))\n");
    {
        int M2 = 16, K2 = 16, N2 = 16;
        NSString *mil16 = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremltools-version\", \"9.0\"}})"
            "]\n{\n"
            "    func main<ios18>(tensor<fp16, [1,1,%d,%d]> A,"
            " tensor<fp16, [1,1,%d,%d]> B) {\n"
            "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
            "        tensor<fp16, [1,1,%d,%d]> C = matmul("
            "transpose_x=bF, transpose_y=bF, x=A, y=B)"
            "[name=string(\"mm\")];\n"
            "    } -> (C);\n}\n",
            M2, K2, K2, N2, M2, N2];
        Model mm16 = compile_mil(mil16, nil);
        if (mm16.model) {
            int n = M2 * K2;
            float *Af = (float*)malloc(n*4), *Bf = (float*)malloc(n*4), *Cf = (float*)malloc(n*4);
            for (int i = 0; i < n; i++) Af[i] = (float)(i+1)*0.1f;
            memset(Bf, 0, n*4); for (int i = 0; i < K2; i++) Bf[i*N2+i] = 1.0f;
            IOSurfaceRef iA = make_surface(n*2), iB = make_surface(n*2), iC = make_surface(n*2);
            ios_write_f16(iA, Af, n); ios_write_f16(iB, Bf, n);
            IOSurfaceRef ins16[] = {iA, iB}, outs16[] = {iC};
            BOOL ok16 = run_model(&mm16, ins16, 2, outs16, 1);
            ios_read_f16(iC, Cf, n);
            float me = 0; for (int i = 0; i < n; i++) { float e=fabsf(Cf[i]-Af[i]); if(e>me) me=e; }
            printf("  M=K=N=16: %s max_err=%.4f\n", ok16?"OK":"FAIL", me);
            CFRelease(iA); CFRelease(iB); CFRelease(iC); free(Af); free(Bf); free(Cf);
            cleanup_model(&mm16);
        }
    }

    /* Also try packed-IOSurface variant: one input IOSurface containing
     * both activations and weight, sliced with slice_by_size */
    printf("\n  Packed-IOSurface variant: single input [1,1,%d,%d],"
           " sliced into A[1,1,%d,%d] + B[1,1,%d,%d]\n",
           M, K + N, M, K, K, N);

    NSString *mil2 = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremltools-version\", \"9.0\"}})"
        "]\n{\n"
        "    func main<ios18>(tensor<fp16, [1,1,%d,%d]> packed) {\n"
        "        tensor<int32, [4]> ba = const()[name=string(\"ba\"),"
        " val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> sa = const()[name=string(\"sa\"),"
        " val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "        tensor<fp16, [1,1,%d,%d]> A = slice_by_size("
        "begin=ba, size=sa, x=packed)[name=string(\"slA\")];\n"
        "        tensor<int32, [4]> bb = const()[name=string(\"bb\"),"
        " val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> sb = const()[name=string(\"sb\"),"
        " val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "        tensor<fp16, [1,1,%d,%d]> B = slice_by_size("
        "begin=bb, size=sb, x=packed)[name=string(\"slB\")];\n"
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        "        tensor<fp16, [1,1,%d,%d]> C = matmul("
        "transpose_x=bF, transpose_y=bF, x=A, y=B)[name=string(\"mm\")];\n"
        "    } -> (C);\n}\n",
        M, K + N,          /* input shape */
        M, K, M, K,        /* slice A */
        K,                 /* B offset */
        K, N, K, N,        /* slice B */
        M, N];             /* output C */

    Model m2 = compile_mil(mil2, nil);
    if (!m2.model) { printf("  packed variant: compile FAIL\n"); return; }
    printf("  packed compile: OK\n");

    int n_packed = M * (K + N);
    float *packed_f = (float *)malloc((size_t)n_packed * sizeof(float));
    float *C2_f     = (float *)malloc((size_t)n_C  * sizeof(float));
    /* First K columns = A, last N columns = identity B */
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++)
            packed_f[r * (K + N) + c] = (float)(r * K + c + 1) * 0.1f;
        for (int c = 0; c < N; c++)
            packed_f[r * (K + N) + K + c] = (r == c) ? 1.0f : 0.0f;
    }
    IOSurfaceRef ios_pk = make_surface((size_t)n_packed * 2);
    IOSurfaceRef ios_C2 = make_surface((size_t)n_C * 2);
    ios_write_f16(ios_pk, packed_f, n_packed);

    IOSurfaceRef ins2[]  = { ios_pk };
    IOSurfaceRef outs2[] = { ios_C2 };
    ok = run_model(&m2, ins2, 1, outs2, 1);
    ios_read_f16(ios_C2, C2_f, n_C);

    float max_err2 = 0.0f;
    for (int i = 0; i < n_C; i++) {
        float e = fabsf(C2_f[i] - packed_f[(i / N) * (K + N) + (i % N)]);
        if (e > max_err2) max_err2 = e;
    }
    printf("  packed execute: %s  max_err=%.4f\n", ok ? "OK" : "FAIL", max_err2);
    printf("  RESULT: packed-IOSurface matmul = %s\n",
           (ok && max_err2 < 0.1f) ? "WORKS ✓" : "BROKEN ✗");

    CFRelease(ios_pk); CFRelease(ios_C2);
    free(packed_f); free(C2_f);
    cleanup_model(&m2);
}

/* ════════════════════════════════════════════════════════════════════════
   Q2: scaled_dot_product_attention
   ════════════════════════════════════════════════════════════════════════ */

static void q2_sdpa(void) {
    printf("\n══════════════════════════════════════════\n");
    printf(" Q2: scaled_dot_product_attention MIL op\n");
    printf("══════════════════════════════════════════\n");

    /* Test 1: basic SDPA — 1 head, seq=32, head_dim=32 (avoid stride bug) */
    int H = 1, S = 32, D = 32;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremltools-version\", \"9.0\"}})"
        "]\n{\n"
        "    func main<ios18>(tensor<fp16, [1,%d,%d,%d]> q,"
        " tensor<fp16, [1,%d,%d,%d]> k, tensor<fp16, [1,%d,%d,%d]> v) {\n"
        "        tensor<fp16, [1,%d,%d,%d]> att = "
        "scaled_dot_product_attention(query=q, key=k, value=v)"
        "[name=string(\"sdpa\")];\n"
        "    } -> (att);\n}\n",
        H, S, D, H, S, D, H, S, D, H, S, D];

    printf("  MIL: sdpa(q,k,v) shape [1,%d,%d,%d] (H=%d, seq=%d, dim=%d)\n",
           H, S, D, H, S, D);

    Model m = compile_mil(mil, nil);
    if (!m.model) {
        printf("  RESULT: SDPA compile FAIL — op unsupported\n");

        /* Try the multi-head variant in case single head is rejected */
        printf("\n  Retry with H=4, S=32, D=32\n");
        H = 4; S = 32; D = 32;
        NSString *mil2 = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremltools-version\", \"9.0\"}})"
            "]\n{\n"
            "    func main<ios18>(tensor<fp16, [1,%d,%d,%d]> q,"
            " tensor<fp16, [1,%d,%d,%d]> k, tensor<fp16, [1,%d,%d,%d]> v) {\n"
            "        tensor<fp16, [1,%d,%d,%d]> att = "
            "scaled_dot_product_attention(query=q, key=k, value=v)"
            "[name=string(\"sdpa\")];\n"
            "    } -> (att);\n}\n",
            H, S, D, H, S, D, H, S, D, H, S, D];
        Model m2 = compile_mil(mil2, nil);
        if (!m2.model) {
            printf("  RESULT: SDPA compile FAIL — op unsupported on this hardware\n");
            return;
        }
        m = m2;
        printf("  compile: OK with H=4\n");
    } else {
        printf("  compile: OK\n");
    }

    /* Run: Q=K=V=identity-ish inputs, verify output is finite */
    int n_elem = H * S * D;
    float *Q_f   = (float *)malloc((size_t)n_elem * sizeof(float));
    float *K_f   = (float *)malloc((size_t)n_elem * sizeof(float));
    float *V_f   = (float *)malloc((size_t)n_elem * sizeof(float));
    float *out_f = (float *)malloc((size_t)n_elem * sizeof(float));
    for (int i = 0; i < n_elem; i++) {
        Q_f[i] = (float)(i % 8) * 0.1f - 0.35f;
        K_f[i] = (float)(i % 5) * 0.1f - 0.2f;
        V_f[i] = (float)(i % 13) * 0.05f;
    }

    IOSurfaceRef ios_Q   = make_surface((size_t)n_elem * 2);
    IOSurfaceRef ios_K   = make_surface((size_t)n_elem * 2);
    IOSurfaceRef ios_V   = make_surface((size_t)n_elem * 2);
    IOSurfaceRef ios_out = make_surface((size_t)n_elem * 2);
    ios_write_f16(ios_Q, Q_f, n_elem);
    ios_write_f16(ios_K, K_f, n_elem);
    ios_write_f16(ios_V, V_f, n_elem);

    IOSurfaceRef ins[]  = { ios_Q, ios_K, ios_V };
    IOSurfaceRef outs[] = { ios_out };
    BOOL ok = run_model(&m, ins, 3, outs, 1);
    ios_read_f16(ios_out, out_f, n_elem);

    int finite = 0, nonzero = 0;
    for (int i = 0; i < n_elem; i++) {
        if (isfinite(out_f[i])) finite++;
        if (out_f[i] != 0.0f)  nonzero++;
    }
    printf("  execute: %s  finite=%d/%d  nonzero=%d/%d\n",
           ok ? "OK" : "FAIL", finite, n_elem, nonzero, n_elem);
    printf("  sample out[0..7]: ");
    for (int i = 0; i < 8 && i < n_elem; i++) printf("%.4f ", out_f[i]);
    printf("\n");

    /* CPU reference: single-head softmax(Q@K^T / sqrt(D)) @ V */
    float scale = 1.0f / sqrtf((float)D);
    float *ref = (float *)calloc((size_t)n_elem, sizeof(float));
    float *scores = (float *)calloc((size_t)S * S, sizeof(float));
    for (int t = 0; t < S; t++) {
        float mx = -1e30f;
        for (int t2 = 0; t2 < S; t2++) {
            float dot = 0;
            for (int d = 0; d < D; d++)
                dot += Q_f[t*D+d] * K_f[t2*D+d];
            scores[t*S+t2] = dot * scale;
            if (scores[t*S+t2] > mx) mx = scores[t*S+t2];
        }
        float sum = 0;
        for (int t2 = 0; t2 < S; t2++) { scores[t*S+t2] = expf(scores[t*S+t2]-mx); sum += scores[t*S+t2]; }
        for (int t2 = 0; t2 < S; t2++) scores[t*S+t2] /= sum;
        for (int d = 0; d < D; d++) {
            float r = 0;
            for (int t2 = 0; t2 < S; t2++) r += scores[t*S+t2] * V_f[t2*D+d];
            ref[t*D+d] = r;
        }
    }
    float max_err = 0;
    for (int i = 0; i < n_elem; i++) {
        float e = fabsf(out_f[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("  vs CPU ref:  max_err=%.4f\n", max_err);
    printf("  RESULT: SDPA = %s\n",
           (ok && finite == n_elem && nonzero > 0 && max_err < 0.1f)
           ? "WORKS ✓" : "BROKEN/INACCURATE ✗");

    CFRelease(ios_Q); CFRelease(ios_K); CFRelease(ios_V); CFRelease(ios_out);
    free(Q_f); free(K_f); free(V_f); free(out_f); free(ref); free(scores);
    cleanup_model(&m);

    /* Test 2: with causal mask */
    printf("\n  Sub-test: SDPA with causal mask (H=1, S=32, D=32)\n");
    {
        H = 1; S = 32; D = 32;
        NSString *mil3 = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremltools-version\", \"9.0\"}})"
            "]\n{\n"
            "    func main<ios18>(tensor<fp16, [1,%d,%d,%d]> q,"
            " tensor<fp16, [1,%d,%d,%d]> k, tensor<fp16, [1,%d,%d,%d]> v,"
            " tensor<fp16, [1,1,%d,%d]> mask) {\n"
            "        tensor<fp16, [1,%d,%d,%d]> att = "
            "scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)"
            "[name=string(\"sdpa_m\")];\n"
            "    } -> (att);\n}\n",
            H, S, D, H, S, D, H, S, D, S, S, H, S, D];
        Model m3 = compile_mil(mil3, nil);
        printf("  masked SDPA compile: %s\n", m3.model ? "OK" : "FAIL");
        if (m3.model) {
            /* Build causal mask IOSurface */
            float *mask_f = (float *)malloc((size_t)S * S * sizeof(float));
            for (int t = 0; t < S; t++)
                for (int t2 = 0; t2 < S; t2++)
                    mask_f[t * S + t2] = (t2 <= t) ? 0.0f : -65504.0f;

            IOSurfaceRef ios_m = make_surface((size_t)S * S * 2);
            ios_write_f16(ios_m, mask_f, S * S);

            n_elem = H * S * D;
            Q_f   = (float *)malloc((size_t)n_elem * sizeof(float));
            K_f   = (float *)malloc((size_t)n_elem * sizeof(float));
            V_f   = (float *)malloc((size_t)n_elem * sizeof(float));
            out_f = (float *)malloc((size_t)n_elem * sizeof(float));
            for (int i = 0; i < n_elem; i++) {
                Q_f[i] = (float)(i % 8) * 0.1f - 0.35f;
                K_f[i] = (float)(i % 5) * 0.1f - 0.2f;
                V_f[i] = (float)(i % 13) * 0.05f;
            }
            ios_Q   = make_surface((size_t)n_elem * 2);
            ios_K   = make_surface((size_t)n_elem * 2);
            ios_V   = make_surface((size_t)n_elem * 2);
            ios_out = make_surface((size_t)n_elem * 2);
            ios_write_f16(ios_Q, Q_f, n_elem);
            ios_write_f16(ios_K, K_f, n_elem);
            ios_write_f16(ios_V, V_f, n_elem);

            IOSurfaceRef ins3[]  = { ios_Q, ios_K, ios_V, ios_m };
            IOSurfaceRef outs3[] = { ios_out };
            ok = run_model(&m3, ins3, 4, outs3, 1);
            ios_read_f16(ios_out, out_f, n_elem);
            finite = 0; nonzero = 0;
            for (int i = 0; i < n_elem; i++) {
                if (isfinite(out_f[i])) finite++;
                if (out_f[i] != 0.0f)  nonzero++;
            }
            printf("  masked execute: %s  finite=%d/%d  nonzero=%d/%d\n",
                   ok ? "OK" : "FAIL", finite, n_elem, nonzero, n_elem);

            CFRelease(ios_Q); CFRelease(ios_K); CFRelease(ios_V);
            CFRelease(ios_out); CFRelease(ios_m);
            free(Q_f); free(K_f); free(V_f); free(out_f); free(mask_f);
            cleanup_model(&m3);
        }
    }
}

int main(void) {
    @autoreleasepool {
        printf("=== probe_dynamic_ops ===\n");
        ane_init();
        if (!g_Desc || !g_Model || !g_IOS || !g_Req) {
            fprintf(stderr, "ANE classes not found — not running on Apple Silicon?\n");
            return 1;
        }
        q3_activation_alignment();
        q1_runtime_matmul();
        q2_sdpa();
        printf("\n=== DONE ===\n");
    }
    return 0;
}
