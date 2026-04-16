/**
 * probe_chaining.m — _ANEChainingRequest + _ANEClient chaining API probe
 *
 * Goals:
 *   1. Dump full method signatures (with ObjC type encodings) for:
 *        _ANEChainingRequest, _ANEClient, _ANEIOSurfaceOutputSets,
 *        _ANEOutputSetEnqueue, _ANEInputBuffersReady
 *   2. Compile two minimal identity kernels (A and B)
 *   3. Attempt progressive chain construction:
 *        a) Create _ANEChainingRequest — known to work
 *        b) Call validate — known to crash with scalar params, need array types
 *        c) prepareChainingWithModel: on _ANEClient
 *        d) buffersReadyWithModel: + enqueueSetsWithModel: dispatch
 *
 * Reference: maderix/ANE training/test_ane_advanced.m + m5result.md
 * Key finding from m5result: validate fails because some NSNumber args must
 * be NSArrays. Type encodings here tell us which ones.
 *
 * Build:
 *   clang -fmodules -framework Foundation -framework IOSurface \
 *         -o probe_chaining probe_chaining.m && ./probe_chaining
 */

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void dump_class_full(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  [NOT FOUND] %s\n\n", name); return; }

    printf("\n╔═══ %s ═══╗\n", name);

    // Class methods
    unsigned int count = 0;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count > 0) {
        printf("  Class methods (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            const char *enc = method_getTypeEncoding(methods[i]);
            printf("    + %-60s  %s\n", sel_getName(s), enc ? enc : "?");
        }
    }
    free(methods);

    // Instance methods
    methods = class_copyMethodList(cls, &count);
    if (count > 0) {
        printf("  Instance methods (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            const char *enc = method_getTypeEncoding(methods[i]);
            printf("    - %-60s  %s\n", sel_getName(s), enc ? enc : "?");
        }
    }
    free(methods);

    // Properties
    objc_property_t *props = class_copyPropertyList(cls, &count);
    if (count > 0) {
        printf("  Properties (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            printf("    @property %-40s  %s\n",
                   property_getName(props[i]),
                   property_getAttributes(props[i]));
        }
    }
    free(props);
    printf("\n");
}

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth:          @(bytes),
        (id)kIOSurfaceHeight:         @1,
        (id)kIOSurfaceBytesPerElement:@1,
        (id)kIOSurfaceBytesPerRow:    @(bytes),
        (id)kIOSurfaceAllocSize:      @(bytes),
        (id)kIOSurfacePixelFormat:    @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

/* Build a minimal fp16 weight blob: 128-byte ANE header + CH×CH fp16 identity */
static NSData *make_identity_weight(int CH) {
    int n_elems = CH * CH;
    int ws = n_elems * 2;              // fp16 bytes
    int total = 128 + ws;
    uint8_t *blob = (uint8_t *)calloc(total, 1);

    /* File header (64 bytes) */
    blob[0] = 1;                        /* version */
    blob[4] = 2;                        /* type */
    /* Chunk header at offset 64 */
    blob[64] = 0xEF; blob[65] = 0xBE;  /* magic 0xDEADBEEF (LE) */
    blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 1;                       /* chunk type */
    *(uint32_t *)(blob + 72) = (uint32_t)ws;   /* data size */
    *(uint32_t *)(blob + 80) = 128;            /* data offset */

    /* fp16 identity matrix */
    uint16_t *fp16 = (uint16_t *)(blob + 128);
    for (int i = 0; i < CH; i++) {
        fp16[i * CH + i] = 0x3C00; /* fp16 1.0 */
    }

    return [NSData dataWithBytesNoCopy:blob length:total freeWhenDone:YES];
}

/* Build MIL text for a CH×CH conv identity, spatial SP */
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

/* Compile + load a model, return the ObjC model object (retained) */
static id compile_and_load(int CH, int SP, const char *tag) {
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class M = NSClassFromString(@"_ANEInMemoryModel");
    if (!D || !M) { printf("[%s] classes not found\n", tag); return nil; }

    NSData *wdata = make_identity_weight(CH);
    NSData *mil_data = [make_mil(CH, SP) dataUsingEncoding:NSUTF8StringEncoding];

    NSDictionary *wdict = @{
        @"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": wdata }
    };

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        D, sel_registerName("modelWithMILText:weights:optionsPlist:"),
        mil_data, wdict, nil);
    if (!desc) { printf("[%s] descriptor failed\n", tag); return nil; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        M, sel_registerName("inMemoryModelWithDescriptor:"), desc);
    if (!mdl) { printf("[%s] model failed\n", tag); return nil; }

    NSString *hx = ((NSString*(*)(id,SEL))objc_msgSend)(
        mdl, sel_registerName("hexStringIdentifier"));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil_data writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata    writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, sel_registerName("compileWithQoS:options:error:"), 21, @{}, &e);
    if (!ok) { printf("[%s] compile failed: %s\n", tag, e ? [[e description] UTF8String] : "?"); return nil; }

    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, sel_registerName("loadWithQoS:options:error:"), 21, @{}, &e);
    if (!ok) { printf("[%s] load failed: %s\n", tag, e ? [[e description] UTF8String] : "?"); return nil; }

    [mdl retain];
    printf("[%s] compiled + loaded OK\n", tag);
    return mdl;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW);

        printf("╔════════════════════════════════════════════════════╗\n");
        printf("║         libane — ChainingRequest Probe             ║\n");
        printf("╚════════════════════════════════════════════════════╝\n\n");

        /* ── Part 1: Full signature dumps ─────────────────────────────────── */
        printf("═══════════════════════════════════════════════════════\n");
        printf("PART 1 — Type-encoding dumps (key for fixing validate)\n");
        printf("═══════════════════════════════════════════════════════\n");

        dump_class_full("_ANEChainingRequest");
        dump_class_full("_ANEClient");
        dump_class_full("_ANEIOSurfaceOutputSets");
        dump_class_full("_ANEOutputSetEnqueue");
        dump_class_full("_ANEInputBuffersReady");
        dump_class_full("_ANEBuffer");

        /* ── Part 2: Compile two identity kernels ─────────────────────────── */
        printf("═══════════════════════════════════════════════════════\n");
        printf("PART 2 — Compile kernel A and kernel B\n");
        printf("═══════════════════════════════════════════════════════\n");

        int CH = 64, SP = 32;
        id mdl_A = compile_and_load(CH, SP, "kernel_A");
        id mdl_B = compile_and_load(CH, SP, "kernel_B");

        if (!mdl_A || !mdl_B) {
            printf("Cannot proceed without both kernels.\n");
            return 1;
        }

        /* ── Part 3: _ANEChainingRequest construction ─────────────────────── */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 3 — _ANEChainingRequest correct construction\n");
        printf("         (informed by type-encoding dump in Part 1)\n");
        printf("═══════════════════════════════════════════════════════\n");
        printf("  Key findings:\n");
        printf("  - +chainingRequestWithInputs:... does NOT exist on this OS\n");
        printf("  - Use alloc / initWithInputs:outputs:lb...:procedureIndex:...\n");
        printf("  - inputBuffer  = NSArray of _ANEBuffer\n");
        printf("  - outputSets   = NSArray of _ANEIOSurfaceOutputSets\n");
        printf("  - loopbackInputSymbolIndex  = NSArray  (not NSNumber!)\n");
        printf("  - loopbackOutputSymbolIndex = NSArray  (not NSNumber!)\n");
        printf("  - validate takes NO argument: -validate  (not -validate:)\n\n");

        Class cls_Chain  = NSClassFromString(@"_ANEChainingRequest");
        Class cls_AIO    = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cls_AR     = NSClassFromString(@"_ANERequest");
        Class cls_OSets  = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class cls_ABuf   = NSClassFromString(@"_ANEBuffer");

        if (!cls_Chain)  { printf("_ANEChainingRequest not found\n"); return 1; }
        if (!cls_OSets)  printf("WARNING: _ANEIOSurfaceOutputSets not found\n");
        if (!cls_ABuf)   printf("WARNING: _ANEBuffer not found\n");

        int io_bytes = CH * SP * 4; /* fp32 */
        IOSurfaceRef stats_surf_live = make_surface(4096); /* kept alive for all output_set uses */
        IOSurfaceRef surf_in  = make_surface(io_bytes);
        IOSurfaceRef surf_mid = make_surface(io_bytes);
        IOSurfaceRef surf_out = make_surface(io_bytes);

        /* Wrap surfaces as _ANEIOSurfaceObject */
        id sio_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
        id sio_mid = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
        id sio_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            cls_AIO, sel_registerName("objectWithIOSurface:"), surf_out);
        printf("  _ANEIOSurfaceObjects: in=%s mid=%s out=%s\n",
               sio_in ? "OK" : "nil", sio_mid ? "OK" : "nil", sio_out ? "OK" : "nil");

        /* Build _ANEBuffer objects wrapping the IOSurfaceObjects
         * +bufferWithIOSurfaceObject:symbolIndex:source:
         *   symbolIndex = NSNumber (index into model's symbol table)
         *   source      = int64_t (0 = external / host)
         */
        id buf_in  = nil, buf_out = nil;
        if (cls_ABuf) {
            buf_in = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                cls_ABuf,
                sel_registerName("bufferWithIOSurfaceObject:symbolIndex:source:"),
                sio_in, @0, (long long)0);
            buf_out = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                cls_ABuf,
                sel_registerName("bufferWithIOSurfaceObject:symbolIndex:source:"),
                sio_out, @0, (long long)0);
            printf("  _ANEBuffers: in=%s out=%s\n",
                   buf_in ? "OK" : "nil", buf_out ? "OK" : "nil");
        }

        /* Build _ANEIOSurfaceOutputSets
         * +objectWithstatsSurRef:outputBuffer:
         *   statsSurRef  = IOSurfaceRef (perf stats surface, can try nil/NULL)
         *   outputBuffer = NSArray of output _ANEBuffer objects
         */
        id output_set = nil;
        if (cls_OSets && buf_out) {
            /* Try with NULL stats surface first */
            @try {
                output_set = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_OSets,
                    sel_registerName("objectWithstatsSurRef:outputBuffer:"),
                    (IOSurfaceRef)NULL,
                    @[buf_out]);
                printf("  _ANEIOSurfaceOutputSets (statsRef=NULL): %s\n",
                       output_set ? [[output_set description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  _ANEIOSurfaceOutputSets (statsRef=NULL) EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            if (!output_set) {
                /* Use stats_surf_live (declared at top, freed at cleanup) */
                @try {
                    output_set = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                        cls_OSets,
                        sel_registerName("objectWithstatsSurRef:outputBuffer:"),
                        stats_surf_live,
                        @[buf_out]);
                    printf("  _ANEIOSurfaceOutputSets (statsRef=real): %s\n",
                           output_set ? [[output_set description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  _ANEIOSurfaceOutputSets (statsRef=real) EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
            }
        }

        /* Build _ANEChainingRequest using the instance init method:
         * -initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:
         *   procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:
         *
         * Types (from property dump):
         *   inputs             → NSArray of _ANEBuffer       (inputBuffer property)
         *   outputs            → NSArray of _ANEIOSurfaceOutputSets (outputSets property)
         *   lbInputSymbolId    → NSArray                     (loopbackInputSymbolIndex)
         *   lbOutputSymbolId   → NSArray                     (loopbackOutputSymbolIndex)
         *   procedureIndex     → NSNumber
         *   signalEvents       → NSArray  (or nil)
         *   transactionHandle  → NSNumber (or nil)
         *   fwEnqueueDelay     → NSNumber
         *   memoryPoolId       → NSNumber
         */
        id live_chain = nil;
        if (buf_in && output_set) {
            printf("\n  [chain construction] using initWithInputs:outputs:... (correct API)\n");
            @try {
                live_chain = [cls_Chain alloc];
                live_chain = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    live_chain,
                    sel_registerName("initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"),
                    @[buf_in],      /* inputs: array of _ANEBuffer */
                    @[output_set],  /* outputs: array of _ANEIOSurfaceOutputSets */
                    @[],            /* lbInputSymbolId: empty NSArray (no loopback) */
                    @[],            /* lbOutputSymbolId: empty NSArray */
                    @0,             /* procedureIndex: NSNumber */
                    nil,            /* signalEvents: nil */
                    nil,            /* transactionHandle: nil */
                    @0,             /* fwEnqueueDelay: NSNumber */
                    @0);            /* memoryPoolId: NSNumber */
                printf("  chain = %s\n",
                       live_chain ? [[live_chain description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  chain EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
                live_chain = nil;
            }
        } else {
            printf("\n  Skipping chain construction (missing buffers or output_set)\n");
        }

        /* validate — NO argument, returns BOOL */
        if (live_chain) {
            printf("\n  [validate] -validate (no arg)\n");
            @try {
                BOOL vok = ((BOOL(*)(id,SEL))objc_msgSend)(
                    live_chain, sel_registerName("validate"));
                printf("  validate: %s\n", vok ? "OK" : "FAIL");
            } @catch (NSException *ex) {
                printf("  validate EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        /* ── Part 4: Find class with getUUID + client load path ──────────── */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 4 — Find getUUID + _ANEClient load path\n");
        printf("         prepareChainingWithModel: calls getUUID on model.\n");
        printf("         _ANEInMemoryModel lacks it — need client-loaded model.\n");
        printf("═══════════════════════════════════════════════════════\n");

        /* Scan all ANE classes for getUUID */
        printf("\n  Classes with 'getUUID':\n");
        unsigned int allCount = 0;
        Class *allCls = objc_copyClassList(&allCount);
        for (unsigned int i = 0; i < allCount; i++) {
            const char *nm = class_getName(allCls[i]);
            if (!strstr(nm, "ANE") && !strstr(nm, "ane")) continue;
            if (class_respondsToSelector(allCls[i], sel_registerName("getUUID"))) {
                printf("    %s  has getUUID\n", nm);
            }
        }
        free(allCls);

        /* Dump _ANEModel if it exists */
        dump_class_full("_ANEModel");
        dump_class_full("_ANEProgramForEvaluation");
        dump_class_full("_ANEModelDescriptor");

        Class cls_Client = NSClassFromString(@"_ANEClient");
        if (!cls_Client) { printf("_ANEClient not found\n"); goto cleanup; }

        id client = ((id(*)(Class,SEL))objc_msgSend)(
            cls_Client, sel_registerName("sharedConnection"));
        printf("  sharedConnection: %s\n", client ? "OK" : "nil");

        /* ── Introspect _ANEInMemoryModel for internal handles ────────────
         * _ANEClient.loadModel: calls connectionForLoadingModel: which needs
         * getUUID — a method _ANEInMemoryModel lacks. _ANEModel has UUID.
         *
         * Instead of the client path, look for _ANEProgramForEvaluation
         * inside the compiled model. It has processInputBuffers: and
         * processOutputSet: which ARE the chaining primitives.
         */
        printf("\n  Introspecting _ANEInMemoryModel internal state:\n");

        /* Dump all methods on the live model object's class */
        {
            Class mdl_cls = [mdl_A class];
            printf("  mdl_A class: %s\n", class_getName(mdl_cls));
            unsigned int mc = 0;
            Method *ms = class_copyMethodList(mdl_cls, &mc);
            printf("  Instance methods (%u):\n", mc);
            for (unsigned int i = 0; i < mc; i++) {
                SEL s = method_getName(ms[i]);
                const char *enc = method_getTypeEncoding(ms[i]);
                printf("    - %-55s  %s\n", sel_getName(s), enc ? enc : "?");
            }
            free(ms);
        }

        /* Try KVC to reach internal program / model handles */
        const char *kvc_keys[] = {
            "model", "_model", "program", "_program", "aneModel",
            "internalModel", "modelHandle", "programHandle",
            "evaluationProgram", "activeProgram", "currentProgram",
            "programForEvaluation", NULL
        };
        for (int i = 0; kvc_keys[i]; i++) {
            @try {
                id val = [mdl_A valueForKey:[NSString stringWithUTF8String:kvc_keys[i]]];
                if (val) {
                    printf("  KVC[%s] = %s  (class: %s)\n",
                           kvc_keys[i],
                           [[val description] UTF8String],
                           class_getName([val class]));
                }
            } @catch (...) {}
        }

        /* Try _ANEProgramForEvaluation path via _ANEProgramIOSurfacesMapper */
        dump_class_full("_ANEProgramIOSurfacesMapper");

        /* ── BREAKTHROUGH: inner _ANEModel has getUUID ────────────────────
         * _ANEInMemoryModel.model (KVC) returns _ANEModel which has:
         *   - UUID (NSUUID) ← what prepareChainingWithModel: needs
         *   - program (_ANEProgramForEvaluation) with programHandle + queueDepth=127
         *   - modelAttributes with ANEFModelInputSymbolIndexArray / OutputSymbolIndexArray
         *     → these are the loopback symbol indices for chaining
         *
         * Fix: pass inner_model_A to prepareChainingWithModel: instead of mdl_A.
         */
        id inner_model_A = nil;
        id inner_model_B = nil;
        @try {
            inner_model_A = [mdl_A valueForKey:@"model"];
            inner_model_B = [mdl_B valueForKey:@"model"];
        } @catch (...) {}

        printf("  inner_model_A (_ANEModel): %s\n",
               inner_model_A ? "OK" : "nil");
        printf("  inner_model_B (_ANEModel): %s\n",
               inner_model_B ? "OK" : "nil");

        if (inner_model_A) {
            id uuid = ((id(*)(id,SEL))objc_msgSend)(inner_model_A, sel_registerName("UUID"));
            printf("  inner_model_A UUID: %s\n", uuid ? [[uuid description] UTF8String] : "nil");
        }

        /* Extract loopback symbol indices from modelAttributes
         * ANEFModelInputSymbolIndexArray / ANEFModelOutputSymbolIndexArray
         * These go into lbInputSymbolId / lbOutputSymbolId of the chain request.
         */
        NSArray *lb_in_syms  = @[@0];  /* default from probe: input symbol 0 */
        NSArray *lb_out_syms = @[@0];  /* default from probe: output symbol 0 */
        if (inner_model_A) {
            @try {
                id attrs = ((id(*)(id,SEL))objc_msgSend)(
                    inner_model_A, sel_registerName("modelAttributes"));
                if (attrs) {
                    id ane_desc = [attrs objectForKey:@"ANEFModelDescription"];
                    if (ane_desc) {
                        id in_syms  = [ane_desc objectForKey:@"ANEFModelInputSymbolIndexArray"];
                        id out_syms = [ane_desc objectForKey:@"ANEFModelOutputSymbolIndexArray"];
                        if (in_syms  && [in_syms  isKindOfClass:[NSArray class]]) lb_in_syms  = in_syms;
                        if (out_syms && [out_syms isKindOfClass:[NSArray class]]) lb_out_syms = out_syms;
                        printf("  lb_in_syms:  %s\n", [[lb_in_syms  description] UTF8String]);
                        printf("  lb_out_syms: %s\n", [[lb_out_syms description] UTF8String]);
                    }
                }
            } @catch (...) {}
        }

        /* Rebuild output_set with the (same) live stats surface */
        id output_set2 = nil;
        if (cls_OSets && buf_out) {
            @try {
                output_set2 = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    cls_OSets,
                    sel_registerName("objectWithstatsSurRef:outputBuffer:"),
                    stats_surf_live,
                    @[buf_out]);
                printf("  output_set2 (live statsRef): %s\n",
                       output_set2 ? "OK" : "nil");
            } @catch (NSException *ex) {
                printf("  output_set2 EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        /* ── Chain v2a: minimal — no inputs/outputs, just validate structure ─ */
        printf("\n  [chain v2a] minimal (nil inputs/outputs — test prepare signature)\n");
        id chain_minimal = nil;
        @try {
            chain_minimal = [cls_Chain alloc];
            chain_minimal = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                chain_minimal,
                sel_registerName("initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"),
                @[],            /* empty inputs */
                @[],            /* empty outputs */
                @[],
                @[],
                @0, nil, nil, @0, @0);
            printf("  chain_minimal: %s\n", chain_minimal ? "OK" : "nil");
        } @catch (NSException *ex) {
            printf("  chain_minimal EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        if (client && inner_model_A && chain_minimal) {
            printf("  [prepareChainingWithModel:] minimal chain\n");
            NSError *ce = nil;
            @try {
                BOOL cok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client,
                    sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
                    inner_model_A, @{}, chain_minimal, (unsigned int)21, &ce);
                printf("  result: %s  err=%s\n",
                       cok ? "OK" : "FAIL",
                       ce ? [[ce description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        /* ── Chain v2b: full — buf_in input + live output_set2 ─────────────── */
        id live_chain2 = nil;
        if (buf_in && output_set2) {
            printf("\n  [chain v2b] full chain with live stats surface\n");
            @try {
                live_chain2 = [cls_Chain alloc];
                live_chain2 = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    live_chain2,
                    sel_registerName("initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"),
                    @[buf_in],
                    @[output_set2],
                    lb_in_syms,
                    lb_out_syms,
                    @0, nil, nil, @0, @0);
                BOOL vok = ((BOOL(*)(id,SEL))objc_msgSend)(live_chain2, sel_registerName("validate"));
                printf("  chain v2b: %s  validate: %s\n",
                       live_chain2 ? "OK" : "nil", vok ? "OK" : "FAIL");
            } @catch (NSException *ex) {
                printf("  chain v2b EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        /* ── Try registering the model with the client before chaining ──────
         * _ANEInMemoryModel.loadWithQoS: loads into SRAM via the model's own
         * internal path.  prepareChainingWithModel: (error 15) may require
         * the model to be registered through _ANEClient.loadModel: instead.
         *
         * Strategy: unload from InMemoryModel path, then reload via client
         * using inner_model_A (_ANEModel with UUID + modelURL).
         */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 4b — client-path load → chaining\n");
        printf("═══════════════════════════════════════════════════════\n");

        /* ── Approach A: reset programHandle on unloaded inner_model_A ──────
         * After unloadWithQoS:, the inner _ANEModel still has its programHandle
         * from the previous load. _ANEClient.loadModel: may be checking that
         * programHandle == 0 as a precondition. Reset it and retry.
         */
        if (client && inner_model_A) {
            printf("  [unload via InMemoryModel]\n");
            @try {
                NSError *ue = nil;
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl_A, sel_registerName("unloadWithQoS:error:"), (unsigned int)21, &ue);
            } @catch (...) {}

            /* Read state + handle before reset */
            uint64_t ph_before = ((uint64_t(*)(id,SEL))objc_msgSend)(
                inner_model_A, sel_registerName("programHandle"));
            uint64_t st_before = ((uint64_t(*)(id,SEL))objc_msgSend)(
                inner_model_A, sel_registerName("state"));
            printf("  after unload: state=%llu  programHandle=%llu\n",
                   (unsigned long long)st_before, (unsigned long long)ph_before);

            /* Try approach A: reset programHandle=0 then client load */
            printf("\n  [Approach A] reset programHandle → 0, then client load\n");
            @try {
                ((void(*)(id,SEL,uint64_t))objc_msgSend)(
                    inner_model_A, sel_registerName("setProgramHandle:"), (uint64_t)0);
                uint64_t ph_after = ((uint64_t(*)(id,SEL))objc_msgSend)(
                    inner_model_A, sel_registerName("programHandle"));
                printf("  programHandle after reset: %llu\n", (unsigned long long)ph_after);
            } @catch (NSException *ex) {
                printf("  setProgramHandle EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            NSError *cle_a = nil;
            BOOL clok_a = NO;
            @try {
                clok_a = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client,
                    sel_registerName("loadModel:options:qos:error:"),
                    inner_model_A, @{}, (unsigned int)21, &cle_a);
                printf("  client loadModel (A): %s  err=%s\n",
                       clok_a ? "OK" : "FAIL",
                       cle_a ? [[cle_a description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  client loadModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            if (clok_a && live_chain2) {
                printf("  [prepareChainingWithModel:] after approach A load\n");
                NSError *ce_a = nil;
                @try {
                    BOOL cok_a = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client,
                        sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
                        inner_model_A, @{}, live_chain2, (unsigned int)21, &ce_a);
                    printf("  prepareChainingWithModel (A): %s  err=%s\n",
                           cok_a ? "OK" : "FAIL",
                           ce_a ? [[ce_a description] UTF8String] : "nil");
                    if (cok_a) goto chaining_success;
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            /* ── Approach B: compile-only model, client load from scratch ──── */
            printf("\n  [Approach B] fresh compile-only model → client load\n");

            /* Compile a new model (no load via InMemoryModel) */
            Class D2 = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class M2 = NSClassFromString(@"_ANEInMemoryModel");
            NSData *w2 = make_identity_weight(CH);
            NSData *m2 = [make_mil(CH, SP) dataUsingEncoding:NSUTF8StringEncoding];
            NSDictionary *wd2 = @{
                @"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": w2 }
            };
            id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                D2, sel_registerName("modelWithMILText:weights:optionsPlist:"), m2, wd2, nil);
            id mdl_C = ((id(*)(Class,SEL,id))objc_msgSend)(
                M2, sel_registerName("inMemoryModelWithDescriptor:"), desc2);

            NSString *hx2 = ((NSString*(*)(id,SEL))objc_msgSend)(
                mdl_C, sel_registerName("hexStringIdentifier"));
            NSString *td2 = [NSTemporaryDirectory() stringByAppendingPathComponent:hx2];
            NSFileManager *fm2 = [NSFileManager defaultManager];
            [fm2 createDirectoryAtPath:[td2 stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [m2 writeToFile:[td2 stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [w2 writeToFile:[td2 stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            NSError *ce2b = nil;
            BOOL comp2 = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl_C, sel_registerName("compileWithQoS:options:error:"), 21, @{}, &ce2b);
            printf("  compile (no load): %s\n", comp2 ? "OK" : "FAIL");

            if (comp2) {
                id inner_C = nil;
                @try { inner_C = [mdl_C valueForKey:@"model"]; } @catch (...) {}
                printf("  inner_C (_ANEModel): %s  programHandle=%llu  state=%llu\n",
                       inner_C ? "OK" : "nil",
                       inner_C ? (unsigned long long)((uint64_t(*)(id,SEL))objc_msgSend)(inner_C, sel_registerName("programHandle")) : 0,
                       inner_C ? (unsigned long long)((uint64_t(*)(id,SEL))objc_msgSend)(inner_C, sel_registerName("state")) : 0);

                NSError *cle_b = nil;
                BOOL clok_b = NO;
                @try {
                    clok_b = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client,
                        sel_registerName("loadModel:options:qos:error:"),
                        inner_C, @{}, (unsigned int)21, &cle_b);
                    printf("  client loadModel (B): %s  err=%s\n",
                           clok_b ? "OK" : "FAIL",
                           cle_b ? [[cle_b description] UTF8String] : "nil");
                    if (inner_C) {
                        printf("  programHandle after client load: %llu\n",
                               (unsigned long long)((uint64_t(*)(id,SEL))objc_msgSend)(inner_C, sel_registerName("programHandle")));
                    }
                } @catch (NSException *ex) {
                    printf("  client loadModel (B) EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                if (clok_b && live_chain2 && inner_C) {
                    printf("  [prepareChainingWithModel:] after approach B load\n");
                    NSError *ce_b = nil;
                    @try {
                        BOOL cok_b = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
                            inner_C, @{}, live_chain2, (unsigned int)21, &ce_b);
                        printf("  prepareChainingWithModel (B): %s  err=%s\n",
                               cok_b ? "OK" : "FAIL",
                               ce_b ? [[ce_b description] UTF8String] : "nil");
                        if (cok_b) goto chaining_success;
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                }
            }
        }
        goto skip_chain_success;

chaining_success:
        printf("\n  *** CHAINING PREPARED SUCCESSFULLY ***\n");
skip_chain_success:;

        /* ── PART 4c: _ANEProgramForEvaluation direct path ──────────────────
         * Key finding: _ANEClient.loadModel: expects model.espresso.net (CoreML
         * Espresso path). Our _ANEInMemoryModel compile path produces E5/HWX
         * directly — these paths are incompatible.
         *
         * The correct chaining primitive for our path is _ANEProgramForEvaluation
         * which is obtained from [inner_model.program]:
         *
         *   program_A = [inner_model_A valueForKey:@"program"]
         *   → _ANEProgramForEvaluation { programHandle, intermediateBufferHandle, queueDepth=127 }
         *
         * It has:
         *   processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:
         *   processInputBuffers:model:options:error:   ← chaining: signal input ready
         *   processOutputSet:model:options:error:      ← chaining: enqueue output
         *
         * Try processRequest: first as a sanity check, then the chaining pair.
         */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 4c — _ANEProgramForEvaluation direct path\n");
        printf("═══════════════════════════════════════════════════════\n");

        /* Use FRESH models — Part 4b mutated mdl_A/inner_model_A */
        id mdl_C = compile_and_load(CH, SP, "kernel_C");
        id mdl_D = compile_and_load(CH, SP, "kernel_D");
        if (!mdl_C || !mdl_D) { printf("  Cannot proceed without C/D\n"); goto cleanup; }

        id inner_model_C = nil, inner_model_D = nil;
        @try { inner_model_C = [mdl_C valueForKey:@"model"]; } @catch (...) {}
        @try { inner_model_D = [mdl_D valueForKey:@"model"]; } @catch (...) {}

        id prog_A = nil;
        id prog_B = nil;
        @try { prog_A = [inner_model_C valueForKey:@"program"]; } @catch (...) {}
        @try { prog_B = [inner_model_D valueForKey:@"program"]; } @catch (...) {}

        printf("  prog_C: %s  (class: %s)\n",
               prog_A ? "OK" : "nil",
               prog_A ? class_getName([prog_A class]) : "?");
        printf("  prog_D: %s  (class: %s)\n",
               prog_B ? "OK" : "nil",
               prog_B ? class_getName([prog_B class]) : "?");

        if (prog_A) {
            uint64_t ph_a = ((uint64_t(*)(id,SEL))objc_msgSend)(prog_A, sel_registerName("programHandle"));
            uint64_t ibh_a = ((uint64_t(*)(id,SEL))objc_msgSend)(prog_A, sel_registerName("intermediateBufferHandle"));
            int8_t  qd_a  = ((int8_t(*)(id,SEL))objc_msgSend)(prog_A, sel_registerName("queueDepth"));
            printf("  prog_A: programHandle=%llu  intermediateBufferHandle=%llu  queueDepth=%d\n",
                   (unsigned long long)ph_a, (unsigned long long)ibh_a, (int)qd_a);
        }

        /* ── Test 1: processRequest: (sanity check, same as evaluateWithQoS:) */
        if (prog_A && inner_model_C) {
            printf("\n  [processRequest:] sanity check\n");

            /* Fill input */
            IOSurfaceLock(surf_in, 0, NULL);
            float *inp = (float *)IOSurfaceGetBaseAddress(surf_in);
            for (int i = 0; i < CH * SP; i++) inp[i] = 1.0f;
            IOSurfaceUnlock(surf_in, 0, NULL);

            id sio_r_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
            id sio_r_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id req_r = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                @[sio_r_in], @[@0], @[sio_r_out], @[@0], nil, nil, (NSUInteger)0);

            uint64_t sid_a = ((uint64_t(*)(id,SEL))objc_msgSend)(inner_model_C, sel_registerName("string_id"));
            uint32_t ret_val = 0;
            NSError *pre = nil;
            @try {
                BOOL prok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,uint32_t*,NSError**))objc_msgSend)(
                    prog_A,
                    sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:"),
                    req_r,
                    inner_model_C,
                    (unsigned int)21,   /* qos */
                    (uint64_t)0,        /* qIndex */
                    (uint64_t)sid_a,    /* modelStringID */
                    @{},
                    &ret_val,
                    &pre);
                printf("  processRequest: %s  retVal=%u  err=%s\n",
                       prok ? "OK" : "FAIL",
                       (unsigned)ret_val,
                       pre ? [[pre description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  processRequest EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        /* ── mapIOSurfacesWithRequest:cacheInference: — does it set intermediateBufferHandle? */
        {
            printf("\n  [mapIOSurfacesWithRequest:cacheInference:YES]\n");
            id sio_map_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
            id sio_map_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id req_map = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                @[sio_map_in], @[@0], @[sio_map_out], @[@0], nil, nil, (NSUInteger)0);
            NSError *me = nil;
            @try {
                BOOL mok = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                    mdl_C,
                    sel_registerName("mapIOSurfacesWithRequest:cacheInference:error:"),
                    req_map, YES, &me);
                printf("  mapIOSurfacesWithRequest: %s  err=%s\n",
                       mok ? "OK" : "FAIL",
                       me ? [[me description] UTF8String] : "nil");
                if (mok) {
                    uint64_t ibh_after = ((uint64_t(*)(id,SEL))objc_msgSend)(
                        prog_A, sel_registerName("intermediateBufferHandle"));
                    printf("  intermediateBufferHandle after map: %llu\n",
                           (unsigned long long)ibh_after);
                }
            } @catch (NSException *ex) {
                printf("  mapIOSurfacesWithRequest EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        /* ── Benchmark: processRequest: vs evaluateWithQoS: overhead ────────
         * processRequest: bypasses _ANEInMemoryModel dispatch overhead.
         * If it's faster, we can use it as the hot eval path.
         */
        if (prog_A && inner_model_C) {
            printf("\n  [Benchmark] processRequest: vs evaluateWithQoS: (10 warm evals)\n");
            id sio_bm_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
            id sio_bm_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id req_bm = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                @[sio_bm_in], @[@0], @[sio_bm_out], @[@0], nil, nil, (NSUInteger)0);

            uint64_t sid_c = ((uint64_t(*)(id,SEL))objc_msgSend)(inner_model_C, sel_registerName("string_id"));

            /* Warm up */
            for (int i = 0; i < 3; i++) {
                uint32_t rv = 0; NSError *we = nil;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,uint32_t*,NSError**))objc_msgSend)(
                    prog_A, sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:"),
                    req_bm, inner_model_C, (unsigned int)21, (uint64_t)0, (uint64_t)sid_c, @{}, &rv, &we);
            }

            /* processRequest: timing */
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < 10; i++) {
                uint32_t rv = 0; NSError *pe = nil;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,uint32_t*,NSError**))objc_msgSend)(
                    prog_A, sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:"),
                    req_bm, inner_model_C, (unsigned int)21, (uint64_t)0, (uint64_t)sid_c, @{}, &rv, &pe);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double pr_ms = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec)) / 10.0 / 1e6;

            /* evaluateWithQoS: timing */
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < 10; i++) {
                id sio_e_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
                id sio_e_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
                id req_e = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                    cls_AR,
                    sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                    @[sio_e_in], @[@0], @[sio_e_out], @[@0], nil, nil, (NSUInteger)0);
                NSError *ee = nil;
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl_C, sel_registerName("evaluateWithQoS:options:request:error:"), 21, @{}, req_e, &ee);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ev_ms = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec)) / 10.0 / 1e6;

            printf("  processRequest:   %.3f ms/eval\n", pr_ms);
            printf("  evaluateWithQoS:  %.3f ms/eval\n", ev_ms);
            printf("  Overhead ratio:   %.2fx\n", ev_ms / pr_ms);
        }

        /* ── Test 2: processInputBuffers: + processOutputSet: (chaining pair) */
        if (prog_A && inner_model_C) {
            printf("\n  [processInputBuffers: + processOutputSet:] chaining primitives\n");

            /* _ANEInputBuffersReady:
             * +inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:
             *   procedureIndex     = uint32_t = 0
             *   inputBufferInfoIndex = NSArray (symbol indices? [0])
             *   inputFreeValue       = NSArray (free value per buffer? [@0])
             *   executionDelay       = uint64_t = 0
             */
            Class cls_IBR = NSClassFromString(@"_ANEInputBuffersReady");
            id ibr = nil;
            if (cls_IBR) {
                @try {
                    ibr = ((id(*)(Class,SEL,uint32_t,id,id,uint64_t))objc_msgSend)(
                        cls_IBR,
                        sel_registerName("inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:"),
                        (uint32_t)0,   /* procedureIndex */
                        @[@0],         /* inputBufferInfoIndex */
                        @[@0],         /* inputFreeValue */
                        (uint64_t)0);  /* executionDelay */
                    printf("  _ANEInputBuffersReady: %s\n",
                           ibr ? [[ibr description] UTF8String] : "nil");
                    if (ibr) {
                        BOOL ibr_valid = ((BOOL(*)(id,SEL))objc_msgSend)(ibr, sel_registerName("validate"));
                        printf("  _ANEInputBuffersReady.validate: %s\n", ibr_valid ? "OK" : "FAIL");
                    }
                } @catch (NSException *ex) {
                    printf("  _ANEInputBuffersReady EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            if (ibr) {
                NSError *pib_e = nil;
                @try {
                    BOOL pib_ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        prog_A,
                        sel_registerName("processInputBuffers:model:options:error:"),
                        ibr, inner_model_C, @{}, &pib_e);
                    printf("  processInputBuffers: %s  err=%s\n",
                           pib_ok ? "OK" : "FAIL",
                           pib_e ? [[pib_e description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  processInputBuffers EXCEPTION: %s — %s\n",
                           [[ex name] UTF8String], [[ex reason] UTF8String]);
                }
            }

            /* _ANEOutputSetEnqueue for processOutputSet:
             * processOutputSet: calls -procedureIndex on its first arg →
             * must use _ANEOutputSetEnqueue (has procedureIndex), NOT
             * _ANEIOSurfaceOutputSets (lacks it).
             *
             * +outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:
             */
            Class cls_OSE = NSClassFromString(@"_ANEOutputSetEnqueue");
            id ose = nil;
            if (cls_OSE) {
                @try {
                    ose = ((id(*)(Class,SEL,uint32_t,uint32_t,uint64_t,BOOL,BOOL))objc_msgSend)(
                        cls_OSE,
                        sel_registerName("outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:"),
                        (uint32_t)0,   /* procedureIndex */
                        (uint32_t)0,   /* setIndex */
                        (uint64_t)0,   /* signalValue */
                        NO,            /* signalNotRequired */
                        NO);           /* isOpenLoop */
                    printf("  _ANEOutputSetEnqueue: %s\n",
                           ose ? [[ose description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  _ANEOutputSetEnqueue EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            if (ose) {
                NSError *pos_e = nil;
                @try {
                    BOOL pos_ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        prog_A,
                        sel_registerName("processOutputSet:model:options:error:"),
                        ose, inner_model_C, @{}, &pos_e);
                    printf("  processOutputSet (OSE): %s  err=%s\n",
                           pos_ok ? "OK" : "FAIL",
                           pos_e ? [[pos_e description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  processOutputSet EXCEPTION: %s — %s\n",
                           [[ex name] UTF8String], [[ex reason] UTF8String]);
                }

                /* Also try prepareChainingWithModel: with _ANEOutputSetEnqueue in outputSets */
                if (client && buf_in) {
                    printf("\n  [chain v3] outputSets = _ANEOutputSetEnqueue (not _ANEIOSurfaceOutputSets)\n");
                    id chain3 = nil;
                    @try {
                        chain3 = [cls_Chain alloc];
                        chain3 = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                            chain3,
                            sel_registerName("initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"),
                            @[buf_in],
                            @[ose],         /* _ANEOutputSetEnqueue instead of _ANEIOSurfaceOutputSets */
                            lb_in_syms,
                            lb_out_syms,
                            @0, nil, nil, @0, @0);
                        BOOL v3ok = ((BOOL(*)(id,SEL))objc_msgSend)(chain3, sel_registerName("validate"));
                        printf("  chain v3 validate: %s\n", v3ok ? "OK" : "FAIL");

                        if (v3ok) {
                            NSError *ce3 = nil;
                            BOOL cok3 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
                                inner_model_C, @{}, chain3, (unsigned int)21, &ce3);
                            printf("  prepareChainingWithModel (v3): %s  err=%s\n",
                                   cok3 ? "OK" : "FAIL",
                                   ce3 ? [[ce3 description] UTF8String] : "nil");
                            if (cok3) printf("  *** prepareChainingWithModel SUCCEEDED! ***\n");
                        }
                    } @catch (NSException *ex) {
                        printf("  chain v3 EXCEPTION: %s — %s\n",
                               [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }
                }
            }
        }

        if (client && inner_model_A && live_chain2) {
            printf("\n  [prepareChainingWithModel:] full chain (live statsRef) — original attempt\n");
            NSError *ce = nil;
            @try {
                BOOL cok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client,
                    sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
                    inner_model_A, @{}, live_chain2, (unsigned int)21, &ce);
                printf("  prepareChainingWithModel: %s  err=%s\n",
                       cok ? "OK" : "FAIL",
                       ce ? [[ce description] UTF8String] : "nil");

                if (cok) {
                    printf("\n  [buffersReadyWithModel:]\n");
                    NSError *be = nil;
                    @try {
                        BOOL bok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            sel_registerName("buffersReadyWithModel:inputBuffers:options:qos:error:"),
                            inner_model_A, @[buf_in], @{}, (unsigned int)21, &be);
                        printf("  buffersReadyWithModel: %s  err=%s\n",
                               bok ? "OK" : "FAIL",
                               be ? [[be description] UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("  buffersReadyWithModel EXCEPTION: %s — %s\n",
                               [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }

                    printf("\n  [enqueueSetsWithModel:]\n");
                    NSError *ee = nil;
                    @try {
                        BOOL eok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            sel_registerName("enqueueSetsWithModel:outputSet:options:qos:error:"),
                            inner_model_A, output_set2, @{}, (unsigned int)21, &ee);
                        printf("  enqueueSetsWithModel: %s  err=%s\n",
                               eok ? "OK" : "FAIL",
                               ee ? [[ee description] UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("  enqueueSetsWithModel EXCEPTION: %s — %s\n",
                               [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }
                }
            } @catch (NSException *ex) {
                printf("  prepareChainingWithModel EXCEPTION: %s — %s\n",
                       [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        /* ── Part 5: fallback — evaluate A→B sequentially via two requests ── */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 5 — Sequential A→B via separate requests (baseline)\n");
        printf("═══════════════════════════════════════════════════════\n");

        {
            /* Fill input */
            IOSurfaceLock(surf_in, 0, NULL);
            float *inp = (float *)IOSurfaceGetBaseAddress(surf_in);
            for (int i = 0; i < CH * SP; i++) inp[i] = (float)(i % CH) * 0.01f + 1.0f;
            IOSurfaceUnlock(surf_in, 0, NULL);

            /* Request A: in → mid */
            id a_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
            id a_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id req_A = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                @[a_in], @[@0], @[a_out], @[@0], nil, nil, (NSUInteger)0);

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl_A, sel_registerName("evaluateWithQoS:options:request:error:"), 21, @{}, req_A, &e);
            printf("  A (in→mid): %s\n", ok ? "OK" : (e ? [[e description] UTF8String] : "FAIL"));

            /* Request B: mid → out */
            id b_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id b_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), surf_out);
            id req_B = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                @[b_in], @[@0], @[b_out], @[@0], nil, nil, (NSUInteger)0);

            e = nil;
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl_B, sel_registerName("evaluateWithQoS:options:request:error:"), 21, @{}, req_B, &e);
            printf("  B (mid→out): %s\n", ok ? "OK" : (e ? [[e description] UTF8String] : "FAIL"));

            if (ok) {
                IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
                float *out = (float *)IOSurfaceGetBaseAddress(surf_out);
                printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]  (expect ≈1.0 identity)\n",
                       out[0], out[1], out[2], out[3]);
                IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);
            }
        }

        /* ═══════════════════════════════════════════════════════════════
           PART 6 — Zero-copy surface chaining via processRequest:
           ═══════════════════════════════════════════════════════════════

           Central question: after processRequest: completes, is the output
           IOSurface's data immediately coherent for a second processRequest:
           on a different kernel?

           If yes: no _ANEChainingRequest, no intermediateBufferHandle, no
           Espresso path needed. Zero-copy A→B is just two sequential
           processRequest: calls sharing surf_mid. The CPU never touches data,
           it only passes the surface handle.

           Test matrix:
             [1] Coherence: A(surf_in→surf_mid) + B(surf_mid→surf_out),
                            verify surf_out ≈ surf_in (identity × 2, fp16 drift only)
             [2] Benchmark 50×(A+B) shared surface — no CPU touch between
             [3] Benchmark 50×(A+copy+B) — explicit IOSurface lock/memcpy between
                 calls to measure the copy overhead as a reference point
        ══════════════════════════════════════════════════════════════════ */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 6 — Zero-copy surface chaining via processRequest:\n");
        printf("═══════════════════════════════════════════════════════\n");

        if (prog_A && prog_B && inner_model_C && inner_model_D) {
            size_t p6_bytes = (size_t)(CH * SP) * sizeof(float);
            IOSurfaceRef surf_copy6 = make_surface(p6_bytes);
            void *p6_tmp = malloc(p6_bytes);

            uint64_t p6_sid_c = ((uint64_t(*)(id,SEL))objc_msgSend)(
                inner_model_C, sel_registerName("string_id"));
            uint64_t p6_sid_d = ((uint64_t(*)(id,SEL))objc_msgSend)(
                inner_model_D, sel_registerName("string_id"));

            SEL p6_pr = sel_registerName(
                "processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:");

            /* Build reusable request objects */
            id p6_sio_in   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_in);
            id p6_sio_mid  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_mid);
            id p6_sio_out  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_out);
            id p6_sio_copy = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                cls_AIO, sel_registerName("objectWithIOSurface:"), surf_copy6);

            SEL p6_build = sel_registerName(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
            id req_p6A = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, p6_build, @[p6_sio_in], @[@0], @[p6_sio_mid], @[@0], nil, nil, (NSUInteger)0);
            id req_p6B = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, p6_build, @[p6_sio_mid], @[@0], @[p6_sio_out], @[@0], nil, nil, (NSUInteger)0);
            id req_p6copy = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, p6_build, @[p6_sio_copy], @[@0], @[p6_sio_out], @[@0], nil, nil, (NSUInteger)0);

            typedef BOOL (*PR6Fn)(id, SEL, id, id, unsigned int,
                                  uint64_t, uint64_t, id, uint32_t*, NSError**);

            /* [1] Coherence check */
            {
                IOSurfaceLock(surf_in, 0, NULL);
                float *inp = (float *)IOSurfaceGetBaseAddress(surf_in);
                for (int i = 0; i < CH * SP; i++) inp[i] = (float)(i % CH) * 0.01f + 1.0f;
                IOSurfaceUnlock(surf_in, 0, NULL);

                uint32_t rv = 0; NSError *ce = nil;
                BOOL ok_A = ((PR6Fn)objc_msgSend)(prog_A, p6_pr,
                    req_p6A, inner_model_C, (unsigned int)21,
                    (uint64_t)0, p6_sid_c, @{}, &rv, &ce);
                ce = nil;
                BOOL ok_B = ((PR6Fn)objc_msgSend)(prog_B, p6_pr,
                    req_p6B, inner_model_D, (unsigned int)21,
                    (uint64_t)0, p6_sid_d, @{}, &rv, &ce);

                printf("  [1] A(in→mid)=%s  B(mid→out)=%s\n",
                       ok_A ? "OK" : "FAIL", ok_B ? "OK" : "FAIL");

                if (ok_A && ok_B) {
                    IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(surf_in,  kIOSurfaceLockReadOnly, NULL);
                    float *out = (float *)IOSurfaceGetBaseAddress(surf_out);
                    float *ref = (float *)IOSurfaceGetBaseAddress(surf_in);
                    float max_diff = 0.0f;
                    for (int i = 0; i < CH * SP; i++) {
                        float d = fabsf(out[i] - ref[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    printf("      surf_out[0..3]: [%.4f, %.4f, %.4f, %.4f]\n",
                           out[0], out[1], out[2], out[3]);
                    printf("      Max diff (out vs in): %.6f  → %s\n", max_diff,
                           max_diff < 0.05f
                               ? "COHERENT (fp16 drift only)"
                               : "INCOHERENT — surface not ready or stale");
                    IOSurfaceUnlock(surf_in,  kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);
                }
            }

            /* [2] Benchmark: shared surface (zero-copy candidate) */
            {
                int N = 50;
                for (int i = 0; i < 3; i++) {  /* warm up */
                    uint32_t rv = 0; NSError *we = nil;
                    ((PR6Fn)objc_msgSend)(prog_A, p6_pr, req_p6A, inner_model_C,
                        (unsigned int)21, (uint64_t)0, p6_sid_c, @{}, &rv, &we);
                    ((PR6Fn)objc_msgSend)(prog_B, p6_pr, req_p6B, inner_model_D,
                        (unsigned int)21, (uint64_t)0, p6_sid_d, @{}, &rv, &we);
                }
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < N; i++) {
                    uint32_t rv = 0; NSError *be = nil;
                    ((PR6Fn)objc_msgSend)(prog_A, p6_pr, req_p6A, inner_model_C,
                        (unsigned int)21, (uint64_t)0, p6_sid_c, @{}, &rv, &be);
                    ((PR6Fn)objc_msgSend)(prog_B, p6_pr, req_p6B, inner_model_D,
                        (unsigned int)21, (uint64_t)0, p6_sid_d, @{}, &rv, &be);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double shared_ms = ((t1.tv_sec - t0.tv_sec)*1e9 +
                                    (t1.tv_nsec - t0.tv_nsec)) / (double)N / 1e6;

                /* [3] Benchmark: explicit CPU memcpy between A and B */
                for (int i = 0; i < 3; i++) {  /* warm up */
                    uint32_t rv = 0; NSError *we = nil;
                    ((PR6Fn)objc_msgSend)(prog_A, p6_pr, req_p6A, inner_model_C,
                        (unsigned int)21, (uint64_t)0, p6_sid_c, @{}, &rv, &we);
                    IOSurfaceLock(surf_mid, kIOSurfaceLockReadOnly, NULL);
                    memcpy(p6_tmp, IOSurfaceGetBaseAddress(surf_mid), p6_bytes);
                    IOSurfaceUnlock(surf_mid, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(surf_copy6, 0, NULL);
                    memcpy(IOSurfaceGetBaseAddress(surf_copy6), p6_tmp, p6_bytes);
                    IOSurfaceUnlock(surf_copy6, 0, NULL);
                    ((PR6Fn)objc_msgSend)(prog_B, p6_pr, req_p6copy, inner_model_D,
                        (unsigned int)21, (uint64_t)0, p6_sid_d, @{}, &rv, &we);
                }
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < N; i++) {
                    uint32_t rv = 0; NSError *ce2 = nil;
                    ((PR6Fn)objc_msgSend)(prog_A, p6_pr, req_p6A, inner_model_C,
                        (unsigned int)21, (uint64_t)0, p6_sid_c, @{}, &rv, &ce2);
                    IOSurfaceLock(surf_mid, kIOSurfaceLockReadOnly, NULL);
                    memcpy(p6_tmp, IOSurfaceGetBaseAddress(surf_mid), p6_bytes);
                    IOSurfaceUnlock(surf_mid, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(surf_copy6, 0, NULL);
                    memcpy(IOSurfaceGetBaseAddress(surf_copy6), p6_tmp, p6_bytes);
                    IOSurfaceUnlock(surf_copy6, 0, NULL);
                    ((PR6Fn)objc_msgSend)(prog_B, p6_pr, req_p6copy, inner_model_D,
                        (unsigned int)21, (uint64_t)0, p6_sid_d, @{}, &rv, &ce2);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double copy_ms = ((t1.tv_sec - t0.tv_sec)*1e9 +
                                  (t1.tv_nsec - t0.tv_nsec)) / (double)N / 1e6;

                printf("\n  [2] Shared surface A→B:  %.3f ms/pair  (%.3f ms/kernel)\n",
                       shared_ms, shared_ms / 2.0);
                printf("  [3] With CPU copy A→B:   %.3f ms/pair  (copy = %.3f ms = %.1f%% overhead)\n",
                       copy_ms, copy_ms - shared_ms,
                       (copy_ms - shared_ms) / copy_ms * 100.0);
                printf("\n  Ratio (copy/shared): %.2fx\n", copy_ms / shared_ms);
                if (copy_ms > shared_ms * 1.05) {
                    printf("  → CPU copy adds measurable overhead — shared surface IS the faster path\n");
                } else {
                    printf("  → No significant latency difference — ANE likely flushes to unified memory synchronously\n");
                }
            }

            free(p6_tmp);
            CFRelease(surf_copy6);
        } else {
            printf("  Skipped — prog_A/prog_B/inner models unavailable\n");
        }

        /* ═══════════════════════════════════════════════════════════════
           PART 7 — Scale test: 768×256 (attention-sized tensors)
           ═══════════════════════════════════════════════════════════════
           Surface size: 768 × 256 × 4 bytes = 768 KB per tensor.
           At this scale, memcpy overhead becomes significant and the
           shared-surface speedup is the number that belongs in the docs.
        ══════════════════════════════════════════════════════════════════ */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 7 — Scale test: 768ch × 256sp (768 KB tensors)\n");
        printf("═══════════════════════════════════════════════════════\n");

        {
            int LCH = 768, LSP = 256;
            printf("  Compiling large kernels (768×768 conv, sp=256) ...\n");
            id lmdl_A = compile_and_load(LCH, LSP, "large_A");
            id lmdl_B = compile_and_load(LCH, LSP, "large_B");

            if (!lmdl_A || !lmdl_B) {
                printf("  SKIP — compile failed (kernel may exceed hardware limits)\n");
                goto cleanup;
            }
            printf("  Compiled OK\n");

            id l_inner_A = nil, l_inner_B = nil, l_prog_A = nil, l_prog_B = nil;
            @try { l_inner_A = [lmdl_A valueForKey:@"model"]; } @catch (...) {}
            @try { l_inner_B = [lmdl_B valueForKey:@"model"]; } @catch (...) {}
            @try { l_prog_A  = [l_inner_A valueForKey:@"program"]; } @catch (...) {}
            @try { l_prog_B  = [l_inner_B valueForKey:@"program"]; } @catch (...) {}

            if (!l_prog_A || !l_prog_B) {
                printf("  SKIP — could not extract _ANEProgramForEvaluation\n");
                goto cleanup;
            }

            uint64_t l_sid_A = ((uint64_t(*)(id,SEL))objc_msgSend)(
                l_inner_A, sel_registerName("string_id"));
            uint64_t l_sid_B = ((uint64_t(*)(id,SEL))objc_msgSend)(
                l_inner_B, sel_registerName("string_id"));

            size_t l_bytes = (size_t)(LCH * LSP) * sizeof(float);
            printf("  Surface size: %zu bytes (%.1f KB) per tensor\n",
                   l_bytes, l_bytes / 1024.0);

            IOSurfaceRef l_surf_in   = make_surface(l_bytes);
            IOSurfaceRef l_surf_mid  = make_surface(l_bytes);
            IOSurfaceRef l_surf_out  = make_surface(l_bytes);
            IOSurfaceRef l_surf_copy = make_surface(l_bytes);
            void *l_tmp = malloc(l_bytes);

            /* Fill input */
            IOSurfaceLock(l_surf_in, 0, NULL);
            float *linp = (float *)IOSurfaceGetBaseAddress(l_surf_in);
            for (int i = 0; i < LCH * LSP; i++) linp[i] = (float)(i % LCH) * 0.001f + 1.0f;
            IOSurfaceUnlock(l_surf_in, 0, NULL);

            SEL l_pr    = sel_registerName(
                "processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:");
            SEL l_build = sel_registerName(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");

            id l_sio_in   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), l_surf_in);
            id l_sio_mid  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), l_surf_mid);
            id l_sio_out  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), l_surf_out);
            id l_sio_copy = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(cls_AIO, sel_registerName("objectWithIOSurface:"), l_surf_copy);

            id l_req_A    = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, l_build, @[l_sio_in],   @[@0], @[l_sio_mid],  @[@0], nil, nil, (NSUInteger)0);
            id l_req_B    = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, l_build, @[l_sio_mid],  @[@0], @[l_sio_out],  @[@0], nil, nil, (NSUInteger)0);
            id l_req_copy = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
                cls_AR, l_build, @[l_sio_copy], @[@0], @[l_sio_out],  @[@0], nil, nil, (NSUInteger)0);

            typedef BOOL (*LPRFn)(id, SEL, id, id, unsigned int,
                                  uint64_t, uint64_t, id, uint32_t*, NSError**);

            /* Coherence check at large scale */
            {
                uint32_t rv = 0; NSError *ce = nil;
                BOOL ok_A = ((LPRFn)objc_msgSend)(l_prog_A, l_pr,
                    l_req_A, l_inner_A, (unsigned int)21, (uint64_t)0, l_sid_A, @{}, &rv, &ce);
                ce = nil;
                BOOL ok_B = ((LPRFn)objc_msgSend)(l_prog_B, l_pr,
                    l_req_B, l_inner_B, (unsigned int)21, (uint64_t)0, l_sid_B, @{}, &rv, &ce);
                if (ok_A && ok_B) {
                    IOSurfaceLock(l_surf_out, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(l_surf_in,  kIOSurfaceLockReadOnly, NULL);
                    float *lout = (float *)IOSurfaceGetBaseAddress(l_surf_out);
                    float *lref = (float *)IOSurfaceGetBaseAddress(l_surf_in);
                    float max_diff = 0.0f;
                    for (int i = 0; i < LCH * LSP; i++) {
                        float d = fabsf(lout[i] - lref[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    IOSurfaceUnlock(l_surf_in,  kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(l_surf_out, kIOSurfaceLockReadOnly, NULL);
                    printf("  Coherence: A=%s B=%s  max_diff=%.6f  → %s\n",
                           ok_A ? "OK" : "FAIL", ok_B ? "OK" : "FAIL", max_diff,
                           max_diff < 0.05f ? "COHERENT" : "INCOHERENT");
                }
            }

            /* Benchmark */
            {
                int N = 30;
                struct timespec t0, t1;

                /* Warm up */
                for (int i = 0; i < 3; i++) {
                    uint32_t rv = 0; NSError *we = nil;
                    ((LPRFn)objc_msgSend)(l_prog_A, l_pr, l_req_A, l_inner_A,
                        (unsigned int)21, (uint64_t)0, l_sid_A, @{}, &rv, &we);
                    ((LPRFn)objc_msgSend)(l_prog_B, l_pr, l_req_B, l_inner_B,
                        (unsigned int)21, (uint64_t)0, l_sid_B, @{}, &rv, &we);
                }

                /* Shared surface */
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < N; i++) {
                    uint32_t rv = 0; NSError *be = nil;
                    ((LPRFn)objc_msgSend)(l_prog_A, l_pr, l_req_A, l_inner_A,
                        (unsigned int)21, (uint64_t)0, l_sid_A, @{}, &rv, &be);
                    ((LPRFn)objc_msgSend)(l_prog_B, l_pr, l_req_B, l_inner_B,
                        (unsigned int)21, (uint64_t)0, l_sid_B, @{}, &rv, &be);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double l_shared_ms = ((t1.tv_sec - t0.tv_sec)*1e9 +
                                      (t1.tv_nsec - t0.tv_nsec)) / (double)N / 1e6;

                /* Warm up copy path */
                for (int i = 0; i < 3; i++) {
                    uint32_t rv = 0; NSError *we = nil;
                    ((LPRFn)objc_msgSend)(l_prog_A, l_pr, l_req_A, l_inner_A,
                        (unsigned int)21, (uint64_t)0, l_sid_A, @{}, &rv, &we);
                    IOSurfaceLock(l_surf_mid, kIOSurfaceLockReadOnly, NULL);
                    memcpy(l_tmp, IOSurfaceGetBaseAddress(l_surf_mid), l_bytes);
                    IOSurfaceUnlock(l_surf_mid, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(l_surf_copy, 0, NULL);
                    memcpy(IOSurfaceGetBaseAddress(l_surf_copy), l_tmp, l_bytes);
                    IOSurfaceUnlock(l_surf_copy, 0, NULL);
                    ((LPRFn)objc_msgSend)(l_prog_B, l_pr, l_req_copy, l_inner_B,
                        (unsigned int)21, (uint64_t)0, l_sid_B, @{}, &rv, &we);
                }

                /* With CPU copy */
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < N; i++) {
                    uint32_t rv = 0; NSError *ce2 = nil;
                    ((LPRFn)objc_msgSend)(l_prog_A, l_pr, l_req_A, l_inner_A,
                        (unsigned int)21, (uint64_t)0, l_sid_A, @{}, &rv, &ce2);
                    IOSurfaceLock(l_surf_mid, kIOSurfaceLockReadOnly, NULL);
                    memcpy(l_tmp, IOSurfaceGetBaseAddress(l_surf_mid), l_bytes);
                    IOSurfaceUnlock(l_surf_mid, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(l_surf_copy, 0, NULL);
                    memcpy(IOSurfaceGetBaseAddress(l_surf_copy), l_tmp, l_bytes);
                    IOSurfaceUnlock(l_surf_copy, 0, NULL);
                    ((LPRFn)objc_msgSend)(l_prog_B, l_pr, l_req_copy, l_inner_B,
                        (unsigned int)21, (uint64_t)0, l_sid_B, @{}, &rv, &ce2);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double l_copy_ms = ((t1.tv_sec - t0.tv_sec)*1e9 +
                                    (t1.tv_nsec - t0.tv_nsec)) / (double)N / 1e6;

                double copy_overhead_ms = l_copy_ms - l_shared_ms;
                double speedup = l_copy_ms / l_shared_ms;
                printf("\n  Shared surface A→B:  %.3f ms/pair  (%.3f ms/kernel)\n",
                       l_shared_ms, l_shared_ms / 2.0);
                printf("  With CPU copy A→B:   %.3f ms/pair  (copy overhead: %.3f ms)\n",
                       l_copy_ms, copy_overhead_ms);
                printf("\n  Speedup (shared vs copy): %.2fx\n", speedup);
                printf("  Copy overhead as %% of total: %.1f%%\n",
                       copy_overhead_ms / l_copy_ms * 100.0);
                printf("\n  → At 768×256 (768 KB), shared surface is %.1f%% faster than copy baseline\n",
                       (speedup - 1.0) * 100.0);
            }

            free(l_tmp);
            CFRelease(l_surf_in);
            CFRelease(l_surf_mid);
            CFRelease(l_surf_out);
            CFRelease(l_surf_copy);
        }

cleanup:
        printf("\nDone.\n");
        CFRelease(surf_in);
        CFRelease(surf_mid);
        CFRelease(surf_out);
        if (stats_surf_live) CFRelease(stats_surf_live);
    }
    return 0;
}
