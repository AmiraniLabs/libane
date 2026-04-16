/**
 * probe_shared_events.m
 *
 * Investigates _ANESharedEvents / _ANESharedSignalEvent / _ANESharedWaitEvent
 * with the in-memory compile path (Path A).
 *
 * FINDINGS SUMMARY:
 *
 *   _ANESharedEvents DOES NOT work on Path A (_ANEInMemoryModel).
 *   Root cause is the same "intermediateBufferHandle=0 gate" that blocks
 *   mapIOSurfacesWithRequest:, _ANEChainingRequest, and startOffset:
 *
 *   ANE's processRequest: completion block maintains a C++ event infrastructure
 *   object for sharedEvents. That object is only initialized when
 *   intermediateBufferHandle != 0 (Path B, _ANEClient.loadModel:). On Path A,
 *   the pointer is nil. The completion block does C++ vtable dispatch on nil
 *   when it processes any event — crashing at:
 *
 *       ldr x9, [x8, #0x10]!   ;; x8 = nil → EXC_BAD_ACCESS at 0x10
 *       blraa x9, x8            ;; C++ virtual call with self=nil
 *
 *   (frame: __100-[_ANEProgramForEvaluation processRequest:model:qos:qIndex:
 *            modelStringID:options:returnValue:error:]_block_invoke + 1320)
 *
 *   What "processing" means:
 *     - Signal events: always processed at completion → always crashes
 *     - Wait events whose condition is MET at completion → crashes
 *     - Wait events whose condition is NOT MET → skipped, no crash
 *     - Empty sharedEvents (@[], @[]): no processing → no crash
 *
 *   WORKING ALTERNATIVE on Path A:
 *     _ANERequest.completionHandler — pure ObjC block, no C++ required.
 *     Fires asynchronously on ANEServicesThread ~0.2ms after evaluateWithQoS:
 *     returns. The IOSurface output is coherent at the time the handler fires.
 *     Use this for post-completion notification, pipelining, and async dispatch.
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <time.h>

/* ── helpers ────────────────────────────────────────────────────────────── */

static Class cls_AIO, cls_AR, cls_Desc, cls_Model;
static Class cls_SE, cls_SSE, cls_SWE;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((CFDictionaryRef)@{
        @"IOSurfaceWidth":          @((int)bytes),
        @"IOSurfaceHeight":         @(1),
        @"IOSurfaceBytesPerElement":@(1),
        @"IOSurfacePixelFormat":    @(0)
    });
}

static NSData *make_identity_weight(int CH) {
    int ws = CH * CH * 2, total = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(total, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE;
    b[68]=1;
    *(uint32_t *)(b+72) = (uint32_t)ws;
    *(uint32_t *)(b+80) = 128;
    uint16_t *fp16 = (uint16_t *)(b+128);
    for (int i = 0; i < CH; i++) fp16[i*CH+i] = 0x3C00;
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

static NSString *make_mil(int CH, int SP) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n{\n"
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
        "    } -> (y);\n}\n",
        CH,SP, CH,SP, CH,CH,CH,CH, CH,SP, CH,SP];
}

static id compile_and_load(int CH, int SP, const char *tag) {
    NSData *mil = [make_mil(CH,SP) dataUsingEncoding:NSUTF8StringEncoding];
    NSData *w   = make_identity_weight(CH);
    NSDictionary *wdict = @{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":w}};
    id desc = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
        cls_Desc, sel_registerName("modelWithMILText:weights:optionsPlist:"), mil, wdict, nil);
    if (!desc) { printf("[%s] descriptor fail\n", tag); return nil; }
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc);
    if (!model) { printf("[%s] model fail\n", tag); return nil; }
    NSString *hid = ((NSString*(*)(id,SEL))objc_msgSend)(model, sel_registerName("hexStringIdentifier"));
    NSString *mdir = [NSTemporaryDirectory() stringByAppendingPathComponent:hid];
    NSString *wdir = [mdir stringByAppendingPathComponent:@"weights"];
    [[NSFileManager defaultManager] createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[mdir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [w   writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"] atomically:YES];
    NSError *e = nil;
    typedef BOOL (*QFn)(id,SEL,unsigned int,id,NSError**);
    BOOL ok = ((QFn)objc_msgSend)(model,sel_registerName("compileWithQoS:options:error:"),21,@{},&e);
    if (!ok){printf("[%s] compile fail: %s\n",tag,e?[[e localizedDescription]UTF8String]:"?");return nil;}
    e=nil;
    ok = ((QFn)objc_msgSend)(model,sel_registerName("loadWithQoS:options:error:"),21,@{},&e);
    if (!ok){printf("[%s] load fail: %s\n",tag,e?[[e localizedDescription]UTF8String]:"?");return nil;}
    printf("[%s] OK\n", tag);
    return model;
}

static id make_request(id in_aio, id out_aio) {
    return ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger))objc_msgSend)(
        cls_AR,
        sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:"
                         "weightsBuffer:perfStats:procedureIndex:"),
        @[in_aio], @[@0], @[out_aio], @[@0], nil, nil, (NSUInteger)0);
}

int main(void) {
    dlopen("/System/Library/PrivateFrameworks/"
           "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    @autoreleasepool {
        cls_AIO   = NSClassFromString(@"_ANEIOSurfaceObject");
        cls_AR    = NSClassFromString(@"_ANERequest");
        cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        cls_SE    = NSClassFromString(@"_ANESharedEvents");
        cls_SSE   = NSClassFromString(@"_ANESharedSignalEvent");
        cls_SWE   = NSClassFromString(@"_ANESharedWaitEvent");

        if (!cls_AIO||!cls_AR||!cls_Desc||!cls_Model||!cls_SE||!cls_SSE||!cls_SWE) {
            fprintf(stderr, "class lookup failed\n"); return 1;
        }

        int CH = 32, SP = 32;
        size_t surf_bytes = (size_t)CH * SP * sizeof(float);

        /* ═══════════════════════════════════════════════════════════════
           PART 1 — _ANESharedEvents object construction
           Confirms all classes exist and construct cleanly.
           IOSurfaceSharedEvent requires initWithOptions: (not bare init) for a
           fully initialized C++ backing, but bare init works for our purposes
           since ANE's crash is NOT caused by a broken IOSurfaceSharedEvent.
        ════════════════════════════════════════════════════════════════ */
        printf("═══════════════════════════════════════════════════════\n");
        printf("PART 1 — _ANESharedEvents construction\n");
        printf("═══════════════════════════════════════════════════════\n");

        Class cls_ISE = NSClassFromString(@"IOSurfaceSharedEvent");
        id ise = ((id(*)(id,SEL,uint64_t))objc_msgSend)(
            [cls_ISE alloc], sel_registerName("initWithOptions:"), (uint64_t)0);
        mach_port_t ep = (mach_port_t)((mach_port_t(*)(id,SEL))objc_msgSend)(
            ise, sel_registerName("eventPort"));
        printf("  IOSurfaceSharedEvent (initWithOptions:0): port=%u\n", ep);

        /* setSignaledValue: works — event is functional */
        ((void(*)(id,SEL,uint64_t))objc_msgSend)(ise, sel_registerName("setSignaledValue:"), (uint64_t)1);
        uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(ise, sel_registerName("signaledValue"));
        printf("  setSignaledValue:1 OK, signaledValue=%llu\n", (unsigned long long)sv);

        /* Reset for later use */
        ((void(*)(id,SEL,uint64_t))objc_msgSend)(ise, sel_registerName("setSignaledValue:"), (uint64_t)0);

        id sig_ev = ((id(*)(Class,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(
            cls_SSE,
            sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
            (uint64_t)1, (uint32_t)0, (int64_t)0, ise);
        printf("  _ANESharedSignalEvent: %s\n", sig_ev ? "OK" : "nil");

        id wait_ev = ((id(*)(Class,SEL,uint64_t,id))objc_msgSend)(
            cls_SWE, sel_registerName("waitEventWithValue:sharedEvent:"),
            (uint64_t)1, ise);
        printf("  _ANESharedWaitEvent: %s\n", wait_ev ? "OK" : "nil");

        id ane_events = ((id(*)(Class,SEL,id,id))objc_msgSend)(
            cls_SE, sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),
            @[sig_ev], @[wait_ev]);
        printf("  _ANESharedEvents (sig+wait): %s\n", ane_events ? "OK" : "nil");

        id mdl = compile_and_load(CH, SP, "se_model");
        if (!mdl) { fprintf(stderr, "model load failed\n"); return 1; }

        IOSurfaceRef s_in  = make_surface(surf_bytes);
        IOSurfaceRef s_out = make_surface(surf_bytes);
        id aio_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            cls_AIO, sel_registerName("objectWithIOSurface:"), s_in);
        id aio_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            cls_AIO, sel_registerName("objectWithIOSurface:"), s_out);

        /* ═══════════════════════════════════════════════════════════════
           PART 2 — _ANERequest.completionHandler: the WORKING async path
           completionHandler fires asynchronously on ANEServicesThread
           after evaluateWithQoS: returns. No C++ infrastructure required.
           This is the correct async dispatch mechanism on Path A.
        ════════════════════════════════════════════════════════════════ */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 2 — _ANERequest.completionHandler (working async path)\n");
        printf("═══════════════════════════════════════════════════════\n");

        {
            IOSurfaceLock(s_in, 0, NULL);
            float *inp = (float *)IOSurfaceGetBaseAddress(s_in);
            for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f + (float)i * 0.001f;
            IOSurfaceUnlock(s_in, 0, NULL);

            id req = make_request(aio_in, aio_out);

            dispatch_semaphore_t sema = dispatch_semaphore_create(0);
            __block double handler_time = 0;
            __block BOOL handler_called = NO;

            typedef void (^CompBlock)(void);
            CompBlock handler = ^{
                handler_called = YES;
                handler_time = now_ms();
                printf("  completionHandler fired on thread: %s\n",
                       [[NSThread currentThread] isMainThread] ? "main" : "background");
                /* Output is coherent here — safe to read/chain */
                dispatch_semaphore_signal(sema);
            };
            ((void(*)(id,SEL,CompBlock))objc_msgSend)(
                req, sel_registerName("setCompletionHandler:"), handler);

            double t0 = now_ms();
            NSError *err = nil;
            typedef BOOL (*EvalFn)(id,SEL,unsigned int,id,id,NSError**);
            BOOL ok = ((EvalFn)objc_msgSend)(
                mdl, sel_registerName("evaluateWithQoS:options:request:error:"),
                21, @{}, req, &err);
            double t1 = now_ms();
            printf("  evaluateWithQoS: ok=%d  err=%s  wall=%.3fms\n",
                   (int)ok, err ? [[err localizedDescription] UTF8String] : "nil", t1-t0);

            long r = dispatch_semaphore_wait(sema,
                dispatch_time(DISPATCH_TIME_NOW, (int64_t)(100 * NSEC_PER_MSEC)));
            if (handler_called) {
                printf("  handler fired at +%.3fms after submit  "
                       "(eval returned at +%.3fms)\n",
                       handler_time - t0, t1 - t0);
                printf("  → completionHandler fires AFTER evaluateWithQoS: returns\n");
                printf("  → Output is coherent at handler invocation\n");
                printf("  → Works on Path A (intermediateBufferHandle=0)\n");

                /* Verify output */
                IOSurfaceLock(s_out, kIOSurfaceLockReadOnly, NULL);
                float *out = (float *)IOSurfaceGetBaseAddress(s_out);
                printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]  (expect ≈ input)\n",
                       out[0], out[1], out[2], out[3]);
                IOSurfaceUnlock(s_out, kIOSurfaceLockReadOnly, NULL);
            } else {
                printf("  ERROR: handler NOT called within 100ms\n");
            }
        }

        /* ═══════════════════════════════════════════════════════════════
           PART 3 — _ANESharedEvents crash proof: signal events
           Any _ANESharedSignalEvent causes EXC_BAD_ACCESS in the
           processRequest: completion block on ANEServicesThread.
           Root cause: C++ event infrastructure nil on Path A.
           THIS TEST WILL CRASH THE PROCESS.
        ════════════════════════════════════════════════════════════════ */
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("PART 3 — _ANESharedEvents crash: signal event\n");
        printf("  (process will crash in ANEServicesThread completion callback)\n");
        printf("═══════════════════════════════════════════════════════\n");

        {
            id crash_ise = ((id(*)(id,SEL,uint64_t))objc_msgSend)(
                [cls_ISE alloc], sel_registerName("initWithOptions:"), (uint64_t)0);
            id crash_sig = ((id(*)(Class,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(
                cls_SSE,
                sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
                (uint64_t)1, (uint32_t)0, (int64_t)0, crash_ise);
            id crash_se = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                cls_SE, sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),
                @[crash_sig], @[]);

            id crash_req = ((id(*)(Class,SEL,id,id,id,id,id,id,NSUInteger,id))objc_msgSend)(
                cls_AR,
                sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:"
                                 "weightsBuffer:perfStats:procedureIndex:sharedEvents:"),
                @[aio_in], @[@0], @[aio_out], @[@0], nil, nil, (NSUInteger)0, crash_se);

            NSError *err = nil;
            typedef BOOL (*EvalFn)(id,SEL,unsigned int,id,id,NSError**);
            BOOL ok = ((EvalFn)objc_msgSend)(
                mdl, sel_registerName("evaluateWithQoS:options:request:error:"),
                21, @{}, crash_req, &err);
            printf("  evaluateWithQoS: ok=%d err=%s\n",
                   (int)ok, err ? [[err localizedDescription] UTF8String] : "nil");
            printf("  Waiting for async crash...\n"); fflush(stdout);
            usleep(100000); /* crash fires here on ANEServicesThread */
            printf("  (should not reach here)\n");
        }

        CFRelease(s_in);
        CFRelease(s_out);
        printf("\nDone.\n");
    }
    return 0;
}
