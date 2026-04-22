/**
 * test_compiler_options_probe.mm — Targeted probe on the highest-value findings
 *
 * Evidence so far (2026-04-22):
 *
 *   ✓ XPC connection to com.apple.ANECompilerService SUCCEEDS from this process
 *   ✓ setModelURL: exists on _ANEInMemoryModel — we can redirect the model directory
 *   ✓ compilerOptionsWithOptions:isCompiledModelCached: exists — returns compiler dict
 *   ✓ kANEFSkipPreparePhaseKey=YES causes ANECCompile to fail (no ANECIR input)
 *   ✓ All kANEF* string values known
 *   ✓ kANEFModelPreCompiled descriptor crashes — wrong parameter slot
 *
 * This probe investigates:
 *
 *   Probe 1 — compilerOptionsWithOptions:isCompiledModelCached: diff
 *     Call with YES vs NO and compare the returned options dict.
 *     Any new key when YES is the "skip compile" signal we need.
 *
 *   Probe 2 — _ANEInMemoryModel ivar layout
 *     Enumerate all ivars to find the XPC connection object.
 *     If found, read its service name + get the interface protocol.
 *
 *   Probe 3 — setModelURL: injection then loadWithQoS: bypass
 *     Compile model A.  Build model B with IDENTICAL MIL (same hexID).
 *     Set B's modelURL = A's modelURL using setModelURL:.
 *     Call B.loadWithQoS: WITHOUT B.compileWithQoS:.
 *     → Tests if aned resolves compiled slot from hexID + matching modelURL.
 *
 *   Probe 4 — ANEFModelDescription dict structure from live model
 *     After compile, attempt to read the "ANEFModelDescription" dict from
 *     the model's internal representation (via any accessible getter).
 *     This dict is what memoryMapModelAtPath:isPrecompiled:modelAttributes:
 *     needs, so knowing its schema unlocks the precompiled load path.
 *
 *   Probe 5 — XPC protocol discovery via NSXPCInterface introspection
 *     Connect to ANECompilerService without specifying protocol.
 *     Use -[NSXPCInterface remoteObjectInterfaceForSelector:] on the
 *     interface returned by _ANEInMemoryModel's internal connection.
 *     Report the protocol name and all selectors.
 *
 * Tags: [pathc][xpc][options][probe]
 */

#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <chrono>
#include <string>
#include <vector>

static const char kReluMIL[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,4,1,16]> x) {\n"
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"opts_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ── Probe 1: compilerOptionsWithOptions:isCompiledModelCached: diff ───────────

TEST_CASE("Opts-1: compilerOptionsWithOptions:isCompiledModelCached: YES vs NO",
          "[pathc][options][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;

        SEL s_opts = sel_registerName("compilerOptionsWithOptions:isCompiledModelCached:");
        if (![model respondsToSelector:s_opts]) {
            WARN("compilerOptionsWithOptions:isCompiledModelCached: NOT present");
            libane_mil_release(h); return;
        }
        WARN("compilerOptionsWithOptions:isCompiledModelCached: IS present");

        typedef id (*OptsFn)(id, SEL, id, BOOL);

        // Call with cached=NO (normal compile scenario)
        id opts_no = ((OptsFn)objc_msgSend)(model, s_opts, @{}, NO);
        // Call with cached=YES (cache-hit scenario)
        id opts_yes = ((OptsFn)objc_msgSend)(model, s_opts, @{}, YES);

        if (opts_no) {
            NSString* d = [opts_no description];
            WARN("opts (cached=NO):  " << (d ? [d UTF8String] : "nil"));
        } else {
            WARN("opts (cached=NO):  nil");
        }

        if (opts_yes) {
            NSString* d = [opts_yes description];
            WARN("opts (cached=YES): " << (d ? [d UTF8String] : "nil"));
        } else {
            WARN("opts (cached=YES): nil");
        }

        // Diff the two dicts
        if ([opts_no isKindOfClass:[NSDictionary class]] &&
            [opts_yes isKindOfClass:[NSDictionary class]]) {
            NSDictionary* d_no  = (NSDictionary*)opts_no;
            NSDictionary* d_yes = (NSDictionary*)opts_yes;

            NSMutableSet* keys_no  = [NSMutableSet setWithArray:[d_no  allKeys]];
            NSMutableSet* keys_yes = [NSMutableSet setWithArray:[d_yes allKeys]];

            NSMutableSet* only_in_yes = [keys_yes mutableCopy];
            [only_in_yes minusSet:keys_no];
            NSMutableSet* only_in_no = [keys_no mutableCopy];
            [only_in_no minusSet:keys_yes];

            if ([only_in_yes count]) {
                WARN("Keys ONLY in cached=YES dict (these enable the cached path):");
                for (id k in only_in_yes)
                    WARN("  " << [[k description] UTF8String]
                         << " = " << [[d_yes[k] description] UTF8String]);
            }
            if ([only_in_no count]) {
                WARN("Keys ONLY in cached=NO dict:");
                for (id k in only_in_no)
                    WARN("  " << [[k description] UTF8String]
                         << " = " << [[d_no[k] description] UTF8String]);
            }
            if (![only_in_yes count] && ![only_in_no count]) {
                WARN("Dicts have identical key sets — comparing values:");
                for (id k in [d_no allKeys]) {
                    id v_no  = d_no[k];
                    id v_yes = d_yes[k];
                    if (![v_no isEqual:v_yes]) {
                        WARN("  " << [[k description] UTF8String]
                             << ":  NO=" << [[v_no description] UTF8String]
                             << "  YES=" << [[v_yes description] UTF8String]);
                    }
                }
            }
        }

        libane_mil_release(h);
    }
}

// ── Probe 2: _ANEInMemoryModel ivar layout ────────────────────────────────────

TEST_CASE("Opts-2: _ANEInMemoryModel ivar layout and XPC connection discovery",
          "[pathc][options][probe]") {
    @autoreleasepool {
        libane_available();

        Class cls = NSClassFromString(@"_ANEInMemoryModel");
        REQUIRE(cls != nil);

        // Walk the class hierarchy listing all ivars
        Class c = cls;
        while (c && c != [NSObject class]) {
            unsigned int count = 0;
            Ivar* ivars = class_copyIvarList(c, &count);
            if (count) {
                WARN("Ivars of " << class_getName(c) << ":");
                for (unsigned int i = 0; i < count; ++i) {
                    const char* name = ivar_getName(ivars[i]);
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    ptrdiff_t   off  = ivar_getOffset(ivars[i]);
                    WARN("  [" << off << "] " << (name ? name : "?")
                         << " : " << (type ? type : "?"));
                }
            }
            free(ivars);
            c = class_getSuperclass(c);
        }

        // Also check _ANEModel (inner model wrapper)
        Class cls_inner = NSClassFromString(@"_ANEModel");
        if (cls_inner) {
            unsigned int count = 0;
            Ivar* ivars = class_copyIvarList(cls_inner, &count);
            WARN("Ivars of _ANEModel (" << count << "):");
            for (unsigned int i = 0; i < count; ++i) {
                const char* name = ivar_getName(ivars[i]);
                const char* type = ivar_getTypeEncoding(ivars[i]);
                ptrdiff_t   off  = ivar_getOffset(ivars[i]);
                WARN("  [" << off << "] " << (name ? name : "?")
                     << " : " << (type ? type : "?"));
            }
            free(ivars);
        }
    }
}

// ── Probe 3: setModelURL: injection → loadWithQoS: without compile ────────────

TEST_CASE("Opts-3: setModelURL: injection then loadWithQoS: on fresh model",
          "[pathc][options][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Compile model A (standard path)
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile A failed"); if (h) libane_mil_release(h); return;
        }

        id model_a = (id)h->prog->objc_model;
        SEL s_url      = sel_registerName("modelURL");
        SEL s_setURL   = sel_registerName("setModelURL:");
        SEL s_hexID    = sel_registerName("hexStringIdentifier");
        SEL s_handle   = sel_registerName("programHandle");
        SEL s_compile  = sel_registerName("compileWithQoS:options:error:");
        SEL s_load     = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload   = sel_registerName("unloadWithQoS:error:");
        SEL s_cme      = sel_registerName("compiledModelExists");

        NSURL* url_a = ((NSURL*(*)(id,SEL))objc_msgSend)(model_a, s_url);
        NSString* hex_a = ((NSString*(*)(id,SEL))objc_msgSend)(model_a, s_hexID);
        INFO("Model A URL:   " << (url_a ? [[url_a absoluteString] UTF8String] : "nil"));
        INFO("Model A hexID: " << (hex_a ? [hex_a UTF8String] : "nil"));

        bool cme_a = [model_a respondsToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model_a, s_cme) : false;
        INFO("Model A compiledModelExists: " << cme_a);

        // Create model B with identical MIL (same hexID)
        Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_inMem = NSClassFromString(@"_ANEInMemoryModel");
        SEL s_milDesc = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem   = sel_registerName("inMemoryModelWithDescriptor:");

        NSData* mil = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
        id desc_b  = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_desc, s_milDesc, mil, @{}, nil);
        id model_b = ((id(*)(Class,SEL,id))objc_msgSend)(cls_inMem, s_inMem, desc_b);
        if (!model_b) { WARN("model B nil"); libane_mil_release(h); return; }

        NSString* hex_b = ((NSString*(*)(id,SEL))objc_msgSend)(model_b, s_hexID);
        INFO("Model B hexID: " << (hex_b ? [hex_b UTF8String] : "nil"));
        INFO("hexIDs match: " << (hex_a && hex_b && [hex_a isEqualToString:hex_b]));

        // Inject model A's URL into model B via setModelURL:
        if (url_a && [model_b respondsToSelector:s_setURL]) {
            ((void(*)(id,SEL,NSURL*))objc_msgSend)(model_b, s_setURL, url_a);
            NSURL* url_b_after = ((NSURL*(*)(id,SEL))objc_msgSend)(model_b, s_url);
            INFO("Model B URL after inject: " << (url_b_after
                 ? [[url_b_after absoluteString] UTF8String] : "nil"));
        } else {
            WARN("setModelURL: not available or no URL from A");
        }

        // Check compiledModelExists on B after URL injection
        bool cme_b = [cls_inMem instancesRespondToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model_b, s_cme) : false;
        INFO("Model B compiledModelExists after URL inject: " << cme_b);
        if (cme_b) WARN("compiledModelExists=YES after URL inject — aned sees existing compile!");

        // Attempt loadWithQoS: WITHOUT compile
        NSError* lerr = nil;
        auto t0 = ms_now();
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model_b, s_load, 33u, @{}, &lerr);
        double elapsed = ms_now() - t0;
        INFO("loadWithQoS: (no compile, URL injected): ok=" << (int)loaded
             << " elapsed=" << elapsed << "ms"
             << " err=" << (lerr ? [[lerr localizedDescription] UTF8String] : "none"));

        if (loaded && elapsed < 500.0) {
            WARN("CACHE HIT via setModelURL: inject! " << elapsed
                 << "ms — Path C reconnect confirmed!");
        } else if (loaded) {
            WARN("Loaded but slow (" << elapsed << "ms) — may have recompiled");
        } else {
            WARN("load FAILED after URL inject: "
                 << (lerr ? [[lerr localizedDescription] UTF8String] : "no error"));

            // Fall back: compile B (should be fast if hexID is cached)
            NSError* cerr = nil;
            auto tc0 = ms_now();
            BOOL compiled = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                model_b, s_compile, 33u, @{}, &cerr);
            double ctime = ms_now() - tc0;
            INFO("Fallback compile on B: ok=" << (int)compiled
                 << " elapsed=" << ctime << "ms");
            if (compiled && ctime < 500.0)
                WARN("FAST compile on B (" << ctime
                     << "ms) — aned returned cached slot for same hexID");
            if (compiled) {
                NSError* l2 = nil;
                BOOL l2ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                    model_b, s_load, 33u, @{}, &l2);
                INFO("load after compile on B: ok=" << (int)l2ok);
                if (l2ok) {
                    NSError* ue = nil;
                    ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
                        model_b, s_unload, 33u, &ue);
                }
            }
        }

        if (loaded) {
            NSError* ue = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
                model_b, s_unload, 33u, &ue);
        }

        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

// ── Probe 4: ANEFModelDescription dict from live model ─────────────────────────

TEST_CASE("Opts-4: ANEFModelDescription dict from live compiled model",
          "[pathc][options][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;

        // Try to read the model description dict via any available method
        const char* desc_key = "ANEFModelDescription";

        // Look for a method called "modelDescription", "modelAttributesDictionary",
        // "attributes", "modelDict", etc.
        static const char* candidates[] = {
            "modelDescription", "modelAttributesDictionary", "modelAttributes",
            "attributes", "modelDict", "modelDictionary", "attributesDictionary",
            "description", "dictionaryRepresentation", "modelInfo",
            "compiledModelDescription", "modelCompiledDescription",
            nullptr
        };

        Class cls = [model class];
        for (int i = 0; candidates[i]; ++i) {
            SEL s = sel_registerName(candidates[i]);
            if (![model respondsToSelector:s]) continue;
            Method m = class_getInstanceMethod(cls, s);
            if (!m) continue;
            const char* types = method_getTypeEncoding(m);
            if (!types || (types[0] != '@')) continue;
            if (strchr(candidates[i], ':')) continue;  // skip methods with args

            @try {
                id result = ((id(*)(id,SEL))objc_msgSend)(model, s);
                if (result && [result isKindOfClass:[NSDictionary class]]) {
                    NSDictionary* dict = (NSDictionary*)result;
                    WARN("  [model " << candidates[i] << "] → NSDictionary("
                         << [dict count] << " keys):");
                    for (id k in dict) {
                        NSString* vdesc = [dict[k] description];
                        std::string vs = vdesc ? [vdesc UTF8String] : "nil";
                        if (vs.size() > 100) vs = vs.substr(0, 100) + "...";
                        WARN("    " << [[k description] UTF8String] << " = " << vs);
                    }
                    // Check if it contains ANEFModelDescription or Procedures
                    for (NSString* key : @[@"ANEFModelDescription",
                                           @"ANEFModelProcedures",
                                           @"ANEFModelProcedureID",
                                           @"kANEFModelType"]) {
                        if (dict[key])
                            WARN("    *** CONTAINS " << [key UTF8String] << " ***");
                    }
                }
            } @catch (...) {}
        }

        // Also try looking at the inner _ANEModel (which may hold the attributes)
        SEL s_model = sel_registerName("model");
        id inner = nil;
        if ([model respondsToSelector:s_model])
            inner = ((id(*)(id,SEL))objc_msgSend)(model, s_model);
        if (inner && inner != model) {
            WARN("Inner _ANEModel found, scanning its methods:");
            Class inner_cls = [inner class];
            unsigned int count = 0;
            Method* methods = class_copyMethodList(inner_cls, &count);
            for (unsigned int i = 0; i < count; ++i) {
                std::string name = sel_getName(method_getName(methods[i]));
                // Look for dict/attribute/description methods
                if (name.find("dict") != std::string::npos ||
                    name.find("Dict") != std::string::npos ||
                    name.find("attr") != std::string::npos ||
                    name.find("Attr") != std::string::npos ||
                    name.find("desc") != std::string::npos ||
                    name.find("Desc") != std::string::npos ||
                    name.find("info") != std::string::npos ||
                    name.find("Info") != std::string::npos) {
                    WARN("  " << name);
                }
            }
            free(methods);
        }

        libane_mil_release(h);
    }
}

// ── Probe 5: All _ANEInMemoryModel methods (complete dump) ────────────────────

TEST_CASE("Opts-5: complete _ANEInMemoryModel method list",
          "[pathc][options][probe]") {
    @autoreleasepool {
        libane_available();

        Class cls = NSClassFromString(@"_ANEInMemoryModel");
        REQUIRE(cls != nil);

        WARN("--- _ANEInMemoryModel class methods ---");
        unsigned int cm = 0;
        Method* cmethods = class_copyMethodList(object_getClass(cls), &cm);
        for (unsigned int i = 0; i < cm; ++i)
            WARN("  +" << sel_getName(method_getName(cmethods[i])));
        free(cmethods);

        WARN("--- _ANEInMemoryModel instance methods (" << 0 << " listed below) ---");
        unsigned int im = 0;
        Method* imethods = class_copyMethodList(cls, &im);
        WARN("Total instance methods: " << im);
        for (unsigned int i = 0; i < im; ++i)
            WARN("  -" << sel_getName(method_getName(imethods[i])));
        free(imethods);
    }
}

// ── Probe 6: inject kANEFInMemoryModelIsCachedKey into compileWithQoS:options: ─
//
// Hypothesis: passing the two keys revealed by Opts-1 directly into the
// compileWithQoS:options: dict signals aned to skip ANECompilerService and
// return the cached compiled slot immediately — a faster warm path than
// setModelURL:+loadWithQoS: because it skips the compile XPC round-trip too.
//
// Tests three sub-scenarios:
//   6a: keys injected WHILE slot is alive (should be fast — slot already present)
//   6b: keys injected AFTER hard-unload (slot purged — do keys revive it?)
//   6c: timing comparison — with keys vs without keys while slot is alive

TEST_CASE("Opts-6: kANEFInMemoryModelIsCachedKey + kANEFIsInMemoryModelTypeKey "
          "injected into compileWithQoS:options:",
          "[pathc][options][probe][cached-key]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Resolve the two new key strings discovered in Opts-1
        void* fw = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_NOLOAD);

        NSString* cached_key    = nil; // kANEFInMemoryModelIsCachedKey
        NSString* hex_type_key  = nil; // kANEFIsInMemoryModelTypeKey

        if (fw) {
            NSString** p1 = (NSString**)dlsym(fw, "kANEFInMemoryModelIsCachedKey");
            NSString** p2 = (NSString**)dlsym(fw, "kANEFIsInMemoryModelTypeKey");
            if (p1) cached_key   = *p1;
            if (p2) hex_type_key = *p2;
        }

        if (!cached_key || !hex_type_key) {
            WARN("kANEFInMemoryModelIsCachedKey or kANEFIsInMemoryModelTypeKey not found "
                 "— trying string literals from Opts-1");
            // Fall back to the string values confirmed by Opts-1
            cached_key   = @"kANEFInMemoryModelIsCachedKey";
            hex_type_key = @"kANEFIsInMemoryModelTypeKey";
        }

        WARN("cached_key:   " << [cached_key UTF8String]);
        WARN("hex_type_key: " << [hex_type_key UTF8String]);

        SEL s_hexID   = sel_registerName("hexStringIdentifier");
        SEL s_compile = sel_registerName("compileWithQoS:options:error:");
        SEL s_load    = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload  = sel_registerName("unloadWithQoS:error:");
        SEL s_cme     = sel_registerName("compiledModelExists");
        SEL s_milDesc = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem   = sel_registerName("inMemoryModelWithDescriptor:");

        Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_inMem = NSClassFromString(@"_ANEInMemoryModel");

        NSData* mil = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];

        auto make_model = [&]() -> id {
            id desc = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                cls_desc, s_milDesc, mil, @{}, nil);
            return ((id(*)(Class,SEL,id))objc_msgSend)(cls_inMem, s_inMem, desc);
        };

        // ── 6a: inject keys while slot is alive ──────────────────────────────
        // Use libane_mil_compile() so the temp dir is set up correctly, then
        // test a fresh raw model with the cached keys injected into compile.
        WARN("\n--- 6a: inject keys while slot ALIVE ---");
        {
            auto* h_a = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            if (!h_a || !h_a->prog || !h_a->prog->objc_model) {
                WARN("6a cold compile failed — skipping");
                if (h_a) libane_mil_release(h_a);
                goto probe6_done;
            }

            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                (id)h_a->prog->objc_model, s_hexID);
            WARN("6a cold compile ok, hexID=" << [hex_id UTF8String]);

            NSDictionary* warm_opts = @{cached_key: @YES, hex_type_key: hex_id};

            // Use ane_compile() (which sets up temp dir) — but pass the extra keys.
            // We do this by calling the runtime directly after descriptor setup.
            // Fresh model via libane_mil_compile skips compile if compiledModelExists=YES;
            // here we want to test what the keys do to compileWithQoS: explicitly.
            // To do that properly we call libane's ane_compile path — but that always
            // passes @{} as options.  Instead, time a second libane_mil_compile
            // (slot alive) vs a third with explicit key-injected compileWithQoS:.

            // Second compile via libane (compiledModelExists path, no keys)
            auto t_no0 = ms_now();
            auto* h_no = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            double t_no = ms_now() - t_no0;
            WARN("6a compile via libane (no keys, slot alive): " << t_no << "ms");

            // Third compile: create model manually, write temp dir, inject keys
            id desc_c = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                cls_desc, s_milDesc,
                [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)], @{}, nil);
            id model_c = ((id(*)(Class,SEL,id))objc_msgSend)(cls_inMem, s_inMem, desc_c);
            if (model_c) {
                // Write temp dir so aned can find the model files
                NSString* tmpdir = [[NSTemporaryDirectory()
                    stringByAppendingPathComponent:hex_id]
                    stringByAppendingString:@""];
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:tmpdir
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
                [mil_data writeToFile:[tmpdir stringByAppendingPathComponent:@"model.mil"]
                           atomically:YES];

                NSError* ey = nil;
                auto t_yes0 = ms_now();
                BOOL ok_yes = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                    model_c, s_compile, 21u, warm_opts, &ey);
                double t_yes = ms_now() - t_yes0;

                if (ok_yes) {
                    WARN("6a compile WITH keys (slot alive): " << t_yes << "ms"
                         << (t_yes < 2.0  ? " ← INSTANT — keys bypass ANECompilerService!"
                            : t_yes < 10.0 ? " ← fast (Layer 3 cache hit)"
                            : t_yes < 80.0 ? " ← Layer 2 disk cache"
                                           : " ← cold path (keys had no effect)"));
                    if (t_no > 0 && t_yes < t_no * 0.5)
                        WARN("6a keys are " << (int)(t_no / t_yes) << "x faster than no-keys");
                    NSError* eu = nil;
                    ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
                        model_c, s_unload, 21u, &eu);
                } else {
                    WARN("6a compile WITH keys FAILED: "
                         << (ey ? [[ey localizedDescription] UTF8String] : "unknown"));
                }
            }

            if (h_no) libane_mil_release(h_no);
            libane_mil_release(h_a);
        }

        // ── 6b: inject keys AFTER hard-unload (slot purged) ──────────────────
        WARN("\n--- 6b: inject keys after hard-unload (slot PURGED) ---");
        {
            auto* h_b = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            if (!h_b || !h_b->prog) {
                WARN("6b compile failed — skipping"); if (h_b) libane_mil_release(h_b);
                goto probe6_done;
            }
            NSString* hex_id_raw = ((NSString*(*)(id,SEL))objc_msgSend)(
                (id)h_b->prog->objc_model, s_hexID);
            NSString* hex_id_b = [NSString stringWithUTF8String:[hex_id_raw UTF8String]];
            std::string tmpdir_b = std::string([NSTemporaryDirectory() UTF8String])
                                 + [hex_id_b UTF8String];
            NSDictionary* warm_opts_b = @{cached_key: @YES, hex_type_key: hex_id_b};

            libane_mil_release(h_b);  // hard-unload: purges slot + deletes temp dir

            // Recreate temp dir (aned may need the files)
            NSString* tmpdir_ns = [NSString stringWithUTF8String:tmpdir_b.c_str()];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:tmpdir_ns
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
            [mil_data writeToFile:[tmpdir_ns stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES];

            id desc_b = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                cls_desc, s_milDesc, mil_data, @{}, nil);
            id model_b2 = ((id(*)(Class,SEL,id))objc_msgSend)(cls_inMem, s_inMem, desc_b);
            if (!model_b2) { WARN("6b model creation failed"); goto probe6_done; }

            NSError* e3 = nil;
            auto t2 = ms_now();
            BOOL ok3 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                model_b2, s_compile, 21u, warm_opts_b, &e3);
            double purged_ms = ms_now() - t2;

            if (ok3) {
                WARN("6b compile WITH keys (slot PURGED): " << purged_ms << "ms"
                     << (purged_ms < 5.0   ? " ← INSTANT — keys revive purged slot!"
                        : purged_ms < 80.0 ? " ← Layer 2 disk cache (keys had no special effect)"
                                           : " ← cold recompile"));
                NSError* eu = nil;
                ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model_b2, s_unload, 21u, &eu);
            } else {
                WARN("6b compile WITH keys FAILED after purge: "
                     << (e3 ? [[e3 localizedDescription] UTF8String] : "unknown")
                     << " — keys appear to require live slot (expected result)");
            }
        }

        // ── 6c: direct timing: compileWithQoS:@{} vs compileWithQoS:warm_opts ─
        WARN("\n--- 6c: timing comparison keys vs no-keys (slot ALIVE) ---");
        {
            // Establish a live slot
            auto* h_c = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            if (!h_c || !h_c->prog) {
                WARN("6c compile failed — skipping"); if (h_c) libane_mil_release(h_c);
                goto probe6_done;
            }
            NSString* hex_id_c = ((NSString*(*)(id,SEL))objc_msgSend)(
                (id)h_c->prog->objc_model, s_hexID);
            NSDictionary* warm_opts_c = @{cached_key: @YES, hex_type_key: hex_id_c};
            NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
            NSString* tmpdir_c = [NSTemporaryDirectory()
                stringByAppendingPathComponent:hex_id_c];

            auto write_tmpdir = [&](NSString* dir) {
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:dir withIntermediateDirectories:YES
                    attributes:nil error:nil];
                [mil_data writeToFile:[dir stringByAppendingPathComponent:@"model.mil"]
                           atomically:YES];
            };
            write_tmpdir(tmpdir_c);

            auto make_raw_model = [&]() -> id {
                id d = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                    cls_desc, s_milDesc, mil_data, @{}, nil);
                return ((id(*)(Class,SEL,id))objc_msgSend)(cls_inMem, s_inMem, d);
            };

            // Without keys
            id m_no = make_raw_model();
            NSError* en = nil;
            auto t_no0 = ms_now();
            ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                m_no, s_compile, 21u, @{}, &en);
            double t_no = ms_now() - t_no0;

            // With keys
            id m_yes = make_raw_model();
            NSError* ey = nil;
            auto t_yes0 = ms_now();
            ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                m_yes, s_compile, 21u, warm_opts_c, &ey);
            double t_yes = ms_now() - t_yes0;

            WARN("6c compile NO keys:  " << t_no  << "ms");
            WARN("6c compile YES keys: " << t_yes << "ms");
            if (t_yes < t_no * 0.5)
                WARN("6c keys are " << (int)(t_no / t_yes) << "x faster than no-keys");
            else
                WARN("6c no significant speedup from keys alone");

            NSError* eu = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(m_no,  s_unload, 21u, &eu);
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(m_yes, s_unload, 21u, &eu);
            libane_mil_release(h_c);
        }

        probe6_done:;
    }
}

#else
TEST_CASE("Compiler options probe: Apple-only", "[pathc][options]") { WARN("Apple-only"); }
#endif
