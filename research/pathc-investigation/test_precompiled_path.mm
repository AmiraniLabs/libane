/**
 * test_precompiled_path.mm — Probe for kANEFModelPreCompiledValue descriptor path
 *
 * Background: ANECompilerService exposes the following model type constants:
 *   kANEFModelCoreMLValue      — CoreML .mlmodelc
 *   kANEFModelMILValue         — MIL text
 *   kANEFModelMLIRValue        — MLIR
 *   kANEFModelANECIRValue      — ANECIR (compiled IR)
 *   kANEFModelLLIRBundleValue  — LLIR bundle (model.llir.bundle)
 *   kANEFModelPreCompiledValue — Pre-compiled binary  ← THIS IS THE TARGET
 *
 * The BEEFFACE HWX binary we have may be directly accepted by
 * _ANEInMemoryModelDescriptor when passed with model type = PreCompiled.
 *
 * Also investigating:
 *   kANEFSkipPreparePhaseKey   — may skip compile when binary already staged
 *   createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:
 *                              — loads pre-compiled binary from disk path
 *
 * Probe A: kANEFModelPreCompiledValue + HWX bytes as networkDescription
 *   Compile a relu, capture .hwx from model_dir (legacy path).
 *   Create descriptor: modelType=PreCompiled, data=hwx_bytes.
 *   Call compileWithQoS: and observe if it's fast (0-1 ms) vs slow (4200 ms).
 *
 * Probe B: kANEFSkipPreparePhaseKey in compile options
 *   Compile with kANEFSkipPreparePhaseKey=YES.
 *   Measure whether compile is faster (skip ANECIR generation).
 *
 * Probe C: createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:
 *   Find _ANECompiler class in runtime.
 *   Write HWX bytes to a temp file.
 *   Call createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:
 *   Observe if it returns a valid program.
 *
 * Probe D: model.llir.bundle as aot binary
 *   After compile, scan model_dir subdirs for a .llir.bundle directory/file.
 *   Read its contents and try passing it to the JIT loader.
 *
 * Tags: [pathc][precompiled][probe]
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
#include <vector>
#include <string>

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
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"precompiled_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Compile and capture any .hwx or .llir.bundle from model_dir
static std::vector<uint8_t> capture_binary_any_ext(libane_mil_program_s* h) {
    if (!h || !h->prog || h->prog->model_dir.empty()) return {};
    @autoreleasepool {
        NSString* dir = [NSString stringWithUTF8String:h->prog->model_dir.c_str()];
        NSArray<NSString*>* exts = @[@".hwx", @".anecir", @".anec", @".llir.bundle"];
        NSDirectoryEnumerator* en = [[NSFileManager defaultManager] enumeratorAtPath:dir];
        for (NSString* f in en) {
            for (NSString* ext in exts) {
                if ([f hasSuffix:ext]) {
                    NSString* full = [dir stringByAppendingPathComponent:f];
                    // If it's a directory (.llir.bundle), report but don't read as bytes
                    BOOL isDir = NO;
                    [[NSFileManager defaultManager] fileExistsAtPath:full isDirectory:&isDir];
                    if (isDir) {
                        WARN("  found bundle (dir): " << [full UTF8String]);
                        // Return a sentinel — 1 byte
                        return {0xFF};
                    }
                    NSData* d = [NSData dataWithContentsOfFile:full];
                    if (d && [d length] > 4) {
                        WARN("  found binary file: " << [f UTF8String]
                             << " (" << [d length] << " bytes)");
                        return std::vector<uint8_t>((uint8_t*)[d bytes],
                                                    (uint8_t*)[d bytes] + [d length]);
                    }
                }
            }
        }
    }
    return {};
}

// ── Probe A: kANEFModelPreCompiledValue descriptor ────────────────────────────

TEST_CASE("PreCompiled-A: descriptor with kANEFModelPreCompiledValue + HWX bytes",
          "[pathc][precompiled][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Step 1: standard compile to get whatever binary exists
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) { WARN("Initial compile failed"); return; }

        auto bytes = capture_binary_any_ext(h);
        if (bytes.empty()) {
            WARN("No binary artifact found in model_dir — running on macOS 26 in-memory-only mode");
        }

        // Regardless: grab the HWX from shape cache via hwx_emitter if available
        // (the shape cache may have been populated by prior test runs)
        // For now use the model_dir bytes or fall back to the MIL text

        // Step 2: look up kANEFModelPreCompiledValue
        // It's a string constant in AppleNeuralEngine framework
        // Try to resolve it via NSClassFromString + attribute lookup
        id precompiled_value = nil;
        {
            // The constant is exported as _kANEFModelPreCompiledValue
            // It's a CFStringRef / NSString* stored in the framework data section.
            // We can try to read it via dlsym.
            void* sym = dlsym(RTLD_DEFAULT, "kANEFModelPreCompiledValue");
            if (sym) {
                id* ptr = (id*)sym;
                precompiled_value = *ptr;
                WARN("kANEFModelPreCompiledValue = " << (precompiled_value
                     ? [[precompiled_value description] UTF8String] : "nil"));
            } else {
                WARN("kANEFModelPreCompiledValue not found via dlsym");
            }
        }

        id model_type_key = nil;
        {
            void* sym = dlsym(RTLD_DEFAULT, "kANEFModelTypeKey");
            if (sym) {
                id* ptr = (id*)sym;
                model_type_key = *ptr;
                WARN("kANEFModelTypeKey = " << (model_type_key
                     ? [[model_type_key description] UTF8String] : "nil"));
            }
        }

        if (!precompiled_value || !model_type_key) {
            WARN("Cannot test PreCompiled path — key resolution failed");
            libane_mil_release(h);
            return;
        }

        // Step 3: create a descriptor with PreCompiled model type
        Class cls_desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_model = NSClassFromString(@"_ANEInMemoryModel");

        SEL s_net   = sel_registerName("modelWithNetworkDescription:weights:optionsPlist:");
        SEL s_mil   = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem = sel_registerName("inMemoryModelWithDescriptor:");

        if (![cls_desc respondsToSelector:s_net]) {
            WARN("modelWithNetworkDescription: absent — cannot test PreCompiled descriptor");
            libane_mil_release(h);
            return;
        }

        // Try passing HWX bytes (if we have them) with PreCompiled type
        NSData* binary_data = bytes.empty()
            ? [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)]  // fallback: MIL text
            : [NSData dataWithBytes:bytes.data() length:bytes.size()];

        NSDictionary* opts = @{ model_type_key: precompiled_value };
        typedef id (*DF)(Class,SEL,NSData*,NSDictionary*,id);
        id desc = ((DF)objc_msgSend)(cls_desc, s_net, binary_data, @{}, opts);
        if (!desc) {
            // Try without passing opts in the 3rd arg — try as optionsPlist
            desc = ((DF)objc_msgSend)(cls_desc, s_net, binary_data, opts, nil);
        }
        INFO("PreCompiled descriptor: " << (desc ? "non-nil" : "nil"));

        if (!desc) {
            WARN("Descriptor returned nil for PreCompiled + binary data");
            // Try with MIL text as a control
            NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
            id desc_mil = ((DF)objc_msgSend)(cls_desc, s_mil, mil_data, opts, nil);
            WARN("MIL descriptor with PreCompiled opts: " << (desc_mil ? "non-nil" : "nil"));
            libane_mil_release(h);
            return;
        }

        SEL s_isMIL = sel_registerName("isMILModel");
        SEL s_hexID = sel_registerName("hexStringIdentifier");
        bool isMIL = ((BOOL(*)(id,SEL))objc_msgSend)(desc, s_isMIL);
        NSString* hex = ((NSString*(*)(id,SEL))objc_msgSend)(desc, s_hexID);
        INFO("PreCompiled desc: isMILModel=" << isMIL
             << " hex=" << (hex ? [hex UTF8String] : "nil"));

        if (isMIL)
            WARN("isMILModel=YES — treated as MIL text, no binary bypass");
        else
            WARN("isMILModel=NO — treated as pre-compiled binary, may bypass compiler!");

        // Step 4: create model and compile — time it
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(cls_model, s_inMem, desc);
        if (!model) { WARN("inMemoryModelWithDescriptor: nil"); libane_mil_release(h); return; }

        SEL s_compile = sel_registerName("compileWithQoS:options:error:");
        SEL s_unload  = sel_registerName("unloadWithQoS:error:");

        NSError* cerr = nil;
        auto t0 = ms_now();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model, s_compile, 33u, @{}, &cerr);
        double elapsed = ms_now() - t0;

        INFO("compileWithQoS: ok=" << (int)ok
             << " elapsed=" << elapsed << "ms"
             << " err=" << (cerr ? [[cerr localizedDescription] UTF8String] : "none"));

        if (ok && elapsed < 500.0)
            WARN("FAST compile (" << elapsed << "ms) — PreCompiled path bypasses ANECompilerService!");
        else if (ok)
            WARN("Compile succeeded but slow (" << elapsed << "ms) — still went through compiler");
        else
            WARN("Compile FAILED with PreCompiled descriptor");

        if (ok) {
            NSError* uerr = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model, s_unload, 33u, &uerr);
        }

        libane_mil_release(h);
    }
}

// ── Probe B: kANEFSkipPreparePhaseKey ────────────────────────────────────────

TEST_CASE("PreCompiled-B: kANEFSkipPreparePhaseKey in compile options",
          "[pathc][precompiled][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Resolve kANEFSkipPreparePhaseKey
        id skip_key = nil;
        void* sym = dlsym(RTLD_DEFAULT, "kANEFSkipPreparePhaseKey");
        if (sym) {
            skip_key = *(id*)sym;
            WARN("kANEFSkipPreparePhaseKey = " << (skip_key
                 ? [[skip_key description] UTF8String] : "nil"));
        } else {
            WARN("kANEFSkipPreparePhaseKey not found via dlsym — skipping");
            return;
        }

        Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_model = NSClassFromString(@"_ANEInMemoryModel");
        SEL s_mil    = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem  = sel_registerName("inMemoryModelWithDescriptor:");
        SEL s_compile= sel_registerName("compileWithQoS:options:error:");
        SEL s_unload = sel_registerName("unloadWithQoS:error:");

        NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];

        // Baseline: compile WITHOUT skip flag
        id desc1  = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_desc, s_mil, mil_data, @{}, nil);
        id model1 = ((id(*)(Class,SEL,id))objc_msgSend)(cls_model, s_inMem, desc1);

        NSError* e1 = nil;
        auto t1 = ms_now();
        BOOL ok1 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model1, s_compile, 33u, @{}, &e1);
        double ms1 = ms_now() - t1;
        INFO("Baseline compile: ok=" << ok1 << " elapsed=" << ms1 << "ms");
        if (ok1) {
            NSError* ue = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model1, s_unload, 33u, &ue);
        }

        // With skip flag
        NSDictionary* skip_opts = @{ skip_key: @YES };
        id desc2  = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_desc, s_mil, mil_data, @{}, nil);
        id model2 = ((id(*)(Class,SEL,id))objc_msgSend)(cls_model, s_inMem, desc2);

        NSError* e2 = nil;
        auto t2 = ms_now();
        BOOL ok2 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model2, s_compile, 33u, skip_opts, &e2);
        double ms2 = ms_now() - t2;
        INFO("SkipPrepare compile: ok=" << ok2 << " elapsed=" << ms2 << "ms"
             << " err=" << (e2 ? [[e2 localizedDescription] UTF8String] : "none"));

        if (ok2 && ms2 < 500.0)
            WARN("FAST compile with SkipPrepare! " << ms2
                 << "ms (baseline=" << ms1 << "ms)");
        else if (ok2)
            WARN("SkipPrepare: compiled but same speed (" << ms2
                 << "ms vs " << ms1 << "ms baseline)");
        else
            WARN("SkipPrepare compile FAILED — key may not apply to in-memory path");

        if (ok2) {
            NSError* ue = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model2, s_unload, 33u, &ue);
        }
    }
}

// ── Probe C: _ANECompiler.createJITNetworkFromModelAtPath:... ─────────────────

TEST_CASE("PreCompiled-C: createJITNetworkFromModelAtPath with staged binary",
          "[pathc][precompiled][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Find _ANECompiler in the runtime (it's defined in ANECompilerService.xpc,
        // but AppleNeuralEngine.framework may re-export a client proxy)
        Class cls_compiler = NSClassFromString(@"_ANECompiler");
        if (!cls_compiler) {
            WARN("_ANECompiler not in runtime — only accessible inside ANECompilerService");
            return;
        }
        WARN("_ANECompiler found in runtime");

        SEL s_createJIT = sel_registerName(
            "createJITNetworkFromModelAtPath:modelFilename:aotModelAtPath:aotModelFilename:");

        if (![cls_compiler instancesRespondToSelector:s_createJIT] &&
            ![cls_compiler respondsToSelector:s_createJIT]) {
            WARN("createJITNetworkFromModelAtPath: not available");
        } else {
            WARN("createJITNetworkFromModelAtPath: IS available on _ANECompiler");
        }

        // List all class and instance methods
        WARN("_ANECompiler class methods:");
        unsigned int cm = 0;
        Method* cmethods = class_copyMethodList(object_getClass(cls_compiler), &cm);
        for (unsigned int i = 0; i < cm; ++i)
            WARN("  +" << sel_getName(method_getName(cmethods[i])));
        free(cmethods);

        WARN("_ANECompiler instance methods:");
        unsigned int im = 0;
        Method* imethods = class_copyMethodList(cls_compiler, &im);
        for (unsigned int i = 0; i < im; ++i)
            WARN("  -" << sel_getName(method_getName(imethods[i])));
        free(imethods);
    }
}

// ── Probe D: model.llir.bundle directory structure ─────────────────────────────

TEST_CASE("PreCompiled-D: model.llir.bundle directory structure after compile",
          "[pathc][precompiled][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) { WARN("compile failed"); return; }

        std::string mdir = h->prog->model_dir;
        INFO("model_dir = " << mdir);

        // Walk the ENTIRE model_dir tree (including subdirs)
        NSString* dir = [NSString stringWithUTF8String:mdir.c_str()];
        NSDirectoryEnumerator* en = [[NSFileManager defaultManager]
            enumeratorAtPath:dir];
        bool any = false;
        for (NSString* f in en) {
            NSString* full = [dir stringByAppendingPathComponent:f];
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:full isDirectory:&isDir];
            NSDictionary* attrs = [[NSFileManager defaultManager]
                attributesOfItemAtPath:full error:nil];
            NSNumber* sz = attrs[NSFileSize];
            WARN((isDir ? "  dir:  " : "  file: ") << [f UTF8String]
                 << (isDir ? "" : std::string("  (") + std::to_string([sz longLongValue]) + " bytes)"));
            any = true;
        }
        if (!any) WARN("model_dir is empty");

        // Also check if an llir.bundle appears adjacent to model_dir
        NSString* parent = [dir stringByDeletingLastPathComponent];
        NSArray* siblings = [[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:parent error:nil];
        WARN("Siblings of model_dir in parent:");
        for (NSString* s in siblings) {
            NSString* full = [parent stringByAppendingPathComponent:s];
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:full isDirectory:&isDir];
            WARN("  " << [s UTF8String] << (isDir ? " (dir)" : ""));
        }

        libane_mil_release(h);
    }
}

// ── Probe E: all kANEF* keys accessible via dlsym ─────────────────────────────

TEST_CASE("PreCompiled-E: enumerate kANEF* key values via dlsym",
          "[pathc][precompiled][probe]") {
    @autoreleasepool {
        libane_available();

        static const char* key_names[] = {
            "kANEFModelTypeKey",
            "kANEFModelPreCompiledValue",
            "kANEFModelMILValue",
            "kANEFModelMLIRValue",
            "kANEFModelANECIRValue",
            "kANEFModelLLIRBundleValue",
            "kANEFModelCoreMLValue",
            "kANEFAOTCacheUrlIdentifierKey",
            "kANEFSkipPreparePhaseKey",
            "kANEFModelIsEncryptedKey",
            "kANEFKeepModelMemoryWiredKey",
            "kANEFIntermediateBufferHandleKey",
            "kANEFMemoryPoolIDKey",
            "kANEFModelHasCacheURLIdentifierKey",
            "kANEFModelCacheIdentifierUsingSourceURLKey",
            "kANEFModelIdentityStrKey",
            "kANEFModelProceduresArrayKey",
            "kANEFModelProcedureIDKey",
            "kANEFModelProcedureNameToIDMapKey",
            "kANEFModelInputSymbolIndexArrayKey",
            "kANEFModelOutputSymbolIndexArrayKey",
            "kANEFModelInputSymbolsArrayKey",
            "kANEFModelOutputSymbolsArrayKey",
            "kANEFModelInput16KAlignmentArrayKey",
            "kANEFModelOutput16KAlignmentArrayKey",
            "kANEFModelDescriptionKey",
            "kANEFNetPlistFilenameKey",
            "kANEFCompilerOptionsFilenameKey",
            "kANEFEnableLateLatchKey",
            "kANEFEnablePowerSavingKey",
            "kANEFDisableIOFencesUseSharedEventsKey",
            "kANEFConstantSurfaceAlignmentKey",
            "kANEFConstantSurfaceIDKey",
            "kANEFBaseModelIdentifierKey",
            "kANEFModelInstanceParameters",
            nullptr
        };

        int resolved = 0, missing = 0;
        for (int i = 0; key_names[i]; ++i) {
            void* sym = dlsym(RTLD_DEFAULT, key_names[i]);
            if (sym) {
                id val = *(id*)sym;
                std::string vstr = val ? [[val description] UTF8String] : "(null ptr)";
                if (vstr.size() > 80) vstr = vstr.substr(0, 80) + "...";
                WARN("  " << key_names[i] << " = \"" << vstr << "\"");
                ++resolved;
            } else {
                INFO("  " << key_names[i] << " = (not found)");
                ++missing;
            }
        }
        WARN("Resolved " << resolved << " / " << (resolved + missing) << " kANEF* keys");
    }
}

#else
TEST_CASE("PreCompiled path probe: Apple-only", "[pathc][precompiled]") { WARN("Apple-only"); }
#endif
