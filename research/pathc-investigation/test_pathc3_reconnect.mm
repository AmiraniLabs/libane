/**
 * test_pathc3_reconnect.mm — Path C tier-3 probe: cache reconnect + handle injection
 *
 * Tests the two most promising mechanisms to skip ANECompilerService:
 *
 * Probe 1 — Cache reconnect via hex ID
 *   Compile relu → get hexID → unload but keep compile slot (purge skipped)
 *   Create fresh _ANEInMemoryModel via initWithModelIdentifier:<hexID>
 *   Call loadWithQoS: WITHOUT calling compileWithQoS: first
 *   → If aned returns the cached compiled slot, we see ~0 ms instead of ~4200 ms
 *
 * Probe 2 — programHandle injection
 *   Compile relu → get programHandle (kernel handle)
 *   Create fresh _ANEInMemoryModel via inMemoryModelWithDescriptor:
 *   Inject handle via setProgramHandle:
 *   Call loadWithQoS: WITHOUT compileWithQoS:
 *   → If aned accepts the injected handle, load succeeds
 *
 * Probe 3 — Same hexID, same descriptor, skip compile
 *   Compile relu once → keep alive
 *   Create SECOND _ANEInMemoryModel with identical MIL content
 *   Skip compileWithQoS: on the second model, call loadWithQoS: directly
 *   → Tests if aned deduplicates by hexID on load
 *
 * Probe 4 — net.plist as networkDescription
 *   After compile, read net.plist from model_dir
 *   Pass net.plist bytes to modelWithNetworkDescription:
 *   → net.plist is the MIL text; isMILModel check confirms descriptor type
 *
 * Tags: [pathc][tier3][reconnect]
 */

#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
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
    "    func main<ios18>(tensor<fp16, [1,8,1,32]> x) {\n"
    "        tensor<fp16, [1,8,1,32]> y = relu(x=x)[name=string(\"p3relu\")];\n"
    "    } -> (y);\n"
    "}\n";

static const char kReluMIL2[] =
    "program(1.3)\n"
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}"
    "})]\n"
    "{\n"
    "    func main<ios18>(tensor<fp16, [1,8,1,32]> x) {\n"
    "        tensor<fp16, [1,8,1,32]> y = relu(x=x)[name=string(\"p3relu\")];\n"
    "    } -> (y);\n"
    "}\n";

static_assert(sizeof(kReluMIL) == sizeof(kReluMIL2),
              "kReluMIL and kReluMIL2 must be identical content");

// ── Timing helper ─────────────────────────────────────────────────────────────
static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ── ObjC helpers ─────────────────────────────────────────────────────────────
static inline id oc(void* p) { return (__bridge id)(void*)p; }

// ── Probe 1: Cache reconnect via initWithModelIdentifier: ────────────────────

TEST_CASE("PathC3-1: loadWithQoS on initWithModelIdentifier reconnects aned cache",
          "[pathc][tier3][reconnect]") {
    @autoreleasepool {
        libane_available();

        // Step A: compile a relu model (standard path)
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) {
            WARN("Initial compile failed (slots exhausted?) — skipping");
            return;
        }

        id model_a = (id)h->prog->objc_model;
        if (!model_a) { libane_mil_release(h); WARN("objc_model nil"); return; }

        SEL s_hexID  = sel_registerName("hexStringIdentifier");
        SEL s_handle = sel_registerName("programHandle");
        SEL s_unload = sel_registerName("unloadWithQoS:error:");
        SEL s_load   = sel_registerName("loadWithQoS:options:error:");
        SEL s_compile= sel_registerName("compileWithQoS:options:error:");
        SEL s_purge  = sel_registerName("purgeCompiledModel");
        SEL s_cme    = sel_registerName("compiledModelExists");

        NSString* hexID = ((NSString*(*)(id,SEL))objc_msgSend)(model_a, s_hexID);
        uint64_t  ph_a  = ((uint64_t(*)(id,SEL))objc_msgSend)(model_a, s_handle);

        INFO("Model A hex ID: " << (hexID ? [hexID UTF8String] : "nil"));
        INFO("Model A programHandle: 0x" << std::hex << ph_a);

        // Step B: unload from SRAM (but keep compile slot — don't purge)
        {
            NSError* err = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
                model_a, s_unload, 33u, &err);
            INFO("Unload A: " << (int)ok
                 << " err=" << (err ? [[err localizedDescription] UTF8String] : "none"));
        }

        // Step C: create a fresh _ANEInMemoryModel via initWithModelIdentifier:
        Class cls_ANEModel = NSClassFromString(@"_ANEModel");
        if (!cls_ANEModel) { WARN("_ANEModel class not found"); libane_mil_release(h); return; }

        SEL s_init_id = sel_registerName("initWithModelIdentifier:");
        id raw_model  = [cls_ANEModel alloc];
        id model_b    = ((id(*)(id,SEL,NSString*))objc_msgSend)(raw_model, s_init_id, hexID);

        if (!model_b) {
            WARN("initWithModelIdentifier: returned nil");
            libane_mil_release(h);
            return;
        }

        INFO("model_b class: " << [NSStringFromClass([model_b class]) UTF8String]);
        NSString* desc_b = [model_b description];
        INFO("model_b description: " << (desc_b ? [desc_b UTF8String] : "nil"));

        // Step D: wrap in _ANEInMemoryModel
        Class cls_InMem = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_InMem) { WARN("_ANEInMemoryModel class not found"); libane_mil_release(h); return; }

        // Try: create InMem model, inject model_b, then load without compile
        SEL s_setModel = sel_registerName("setModel:");
        SEL s_getModel = sel_registerName("model");

        // Create an in-memory model shell via inMemoryModelWithDescriptor:
        // We need a descriptor to create a shell. Use the minimal MIL descriptor.
        Class cls_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        SEL s_milDesc  = sel_registerName("modelWithMILText:weights:optionsPlist:");
        NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
        id desc = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_Desc, s_milDesc, mil_data, @{}, nil);
        SEL s_inMem = sel_registerName("inMemoryModelWithDescriptor:");
        id model_c  = ((id(*)(Class,SEL,id))objc_msgSend)(cls_InMem, s_inMem, desc);
        if (!model_c) { WARN("inMemoryModelWithDescriptor: nil"); libane_mil_release(h); return; }

        // Inject the initWithModelIdentifier _ANEModel via setModel:
        if ([model_c respondsToSelector:s_setModel]) {
            ((void(*)(id,SEL,id))objc_msgSend)(model_c, s_setModel, model_b);
            INFO("Injected model_b into model_c via setModel:");
        } else {
            WARN("setModel: not available on _ANEInMemoryModel");
        }

        // Check compiledModelExists on model_c
        bool cme = [cls_InMem instancesRespondToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model_c, s_cme) : false;
        INFO("compiledModelExists on model_c: " << cme);

        // Step E: attempt loadWithQoS: WITHOUT compileWithQoS:
        NSError* lerr = nil;
        auto t0 = ms_now();
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model_c, s_load, 33u, @{}, &lerr);
        auto elapsed = ms_now() - t0;
        INFO("loadWithQoS: (no prior compile): ok=" << (int)loaded
             << " elapsed=" << elapsed << "ms"
             << " err=" << (lerr ? [[lerr localizedDescription] UTF8String] : "none"));

        if (loaded) {
            uint64_t ph_c = ((uint64_t(*)(id,SEL))objc_msgSend)(model_c, s_handle);
            INFO("model_c programHandle: 0x" << std::hex << ph_c);
            if (elapsed < 500.0) {
                WARN("CACHE HIT! loadWithQoS: completed in " << elapsed
                     << "ms without compile — Path C via reconnect WORKS!");
            } else {
                WARN("Load succeeded but slow (" << elapsed
                     << "ms) — may have triggered recompile via cache ID path");
            }
            // Unload model_c
            NSError* uerr2 = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model_c, s_unload, 33u, &uerr2);
        } else {
            WARN("loadWithQoS: FAILED without compile — cache reconnect does not bypass compile");
        }

        // Step F: now try the standard compile → load on model_c to verify it works normally
        NSError* cerr = nil;
        auto tc0 = ms_now();
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model_c, s_compile, 33u, @{}, &cerr);
        auto compile_ms = ms_now() - tc0;
        INFO("compileWithQoS: (baseline): ok=" << (int)compiled
             << " elapsed=" << compile_ms << "ms"
             << " err=" << (cerr ? [[cerr localizedDescription] UTF8String] : "none"));
        if (compiled) {
            NSError* l2err = nil;
            auto tl0 = ms_now();
            BOOL l2 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                model_c, s_load, 33u, @{}, &l2err);
            auto load2_ms = ms_now() - tl0;
            INFO("loadWithQoS: after compile: ok=" << (int)l2
                 << " elapsed=" << load2_ms << "ms");
            if (l2) {
                NSError* u2err = nil;
                ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model_c, s_unload, 33u, &u2err);
            }
        }

        // Clean up
        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

// ── Probe 2: programHandle injection ─────────────────────────────────────────

TEST_CASE("PathC3-2: setProgramHandle injection then loadWithQoS bypasses compile",
          "[pathc][tier3][reconnect]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Compile and load a model to get a live programHandle
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) { WARN("Compile failed — skipping"); return; }

        id model_src = (id)h->prog->objc_model;
        if (!model_src) { libane_mil_release(h); WARN("objc_model nil"); return; }

        SEL s_handle    = sel_registerName("programHandle");
        SEL s_setHandle = sel_registerName("setProgramHandle:");
        SEL s_hexID     = sel_registerName("hexStringIdentifier");
        SEL s_load      = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload    = sel_registerName("unloadWithQoS:error:");
        SEL s_compile   = sel_registerName("compileWithQoS:options:error:");
        SEL s_cme       = sel_registerName("compiledModelExists");

        uint64_t ph = ((uint64_t(*)(id,SEL))objc_msgSend)(model_src, s_handle);
        NSString* hexID = ((NSString*(*)(id,SEL))objc_msgSend)(model_src, s_hexID);
        INFO("Source programHandle: 0x" << std::hex << ph);
        INFO("Source hexID: " << (hexID ? [hexID UTF8String] : "nil"));

        if (ph == 0) {
            WARN("programHandle=0 — model not loaded? Check if libane_mil_compile triggers load");
            libane_mil_release(h);
            return;
        }

        // Create a fresh _ANEInMemoryModel shell
        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_InMem = NSClassFromString(@"_ANEInMemoryModel");
        SEL s_milDesc   = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem     = sel_registerName("inMemoryModelWithDescriptor:");

        NSData* mil_data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
        id desc2  = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_Desc, s_milDesc, mil_data, @{}, nil);
        id model2 = ((id(*)(Class,SEL,id))objc_msgSend)(cls_InMem, s_inMem, desc2);
        if (!model2) { WARN("model2 = nil"); libane_mil_release(h); return; }

        NSString* hexID2 = ((NSString*(*)(id,SEL))objc_msgSend)(model2, s_hexID);
        INFO("Fresh model hexID: " << (hexID2 ? [hexID2 UTF8String] : "nil"));
        bool same_hex = hexID && hexID2 && [hexID isEqualToString:hexID2];
        INFO("Same hexID as source: " << same_hex);

        // Inject the programHandle
        if ([model2 respondsToSelector:s_setHandle]) {
            ((void(*)(id,SEL,uint64_t))objc_msgSend)(model2, s_setHandle, ph);
            uint64_t ph2 = ((uint64_t(*)(id,SEL))objc_msgSend)(model2, s_handle);
            INFO("After setProgramHandle: model2.programHandle = 0x" << std::hex << ph2);
        } else {
            WARN("setProgramHandle: not available");
            libane_mil_release(h);
            return;
        }

        // compiledModelExists?
        bool cme2 = [cls_InMem instancesRespondToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model2, s_cme) : false;
        INFO("compiledModelExists on model2 (post-handle-inject): " << cme2);

        // Attempt loadWithQoS: WITHOUT compileWithQoS:
        NSError* lerr = nil;
        auto t0 = ms_now();
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model2, s_load, 33u, @{}, &lerr);
        auto elapsed = ms_now() - t0;
        INFO("loadWithQoS: after handle inject: ok=" << (int)loaded
             << " elapsed=" << elapsed << "ms"
             << " err=" << (lerr ? [[lerr localizedDescription] UTF8String] : "none"));

        if (loaded) {
            uint64_t ph2_after = ((uint64_t(*)(id,SEL))objc_msgSend)(model2, s_handle);
            INFO("model2.programHandle after load: 0x" << std::hex << ph2_after);
            if (elapsed < 500.0) {
                WARN("FAST LOAD via handle inject! " << elapsed
                     << "ms — kernel accepted injected handle, no ANECompilerService call");
            } else {
                WARN("Load ok but slow (" << elapsed << "ms)");
            }
            NSError* uerr = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model2, s_unload, 33u, &uerr);
        } else {
            WARN("loadWithQoS: FAILED with injected handle: "
                 << (lerr ? [[lerr localizedDescription] UTF8String] : "no error"));
        }

        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

// ── Probe 3: Same hexID, skip compile on second model ────────────────────────

TEST_CASE("PathC3-3: second _ANEInMemoryModel with same hexID skips compile",
          "[pathc][tier3][reconnect]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // First model: compile + keep alive
        auto* h1 = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h1 || !h1->prog) { WARN("First compile failed"); return; }

        id model1 = (id)h1->prog->objc_model;
        if (!model1) { libane_mil_release(h1); WARN("objc_model nil"); return; }

        SEL s_hexID  = sel_registerName("hexStringIdentifier");
        SEL s_compile= sel_registerName("compileWithQoS:options:error:");
        SEL s_load   = sel_registerName("loadWithQoS:options:error:");
        SEL s_unload = sel_registerName("unloadWithQoS:error:");
        SEL s_cme    = sel_registerName("compiledModelExists");

        NSString* hexID1 = ((NSString*(*)(id,SEL))objc_msgSend)(model1, s_hexID);
        INFO("Model1 hexID: " << (hexID1 ? [hexID1 UTF8String] : "nil"));

        // Second model from identical content — should have same hexID
        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_InMem = NSClassFromString(@"_ANEInMemoryModel");
        SEL s_milDesc   = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_inMem     = sel_registerName("inMemoryModelWithDescriptor:");

        // Use kReluMIL2 (identical bytes, different C string ptr)
        NSData* mil2 = [NSData dataWithBytes:kReluMIL2 length:strlen(kReluMIL2)];
        id desc2     = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
            cls_Desc, s_milDesc, mil2, @{}, nil);
        id model2    = ((id(*)(Class,SEL,id))objc_msgSend)(cls_InMem, s_inMem, desc2);
        if (!model2) { WARN("model2 nil"); libane_mil_release(h1); return; }

        NSString* hexID2 = ((NSString*(*)(id,SEL))objc_msgSend)(model2, s_hexID);
        INFO("Model2 hexID: " << (hexID2 ? [hexID2 UTF8String] : "nil"));

        bool same = hexID1 && hexID2 && [hexID1 isEqualToString:hexID2];
        if (!same) {
            WARN("hexIDs differ — identical MIL text produces different hash. Cannot test dedup");
            libane_mil_release(h1);
            return;
        }
        INFO("hexIDs MATCH — aned uses content hash");

        // compiledModelExists on model2 (before compile)?
        bool cme_before = [cls_InMem instancesRespondToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model2, s_cme) : false;
        INFO("compiledModelExists on model2 (before compile): " << cme_before);

        if (cme_before) {
            WARN("compiledModelExists=YES before any compile on model2 — aned dedup confirmed!");
        }

        // Attempt load WITHOUT compile
        NSError* lerr = nil;
        auto t0 = ms_now();
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model2, s_load, 33u, @{}, &lerr);
        auto elapsed = ms_now() - t0;
        INFO("loadWithQoS: (no compile, model1 alive): ok=" << (int)loaded
             << " elapsed=" << elapsed << "ms"
             << " err=" << (lerr ? [[lerr localizedDescription] UTF8String] : "none"));

        if (loaded) {
            if (elapsed < 500.0) {
                WARN("CACHE HIT CONFIRMED: " << elapsed
                     << "ms load without compile — aned deduplicates by hexID!");
            } else {
                WARN("Load succeeded in " << elapsed
                     << "ms — may have triggered internal compile via hexID lookup");
            }
            NSError* uerr = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model2, s_unload, 33u, &uerr);
        } else {
            WARN("load FAILED without compile even with same hexID: "
                 << (lerr ? [[lerr localizedDescription] UTF8String] : "no error"));

            // Now try compile on model2 to check timing (should be fast if cached)
            NSError* cerr = nil;
            auto tc0 = ms_now();
            BOOL compiled2 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                model2, s_compile, 33u, @{}, &cerr);
            auto compile_ms = ms_now() - tc0;
            INFO("compileWithQoS: on model2 (while model1 alive): ok=" << (int)compiled2
                 << " elapsed=" << compile_ms << "ms"
                 << " err=" << (cerr ? [[cerr localizedDescription] UTF8String] : "none"));
            if (compiled2 && compile_ms < 500.0) {
                WARN("FAST COMPILE via dedup! " << compile_ms
                     << "ms — aned returned cached compile for same hexID");
            } else if (compiled2) {
                WARN("Compile succeeded but slow (" << compile_ms
                     << "ms) — no dedup, full ANECompilerService call");
            }
            if (compiled2) {
                NSError* l2err = nil;
                BOOL l2 = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                    model2, s_load, 33u, @{}, &l2err);
                INFO("load after compile on model2: ok=" << (int)l2);
                if (l2) {
                    NSError* u2 = nil;
                    ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model2, s_unload, 33u, &u2);
                }
            }
        }

        h1->prog->objc_model = nullptr;
        libane_mil_release(h1);
    }
}

// ── Probe 4: net.plist as networkDescription bytes ────────────────────────────

TEST_CASE("PathC3-4: net.plist passed to modelWithNetworkDescription is MIL text",
          "[pathc][tier3][reconnect]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) { WARN("Compile failed"); return; }

        std::string mdir = h->prog->model_dir;
        if (mdir.empty()) { libane_mil_release(h); WARN("no model_dir"); return; }

        // List all files in model_dir
        @autoreleasepool {
            NSString* dir = [NSString stringWithUTF8String:mdir.c_str()];
            NSArray* items = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:dir error:nil];
            INFO("Files in model_dir after compile:");
            for (NSString* item in items) {
                NSString* fullPath = [dir stringByAppendingPathComponent:item];
                NSDictionary* attrs = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:fullPath error:nil];
                NSNumber* sz = attrs[NSFileSize];
                INFO("  " << [item UTF8String] << "  (" << [sz longLongValue] << " bytes)");

                // Also list subdirectories
                BOOL isDir = NO;
                [[NSFileManager defaultManager] fileExistsAtPath:fullPath isDirectory:&isDir];
                if (isDir) {
                    NSArray* sub = [[NSFileManager defaultManager]
                        contentsOfDirectoryAtPath:fullPath error:nil];
                    for (NSString* s in sub) {
                        NSString* sp = [fullPath stringByAppendingPathComponent:s];
                        NSDictionary* sa = [[NSFileManager defaultManager]
                            attributesOfItemAtPath:sp error:nil];
                        INFO("    " << [s UTF8String] << "  (" << [sa[NSFileSize] longLongValue] << " bytes)");
                    }
                }
            }
        }

        // Try to find net.plist
        @autoreleasepool {
            NSString* dir = [NSString stringWithUTF8String:mdir.c_str()];
            NSString* netPlist = [dir stringByAppendingPathComponent:@"net.plist"];
            NSData* plistData = [NSData dataWithContentsOfFile:netPlist];

            if (!plistData) {
                WARN("net.plist not found at: " << mdir << "/net.plist");
                // Try subdirs
                NSDirectoryEnumerator* en = [[NSFileManager defaultManager]
                    enumeratorAtPath:dir];
                for (NSString* f in en) {
                    if ([f hasSuffix:@"net.plist"] || [f hasSuffix:@".plist"]) {
                        NSString* fp = [dir stringByAppendingPathComponent:f];
                        plistData = [NSData dataWithContentsOfFile:fp];
                        if (plistData) {
                            INFO("Found plist at: " << [f UTF8String]
                                 << " (" << [plistData length] << " bytes)");
                            break;
                        }
                    }
                }
            } else {
                INFO("net.plist found: " << [plistData length] << " bytes");
                // Print first 200 bytes as string to see what it contains
                NSString* preview = [[NSString alloc]
                    initWithData:[plistData subdataWithRange:NSMakeRange(0, MIN(200UL, [plistData length]))]
                    encoding:NSUTF8StringEncoding];
                if (preview) INFO("net.plist preview: " << [preview UTF8String]);
            }

            if (plistData) {
                // Pass to modelWithNetworkDescription:
                Class cls_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
                SEL s_net  = sel_registerName("modelWithNetworkDescription:weights:optionsPlist:");
                SEL s_mil  = sel_registerName("modelWithMILText:weights:optionsPlist:");
                SEL s_isMIL= sel_registerName("isMILModel");
                SEL s_hexID= sel_registerName("hexStringIdentifier");

                if (![cls_Desc respondsToSelector:s_net]) {
                    WARN("modelWithNetworkDescription: absent");
                } else {
                    id d_net = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                        cls_Desc, s_net, plistData, @{}, nil);
                    INFO("networkDescription descriptor from net.plist: " << (d_net ? "non-nil" : "nil"));
                    if (d_net) {
                        bool isMIL = ((BOOL(*)(id,SEL))objc_msgSend)(d_net, s_isMIL);
                        NSString* hex = ((NSString*(*)(id,SEL))objc_msgSend)(d_net, s_hexID);
                        INFO("net.plist descriptor: isMILModel=" << isMIL
                             << " hex=" << (hex ? [hex UTF8String] : "nil"));
                        NSString* hexID_orig = ((NSString*(*)(id,SEL))objc_msgSend)(
                            (id)h->prog->objc_model,
                            sel_registerName("hexStringIdentifier"));
                        bool same = hex && hexID_orig && [hex isEqualToString:hexID_orig];
                        INFO("Same hex as compiled model: " << same);
                        if (same && !isMIL) {
                            WARN("net.plist produces SAME HEX and isMIL=NO — this IS the pre-compiled descriptor format!");
                        } else if (same && isMIL) {
                            WARN("net.plist same hex but isMIL=YES — it's the MIL text again, not a compiled format");
                        }
                    }
                }
            }
        }

        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

// ── Probe 5: Scan ALL writable dirs for .hwx after compile ───────────────────

TEST_CASE("PathC3-5: filesystem scan for aned-written compiled binary",
          "[pathc][tier3][reconnect]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Snapshot of .hwx / .hwbin / .anec files BEFORE compile
        NSArray* search_dirs = @[
            @"/tmp",
            @"/var/folders",
            [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"],
            @"/private/var/folders",
        ];

        auto scan_for_compiled = [](NSString* root, NSMutableSet* found) {
            NSArray* exts = @[@".hwx", @".hwbin", @".anec", @".mlmodelc"];
            NSDirectoryEnumerator* en = [[NSFileManager defaultManager]
                enumeratorAtPath:root];
            [en skipDescendants]; // don't recurse too deep
            // Actually use a simpler approach: just look for ANE-related files
            NSDirectoryEnumerator* en2 = [[NSFileManager defaultManager]
                enumeratorAtPath:root];
            for (NSString* f in en2) {
                for (NSString* ext in exts) {
                    if ([f hasSuffix:ext]) {
                        [found addObject:[root stringByAppendingPathComponent:f]];
                    }
                }
            }
        };

        NSMutableSet* before = [NSMutableSet set];
        for (NSString* d in search_dirs) {
            @try { scan_for_compiled(d, before); } @catch (...) {}
        }
        INFO("Compiled-looking files before: " << [before count]);

        // Compile
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) { WARN("Compile failed"); return; }

        NSMutableSet* after = [NSMutableSet set];
        for (NSString* d in search_dirs) {
            @try { scan_for_compiled(d, after); } @catch (...) {}
        }
        INFO("Compiled-looking files after: " << [after count]);

        // Find new files
        NSMutableSet* newFiles = [after mutableCopy];
        [newFiles minusSet:before];

        if ([newFiles count] == 0) {
            WARN("NO new compiled binary files appeared after compile — binary stays in aned kernel memory only");
        } else {
            INFO("NEW FILES after compile:");
            for (NSString* f in newFiles) {
                NSDictionary* attrs = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:f error:nil];
                INFO("  " << [f UTF8String] << " (" << [attrs[NSFileSize] longLongValue] << " bytes)");
                WARN("New compiled file found: " << [f UTF8String]);
            }
        }

        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

#else
TEST_CASE("PathC3 probe: Apple-only", "[pathc]") { WARN("Apple-only"); }
#endif
