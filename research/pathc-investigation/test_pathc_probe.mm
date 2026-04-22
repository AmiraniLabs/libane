/**
 * test_pathc_probe.mm — Probe for a true Path C (direct HWX loading)
 *
 * Investigates two questions inside the libane test binary (which already
 * carries the entitlements needed to initialise the ANE framework):
 *
 *   A) Does _ANEInMemoryModelDescriptor.modelWithNetworkDescription:...
 *      accept pre-compiled HWX bytes and bypass compilation?
 *
 *   B) Does _ANEClient.purgeCompiledModelMatchingHash: exist and explicitly
 *      free a compile slot (better than directory deletion alone)?
 *
 * Tags: [pathc][tier3]
 */

#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"   // libane_mil_program_s → prog
#include "../src/runtime/ane_runtime.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>

// program(1.3) format — same as MilBuilder::header() + func main<ios18>
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
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"probe_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

// Compile a relu, capture its .hwx bytes, release the program.
static std::vector<uint8_t> capture_hwx() {
    libane_set_log_level(LIBANE_LOG_SILENT);
    auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
    if (!h || !h->prog) return {};
    std::string mdir = h->prog->model_dir;
    std::vector<uint8_t> result;
    if (!mdir.empty()) {
        @autoreleasepool {
            NSString* dir = [NSString stringWithUTF8String:mdir.c_str()];
            NSDirectoryEnumerator* en =
                [[NSFileManager defaultManager] enumeratorAtPath:dir];
            for (NSString* f in en) {
                if ([f hasSuffix:@".hwx"]) {
                    NSData* d = [NSData dataWithContentsOfFile:
                                 [dir stringByAppendingPathComponent:f]];
                    if (d) result.assign((uint8_t*)[d bytes],
                                         (uint8_t*)[d bytes] + [d length]);
                    break;
                }
            }
        }
    }
    libane_mil_release(h);
    return result;
}

// ── Probe A: factory method availability and isMILModel flag ─────────────────

TEST_CASE("PathC A: _ANEInMemoryModelDescriptor factory methods",
          "[pathc][tier3]") {
    @autoreleasepool {
        libane_available(); // trigger framework initialization
        Class cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        if (!cls) { WARN("_ANEInMemoryModelDescriptor not in runtime after init"); return; }

        SEL s_net = sel_registerName("modelWithNetworkDescription:weights:optionsPlist:");
        SEL s_mil = sel_registerName("modelWithMILText:weights:optionsPlist:");
        SEL s_mil_flag = sel_registerName("isMILModel");
        SEL s_hex = sel_registerName("hexStringIdentifier");

        bool has_net = [cls respondsToSelector:s_net];
        bool has_mil = [cls respondsToSelector:s_mil];
        CHECK(has_mil);
        if (has_net) WARN("+ modelWithNetworkDescription present — investigating Path C");
        else         WARN("+ modelWithNetworkDescription NOT present — no alternate descriptor path");

        if (!has_net || !has_mil) return;

        NSData* data = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
        typedef id (*DF)(Class,SEL,NSData*,NSDictionary*,id);

        id d_net = ((DF)objc_msgSend)(cls, s_net, data, @{}, nil);
        id d_mil = ((DF)objc_msgSend)(cls, s_mil, data, @{}, nil);
        REQUIRE(d_net != nil);
        REQUIRE(d_mil != nil);

        bool net_isMIL = ((BOOL(*)(id,SEL))objc_msgSend)(d_net, s_mil_flag);
        bool mil_isMIL = ((BOOL(*)(id,SEL))objc_msgSend)(d_mil, s_mil_flag);
        NSString* hex_net = ((NSString*(*)(id,SEL))objc_msgSend)(d_net, s_hex);
        NSString* hex_mil = ((NSString*(*)(id,SEL))objc_msgSend)(d_mil, s_hex);

        INFO("milText descriptor:            isMILModel=" << mil_isMIL
             << "  hex=" << (hex_mil ? [hex_mil UTF8String] : "nil"));
        INFO("networkDescription descriptor: isMILModel=" << net_isMIL
             << "  hex=" << (hex_net ? [hex_net UTF8String] : "nil"));

        CHECK(mil_isMIL == true);
        if (net_isMIL) WARN("networkDescription isMILModel=YES — treated same as MIL, not a compiled path");
        else           WARN("networkDescription isMILModel=NO  — potential pre-compiled binary path!");
        bool same_hex = hex_net && hex_mil && [hex_net isEqualToString:hex_mil];
        if (same_hex) WARN("Both factories produce same hex (content hash) — compiler will deduplicate");
        else          WARN("Different hex IDs — networkDescription uses a distinct hash");
    }
}

// ── Probe B: pass real HWX bytes as networkDescription ───────────────────────

TEST_CASE("PathC B: networkDescription with real HWX bytes skips compilation",
          "[pathc][tier3]") {
    @autoreleasepool {
        libane_available();
        auto hwx = capture_hwx();
        if (hwx.empty()) {
            WARN("HWX capture failed (slots exhausted or compile error) — skipping");
            return;
        }
        INFO("Captured HWX: " << hwx.size() << " bytes");

        Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_model = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_desc || !cls_model) { WARN("Classes not found"); return; }

        SEL s_net     = sel_registerName("modelWithNetworkDescription:weights:optionsPlist:");
        SEL s_inmem   = sel_registerName("inMemoryModelWithDescriptor:");
        SEL s_compile = sel_registerName("compileWithQoS:options:error:");
        SEL s_isMIL   = sel_registerName("isMILModel");
        SEL s_hex     = sel_registerName("hexStringIdentifier");
        SEL s_cme     = sel_registerName("compiledModelExists");

        if (![cls_desc respondsToSelector:s_net]) {
            WARN("networkDescription factory absent — skipping");
            return;
        }

        NSData* hwx_data = [NSData dataWithBytes:hwx.data() length:hwx.size()];
        typedef id (*DF)(Class,SEL,NSData*,NSDictionary*,id);
        id desc = ((DF)objc_msgSend)(cls_desc, s_net, hwx_data, @{}, nil);
        if (!desc) { WARN("descriptor returned nil for HWX data"); return; }

        bool is_mil = ((BOOL(*)(id,SEL))objc_msgSend)(desc, s_isMIL);
        NSString* hex = ((NSString*(*)(id,SEL))objc_msgSend)(desc, s_hex);
        INFO("HWX descriptor: isMILModel=" << is_mil
             << "  hex=" << (hex ? [hex UTF8String] : "nil"));

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(cls_model, s_inmem, desc);
        if (!model) { WARN("inMemoryModelWithDescriptor: nil for HWX descriptor"); return; }

        bool cme_before = [cls_model instancesRespondToSelector:s_cme]
            ? (bool)((BOOL(*)(id,SEL))objc_msgSend)(model, s_cme) : false;
        INFO("compiledModelExists before compileWithQoS: = " << cme_before);

        NSError* err = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model, s_compile, 33u, @{}, &err);
        INFO("compileWithQoS: ok=" << (int)ok
             << "  error=" << (err ? [[err localizedDescription] UTF8String] : "none"));

        if (ok) WARN("compileWithQoS: SUCCEEDED with HWX descriptor — Path C is viable!");
        else    WARN("compileWithQoS: FAILED with HWX descriptor — cannot bypass MIL compilation");

        // Clean up
        if (ok) {
            SEL s_unload = sel_registerName("unloadWithQoS:error:");
            NSError* uerr = nil;
            ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model, s_unload, 33u, &uerr);
        }
    }
}

// ── Probe C: purgeCompiledModelMatchingHash as explicit slot release ──────────

TEST_CASE("PathC C: _ANEClient purgeCompiledModelMatchingHash frees slot",
          "[pathc][tier3]") {
    @autoreleasepool {
        libane_available();
        Class cls_client = NSClassFromString(@"_ANEClient");
        if (!cls_client) { WARN("_ANEClient not found after init"); return; }

        SEL s_purgeHash = sel_registerName("purgeCompiledModelMatchingHash:");
        SEL s_shared    = sel_registerName("sharedConnection");
        SEL s_exists    = sel_registerName("compiledModelExistsMatchingHash:");

        bool has_purge  = [cls_client instancesRespondToSelector:s_purgeHash];
        bool has_exists = [cls_client instancesRespondToSelector:s_exists];
        INFO("purgeCompiledModelMatchingHash: " << (has_purge  ? "YES" : "NO"));
        INFO("compiledModelExistsMatchingHash:" << (has_exists ? "YES" : "NO"));

        CHECK(has_purge);

        // Compile a sentinel model to get a real hex ID to purge
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog) {
            WARN("Compile failed (slots exhausted) — skipping live purge test");
            return;
        }

        id objc_model = (id)h->prog->objc_model;
        NSString* hex_id = nil;
        if (objc_model) {
            SEL s_hex = sel_registerName("hexStringIdentifier");
            hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(objc_model, s_hex);
        }
        INFO("Model hex ID: " << (hex_id ? [hex_id UTF8String] : "nil"));

        if (!hex_id) { libane_mil_release(h); return; }

        // Unload from SRAM
        SEL s_unload = sel_registerName("unloadWithQoS:error:");
        NSError* uerr = nil;
        ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
            objc_model, s_unload, 33u, &uerr);

        // Get the shared ANEClient and call purge
        typedef id (*SharedFn)(Class, SEL);
        id client = ((SharedFn)objc_msgSend)(cls_client, s_shared);
        if (!client) { WARN("_ANEClient.sharedConnection returned nil"); libane_mil_release(h); return; }

        // Check exists before purge
        bool exists_before = has_exists
            ? (bool)((BOOL(*)(id,SEL,NSString*))objc_msgSend)(client, s_exists, hex_id)
            : true;
        INFO("compiledModelExists before purge: " << exists_before);

        NSError* perr = nil;
        BOOL purged = ((BOOL(*)(id,SEL,NSString*,NSError**))objc_msgSend)(
            client, s_purgeHash, hex_id, &perr);
        INFO("purgeCompiledModelMatchingHash: ok=" << (int)purged
             << "  error=" << (perr ? [[perr localizedDescription] UTF8String] : "none"));

        if (has_exists) {
            bool exists_after = (bool)((BOOL(*)(id,SEL,NSString*))objc_msgSend)(
                client, s_exists, hex_id);
            INFO("compiledModelExists after purge:  " << exists_after);
            if (exists_before && !exists_after)
                WARN("Slot was registered before purge and gone after — purge confirmed freeing slot!");
            else
                WARN("Slot status did not change as expected");
        }

        if (purged) WARN("purgeCompiledModelMatchingHash SUCCEEDED — wire into ane_unload for true slot release");
        else        WARN("purgeCompiledModelMatchingHash FAILED — likely entitlement restriction");

        // Prevent double-unload: clear objc_model before libane_mil_release
        h->prog->objc_model = nullptr;
        libane_mil_release(h);
    }
}

#else
TEST_CASE("PathC probe: Apple-only", "[pathc]") { WARN("Apple-only"); }
#endif
