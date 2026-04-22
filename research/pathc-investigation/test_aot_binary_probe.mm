/**
 * test_aot_binary_probe.mm — Find where aned writes the AOT compiled binary on macOS 26
 *
 * Background: On macOS ≤15, compiled ANE binaries landed at
 *   /Library/Caches/com.apple.aned/…/ModelAssetsCache/…/model.hwx
 * On macOS 26, model_dir contains only model.mil + net.plist + weights/.
 * No .hwx file is written anywhere observable (confirmed test_hwx_search.mm).
 *
 * New evidence from ANECompilerService strings (2026-04-22):
 *   - "model.llir.bundle"     — default LLIR bundle filename
 *   - "model.espresso.net"    — Espresso IR source filename (legacy)
 *   - "defaultANECIRFileName" — method returning the ANECIR output filename
 *   - "modelDataVaultDirectory"      — protected directory for model binaries
 *   - "userModelDataVaultDirectory"  — user-scoped variant
 *   - "systemModelsCacheDirectory"   — system-level model cache
 *   - "kANEFAOTCacheUrlIdentifierKey"— AOT binary cache URL key
 *   - "compilerRequest.aotModelBinaryPath=%s" — confirms binary path in JIT request
 *
 * This probe investigates five questions:
 *
 *   Probe 1 — model.llir.bundle scan
 *     Search all candidate directories for .llir.bundle files before/after compile.
 *     Also searches for .anecir and .anec extensions.
 *
 *   Probe 2 — DataVault and system cache directories
 *     Look for known Apple DataVault patterns:
 *       ~/Library/Application Support/com.apple.NeuralEngine/
 *       ~/Library/Group Containers/…/Library/…
 *       /private/var/db/com.apple.NeuralEngineAssetsD/
 *       ~/Library/Caches/com.apple.aned/
 *
 *   Probe 3 — _ANEInMemoryModel method scan for binary/aot/vault keywords
 *     Dump all instance methods containing: vault, data, aot, llir, binary,
 *     anecir, cache, path, url, identifier, compiled, binary, output.
 *     Call zero-arg methods that return id/NSURL and log what they return.
 *
 *   Probe 4 — kANEFAOTCacheUrlIdentifierKey extraction
 *     After compile, read kANEFAOTCacheUrlIdentifierKey from the model's
 *     internal attribute dict (via any accessible accessor).
 *     Then call URLForModel:bundleID:aotCacheUrlIdentifier: to resolve the path.
 *
 *   Probe 5 — _ANEInMemoryModelCacheManager inspection
 *     Find sharedInstance/sharedManager selector.
 *     List all cached model identifiers and their binary paths.
 *
 * Tags: [pathc][aot][probe]
 */

#include <catch2/catch_test_macros.hpp>

#ifdef __APPLE__

#include "libane.h"
#include "../src/libane_internal.hpp"
#include "../src/runtime/ane_runtime.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>

#include <set>
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
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"aot_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

// ── helpers ──────────────────────────────────────────────────────────────────

static std::set<std::string> snapshot_dir_deep(NSString* root) {
    std::set<std::string> out;
    if (![[NSFileManager defaultManager] fileExistsAtPath:root]) return out;
    NSDirectoryEnumerator* en = [[NSFileManager defaultManager] enumeratorAtPath:root];
    for (NSString* f in en) {
        NSString* full = [root stringByAppendingPathComponent:f];
        BOOL isDir = NO;
        [[NSFileManager defaultManager] fileExistsAtPath:full isDirectory:&isDir];
        if (!isDir) out.insert([full UTF8String]);
    }
    return out;
}

static std::vector<std::string> new_files(const std::set<std::string>& before,
                                          const std::set<std::string>& after) {
    std::vector<std::string> out;
    for (auto& p : after)
        if (!before.count(p)) out.push_back(p);
    return out;
}

static bool is_interesting_for_aot(const std::string& name) {
    static const char* kw[] = {
        "vault", "aot", "llir", "anecir", "cache", "url", "identifier",
        "binary", "output", "compiled", "path", "directory", "source",
        nullptr
    };
    std::string lower = name;
    for (char& c : lower) c = (char)tolower(c);
    for (int i = 0; kw[i]; ++i)
        if (lower.find(kw[i]) != std::string::npos) return true;
    return false;
}

// ── Probe 1: model.llir.bundle / .anecir scan ─────────────────────────────────

TEST_CASE("AOT-1: scan for model.llir.bundle and .anecir after compile",
          "[pathc][aot][probe]") {
    @autoreleasepool {
        libane_available();
        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* home = NSHomeDirectory();

        NSArray<NSString*>* watched = @[
            @"/tmp",
            [NSString stringWithUTF8String: P_tmpdir ? P_tmpdir : "/tmp"],
            NSTemporaryDirectory(),
            [home stringByAppendingPathComponent:@"Library/Caches/com.apple.aned"],
            [home stringByAppendingPathComponent:@"Library/Caches/com.apple.ANECompilerService"],
            [home stringByAppendingPathComponent:@"Library/Caches"],
            [home stringByAppendingPathComponent:@"Library/Application Support/com.apple.NeuralEngine"],
            [home stringByAppendingPathComponent:@"Library/Application Support/com.apple.aned"],
            @"/private/var/db/com.apple.NeuralEngineAssetsD",
            @"/private/var/folders",
        ];

        // Create dirs that may not exist yet so snapshot works
        for (NSString* d in watched) {
            BOOL exists = [fm fileExistsAtPath:d];
            if (!exists) {
                WARN("Dir does not exist (pre-compile): " << [d UTF8String]);
            }
        }

        // Snapshot before
        std::vector<std::pair<std::string, std::set<std::string>>> before;
        for (NSString* d in watched)
            before.push_back({ [d UTF8String], snapshot_dir_deep(d) });

        // Compile
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        INFO("model_dir = " << (h->prog ? h->prog->model_dir : "(nil)"));

        // Wait for aned/ANECompilerService to finish writing
        usleep(500000); // 500 ms

        // Snapshot after
        std::vector<std::pair<std::string, std::set<std::string>>> after;
        for (NSString* d in watched)
            after.push_back({ [d UTF8String], snapshot_dir_deep(d) });

        // Extensions of interest
        NSArray<NSString*>* exts = @[@".llir.bundle", @".anecir", @".anec", @".hwx", @".bundle"];

        bool found_any = false;
        for (size_t i = 0; i < before.size(); ++i) {
            auto news = new_files(before[i].second, after[i].second);
            for (auto& f : news) {
                NSString* nsf = [NSString stringWithUTF8String:f.c_str()];
                bool interesting = false;
                for (NSString* ext in exts)
                    if ([nsf hasSuffix:ext]) interesting = true;
                // Also flag any new file in an ANE-related subdirectory
                if ([nsf containsString:@"aned"] || [nsf containsString:@"ANE"] ||
                    [nsf containsString:@"NeuralEngine"])
                    interesting = true;
                if (interesting) {
                    found_any = true;
                    NSDictionary* attrs = [fm attributesOfItemAtPath:nsf error:nil];
                    WARN("NEW: " << f << " (" << [attrs[NSFileSize] longLongValue] << " bytes)");
                } else {
                    INFO("new (not ANE): " << f);
                }
            }
        }

        // Also do a targeted search for .llir.bundle anywhere under model_dir
        if (h->prog && !h->prog->model_dir.empty()) {
            NSString* mdir = [NSString stringWithUTF8String:h->prog->model_dir.c_str()];
            NSDirectoryEnumerator* en = [fm enumeratorAtPath:mdir];
            for (NSString* f in en) {
                WARN("model_dir file: " << [f UTF8String]);
            }
        }

        if (!found_any)
            WARN("No AOT-relevant new files found — binary not written to any watched directory");

        libane_mil_release(h);
    }
}

// ── Probe 2: _ANEInMemoryModel method scan for aot/vault/path selectors ───────

TEST_CASE("AOT-2: _ANEInMemoryModel aot/vault/path method scan",
          "[pathc][aot][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls = NSClassFromString(@"_ANEInMemoryModel");
        REQUIRE(cls != nil);

        unsigned int count = 0;
        Method* methods = class_copyMethodList(cls, &count);
        std::vector<std::string> interesting;
        for (unsigned int i = 0; i < count; ++i) {
            std::string name = sel_getName(method_getName(methods[i]));
            if (is_interesting_for_aot(name))
                interesting.push_back(name);
        }
        free(methods);

        WARN("AOT/vault/path-related methods on _ANEInMemoryModel (" << interesting.size() << "):");
        for (auto& m : interesting)
            WARN("  " << m);

        // Compile a live model to call these methods on
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed — skipping live calls");
            if (h) libane_mil_release(h);
            return;
        }

        id model = (id)h->prog->objc_model;

        for (auto& name : interesting) {
            if (name.find(':') != std::string::npos) continue;
            SEL s = sel_registerName(name.c_str());
            if (![model respondsToSelector:s]) continue;
            Method m = class_getInstanceMethod(cls, s);
            if (!m) continue;
            const char* types = method_getTypeEncoding(m);
            if (!types) continue;
            char ret = types[0];
            if (ret != '@' && ret != 'v' && ret != '#') continue;

            @try {
                if (ret == 'v') {
                    ((void(*)(id,SEL))objc_msgSend)(model, s);
                    WARN("  [model " << name << "] → void");
                } else {
                    id result = ((id(*)(id,SEL))objc_msgSend)(model, s);
                    if (result) {
                        NSString* desc = [result description];
                        std::string d = desc ? [desc UTF8String] : "";
                        if (d.size() > 300) d = d.substr(0, 300) + "...";
                        WARN("  [model " << name << "] → " << [[result className] UTF8String]
                             << ": " << d);
                        // If it looks like a path, check if the file exists
                        if ([result isKindOfClass:[NSString class]]) {
                            BOOL exists = [[NSFileManager defaultManager]
                                fileExistsAtPath:(NSString*)result];
                            if (exists) WARN("    *** FILE EXISTS at this path!");
                        }
                        if ([result isKindOfClass:[NSURL class]]) {
                            BOOL exists = [[NSFileManager defaultManager]
                                fileExistsAtPath:[(NSURL*)result path]];
                            if (exists) WARN("    *** FILE EXISTS at this URL!");
                        }
                    } else {
                        WARN("  [model " << name << "] → nil");
                    }
                }
            } @catch (NSException* ex) {
                WARN("  [model " << name << "] → EXCEPTION: " << [[ex reason] UTF8String]);
            }
        }

        libane_mil_release(h);
    }
}

// ── Probe 3: _ANEInMemoryModelDescriptor scan for aot/vault/path selectors ────

TEST_CASE("AOT-3: _ANEInMemoryModelDescriptor aot/vault/path method scan",
          "[pathc][aot][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        REQUIRE(cls != nil);

        // Class methods first
        unsigned int cm_count = 0;
        Method* cm = class_copyMethodList(object_getClass(cls), &cm_count);
        WARN("AOT-interesting class methods on _ANEInMemoryModelDescriptor:");
        for (unsigned int i = 0; i < cm_count; ++i) {
            std::string name = sel_getName(method_getName(cm[i]));
            if (is_interesting_for_aot(name))
                WARN("  +" << name);
        }
        free(cm);

        // Instance methods
        unsigned int count = 0;
        Method* methods = class_copyMethodList(cls, &count);
        std::vector<std::string> interesting;
        for (unsigned int i = 0; i < count; ++i) {
            std::string name = sel_getName(method_getName(methods[i]));
            if (is_interesting_for_aot(name))
                interesting.push_back(name);
        }
        free(methods);
        WARN("AOT-interesting instance methods:");
        for (auto& m : interesting) WARN("  " << m);

        // Get a live descriptor
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;
        id desc = nil;
        SEL s_desc = sel_registerName("modelDescriptor");
        if ([model respondsToSelector:s_desc])
            desc = ((id(*)(id,SEL))objc_msgSend)(model, s_desc);
        if (!desc) { WARN("no descriptor"); libane_mil_release(h); return; }

        for (auto& name : interesting) {
            if (name.find(':') != std::string::npos) continue;
            SEL s = sel_registerName(name.c_str());
            if (![desc respondsToSelector:s]) continue;
            Method m = class_getInstanceMethod(cls, s);
            if (!m) continue;
            const char* types = method_getTypeEncoding(m);
            if (!types || (types[0] != '@' && types[0] != 'v')) continue;
            @try {
                if (types[0] == 'v') {
                    ((void(*)(id,SEL))objc_msgSend)(desc, s);
                    WARN("  [desc " << name << "] → void");
                } else {
                    id result = ((id(*)(id,SEL))objc_msgSend)(desc, s);
                    if (result) {
                        NSString* d = [result description];
                        std::string ds = d ? [d UTF8String] : "";
                        if (ds.size() > 300) ds = ds.substr(0, 300) + "...";
                        WARN("  [desc " << name << "] → " << [[result className] UTF8String]
                             << ": " << ds);
                    } else {
                        WARN("  [desc " << name << "] → nil");
                    }
                }
            } @catch (NSException* ex) {
                WARN("  [desc " << name << "] → EXCEPTION: " << [[ex reason] UTF8String]);
            }
        }
        libane_mil_release(h);
    }
}

// ── Probe 4: kANEFAOTCacheUrlIdentifierKey extraction ────────────────────────

TEST_CASE("AOT-4: kANEFAOTCacheUrlIdentifierKey from compiled model",
          "[pathc][aot][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;

        // Try getCacheURLIdentifier on model
        SEL s_getCacheID = sel_registerName("getCacheURLIdentifier");
        SEL s_hexID      = sel_registerName("hexStringIdentifier");
        SEL s_cacheURL   = sel_registerName("cacheURLIdentifier");
        SEL s_modelDesc  = sel_registerName("modelDescriptor");

        NSString* hexID = ((NSString*(*)(id,SEL))objc_msgSend)(model, s_hexID);
        INFO("Model hexID: " << (hexID ? [hexID UTF8String] : "nil"));

        // Try getCacheURLIdentifier
        if ([model respondsToSelector:s_getCacheID]) {
            id cid = ((id(*)(id,SEL))objc_msgSend)(model, s_getCacheID);
            WARN("getCacheURLIdentifier: " << (cid ? [[cid description] UTF8String] : "nil"));
        } else {
            WARN("getCacheURLIdentifier: not present on _ANEInMemoryModel");
        }

        if ([model respondsToSelector:s_cacheURL]) {
            id cid = ((id(*)(id,SEL))objc_msgSend)(model, s_cacheURL);
            WARN("cacheURLIdentifier: " << (cid ? [[cid description] UTF8String] : "nil"));
        }

        // Try on the descriptor
        id desc = nil;
        if ([model respondsToSelector:s_modelDesc])
            desc = ((id(*)(id,SEL))objc_msgSend)(model, s_modelDesc);
        if (desc) {
            if ([desc respondsToSelector:s_getCacheID]) {
                id cid = ((id(*)(id,SEL))objc_msgSend)(desc, s_getCacheID);
                WARN("desc.getCacheURLIdentifier: " << (cid ? [[cid description] UTF8String] : "nil"));
            }
        }

        // Try to reach _ANEModelCacheManager and query it
        Class cls_mgr = NSClassFromString(@"_ANEModelCacheManager");
        Class cls_imm = NSClassFromString(@"_ANEInMemoryModelCacheManager");

        for (Class mgr_cls : {cls_mgr, cls_imm}) {
            if (!mgr_cls) continue;
            const char* cn = class_getName(mgr_cls);
            WARN("--- " << cn << " class methods:");

            unsigned int cm_count = 0;
            Method* cms = class_copyMethodList(object_getClass(mgr_cls), &cm_count);
            for (unsigned int i = 0; i < cm_count; ++i) {
                std::string name = sel_getName(method_getName(cms[i]));
                WARN("  +" << name);
            }
            free(cms);

            // Try sharedInstance/sharedManager
            for (const char* sel_name : {"sharedInstance", "sharedManager",
                                          "defaultManager", "sharedCache"}) {
                SEL s = sel_registerName(sel_name);
                if ([mgr_cls respondsToSelector:s]) {
                    id inst = ((id(*)(Class,SEL))objc_msgSend)(mgr_cls, s);
                    WARN("  " << cn << "." << sel_name << " = " << (inst ? "non-nil" : "nil"));
                    if (inst) {
                        // List instance methods
                        unsigned int ic = 0;
                        Method* im = class_copyMethodList(mgr_cls, &ic);
                        WARN("  Instance methods:");
                        for (unsigned int j = 0; j < ic; ++j) {
                            std::string n = sel_getName(method_getName(im[j]));
                            WARN("    " << n);
                        }
                        free(im);
                    }
                }
            }
        }

        libane_mil_release(h);
    }
}

// ── Probe 5: Scan for XPC service connection to ANECompilerService ────────────

TEST_CASE("AOT-5: NSXPCConnection to com.apple.ANECompilerService",
          "[pathc][aot][probe]") {
    @autoreleasepool {
        libane_available();

        // Attempt to establish direct XPC connection to ANECompilerService
        NSXPCConnection* conn = [[NSXPCConnection alloc]
            initWithMachServiceName:@"com.apple.ANECompilerService"
                            options:0];

        if (!conn) {
            WARN("NSXPCConnection alloc/init returned nil");
            return;
        }

        WARN("NSXPCConnection created (service=com.apple.ANECompilerService)");

        // Set up a minimal interface — we don't know the protocol, so we probe
        // by observing what interface the framework client normally uses.
        // Look for _ANECompilerProtocol
        Protocol* proto = objc_getProtocol("_ANECompilerProtocol");
        if (proto) {
            WARN("Found _ANECompilerProtocol — can set up typed interface");
        } else {
            WARN("_ANECompilerProtocol not in runtime — will try untyped");
        }

        // Try to set remoteObjectInterface
        // Even without a protocol, we can use NSXPCInterface with NSObject
        NSXPCInterface* iface = [NSXPCInterface interfaceWithProtocol:@protocol(NSObject)];
        [conn setRemoteObjectInterface:iface];

        __block BOOL connected = NO;
        __block NSError* connError = nil;
        conn.interruptionHandler = ^{ WARN("XPC connection interrupted"); };
        conn.invalidationHandler = ^{ WARN("XPC connection invalidated"); };

        [conn resume];

        // Try to get a proxy and call cacheURLIdentifierForModel:useSourceURL:withReply:
        id proxy = [conn remoteObjectProxyWithErrorHandler:^(NSError* err) {
            connError = err;
            WARN("XPC remote error: " << [[err localizedDescription] UTF8String]);
        }];

        if (!proxy) {
            WARN("Remote proxy is nil");
        } else {
            WARN("Got remote XPC proxy");
            connected = YES;

            // Check what selectors are available
            SEL s_cacheURL = sel_registerName("cacheURLIdentifierForModel:useSourceURL:withReply:");
            SEL s_compile  = sel_registerName("compileModel:options:ok:error:");
            SEL s_compileJIT = sel_registerName("compileModelJIT:ok:error:");

            WARN("cacheURLIdentifierForModel:useSourceURL:withReply: "
                 << ([proxy respondsToSelector:s_cacheURL] ? "YES" : "NO"));
            WARN("compileModel:options:ok:error: "
                 << ([proxy respondsToSelector:s_compile] ? "YES" : "NO"));
            WARN("compileModelJIT:ok:error: "
                 << ([proxy respondsToSelector:s_compileJIT] ? "YES" : "NO"));
        }

        [conn invalidate];

        if (!connected && !connError)
            WARN("Connection failed (likely entitlement restriction or wrong service name)");
        else if (connected)
            WARN("XPC connection to ANECompilerService SUCCEEDED — direct compiler access is viable!");
    }
}

#else
TEST_CASE("AOT binary probe: Apple-only", "[pathc][aot]") { WARN("Apple-only"); }
#endif
