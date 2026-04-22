/**
 * test_hwx_search.mm — Find where aned puts compiled binaries on macOS 26
 *
 * Background: ane_serialize_program and ane_load_hwx both scan model_dir for
 * a .hwx file written by ANECompilerService after compileWithQoS:.  On
 * macOS 26 no such file appears, breaking tests 442-445.  This probe
 * investigates three questions:
 *
 *   Probe 1 — Filesystem scan
 *     Snapshot ANE-related cache dirs before/after compileWithQoS: and list
 *     every new file.  Covers:
 *       ~/Library/Caches/com.apple.ANECompilerService/
 *       ~/Library/Caches/com.apple.aned/
 *       NSTemporaryDirectory()   (includes /tmp/<hexID>/ == model_dir)
 *       NSCachesDirectory (user)
 *
 *   Probe 2 — _ANEInMemoryModel method scan
 *     Dump every instance method whose name contains a keyword associated
 *     with serialization or binary data:
 *       data, bytes, hwx, compiled, binary, export, serial, flat, buffer,
 *       network, description, encode, archive, write
 *     For each hit, try calling it (no-arg or returning id/NSData*) and log
 *     what comes back.
 *
 *   Probe 3 — _ANEInMemoryModelDescriptor method scan
 *     Same as Probe 2 but on the descriptor class.
 *     modelWithNetworkDescription: was a promising path; are there others?
 *
 *   Probe 4 — model_dir deep listing
 *     List EVERY file in model_dir after compile (not just .hwx).  On older
 *     macOS the directory contained model.mil, weights/, compiled/<hex>.hwx.
 *     On macOS 26 it may contain different artifacts.
 *
 * Tags: [hwx][serialization][probe]
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
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"hwxsearch_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

// ── helpers ──────────────────────────────────────────────────────────────────

// Recursively collect all file paths under root into a set<string>.
static std::set<std::string> snapshot_dir(NSString* root) {
    std::set<std::string> out;
    NSFileManager* fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:root]) return out;
    NSDirectoryEnumerator* en = [fm enumeratorAtPath:root];
    for (NSString* f in en) {
        NSString* full = [root stringByAppendingPathComponent:f];
        // Only record files, not directories
        BOOL isDir = NO;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        if (!isDir)
            out.insert([full UTF8String]);
    }
    return out;
}

// Return all files in 'after' that were not in 'before'.
static std::vector<std::string> new_files(const std::set<std::string>& before,
                                          const std::set<std::string>& after) {
    std::vector<std::string> out;
    for (auto& p : after)
        if (!before.count(p)) out.push_back(p);
    return out;
}

// Keywords that suggest compiled binary / serialization
static bool is_interesting_method(const std::string& name) {
    static const char* kw[] = {
        "data", "bytes", "hwx", "compiled", "binary", "export",
        "serial", "flat", "buffer", "network", "description",
        "encode", "archive", "write", "program", "handle",
        nullptr
    };
    std::string lower = name;
    for (char& c : lower) c = (char)tolower(c);
    for (int i = 0; kw[i]; ++i)
        if (lower.find(kw[i]) != std::string::npos) return true;
    return false;
}

// ── Probe 1: Filesystem scan ─────────────────────────────────────────────────

TEST_CASE("HWXSearch 1: filesystem scan around compileWithQoS:", "[hwx][probe]") {
    @autoreleasepool {
        libane_available(); // ensure ANE framework is initialised

        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* home = NSHomeDirectory();

        // Directories to watch
        NSMutableArray<NSString*>* watched = [NSMutableArray array];

        // ANECompilerService cache
        [watched addObject:[home stringByAppendingPathComponent:
            @"Library/Caches/com.apple.ANECompilerService"]];
        // aned cache
        [watched addObject:[home stringByAppendingPathComponent:
            @"Library/Caches/com.apple.aned"]];
        // Generic user caches dir
        NSArray* cacheDirs = NSSearchPathForDirectoriesInDomains(
            NSCachesDirectory, NSUserDomainMask, YES);
        if (cacheDirs.count)
            [watched addObject:cacheDirs[0]];
        // /tmp
        [watched addObject:[NSString stringWithUTF8String:
            P_tmpdir ? P_tmpdir : "/tmp"]];
        // NSTemporaryDirectory
        [watched addObject:NSTemporaryDirectory()];

        // Ensure dirs exist before snapshotting (creates them if missing so
        // the snapshot is not nil)
        for (NSString* d in watched)
            [fm createDirectoryAtPath:d withIntermediateDirectories:YES
                           attributes:nil error:nil];

        // Snapshot BEFORE
        std::vector<std::pair<std::string, std::set<std::string>>> before_snaps;
        for (NSString* d in watched)
            before_snaps.push_back({ [d UTF8String], snapshot_dir(d) });

        // Compile
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        REQUIRE(h->prog != nullptr);

        std::string model_dir = h->prog->model_dir;
        INFO("model_dir = " << model_dir);

        // Snapshot AFTER (give aned a moment to flush)
        usleep(200000); // 200 ms
        std::vector<std::pair<std::string, std::set<std::string>>> after_snaps;
        for (NSString* d in watched)
            after_snaps.push_back({ [d UTF8String], snapshot_dir(d) });

        // Report new files in each watched dir
        bool found_any = false;
        for (size_t i = 0; i < before_snaps.size(); ++i) {
            auto news = new_files(before_snaps[i].second, after_snaps[i].second);
            if (!news.empty()) {
                found_any = true;
                WARN("NEW files in " << before_snaps[i].first << ":");
                for (auto& f : news)
                    WARN("  " << f);
            }
        }

        if (!found_any)
            WARN("No new files found in any watched directory after compile");

        // Also: look for .hwx anywhere in watched dirs right now
        bool found_hwx = false;
        for (NSString* d in watched) {
            NSDirectoryEnumerator* en = [fm enumeratorAtPath:d];
            for (NSString* f in en) {
                if ([f hasSuffix:@".hwx"]) {
                    found_hwx = true;
                    WARN("Found .hwx: " << [[d stringByAppendingPathComponent:f] UTF8String]);
                }
            }
        }
        if (!found_hwx)
            WARN("No .hwx files found in any watched directory");

        libane_mil_release(h);
    }
}

// ── Probe 2: model_dir deep listing ─────────────────────────────────────────

TEST_CASE("HWXSearch 2: complete model_dir listing after compile", "[hwx][probe]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        REQUIRE(h->prog != nullptr);

        std::string mdir = h->prog->model_dir;
        INFO("model_dir = " << mdir);

        NSString* dir_ns = [NSString stringWithUTF8String:mdir.c_str()];
        NSFileManager* fm = [NSFileManager defaultManager];

        BOOL exists = [fm fileExistsAtPath:dir_ns];
        INFO("model_dir exists: " << (exists ? "YES" : "NO"));

        if (exists) {
            NSDirectoryEnumerator* en = [fm enumeratorAtPath:dir_ns];
            bool any = false;
            for (NSString* f in en) {
                NSString* full = [dir_ns stringByAppendingPathComponent:f];
                NSDictionary* attrs = [fm attributesOfItemAtPath:full error:nil];
                NSNumber* sz = attrs[NSFileSize];
                WARN("  " << [f UTF8String] << "  (" << (sz ? sz.longLongValue : -1LL) << " bytes)");
                any = true;
            }
            if (!any)
                WARN("model_dir is empty — no files written");
        } else {
            WARN("model_dir does not exist on disk");
        }

        libane_mil_release(h);
    }
}

// ── Probe 3: _ANEInMemoryModel interesting method scan ───────────────────────

TEST_CASE("HWXSearch 3: _ANEInMemoryModel serialization-related methods", "[hwx][probe]") {
    @autoreleasepool {
        libane_available();

        Class cls = NSClassFromString(@"_ANEInMemoryModel");
        REQUIRE(cls != nil);

        // Collect all instance methods
        unsigned int count = 0;
        Method* methods = class_copyMethodList(cls, &count);

        std::vector<std::string> interesting;
        for (unsigned int i = 0; i < count; ++i) {
            std::string name = sel_getName(method_getName(methods[i]));
            if (is_interesting_method(name))
                interesting.push_back(name);
        }
        free(methods);

        INFO("Total instance methods on _ANEInMemoryModel: " << count);
        WARN("Interesting methods (" << interesting.size() << "):");
        for (auto& m : interesting)
            WARN("  " << m);

        // Compile a real model so we can call methods on a live instance
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed — skipping live method calls");
            if (h) libane_mil_release(h);
            return;
        }

        id model = (id)h->prog->objc_model;

        // Try zero-argument methods that return id (could be NSData*)
        for (auto& name : interesting) {
            SEL s = sel_registerName(name.c_str());
            if (![model respondsToSelector:s]) continue;

            // Only call zero-arg methods (signature: id (*)(id, SEL))
            Method m = class_getInstanceMethod(cls, s);
            if (!m) continue;
            const char* types = method_getTypeEncoding(m);
            if (!types) continue;
            // The return type is the first char of the encoding
            // '@' = object, 'v' = void, 'B' = BOOL, 'c' = char, etc.
            // Only try to call if return type looks like an object or void
            char ret = types[0];
            if (ret != '@' && ret != 'v' && ret != '#') continue;
            // Must be no-arg (id, SEL only) — encoding starts with @8@0:8
            // Length check: a no-arg selector name has no ':'
            if (name.find(':') != std::string::npos) continue;

            @try {
                if (ret == 'v') {
                    ((void(*)(id,SEL))objc_msgSend)(model, s);
                    WARN("  [model " << name << "] → void (no crash)");
                } else {
                    id result = ((id(*)(id,SEL))objc_msgSend)(model, s);
                    if (result) {
                        NSString* desc = [result description];
                        std::string d = desc ? [desc UTF8String] : "(no description)";
                        if (d.size() > 200) d = d.substr(0, 200) + "...";
                        WARN("  [model " << name << "] → " << [[result className] UTF8String]
                             << ": " << d);
                        // If it's NSData, report size
                        if ([result isKindOfClass:[NSData class]])
                            WARN("    NSData length = " << ((NSData*)result).length << " bytes");
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

// ── Probe 4: _ANEInMemoryModelDescriptor interesting method scan ─────────────

TEST_CASE("HWXSearch 4: _ANEInMemoryModelDescriptor serialization-related methods", "[hwx][probe]") {
    @autoreleasepool {
        libane_available();

        Class cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        REQUIRE(cls != nil);

        // Instance methods
        unsigned int count = 0;
        Method* methods = class_copyMethodList(cls, &count);
        std::vector<std::string> interesting;
        for (unsigned int i = 0; i < count; ++i) {
            std::string name = sel_getName(method_getName(methods[i]));
            if (is_interesting_method(name))
                interesting.push_back(name);
        }
        free(methods);
        INFO("Instance methods on _ANEInMemoryModelDescriptor: " << count);
        WARN("Interesting instance methods:");
        for (auto& m : interesting)
            WARN("  " << m);

        // Class methods
        unsigned int cm_count = 0;
        Method* cm = class_copyMethodList(object_getClass(cls), &cm_count);
        std::vector<std::string> cm_interesting;
        for (unsigned int i = 0; i < cm_count; ++i) {
            std::string name = sel_getName(method_getName(cm[i]));
            if (is_interesting_method(name))
                cm_interesting.push_back(name);
        }
        free(cm);
        WARN("Interesting class methods:");
        for (auto& m : cm_interesting)
            WARN("  +" << m);

        // Compile and call zero-arg instance methods on a live descriptor
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("compile failed — skipping live descriptor calls");
            if (h) libane_mil_release(h);
            return;
        }

        // Get the descriptor from the model if possible
        id model = (id)h->prog->objc_model;
        SEL s_desc = sel_registerName("modelDescriptor");
        id descriptor = nil;
        if ([model respondsToSelector:s_desc])
            descriptor = ((id(*)(id,SEL))objc_msgSend)(model, s_desc);

        if (!descriptor) {
            WARN("modelDescriptor returned nil or selector absent — skipping descriptor calls");
        } else {
            WARN("Got descriptor from live model, calling interesting zero-arg methods:");
            for (auto& name : interesting) {
                if (name.find(':') != std::string::npos) continue;
                SEL s = sel_registerName(name.c_str());
                if (![descriptor respondsToSelector:s]) continue;
                Method m = class_getInstanceMethod(cls, s);
                if (!m) continue;
                const char* types = method_getTypeEncoding(m);
                if (!types || (types[0] != '@' && types[0] != 'v')) continue;
                @try {
                    if (types[0] == 'v') {
                        ((void(*)(id,SEL))objc_msgSend)(descriptor, s);
                        WARN("  [desc " << name << "] → void");
                    } else {
                        id result = ((id(*)(id,SEL))objc_msgSend)(descriptor, s);
                        if (result) {
                            NSString* d = [result description];
                            std::string ds = d ? [d UTF8String] : "";
                            if (ds.size() > 200) ds = ds.substr(0, 200) + "...";
                            WARN("  [desc " << name << "] → " << [[result className] UTF8String]
                                 << ": " << ds);
                            if ([result isKindOfClass:[NSData class]])
                                WARN("    NSData length = " << ((NSData*)result).length);
                        } else {
                            WARN("  [desc " << name << "] → nil");
                        }
                    }
                } @catch (NSException* ex) {
                    WARN("  [desc " << name << "] → EXCEPTION: " << [[ex reason] UTF8String]);
                }
            }
        }

        libane_mil_release(h);
    }
}

// ── Probe 5: ANECompilerService XPC cache directory ─────────────────────────
// ANECompilerService is an XPC service. Its compiled outputs may land in the
// container/cache of the *service*, not the calling process.  Check known
// locations for the service's sandbox container.

TEST_CASE("HWXSearch 5: ANECompilerService XPC service cache directories", "[hwx][probe]") {
    @autoreleasepool {
        libane_available();
        NSFileManager* fm = [NSFileManager defaultManager];

        // Known locations for XPC helper / system service caches
        NSArray<NSString*>* candidates = @[
            @"/Library/Caches/com.apple.ANECompilerService",
            @"/var/root/Library/Caches/com.apple.ANECompilerService",
            @"/private/var/root/Library/Caches/com.apple.ANECompilerService",
            // XPC service containers on macOS are often under
            // /private/var/folders/<XX>/<YYYYYY>/T/com.apple.*
            // We can't enumerate those without knowing the folder IDs, but
            // we can check the user's own var/folders path:
        ];

        // Also derive the user's NSTemporaryDirectory parent
        NSString* tmpDir = NSTemporaryDirectory();
        // /var/folders/xx/yyyyyy/T/ → strip "T/" to get the container dir
        NSString* containerDir = [tmpDir stringByDeletingLastPathComponent];
        NSArray<NSString*>* extra = @[
            [containerDir stringByAppendingPathComponent:@"C/com.apple.ANECompilerService"],
            [containerDir stringByAppendingPathComponent:@"0/com.apple.ANECompilerService"],
        ];

        NSMutableArray<NSString*>* all = [NSMutableArray arrayWithArray:candidates];
        [all addObjectsFromArray:extra];

        // Snapshot before
        std::vector<std::pair<std::string, std::set<std::string>>> before;
        for (NSString* d in all)
            before.push_back({ [d UTF8String], snapshot_dir(d) });

        // Compile
        libane_set_log_level(LIBANE_LOG_SILENT);
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        REQUIRE(h != nullptr);
        usleep(500000); // wait 500ms for XPC service to finish writing

        // Snapshot after
        std::vector<std::pair<std::string, std::set<std::string>>> after;
        for (NSString* d in all)
            after.push_back({ [d UTF8String], snapshot_dir(d) });

        for (size_t i = 0; i < before.size(); ++i) {
            bool dir_exists = [fm fileExistsAtPath:
                [NSString stringWithUTF8String:before[i].first.c_str()]];
            auto news = new_files(before[i].second, after[i].second);
            if (dir_exists || !news.empty()) {
                WARN("Dir: " << before[i].first
                     << (dir_exists ? " (EXISTS)" : " (absent)"));
                if (!news.empty()) {
                    for (auto& f : news)
                        WARN("  NEW: " << f);
                } else {
                    WARN("  (no new files)");
                }
            }
        }

        libane_mil_release(h);
    }
}

#else
TEST_CASE("HWXSearch: Apple-only", "[hwx]") { WARN("Apple-only"); }
#endif
