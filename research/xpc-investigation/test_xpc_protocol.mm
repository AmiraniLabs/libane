/**
 * test_xpc_protocol.mm — XPC protocol discovery for com.apple.ANECompilerService
 *
 * Background: AOT-5 confirmed NSXPCConnection to ANECompilerService succeeds
 * and a remote proxy is obtainable.  The connection invalidated because we used
 * @protocol(NSObject) — the wrong interface.  _ANEClient already has a live,
 * working XPC connection to the same service using the CORRECT protocol.
 *
 * Strategy: read the protocol out of _ANEClient's internal NSXPCConnection
 * rather than guessing.  Then use that protocol to make our own typed connection
 * and call the compile method with a user-controlled output path.
 *
 *   XPC-1  Enumerate all registered ObjC protocols — filter ANE/Compiler names
 *   XPC-2  _ANEClient ivar walk → find NSXPCConnection → remoteObjectInterface
 *          → Protocol* → list all method signatures
 *   XPC-3  Try candidate protocol names via objc_getProtocol()
 *   XPC-4  Establish our own typed connection with the discovered protocol
 *          and call a zero-arg or discovery method to confirm it works
 *   XPC-5  Compile via the typed proxy with a user-controlled aotModelBinaryPath
 *          and scan for the resulting binary
 *
 * Tags: [pathc][xpc][protocol][probe]
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

static double ms_now() {
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

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
    "        tensor<fp16, [1,4,1,16]> y = relu(x=x)[name=string(\"xpc_relu\")];\n"
    "    } -> (y);\n"
    "}\n";

// ── XPC-1: Enumerate all registered protocols ─────────────────────────────────

TEST_CASE("XPC-1: enumerate registered ObjC protocols for ANE/Compiler names",
          "[pathc][xpc][protocol]") {
    @autoreleasepool {
        libane_available(); // loads AppleNeuralEngine.framework

        unsigned int total = 0;
        Protocol** all = objc_copyProtocolList(&total);
        WARN("Total registered protocols: " << total);

        std::vector<std::string> hits;
        for (unsigned int i = 0; i < total; ++i) {
            const char* name = protocol_getName(all[i]);
            if (!name) continue;
            std::string n(name);
            if (n.find("ANE") != std::string::npos  ||
                n.find("Ane") != std::string::npos  ||
                n.find("ane") != std::string::npos  ||
                n.find("Compiler") != std::string::npos ||
                n.find("Neural") != std::string::npos   ||
                n.find("neural") != std::string::npos) {
                hits.push_back(n);
            }
        }
        free(all);

        if (hits.empty()) {
            WARN("No ANE/Compiler protocols found in runtime");
        } else {
            WARN("ANE/Compiler-related protocols (" << hits.size() << "):");
            for (auto& h : hits) {
                Protocol* p = objc_getProtocol(h.c_str());
                WARN("  " << h);

                // List required instance methods
                unsigned int mc = 0;
                objc_method_description* descs =
                    protocol_copyMethodDescriptionList(p, YES, YES, &mc);
                for (unsigned int j = 0; j < mc; ++j)
                    WARN("    -" << sel_getName(descs[j].name)
                         << "  " << (descs[j].types ? descs[j].types : ""));
                free(descs);
            }
        }
    }
}

// ── XPC-2: _ANEClient ivar walk → NSXPCConnection → remoteObjectInterface ─────

TEST_CASE("XPC-2: extract protocol from _ANEClient's internal NSXPCConnection",
          "[pathc][xpc][protocol]") {
    @autoreleasepool {
        libane_available();

        Class cls_client = NSClassFromString(@"_ANEClient");
        if (!cls_client) { WARN("_ANEClient not found"); return; }

        // Get the shared client object
        SEL sel_shared = sel_registerName("sharedConnection");
        id client = nil;
        if ([cls_client respondsToSelector:sel_shared])
            client = ((id(*)(Class,SEL))objc_msgSend)(cls_client, sel_shared);
        if (!client) { WARN("_ANEClient.sharedConnection returned nil"); return; }

        WARN("Got _ANEClient: " << [[client description] UTF8String]);

        // Walk the full class hierarchy dumping ivars
        NSXPCConnection* found_conn = nil;
        Class c = [client class];
        while (c && c != [NSObject class]) {
            unsigned int count = 0;
            Ivar* ivars = class_copyIvarList(c, &count);
            WARN("ivars of " << class_getName(c) << " (" << count << "):");
            for (unsigned int i = 0; i < count; ++i) {
                const char* name = ivar_getName(ivars[i]);
                const char* type = ivar_getTypeEncoding(ivars[i]);
                ptrdiff_t   off  = ivar_getOffset(ivars[i]);
                WARN("  [" << off << "] " << (name ? name : "?")
                     << " : " << (type ? type : "?"));

                // Try to read object ivars that look like XPC connections
                if (!name || !type) continue;
                std::string sname(name), stype(type);

                bool looks_like_conn =
                    sname.find("onnection") != std::string::npos ||
                    sname.find("xpc") != std::string::npos ||
                    sname.find("XPC") != std::string::npos ||
                    sname.find("channel") != std::string::npos;

                if (looks_like_conn && stype.size() > 0 && stype[0] == '@') {
                    @try {
                        // Read the ivar via direct offset arithmetic (MRC-safe)
                        void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                        id val = (__bridge id)*slot;
                        if (val) {
                            WARN("    → value: " << [[val description] UTF8String]);
                            if ([val isKindOfClass:[NSXPCConnection class]]) {
                                WARN("    *** NSXPCConnection FOUND at ivar '" << name << "' ***");
                                found_conn = (NSXPCConnection*)val;
                            }
                        }
                    } @catch (...) {}
                }
            }
            free(ivars);
            c = class_getSuperclass(c);
        }

        // If we didn't find it via ivar name heuristic, brute-force scan all ivars
        if (!found_conn) {
            WARN("No NSXPCConnection found by name heuristic — brute-force scanning");
            c = [client class];
            while (c && c != [NSObject class] && !found_conn) {
                unsigned int count = 0;
                Ivar* ivars = class_copyIvarList(c, &count);
                for (unsigned int i = 0; i < count && !found_conn; ++i) {
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    if (!type || type[0] != '@') continue;
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    @try {
                        void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                        id val = (__bridge id)*slot;
                        if (val && [val isKindOfClass:[NSXPCConnection class]]) {
                            found_conn = (NSXPCConnection*)val;
                            WARN("Brute-force found NSXPCConnection at offset " << off);
                        }
                    } @catch (...) {}
                }
                free(ivars);
                c = class_getSuperclass(c);
            }
        }

        if (!found_conn) {
            WARN("No NSXPCConnection in _ANEClient ivars — may use raw Mach port");
            // Fall back: list all methods of _ANEClient for clues
            unsigned int mc = 0;
            Method* methods = class_copyMethodList(cls_client, &mc);
            WARN("_ANEClient instance methods:");
            for (unsigned int i = 0; i < mc; ++i)
                WARN("  -" << sel_getName(method_getName(methods[i])));
            free(methods);
            return;
        }

        // Extract remoteObjectInterface → Protocol
        SEL sel_roi = sel_registerName("remoteObjectInterface");
        id iface = nil;
        if ([found_conn respondsToSelector:sel_roi])
            iface = ((id(*)(id,SEL))objc_msgSend)(found_conn, sel_roi);

        if (!iface) {
            WARN("remoteObjectInterface returned nil");
            // Also check exportedInterface
            SEL sel_ei = sel_registerName("exportedInterface");
            iface = [found_conn respondsToSelector:sel_ei]
                ? ((id(*)(id,SEL))objc_msgSend)(found_conn, sel_ei) : nil;
            if (iface) WARN("Got exportedInterface instead");
        }

        if (!iface) { WARN("No interface found on NSXPCConnection"); return; }

        WARN("NSXPCInterface: " << [[iface description] UTF8String]);

        // Get the Protocol from NSXPCInterface
        SEL sel_proto = sel_registerName("protocol");
        Protocol* proto = nil;
        if ([iface respondsToSelector:sel_proto])
            proto = ((Protocol*(*)(id,SEL))objc_msgSend)(iface, sel_proto);

        if (!proto) { WARN("interface.protocol returned nil"); return; }

        const char* proto_name = protocol_getName(proto);
        WARN("*** PROTOCOL FOUND: " << (proto_name ? proto_name : "unnamed") << " ***");

        // List all required instance methods
        WARN("Required instance methods:");
        unsigned int mc = 0;
        objc_method_description* descs =
            protocol_copyMethodDescriptionList(proto, YES, YES, &mc);
        for (unsigned int i = 0; i < mc; ++i)
            WARN("  -" << sel_getName(descs[i].name)
                 << "  types=" << (descs[i].types ? descs[i].types : "?"));
        free(descs);

        // Optional instance methods
        mc = 0;
        descs = protocol_copyMethodDescriptionList(proto, NO, YES, &mc);
        if (mc) {
            WARN("Optional instance methods:");
            for (unsigned int i = 0; i < mc; ++i)
                WARN("  -" << sel_getName(descs[i].name)
                     << "  types=" << (descs[i].types ? descs[i].types : "?"));
            free(descs);
        }
    }
}

// ── XPC-3: Try candidate protocol names ──────────────────────────────────────

TEST_CASE("XPC-3: candidate protocol names via objc_getProtocol",
          "[pathc][xpc][protocol]") {
    @autoreleasepool {
        libane_available();

        static const char* candidates[] = {
            "_ANECompilerProtocol",
            "ANECompilerProtocol",
            "_ANECompilerServiceProtocol",
            "ANECompilerServiceProtocol",
            "_ANECompileServiceProtocol",
            "ANECompileServiceProtocol",
            "_ANEClientProtocol",
            "ANEClientProtocol",
            "_ANECompilerXPCProtocol",
            "ANECompilerXPCProtocol",
            "_ANEServiceProtocol",
            "ANEServiceProtocol",
            "_ANEModelCompilerProtocol",
            "ANEModelCompilerProtocol",
            nullptr
        };

        bool any_found = false;
        for (int i = 0; candidates[i]; ++i) {
            Protocol* p = objc_getProtocol(candidates[i]);
            if (!p) continue;
            any_found = true;
            WARN("FOUND: " << candidates[i]);
            unsigned int mc = 0;
            objc_method_description* descs =
                protocol_copyMethodDescriptionList(p, YES, YES, &mc);
            for (unsigned int j = 0; j < mc; ++j)
                WARN("  -" << sel_getName(descs[j].name));
            free(descs);
        }
        if (!any_found)
            WARN("None of the candidate protocol names found in runtime");
    }
}

// ── XPC-4: Establish typed connection with discovered protocol ────────────────

TEST_CASE("XPC-4: connect with discovered protocol and confirm typed RPC works",
          "[pathc][xpc][protocol]") {
    @autoreleasepool {
        libane_available();

        // Try to get the protocol from _ANEClient's connection (same as XPC-2)
        Protocol* proto = nil;
        @try {
            Class cls_client = NSClassFromString(@"_ANEClient");
            if (!cls_client) { WARN("_ANEClient not found"); return; }
            SEL sel_shared = sel_registerName("sharedConnection");
            id client = ((id(*)(Class,SEL))objc_msgSend)(cls_client, sel_shared);
            if (!client) { WARN("sharedConnection nil"); return; }

            // Brute-force scan for NSXPCConnection in all ivars
            Class c = [client class];
            NSXPCConnection* found_conn = nil;
            while (c && c != [NSObject class] && !found_conn) {
                unsigned int count = 0;
                Ivar* ivars = class_copyIvarList(c, &count);
                for (unsigned int i = 0; i < count && !found_conn; ++i) {
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    if (!type || type[0] != '@') continue;
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    @try {
                        void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                        id val = (__bridge id)*slot;
                        if (val && [val isKindOfClass:[NSXPCConnection class]])
                            found_conn = (NSXPCConnection*)val;
                    } @catch (...) {}
                }
                free(ivars);
                c = class_getSuperclass(c);
            }

            if (!found_conn) { WARN("XPC-4: NSXPCConnection not found in _ANEClient"); return; }

            SEL sel_roi = sel_registerName("remoteObjectInterface");
            id iface = [found_conn respondsToSelector:sel_roi]
                ? ((id(*)(id,SEL))objc_msgSend)(found_conn, sel_roi) : nil;
            if (!iface) { WARN("XPC-4: remoteObjectInterface nil"); return; }

            SEL sel_proto = sel_registerName("protocol");
            proto = [iface respondsToSelector:sel_proto]
                ? ((Protocol*(*)(id,SEL))objc_msgSend)(iface, sel_proto) : nil;
        } @catch (...) {}

        if (!proto) { WARN("XPC-4: could not recover protocol — skipping"); return; }
        WARN("XPC-4: using protocol " << protocol_getName(proto));

        // Open our own connection with the real protocol
        NSXPCConnection* conn = [[NSXPCConnection alloc]
            initWithMachServiceName:@"com.apple.ANECompilerService"
                            options:0];
        if (!conn) { WARN("XPC-4: connection alloc failed"); return; }

        NSXPCInterface* iface = [NSXPCInterface interfaceWithProtocol:proto];
        conn.remoteObjectInterface = iface;

        __block NSError* rpc_error = nil;
        __block BOOL invalidated = NO;
        conn.interruptionHandler = ^{ WARN("XPC-4: interrupted"); };
        conn.invalidationHandler = ^{ invalidated = YES; WARN("XPC-4: invalidated"); };
        [conn resume];

        id proxy = [conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
            rpc_error = e;
            WARN("XPC-4 proxy error: " << [[e localizedDescription] UTF8String]);
        }];

        if (!proxy) { WARN("XPC-4: proxy nil"); [conn invalidate]; return; }
        WARN("XPC-4: got typed proxy — connection established with real protocol");

        // Probe: try any zero-arg or trivial discovery methods we know about
        // These come from the method dump in test_aot_binary_probe / ivar scans
        static const char* discovery_sels[] = {
            "defaultANECIRFileName",
            "systemModelsCacheDirectory",
            "userModelDataVaultDirectory",
            "modelDataVaultDirectory",
            "cacheDirectory",
            "version",
            nullptr
        };
        for (int i = 0; discovery_sels[i]; ++i) {
            SEL s = sel_registerName(discovery_sels[i]);
            if (![proxy respondsToSelector:s]) continue;
            @try {
                id result = ((id(*)(id,SEL))objc_msgSend)(proxy, s);
                WARN("XPC-4: " << discovery_sels[i] << " → "
                     << (result ? [[result description] UTF8String] : "nil"));
            } @catch (...) {
                WARN("XPC-4: " << discovery_sels[i] << " threw");
            }
        }

        [conn invalidate];

        if (!invalidated)
            WARN("XPC-4: connection stayed valid — typed RPC is viable!");
        else
            WARN("XPC-4: connection invalidated after typed setup — protocol still wrong");
    }
}

// ── XPC-5: Compile with user-controlled output path ──────────────────────────

TEST_CASE("XPC-5: compile via typed proxy with aotModelBinaryPath to capture binary",
          "[pathc][xpc][protocol]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // First compile normally to establish a model + get hexID + mil tempdir
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("XPC-5: cold compile failed — skipping"); if (h) libane_mil_release(h); return;
        }

        id model = (id)h->prog->objc_model;
        NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
            model, sel_registerName("hexStringIdentifier"));
        std::string model_dir = h->prog->model_dir;
        WARN("XPC-5: hexID=" << [hex_id UTF8String]);
        WARN("XPC-5: model_dir=" << model_dir);

        // Target output path for the compiled binary
        NSString* out_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc5_aot_output"];
        [[NSFileManager defaultManager]
            createDirectoryAtPath:out_dir withIntermediateDirectories:YES
            attributes:nil error:nil];
        NSString* out_path = [out_dir stringByAppendingPathComponent:@"model.llir.bundle"];
        WARN("XPC-5: target output path=" << [out_path UTF8String]);

        // Get the protocol from _ANEClient
        Protocol* proto = nil;
        @try {
            Class cls_client = NSClassFromString(@"_ANEClient");
            id client = ((id(*)(Class,SEL))objc_msgSend)(
                cls_client, sel_registerName("sharedConnection"));
            if (!client) goto xpc5_done;

            Class c = [client class];
            NSXPCConnection* found_conn = nil;
            while (c && c != [NSObject class] && !found_conn) {
                unsigned int count = 0;
                Ivar* ivars = class_copyIvarList(c, &count);
                for (unsigned int i = 0; i < count && !found_conn; ++i) {
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    if (!type || type[0] != '@') continue;
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    @try {
                        void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                        id val = (__bridge id)*slot;
                        if (val && [val isKindOfClass:[NSXPCConnection class]])
                            found_conn = (NSXPCConnection*)val;
                    } @catch (...) {}
                }
                free(ivars);
                c = class_getSuperclass(c);
            }
            if (!found_conn) goto xpc5_done;

            id iface = ((id(*)(id,SEL))objc_msgSend)(
                found_conn, sel_registerName("remoteObjectInterface"));
            if (!iface) goto xpc5_done;
            proto = ((Protocol*(*)(id,SEL))objc_msgSend)(
                iface, sel_registerName("protocol"));
        } @catch (...) {}

        if (proto) {
            WARN("XPC-5: protocol=" << protocol_getName(proto));

            NSXPCConnection* conn = [[NSXPCConnection alloc]
                initWithMachServiceName:@"com.apple.ANECompilerService"
                                options:0];
            conn.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:proto];
            __block BOOL conn_ok = YES;
            conn.invalidationHandler = ^{ conn_ok = NO; };
            [conn resume];

            id proxy = [conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
                WARN("XPC-5 proxy error: " << [[e localizedDescription] UTF8String]);
            }];

            if (proxy) {
                // Try methods that take a model path + output path
                // Known candidates from binary strings:
                //   compileModelAt:outputPath:options:completionHandler:
                //   compileModel:options:aotModelBinaryPath:completionHandler:
                //   compileModelJIT:aotModelBinaryPath:completionHandler:
                NSURL* model_url = [NSURL fileURLWithPath:
                    [NSString stringWithUTF8String:model_dir.c_str()]];
                NSURL* out_url   = [NSURL fileURLWithPath:out_path];

                static const char* compile_sels[] = {
                    "compileModelAt:outputPath:options:completionHandler:",
                    "compileModel:options:aotModelBinaryPath:completionHandler:",
                    "compileModelJIT:aotModelBinaryPath:completionHandler:",
                    "compileModelAtURL:outputURL:options:reply:",
                    "compileModelAtPath:outputPath:options:reply:",
                    nullptr
                };

                for (int i = 0; compile_sels[i]; ++i) {
                    SEL s = sel_registerName(compile_sels[i]);
                    if (![proxy respondsToSelector:s]) continue;

                    WARN("XPC-5: found selector " << compile_sels[i] << " on proxy");

                    // Count args from selector (number of colons)
                    std::string sel_str(compile_sels[i]);
                    int nargs = (int)std::count(sel_str.begin(), sel_str.end(), ':');
                    WARN("XPC-5: " << nargs << " args");

                    // Try calling with (modelURL, outputURL, options, reply)
                    // The exact signature depends on what the protocol says
                    if (nargs == 4) {
                        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                        __block BOOL compile_ok = NO;
                        __block NSError* compile_err = nil;

                        @try {
                            typedef void (*Fn4)(id, SEL, id, id, id, id);
                            ((Fn4)objc_msgSend)(proxy, s, model_url, out_url, @{},
                                ^(NSError* e) {
                                    compile_ok = (e == nil);
                                    compile_err = e;
                                    dispatch_semaphore_signal(sem);
                                });
                            dispatch_semaphore_wait(sem,
                                dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
                        } @catch (NSException* ex) {
                            WARN("XPC-5: exception: " << [[ex reason] UTF8String]);
                        } @catch (...) {
                            WARN("XPC-5: unknown exception");
                        }

                        if (compile_ok) {
                            WARN("XPC-5: COMPILE SUCCEEDED via " << compile_sels[i]);
                            // Check if output file was written
                            BOOL exists = [[NSFileManager defaultManager]
                                fileExistsAtPath:out_path];
                            WARN("XPC-5: output at " << [out_path UTF8String]
                                 << " exists=" << (int)exists);
                            if (exists) {
                                NSDictionary* attrs = [[NSFileManager defaultManager]
                                    attributesOfItemAtPath:out_path error:nil];
                                WARN("XPC-5: output size="
                                     << [[attrs[NSFileSize] description] UTF8String]);
                                WARN("XPC-5: *** BINARY CAPTURED — cross-op patching viable! ***");
                            }
                        } else if (compile_err) {
                            WARN("XPC-5: compile failed via " << compile_sels[i]
                                 << ": " << [[compile_err localizedDescription] UTF8String]);
                        }
                    }
                }

                // If none of the typed selectors matched, log what methods ARE available
                WARN("XPC-5: probing which compile-related selectors proxy accepts:");
                static const char* probe_sels[] = {
                    "compileModel:completionHandler:",
                    "compileModelAtPath:completionHandler:",
                    "compileModelAtURL:completionHandler:",
                    "compileRequest:completionHandler:",
                    "compileModel:withReply:",
                    "compileModelAt:withReply:",
                    nullptr
                };
                for (int i = 0; probe_sels[i]; ++i) {
                    SEL s = sel_registerName(probe_sels[i]);
                    if ([proxy respondsToSelector:s])
                        WARN("  ACCEPTS: " << probe_sels[i]);
                }
            }

            [conn invalidate];
            if (conn_ok)
                WARN("XPC-5: connection held — typed RPC confirmed");
        }

        xpc5_done:
        libane_mil_release(h);
    }
}

// ── XPC-6: _ANEDaemonConnection service name + echo + compile with output path ─
//
// XPC-1 found _ANEDaemonProtocol with compileModel:sandboxExtension:options:qos:withReply:
// XPC-2 showed _ANEClient holds _ANEDaemonConnection objects, not raw NSXPCConnection.
//
// This probe:
//   6a  Walk _ANEDaemonConnection ivars to find the Mach service name and the
//       underlying NSXPCConnection (one layer deeper than _ANEClient)
//   6b  Connect to aned with _ANEDaemonProtocol, fire echo:withReply: to confirm
//       the channel works before attempting anything destructive
//   6c  Try passing aotModelBinaryPath in compileWithQoS:options: directly —
//       the cheapest possible path (no daemon connection needed if it propagates)
//   6d  Call compileModel:sandboxExtension:options:qos:withReply: on our own
//       typed proxy with options containing the output path, then scan for the binary

TEST_CASE("XPC-6: _ANEDaemonConnection service name, echo, and compile with output path",
          "[pathc][xpc][compile]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        // ── 6a: walk _ANEDaemonConnection for service name + NSXPCConnection ─
        WARN("\n--- 6a: _ANEDaemonConnection ivar walk ---");

        Class cls_client = NSClassFromString(@"_ANEClient");
        Class cls_daemon_conn = NSClassFromString(@"_ANEDaemonConnection");

        if (!cls_client || !cls_daemon_conn) {
            WARN("6a: _ANEClient or _ANEDaemonConnection not found"); return;
        }

        SEL sel_shared = sel_registerName("sharedConnection");
        id client = ((id(*)(Class,SEL))objc_msgSend)(cls_client, sel_shared);
        if (!client) { WARN("6a: sharedConnection nil"); return; }

        // Read _conn ivar (offset 32) from _ANEClient
        id daemon_conn = nil;
        {
            unsigned int count = 0;
            Ivar* ivars = class_copyIvarList(cls_client, &count);
            for (unsigned int i = 0; i < count; ++i) {
                const char* name = ivar_getName(ivars[i]);
                if (!name) continue;
                if (strcmp(name, "_conn") == 0 || strcmp(name, "_fastConn") == 0) {
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                    id val = (__bridge id)*slot;
                    if (val) {
                        WARN("6a: _ANEClient." << name << " = "
                             << class_getName([val class]));
                        if (!daemon_conn) daemon_conn = val;
                    }
                }
            }
            free(ivars);
        }

        if (!daemon_conn) { WARN("6a: no _ANEDaemonConnection found"); return; }

        // Walk _ANEDaemonConnection ivars — find NSXPCConnection + service name
        NSXPCConnection* daemon_xpc = nil;
        NSString* service_name = nil;

        {
            Class c = [daemon_conn class];
            while (c && c != [NSObject class]) {
                unsigned int count = 0;
                Ivar* ivars = class_copyIvarList(c, &count);
                WARN("6a: ivars of " << class_getName(c) << ":");
                for (unsigned int i = 0; i < count; ++i) {
                    const char* name = ivar_getName(ivars[i]);
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    ptrdiff_t   off  = ivar_getOffset(ivars[i]);
                    WARN("  [" << off << "] " << (name ? name : "?")
                         << " : " << (type ? type : "?"));

                    if (!name || !type || type[0] != '@') continue;
                    ptrdiff_t o = ivar_getOffset(ivars[i]);
                    @try {
                        void** slot = (void**)((uint8_t*)(__bridge void*)daemon_conn + o);
                        id val = (__bridge id)*slot;
                        if (!val) continue;

                        if ([val isKindOfClass:[NSXPCConnection class]]) {
                            WARN("  *** NSXPCConnection at '" << name << "' ***");
                            daemon_xpc = (NSXPCConnection*)val;
                        }
                        if ([val isKindOfClass:[NSString class]]) {
                            std::string sv = [(NSString*)val UTF8String];
                            if (sv.find("apple") != std::string::npos ||
                                sv.find("ane")   != std::string::npos ||
                                sv.find("ANE")   != std::string::npos ||
                                sv.find("daemon") != std::string::npos) {
                                WARN("  *** service candidate: " << sv << " ***");
                                if (!service_name) service_name = (NSString*)val;
                            }
                        }
                    } @catch (...) {}
                }
                free(ivars);
                c = class_getSuperclass(c);
            }
        }

        // If NSXPCConnection found, pull remoteObjectInterface → protocol
        if (daemon_xpc) {
            WARN("6a: reading remoteObjectInterface from found NSXPCConnection");
            SEL sel_roi = sel_registerName("remoteObjectInterface");
            id iface = [daemon_xpc respondsToSelector:sel_roi]
                ? ((id(*)(id,SEL))objc_msgSend)(daemon_xpc, sel_roi) : nil;
            if (iface) {
                Protocol* p = ((Protocol*(*)(id,SEL))objc_msgSend)(
                    iface, sel_registerName("protocol"));
                if (p) WARN("6a: protocol from daemon NSXPCConnection = "
                            << protocol_getName(p));
            }

            // Also get the service name from the connection itself
            SEL sel_svc = sel_registerName("serviceName");
            if ([daemon_xpc respondsToSelector:sel_svc]) {
                id sn = ((id(*)(id,SEL))objc_msgSend)(daemon_xpc, sel_svc);
                if (sn) WARN("6a: serviceName = " << [(NSString*)sn UTF8String]);
                if (sn && !service_name) service_name = (NSString*)sn;
            }
            SEL sel_ep = sel_registerName("endpoint");
            if ([daemon_xpc respondsToSelector:sel_ep]) {
                id ep = ((id(*)(id,SEL))objc_msgSend)(daemon_xpc, sel_ep);
                if (ep) WARN("6a: endpoint = " << [[ep description] UTF8String]);
            }
        }

        if (service_name)
            WARN("6a: resolved service name = " << [service_name UTF8String]);
        else
            WARN("6a: service name not found — will try known candidate names");

        // ── 6b: echo:withReply: to confirm channel ────────────────────────────
        WARN("\n--- 6b: echo:withReply: via _ANEDaemonProtocol ---");

        Protocol* daemon_proto = objc_getProtocol("_ANEDaemonProtocol");
        REQUIRE(daemon_proto != nil); // confirmed present in XPC-1

        // Candidate service names to try in order
        NSArray* service_candidates = service_name
            ? @[service_name,
                @"com.apple.aned",
                @"com.apple.aneka",
                @"com.apple.ANEDaemon",
                @"com.apple.NeuralEngine"]
            : @[@"com.apple.aned",
                @"com.apple.aneka",
                @"com.apple.ANEDaemon",
                @"com.apple.NeuralEngine"];

        NSXPCConnection* working_conn = nil;
        NSString* working_service = nil;

        for (NSString* svc in service_candidates) {
            NSXPCConnection* conn = [[NSXPCConnection alloc]
                initWithMachServiceName:svc options:0];
            if (!conn) continue;
            conn.remoteObjectInterface =
                [NSXPCInterface interfaceWithProtocol:daemon_proto];
            __block BOOL ok = YES;
            conn.invalidationHandler = ^{ ok = NO; };
            [conn resume];

            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block BOOL echo_ok = NO;
            __block NSError* echo_err = nil;

            id proxy = [conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
                echo_err = e; dispatch_semaphore_signal(sem);
            }];

            SEL sel_echo = sel_registerName("echo:withReply:");
            @try {
                typedef void (*EchoFn)(id, SEL, NSString*, void(^)(NSString*));
                ((EchoFn)objc_msgSend)(proxy, sel_echo, @"ping",
                    ^(NSString* reply) {
                        echo_ok = YES;
                        WARN("6b: echo reply from " << [svc UTF8String]
                             << ": " << (reply ? [reply UTF8String] : "nil"));
                        dispatch_semaphore_signal(sem);
                    });
                dispatch_semaphore_wait(sem,
                    dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC));
            } @catch (NSException* ex) {
                WARN("6b: exception on " << [svc UTF8String]
                     << ": " << [[ex reason] UTF8String]);
                dispatch_semaphore_signal(sem);
            }

            if (echo_ok) {
                WARN("6b: *** ECHO SUCCEEDED on " << [svc UTF8String]
                     << " — daemon channel confirmed! ***");
                working_conn    = conn;
                working_service = svc;
                break;
            } else if (echo_err) {
                WARN("6b: " << [svc UTF8String] << " → error: "
                     << [[echo_err localizedDescription] UTF8String]);
            } else {
                WARN("6b: " << [svc UTF8String] << " → timeout or invalidated");
            }

            if (!echo_ok) { [conn invalidate]; }
        }

        // ── 6c: try aotModelBinaryPath directly in compileWithQoS:options: ───
        WARN("\n--- 6c: aotModelBinaryPath in compileWithQoS:options: ---");
        {
            // Compile a fresh model with an output path key injected into options
            NSString* out_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc6c_output"];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:out_dir withIntermediateDirectories:YES
                attributes:nil error:nil];
            NSURL* out_url = [NSURL fileURLWithPath:
                [out_dir stringByAppendingPathComponent:@"model.llir.bundle"]];

            // Candidate keys for the output path
            NSArray* key_candidates = @[
                @"aotModelBinaryPath",
                @"ANEFAOTBinaryPath",
                @"kANEFAOTBinaryPath",
                @"outputPath",
                @"compiledModelPath",
                @"binaryOutputPath",
            ];

            for (NSString* key in key_candidates) {
                // Create fresh model
                Class cls_desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
                Class cls_inMem = NSClassFromString(@"_ANEInMemoryModel");
                NSData* mil = [NSData dataWithBytes:kReluMIL length:strlen(kReluMIL)];
                id desc = ((id(*)(Class,SEL,NSData*,id,id))objc_msgSend)(
                    cls_desc,
                    sel_registerName("modelWithMILText:weights:optionsPlist:"),
                    mil, @{}, nil);
                id model = ((id(*)(Class,SEL,id))objc_msgSend)(
                    cls_inMem,
                    sel_registerName("inMemoryModelWithDescriptor:"),
                    desc);
                if (!model) continue;

                // Write temp dir
                NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                    model, sel_registerName("hexStringIdentifier"));
                NSString* tmpdir = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:hex_id];
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:tmpdir
                    withIntermediateDirectories:YES attributes:nil error:nil];
                [mil writeToFile:[tmpdir stringByAppendingPathComponent:@"model.mil"]
                      atomically:YES];

                NSDictionary* opts = @{key: out_url};
                NSError* err = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                    model,
                    sel_registerName("compileWithQoS:options:error:"),
                    21u, opts, &err);

                BOOL exists = [[NSFileManager defaultManager]
                    fileExistsAtPath:[out_url path]];
                WARN("6c: key=" << [key UTF8String]
                     << "  compile=" << (ok ? "OK" : "FAIL")
                     << "  binary_exists=" << (int)exists
                     << (err ? [[@"  err=" stringByAppendingString:
                                 [err localizedDescription]] UTF8String] : ""));
                if (exists)
                    WARN("6c: *** BINARY WRITTEN via key " << [key UTF8String] << " ***");

                // Unload
                ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(
                    model, sel_registerName("unloadWithQoS:error:"), 21u, nil);
            }
        }

        // ── 6d: call compileModel:sandboxExtension:options:qos:withReply: ─────
        WARN("\n--- 6d: compileModel:sandboxExtension:options:qos:withReply: ---");

        if (!working_conn) {
            WARN("6d: no working daemon connection from 6b — skipping");
            return;
        }

        // Compile a model via libane first to get the model dict + hex_id
        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("6d: libane compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;

        // Get the model attributes dict (_ANEInMemoryModel.modelAttributes)
        id model_attrs = nil;
        SEL sel_ma = sel_registerName("modelAttributes");
        if ([model respondsToSelector:sel_ma])
            model_attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_ma);
        WARN("6d: modelAttributes = "
             << (model_attrs ? [[model_attrs description] UTF8String] : "nil"));

        // Output path for the compiled binary
        NSString* out_dir_d = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc6d_output"];
        [[NSFileManager defaultManager]
            createDirectoryAtPath:out_dir_d withIntermediateDirectories:YES
            attributes:nil error:nil];
        NSString* out_path_d = [out_dir_d
            stringByAppendingPathComponent:@"model.llir.bundle"];

        // Build options dict with all candidate output path keys
        NSMutableDictionary* compile_opts = [NSMutableDictionary dictionary];
        NSURL* out_url_d = [NSURL fileURLWithPath:out_path_d];
        compile_opts[@"aotModelBinaryPath"]    = out_url_d;
        compile_opts[@"ANEFAOTBinaryPath"]     = out_url_d;
        compile_opts[@"outputPath"]            = out_url_d;
        compile_opts[@"binaryOutputPath"]      = out_url_d;

        // Generate sandbox extension for model_dir (may or may not be needed)
        NSString* model_dir_ns = [NSString
            stringWithUTF8String:h->prog->model_dir.c_str()];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block BOOL compile_ok = NO;
        __block id   compile_result = nil;

        id proxy = [working_conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
            WARN("6d proxy error: " << [[e localizedDescription] UTF8String]);
            dispatch_semaphore_signal(sem);
        }];

        // compileModel:sandboxExtension:options:qos:withReply:
        // type: v52@0:8 @16(model) @24(sandboxExt) @32(options) I40(qos) @?44(reply)
        SEL sel_compile = sel_registerName(
            "compileModel:sandboxExtension:options:qos:withReply:");
        @try {
            typedef void (*CompileFn)(id, SEL, id, id, id, unsigned int,
                                     void(^)(id, NSError*));
            ((CompileFn)objc_msgSend)(proxy, sel_compile,
                model_attrs,   // model description dict
                nil,           // sandbox extension (try nil first)
                compile_opts,  // options with output path candidates
                21u,           // QoS DEFAULT
                ^(id result, NSError* err) {
                    compile_ok = (err == nil);
                    compile_result = result;
                    if (err)
                        WARN("6d compile reply error: "
                             << [[err localizedDescription] UTF8String]);
                    else
                        WARN("6d compile reply: "
                             << (result ? [[result description] UTF8String] : "nil"));
                    dispatch_semaphore_signal(sem);
                });
            dispatch_semaphore_wait(sem,
                dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC));
        } @catch (NSException* ex) {
            WARN("6d: exception: " << [[ex reason] UTF8String]);
        }

        // Scan for binary output
        BOOL bin_exists = [[NSFileManager defaultManager]
            fileExistsAtPath:out_path_d];
        WARN("6d: compile_ok=" << (int)compile_ok
             << "  binary_at_target=" << (int)bin_exists);

        if (bin_exists) {
            WARN("6d: *** BINARY CAPTURED at " << [out_path_d UTF8String] << " ***");
            NSDictionary* attrs = [[NSFileManager defaultManager]
                attributesOfItemAtPath:out_path_d error:nil];
            WARN("6d: binary size=" << [[attrs[NSFileSize] description] UTF8String]);
        } else {
            // Scan nearby locations
            WARN("6d: scanning temp dir for any new binary files...");
            NSArray* tmpContents = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString* entry in tmpContents) {
                if ([entry hasSuffix:@".llir.bundle"] ||
                    [entry hasSuffix:@".hwx"] ||
                    [entry hasSuffix:@".anecir"] ||
                    [entry hasSuffix:@".anec"]) {
                    WARN("6d: found " << [entry UTF8String]);
                }
            }
        }

        [working_conn invalidate];
        libane_mil_release(h);
    }
}

// ── XPC-7: proper NSXPCInterface setup + echo + compile with output path ─────
//
// XPC-6 findings:
//   Service:  com.apple.appleneuralengine  (confirmed from _ANEDaemonConnection)
//   Protocol: _ANEDaemonProtocol           (confirmed from NSXPCConnection.remoteObjectInterface)
//   6b failure: "unrecognized selector" is an NSXPCInterface encoding error —
//     the proxy rejects any method whose reply-block argument types haven't been
//     declared via setClasses:forSelector:argumentIndex:ofReply:.
//     This is a client-side encoding guard, not a server-side entitlement reject.
//   6c: unknown keys in compileWithQoS:options: cause compile failure — the
//     output path cannot be injected that way.
//
//   Fix: call setClasses:forSelector:argumentIndex:ofReply: for every method
//   before connecting, then retry echo + compile.

TEST_CASE("XPC-7: _ANEDaemonProtocol with proper NSXPCInterface block type setup",
          "[pathc][xpc][compile]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Protocol* daemon_proto = objc_getProtocol("_ANEDaemonProtocol");
        REQUIRE(daemon_proto != nil);

        // Build a properly configured NSXPCInterface.
        // For every method in the protocol that has a reply block (@?),
        // we must declare what classes are allowed in each block argument.
        // Without this, the proxy throws "unrecognized selector" locally.
        NSXPCInterface* iface =
            [NSXPCInterface interfaceWithProtocol:daemon_proto];

        // Helper: declare that a reply block's Nth arg carries a set of classes
        auto setReplyArg = [&](const char* selStr, int argIdx,
                                NSArray* classes) {
            SEL s = sel_registerName(selStr);
            NSSet* cls_set = [NSSet setWithArray:classes];
            @try {
                [iface setClasses:cls_set
                      forSelector:s
                    argumentIndex:argIdx
                           ofReply:YES];
            } @catch (...) {}
        };

        // echo:withReply: — reply(NSString*)
        setReplyArg("echo:withReply:", 0, @[[NSString class]]);

        // compiledModelExistsFor:withReply: — reply(BOOL or NSNumber)
        setReplyArg("compiledModelExistsFor:withReply:", 0,
                    @[[NSNumber class]]);
        setReplyArg("compiledModelExistsMatchingHash:withReply:", 0,
                    @[[NSNumber class]]);

        // compileModel:sandboxExtension:options:qos:withReply: — reply(NSError*)
        setReplyArg("compileModel:sandboxExtension:options:qos:withReply:", 0,
                    @[[NSError class]]);

        // loadModel:sandboxExtension:options:qos:withReply: — reply(NSError*)
        setReplyArg("loadModel:sandboxExtension:options:qos:withReply:", 0,
                    @[[NSError class]]);

        // loadModelNewInstance:options:modelInstParams:qos:withReply:
        setReplyArg("loadModelNewInstance:options:modelInstParams:qos:withReply:",
                    0, @[[NSError class]]);

        // purgeCompiledModel:withReply: / purgeCompiledModelMatchingHash:withReply:
        setReplyArg("purgeCompiledModel:withReply:", 0, @[[NSError class]]);
        setReplyArg("purgeCompiledModelMatchingHash:withReply:", 0,
                    @[[NSError class]]);

        // unloadModel:options:qos:withReply:
        setReplyArg("unloadModel:options:qos:withReply:", 0, @[[NSError class]]);

        // prepareChainingWithModel:options:chainingReq:qos:withReply:
        setReplyArg("prepareChainingWithModel:options:chainingReq:qos:withReply:",
                    0, @[[NSError class]]);

        // beginRealTimeTaskWithReply: / endRealTimeTaskWithReply:
        setReplyArg("beginRealTimeTaskWithReply:", 0, @[[NSError class]]);
        setReplyArg("endRealTimeTaskWithReply:", 0, @[[NSError class]]);

        // Connect to the confirmed service
        NSXPCConnection* conn = [[NSXPCConnection alloc]
            initWithMachServiceName:@"com.apple.appleneuralengine"
                            options:0];
        REQUIRE(conn != nil);
        conn.remoteObjectInterface = iface;

        __block BOOL invalidated = NO;
        conn.interruptionHandler  = ^{ WARN("XPC-7: interrupted"); };
        conn.invalidationHandler  = ^{ invalidated = YES;
                                       WARN("XPC-7: invalidated"); };
        [conn resume];

        id proxy = [conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
            WARN("XPC-7 proxy error: " << [[e localizedDescription] UTF8String]);
        }];
        REQUIRE(proxy != nil);
        WARN("XPC-7: proxy obtained — " << class_getName([proxy class]));

        // ── 7a: echo ─────────────────────────────────────────────────────────
        WARN("\n--- 7a: echo:withReply: ---");
        {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block NSString* echo_reply = nil;
            SEL s = sel_registerName("echo:withReply:");
            @try {
                typedef void (*EchoFn)(id, SEL, NSString*, void(^)(NSString*));
                ((EchoFn)objc_msgSend)(proxy, s, @"ping-libane",
                    ^(NSString* r) {
                        echo_reply = r;
                        dispatch_semaphore_signal(sem);
                    });
                long rc = dispatch_semaphore_wait(sem,
                    dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
                if (rc != 0) WARN("7a: echo timed out");
            } @catch (NSException* ex) {
                WARN("7a: exception: " << [[ex reason] UTF8String]);
                dispatch_semaphore_signal(sem);
            }

            if (echo_reply) {
                WARN("7a: *** ECHO SUCCEEDED: " << [echo_reply UTF8String]
                     << " — daemon channel is live! ***");
            } else {
                WARN("7a: echo returned nil (service may require entitlement)");
            }
        }

        if (invalidated) {
            WARN("XPC-7: connection invalidated after echo — likely entitlement");
            [conn invalidate];
            return;
        }

        // ── 7b: compiledModelExistsMatchingHash: (read-only sanity check) ────
        WARN("\n--- 7b: compiledModelExistsMatchingHash: ---");
        {
            // Compile a model to get its hexID
            auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            if (!h || !h->prog || !h->prog->objc_model) {
                WARN("7b: libane compile failed"); if (h) libane_mil_release(h);
                goto xpc7_done;
            }
            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                (id)h->prog->objc_model, sel_registerName("hexStringIdentifier"));
            WARN("7b: checking hexID " << [hex_id UTF8String]);

            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block id exists_result = nil;
            SEL s = sel_registerName("compiledModelExistsMatchingHash:withReply:");
            @try {
                typedef void (*CmeFn)(id, SEL, NSString*, void(^)(id));
                ((CmeFn)objc_msgSend)(proxy, s, hex_id,
                    ^(id result) {
                        exists_result = result;
                        dispatch_semaphore_signal(sem);
                    });
                long rc = dispatch_semaphore_wait(sem,
                    dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
                if (rc != 0) WARN("7b: timed out");
            } @catch (NSException* ex) {
                WARN("7b: exception: " << [[ex reason] UTF8String]);
                dispatch_semaphore_signal(sem);
            }

            if (exists_result)
                WARN("7b: compiledModelExistsMatchingHash: → "
                     << [[exists_result description] UTF8String]);
            else
                WARN("7b: no result (timeout or error)");

            libane_mil_release(h);
        }

        if (invalidated) {
            WARN("XPC-7: connection invalidated after 7b — entitlement wall");
            [conn invalidate];
            return;
        }

        // ── 7c: compileModel:sandboxExtension:options:qos:withReply: ─────────
        WARN("\n--- 7c: compileModel with aotModelBinaryPath in options ---");
        {
            auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
            if (!h || !h->prog) {
                WARN("7c: compile failed"); if (h) libane_mil_release(h);
                goto xpc7_done;
            }
            id model = (id)h->prog->objc_model;

            // Get modelAttributes — the NSDictionary the daemon expects
            id model_attrs = nil;
            SEL sel_ma = sel_registerName("modelAttributes");
            if ([model respondsToSelector:sel_ma])
                model_attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_ma);
            if (!model_attrs) {
                WARN("7c: modelAttributes nil — trying compilerOptionsWithOptions:isCompiledModelCached:");
                // Fall back to the full options dict
                SEL sel_co = sel_registerName(
                    "compilerOptionsWithOptions:isCompiledModelCached:");
                if ([model respondsToSelector:sel_co])
                    model_attrs = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                        model, sel_co, @{}, NO);
            }
            WARN("7c: model_attrs = "
                 << (model_attrs ? [[model_attrs description] UTF8String] : "nil"));

            // Target output path
            NSString* out_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc7c_output"];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:out_dir withIntermediateDirectories:YES
                attributes:nil error:nil];
            NSString* out_path = [out_dir
                stringByAppendingPathComponent:@"model.llir.bundle"];
            NSURL* out_url = [NSURL fileURLWithPath:out_path];

            NSMutableDictionary* opts = [NSMutableDictionary dictionary];
            opts[@"aotModelBinaryPath"] = out_url;
            opts[@"ANEFAOTBinaryPath"]  = out_url;

            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block NSError* compile_err = nil;
            __block BOOL    compile_called = NO;

            SEL sel_compile = sel_registerName(
                "compileModel:sandboxExtension:options:qos:withReply:");
            @try {
                typedef void (*CompFn)(id, SEL, id, id, id, unsigned int,
                                       void(^)(NSError*));
                ((CompFn)objc_msgSend)(proxy, sel_compile,
                    model_attrs,  // model description
                    nil,          // sandbox extension (nil first)
                    opts,
                    21u,          // QOS_CLASS_DEFAULT
                    ^(NSError* err) {
                        compile_called = YES;
                        compile_err    = err;
                        dispatch_semaphore_signal(sem);
                    });
                long rc = dispatch_semaphore_wait(sem,
                    dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC));
                if (rc != 0) WARN("7c: timed out waiting for compile reply");
            } @catch (NSException* ex) {
                WARN("7c: exception: " << [[ex reason] UTF8String]);
                dispatch_semaphore_signal(sem);
            }

            BOOL bin_exists = [[NSFileManager defaultManager]
                fileExistsAtPath:out_path];
            WARN("7c: compile_called=" << (int)compile_called
                 << "  err=" << (compile_err
                    ? [[compile_err localizedDescription] UTF8String] : "none")
                 << "  binary_at_target=" << (int)bin_exists);

            if (bin_exists) {
                NSDictionary* attrs = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:out_path error:nil];
                WARN("7c: *** BINARY CAPTURED ("
                     << [[attrs[NSFileSize] description] UTF8String]
                     << " bytes) — cross-op patching restored! ***");
            }

            // Also scan tmp for any new binary files
            NSArray* tmp = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString* f in tmp) {
                if ([f hasSuffix:@".llir.bundle"] || [f hasSuffix:@".hwx"] ||
                    [f hasSuffix:@".anecir"]      || [f hasSuffix:@".anec"])
                    WARN("7c: found in /tmp: " << [f UTF8String]);
            }

            libane_mil_release(h);
        }

        xpc7_done:
        [conn invalidate];
    }
}

// ── XPC-8: Reuse _ANEClient's existing authenticated proxy ──────────────────
//
// New hypothesis: the "Couldn't communicate" in XPC-7 was NOT an entitlement
// wall — it was a malformed XPC message caused by our NSXPCInterface setup
// being wrong (different from what _ANEClient actually configured).
//
// Instead of building a new connection that has to re-authenticate, we reach
// into _ANEClient → _ANEDaemonConnection → _daemonConnection (NSXPCConnection)
// and call [remoteObjectProxy ...] on the ALREADY AUTHENTICATED channel.
// _ANEClient set up the interface correctly; we just use what it built.
//
// Goal: call compileModel:sandboxExtension:options:qos:withReply: on the live
// proxy, passing options with candidate output-path keys, then check whether
// the compiled binary lands somewhere on disk.

TEST_CASE("XPC-8: reuse _ANEClient authenticated proxy for compile + binary capture",
          "[pathc][xpc][compile][xpc8]") {
    @autoreleasepool {
        // Ensure _ANEClient is initialized and connected.
        auto* seed = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!seed) { WARN("XPC-8: seed compile failed — " << libane_last_error()); return; }
        libane_set_log_level(LIBANE_LOG_SILENT);

        // ── Step 1: get _ANEClient's NSXPCConnection ─────────────────────────
        Class cls_client = NSClassFromString(@"_ANEClient");
        if (!cls_client) { WARN("XPC-8: _ANEClient not found"); libane_mil_release(seed); return; }

        SEL sel_shared = sel_registerName("sharedConnection");
        id client = ((id(*)(Class,SEL))objc_msgSend)(cls_client, sel_shared);
        if (!client) { WARN("XPC-8: sharedConnection nil"); libane_mil_release(seed); return; }

        // Walk _ANEClient ivars → find _ANEDaemonConnection
        id daemon_conn_obj = nil;
        {
            Class c = cls_client;
            while (c && c != [NSObject class] && !daemon_conn_obj) {
                unsigned int cnt = 0;
                Ivar* ivars = class_copyIvarList(c, &cnt);
                for (unsigned int i = 0; i < cnt; ++i) {
                    const char* name = ivar_getName(ivars[i]);
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    if (!name || !type || type[0] != '@') continue;
                    if (strstr(name, "conn") || strstr(name, "Conn")) {
                        ptrdiff_t off = ivar_getOffset(ivars[i]);
                        void** slot = (void**)((uint8_t*)(__bridge void*)client + off);
                        id val = (__bridge id)*slot;
                        if (val) { daemon_conn_obj = val; break; }
                    }
                }
                free(ivars);
                c = class_getSuperclass(c);
            }
        }

        if (!daemon_conn_obj) {
            WARN("XPC-8: no _ANEDaemonConnection in _ANEClient");
            libane_mil_release(seed); return;
        }
        WARN("XPC-8: daemon_conn_obj class = " << class_getName([daemon_conn_obj class]));

        // Walk _ANEDaemonConnection ivars → find NSXPCConnection
        NSXPCConnection* live_conn = nil;
        {
            Class c = [daemon_conn_obj class];
            while (c && c != [NSObject class] && !live_conn) {
                unsigned int cnt = 0;
                Ivar* ivars = class_copyIvarList(c, &cnt);
                for (unsigned int i = 0; i < cnt; ++i) {
                    const char* type = ivar_getTypeEncoding(ivars[i]);
                    if (!type || type[0] != '@') continue;
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    @try {
                        void** slot = (void**)((uint8_t*)(__bridge void*)daemon_conn_obj + off);
                        id val = (__bridge id)*slot;
                        if (val && [val isKindOfClass:[NSXPCConnection class]])
                            live_conn = (NSXPCConnection*)val;
                    } @catch (...) {}
                }
                free(ivars);
                c = class_getSuperclass(c);
            }
        }

        if (!live_conn) {
            WARN("XPC-8: no NSXPCConnection found in _ANEDaemonConnection");
            libane_mil_release(seed); return;
        }
        WARN("XPC-8: live_conn serviceName = "
             << [[live_conn serviceName] UTF8String]);
        WARN("XPC-8: remoteObjectInterface protocol = "
             << [[live_conn.remoteObjectInterface.protocol description] UTF8String]);

        // ── Step 2: get the existing proxy ────────────────────────────────────
        __block BOOL proxy_err = NO;
        id proxy = [live_conn remoteObjectProxyWithErrorHandler:^(NSError* e) {
            proxy_err = YES;
            WARN("XPC-8 proxy error: " << [[e localizedDescription] UTF8String]);
        }];

        if (!proxy) { WARN("XPC-8: proxy nil"); libane_mil_release(seed); return; }
        WARN("XPC-8: proxy class = " << class_getName([proxy class]));

        // ── Step 3: compiledModelExistsMatchingHash: (read-only sanity) ──────
        WARN("\n--- 8a: compiledModelExistsMatchingHash: ---");
        if (seed->prog && seed->prog->objc_model) {
            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                (id)seed->prog->objc_model,
                sel_registerName("hexStringIdentifier"));
            WARN("8a: hexID = " << [hex_id UTF8String]);

            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block id exists_result = nil;
            SEL sel_cme = sel_registerName("compiledModelExistsMatchingHash:withReply:");
            @try {
                typedef void (*CmeFn)(id, SEL, NSString*, void(^)(id));
                ((CmeFn)objc_msgSend)(proxy, sel_cme, hex_id, ^(id r) {
                    exists_result = r;
                    dispatch_semaphore_signal(sem);
                });
                long rc = dispatch_semaphore_wait(sem,
                    dispatch_time(DISPATCH_TIME_NOW, 8 * NSEC_PER_SEC));
                if (rc != 0) WARN("8a: timed out");
            } @catch (NSException* ex) {
                WARN("8a: exception: " << [[ex reason] UTF8String]);
                dispatch_semaphore_signal(sem);
            }
            if (exists_result)
                WARN("8a: *** EXISTS = " << [[exists_result description] UTF8String]
                     << " — proxy is live! ***");
            else
                WARN("8a: no result");
        }

        if (proxy_err) { WARN("XPC-8: proxy error — aborting"); libane_mil_release(seed); return; }

        // ── Step 4: compileModel: with output path options ────────────────────
        WARN("\n--- 8b: compileModel with output path candidates ---");
        {
            // modelAttributes from the seed model
            id model = (id)seed->prog->objc_model;
            id model_attrs = nil;
            SEL sel_ma = sel_registerName("modelAttributes");
            if ([model respondsToSelector:sel_ma])
                model_attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_ma);
            if (!model_attrs) { WARN("8b: modelAttributes nil"); goto xpc8_done; }

            // Target output path
            NSString* out_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc8_binary_output"];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:out_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSString* out_path = [out_dir
                stringByAppendingPathComponent:@"model.llir"];
            NSURL*    out_url  = [NSURL fileURLWithPath:out_path];
            WARN("8b: target output path: " << [out_path UTF8String]);

            // Try several candidate key names for the binary output location
            NSArray* key_candidates = @[
                @"aotModelBinaryPath",
                @"ANEFAOTBinaryPath",
                @"kANEFAOTBinaryPath",
                @"outputModelPath",
                @"compiledModelBinaryPath",
                @"binaryPath",
                @"ANEFOutputPath",
            ];

            for (NSString* key in key_candidates) {
                // Remove any file from previous iteration
                [[NSFileManager defaultManager] removeItemAtPath:out_path error:nil];

                NSMutableDictionary* opts = [NSMutableDictionary dictionary];
                opts[key] = out_url;

                dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                __block NSError* compile_err = nil;
                __block BOOL    compile_called = NO;

                SEL sel_compile = sel_registerName(
                    "compileModel:sandboxExtension:options:qos:withReply:");
                @try {
                    typedef void (*CompFn)(id, SEL, id, id, id, unsigned int,
                                           void(^)(NSError*));
                    ((CompFn)objc_msgSend)(proxy, sel_compile,
                        model_attrs,
                        nil,      // no sandbox extension
                        opts,
                        21u,      // QOS_CLASS_DEFAULT
                        ^(NSError* err) {
                            compile_called = YES;
                            compile_err    = err;
                            dispatch_semaphore_signal(sem);
                        });
                    long rc = dispatch_semaphore_wait(sem,
                        dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC));
                    if (rc != 0) WARN("8b[" << [key UTF8String] << "]: timed out");
                } @catch (NSException* ex) {
                    WARN("8b[" << [key UTF8String] << "]: exception — "
                         << [[ex reason] UTF8String]);
                    dispatch_semaphore_signal(sem);
                }

                BOOL bin_exists = [[NSFileManager defaultManager]
                    fileExistsAtPath:out_path];
                NSString* err_str = compile_err
                    ? [compile_err localizedDescription] : @"none";

                WARN("8b key=" << [key UTF8String]
                     << "  called=" << (int)compile_called
                     << "  err=" << [err_str UTF8String]
                     << "  bin=" << (int)bin_exists);

                if (bin_exists) {
                    NSDictionary* attrs = [[NSFileManager defaultManager]
                        attributesOfItemAtPath:out_path error:nil];
                    WARN("*** BINARY CAPTURED at " << [out_path UTF8String]
                         << " (" << [[attrs[NSFileSize] description] UTF8String]
                         << " bytes) — cross-op patching restored! ***");
                }

                if (proxy_err) { WARN("8b: proxy error — stopping"); break; }
            }

            // Broad scan: any new .llir, .hwx, .anecir, .anec in /tmp or model_dir
            WARN("\n--- 8c: scan /tmp for any compiled binary ---");
            NSArray* tmp_contents = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString* f in tmp_contents) {
                if ([f hasSuffix:@".llir"]        || [f hasSuffix:@".llir.bundle"] ||
                    [f hasSuffix:@".hwx"]          || [f hasSuffix:@".anecir"] ||
                    [f hasSuffix:@".anec"]          || [f hasSuffix:@".espresso.net"])
                    WARN("8c: /tmp/" << [f UTF8String]);
            }
            // Also scan the model_dir itself if we know it
            if (seed->prog && !seed->prog->model_dir.empty()) {
                WARN("8c: model_dir = " << seed->prog->model_dir);
                NSString* md = [NSString stringWithUTF8String:seed->prog->model_dir.c_str()];
                NSArray* md_contents = [[NSFileManager defaultManager]
                    contentsOfDirectoryAtPath:md error:nil];
                for (NSString* f in md_contents)
                    WARN("8c:   model_dir/" << [f UTF8String]);
            }
        }

        xpc8_done:
        libane_mil_release(seed);
    }
}

// ── XPC-9b: _ANEInMemoryModel method probe — localModelPath, saveModelFiles ──
//
// After a successful compile, these model methods may reveal where the compiled
// binary lives or allow us to save it to a readable location.

TEST_CASE("XPC-9b: _ANEInMemoryModel promising methods",
          "[pathc][xpc][xpc9b]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        auto* h = libane_mil_compile(kReluMIL, nullptr, nullptr, nullptr, 0);
        if (!h || !h->prog || !h->prog->objc_model) {
            WARN("9b: compile failed"); if (h) libane_mil_release(h); return;
        }
        id model = (id)h->prog->objc_model;
        WARN("9b: model class = " << class_getName([model class]));

        // programHandle/program may return raw C pointers — keep them out of
        // the ObjC-call loop; probe their type encodings separately below.
        NSArray* sel_names = @[
            @"localModelPath",
            @"saveModelFiles",
            @"string_id",
            @"descriptor",
            @"modelURL",
            @"model",
        ];

        for (NSString* sn in sel_names) {
            SEL s = sel_registerName([sn UTF8String]);
            if (![model respondsToSelector:s]) {
                WARN("9b: " << [sn UTF8String] << " → (not found)");
                continue;
            }
            @try {
                id r = ((id(*)(id,SEL))objc_msgSend)(model, s);
                if (!r) {
                    WARN("9b: " << [sn UTF8String] << " → nil");
                } else {
                    NSString* desc = [r description];
                    if ([desc length] > 200)
                        desc = [[desc substringToIndex:200]
                            stringByAppendingString:@"..."];
                    WARN("9b: " << [sn UTF8String] << " → "
                         << class_getName([r class]) << ": " << [desc UTF8String]);
                }
            } @catch(NSException* ex) {
                WARN("9b: " << [sn UTF8String] << " → exception: "
                     << [[ex reason] UTF8String]);
            }
        }

        // programHandle type encoding (may be a raw C pointer, not ObjC)
        WARN("\n--- 9b: programHandle type encoding ---");
        {
            Method m = class_getInstanceMethod([model class],
                sel_registerName("programHandle"));
            if (m) {
                WARN("9b: programHandle type = " << method_getTypeEncoding(m));
            }
            // Also check intermediateBufferHandle type
            m = class_getInstanceMethod([model class],
                sel_registerName("intermediateBufferHandle"));
            if (m) {
                WARN("9b: intermediateBufferHandle type = " << method_getTypeEncoding(m));
            }
        }

        // saveModelFiles: check if any new files appear
        WARN("\n--- 9b: saveModelFiles scan ---");
        {
            SEL s = sel_registerName("saveModelFiles");
            if ([model respondsToSelector:s]) {
                NSArray* before = [[NSFileManager defaultManager]
                    contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
                NSSet* before_set = [NSSet setWithArray:before];

                @try {
                    id r = ((id(*)(id,SEL))objc_msgSend)(model, s);
                    WARN("9b: saveModelFiles → " << (r ? [[r description] UTF8String] : "nil"));
                } @catch(NSException* ex) {
                    WARN("9b: saveModelFiles exception: " << [[ex reason] UTF8String]);
                }

                NSArray* after = [[NSFileManager defaultManager]
                    contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
                for (NSString* f in after) {
                    if (![before_set containsObject:f])
                        WARN("9b: NEW /tmp/" << [f UTF8String]);
                }
                if ([after count] == [before count])
                    WARN("9b: saveModelFiles: no new files in /tmp");
            }
        }

        // Inner _ANEModel — enumerate its methods and check its modelURL directory
        WARN("\n--- 9b: inner _ANEModel ---");
        {
            SEL s = sel_registerName("model");
            if ([model respondsToSelector:s]) {
                id inner = ((id(*)(id,SEL))objc_msgSend)(model, s);
                if (inner) {
                    WARN("9b: _ANEModel class = " << class_getName([inner class]));

                    // Methods
                    Class c = [inner class];
                    while (c && c != [NSObject class]) {
                        unsigned int cnt = 0;
                        Method* ms = class_copyMethodList(c, &cnt);
                        for (unsigned int i = 0; i < cnt; ++i)
                            WARN("9b:   -" << sel_getName(method_getName(ms[i])));
                        free(ms);
                        c = class_getSuperclass(c);
                    }

                    // Probe key identifier methods — first log type encodings,
                    // then only call ObjC-returning ones (type starts with '@')
                    NSArray* inner_sels = @[
                        @"cacheURLIdentifier",
                        @"getCacheURLIdentifier",
                        @"identifierSource",
                        @"sourceURL",
                        @"key",
                        @"UUID",
                        @"getUUID",
                        @"string_id",
                        @"l",
                    ];
                    for (NSString* isn in inner_sels) {
                        SEL is2 = sel_registerName([isn UTF8String]);
                        Method m2 = class_getInstanceMethod([inner class], is2);
                        if (!m2) continue;
                        const char* enc = method_getTypeEncoding(m2);
                        WARN("9b: _ANEModel." << [isn UTF8String]
                             << " enc=" << (enc ? enc : "?"));
                        // Only call if return type is @ (ObjC object)
                        if (!enc || enc[0] != '@') continue;
                        @try {
                            id ir = ((id(*)(id,SEL))objc_msgSend)(inner, is2);
                            WARN("9b: _ANEModel." << [isn UTF8String] << " → "
                                 << (ir ? class_getName([ir class]) : "nil") << ": "
                                 << (ir ? [[ir description] UTF8String] : "nil"));
                        } @catch (...) {
                            WARN("9b: _ANEModel." << [isn UTF8String] << " → exception");
                        }
                    }

                    // What's at the _ANEModel.modelURL directory?
                    SEL su = sel_registerName("modelURL");
                    if ([inner respondsToSelector:su]) {
                        NSURL* u = ((NSURL*(*)(id,SEL))objc_msgSend)(inner, su);
                        WARN("9b: _ANEModel.modelURL = " << [[u absoluteString] UTF8String]);
                        NSString* path = [u path];
                        NSArray* contents = [[NSFileManager defaultManager]
                            contentsOfDirectoryAtPath:path error:nil];
                        if (contents) {
                            WARN("9b: _ANEModel dir contents:");
                            for (NSString* f in contents) {
                                NSString* fp = [path stringByAppendingPathComponent:f];
                                BOOL isd = NO;
                                [[NSFileManager defaultManager]
                                    fileExistsAtPath:fp isDirectory:&isd];
                                NSDictionary* attrs = [[NSFileManager defaultManager]
                                    attributesOfItemAtPath:fp error:nil];
                                long long sz = [attrs[NSFileSize] longLongValue];
                                WARN("9b:   " << [f UTF8String]
                                     << "  (" << sz << " bytes, dir=" << (int)isd << ")");
                                if (isd) {
                                    NSArray* sub = [[NSFileManager defaultManager]
                                        contentsOfDirectoryAtPath:fp error:nil];
                                    for (NSString* sf in sub) {
                                        NSString* sfp = [fp stringByAppendingPathComponent:sf];
                                        NSDictionary* sa = [[NSFileManager defaultManager]
                                            attributesOfItemAtPath:sfp error:nil];
                                        WARN("9b:     " << [sf UTF8String]
                                             << "  (" << [sa[NSFileSize] longLongValue]
                                             << " bytes)");
                                    }
                                }
                            }
                        } else {
                            WARN("9b: _ANEModel dir not accessible or empty");
                        }
                    }
                }
            }
        }

        libane_mil_release(h);
    }
}

// ── XPC-9c: compile via _ANEClient with empty options, scan for binary ────────
//
// Hypothesis: after a successful compile, the daemon writes the compiled binary
// somewhere on disk.  We call compileModel:options:qos:error: with no output
// path (so no sandbox extension issue) and then scan:
//   (a) the model_dir
//   (b) ANECompilerService cache dirs
//   (c) all /tmp entries with binary suffixes

TEST_CASE("XPC-9c: compile via _ANEClient, scan for written binary",
          "[pathc][xpc][xpc9c]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls_client = NSClassFromString(@"_ANEClient");
        Class cls_Desc   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model  = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_client || !cls_Desc || !cls_Model) {
            WARN("9c: missing required classes"); return;
        }

        id client = ((id(*)(Class,SEL))objc_msgSend)(
            cls_client, sel_registerName("sharedConnection"));
        if (!client) { WARN("9c: sharedConnection nil"); return; }

        // Snapshot /tmp before compile
        NSFileManager* fm = [NSFileManager defaultManager];
        NSArray* before_tmp = [fm contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
        NSSet* before_set = [NSSet setWithArray:before_tmp];

        // Use a unique MIL (different name) so we're sure this is a cold compile
        const char* kMIL_9c =
            "program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}"
            "})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1,8,1,64]> x) {\n"
            "        tensor<fp16, [1,8,1,64]> y = relu(x=x)[name=string(\"xpc9c\")];\n"
            "    } -> (y);\n"
            "}\n";

        // Build descriptor + model
        NSData* mil_data = [NSData dataWithBytes:kMIL_9c length:strlen(kMIL_9c)];
        id desc = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
            cls_Desc, sel_registerName("modelWithMILText:weights:optionsPlist:"),
            mil_data, @{}, nil);
        if (!desc) { WARN("9c: descriptor nil"); return; }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc);
        if (!model) { WARN("9c: model nil"); return; }

        NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
            model, sel_registerName("hexStringIdentifier"));
        WARN("9c: hexID = " << [hex_id UTF8String]);

        // Write model_dir (required by compile path)
        NSString* model_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:hex_id];
        [fm createDirectoryAtPath:model_dir withIntermediateDirectories:YES
            attributes:nil error:nil];
        [mil_data writeToFile:
            [model_dir stringByAppendingPathComponent:@"model.mil"]
            atomically:YES];

        // Compile via _ANEInMemoryModel.compileWithQoS:options:error:
        // First with empty options (baseline), then with output path options.
        NSError* err = nil;
        SEL sel_compile = sel_registerName("compileWithQoS:options:error:");

        // Baseline: empty options
        BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model, sel_compile, 21u, @{}, &err);
        WARN("9c: compileWithQoS (empty opts) ok=" << (int)ok
             << "  err=" << (err ? [[err localizedDescription] UTF8String] : "none"));

        // Now try: use _ANEInMemoryModel.sharedConnection (may differ from
        // _ANEClient.sharedConnection) and call compileModel: with output path.
        WARN("\n--- 9c: _ANEInMemoryModel.sharedConnection client ---");
        {
            SEL sel_sc = sel_registerName("sharedConnection");
            id imm_client = nil;
            if ([cls_Model respondsToSelector:sel_sc]) {
                imm_client = ((id(*)(Class,SEL))objc_msgSend)(cls_Model, sel_sc);
            }
            WARN("9c: _ANEInMemoryModel.sharedConnection = "
                 << (imm_client ? class_getName([imm_client class]) : "nil")
                 << (imm_client == client ? " (same as _ANEClient.shared)" : " (DIFFERENT)"));

            if (imm_client && imm_client != client) {
                NSString* out_file2 = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:@"xpc9c_out2.llir"];
                [@"" writeToFile:out_file2 atomically:NO
                        encoding:NSUTF8StringEncoding error:nil];
                NSURL* out_url2 = [NSURL fileURLWithPath:out_file2];

                for (NSString* key in @[@"aotModelBinaryPath", @"ANEFAOTBinaryPath"]) {
                    NSError* err3 = nil;
                    BOOL ok3 = ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(
                        imm_client,
                        sel_registerName("compileModel:options:qos:error:"),
                        model, @{key: out_url2}, 21u, &err3);
                    NSDictionary* a3 = [[NSFileManager defaultManager]
                        attributesOfItemAtPath:out_file2 error:nil];
                    long long sz3 = [a3[NSFileSize] longLongValue];
                    WARN("9c imm_client key=" << [key UTF8String]
                         << "  ok=" << (int)ok3
                         << "  err=" << (err3 ? [[err3 localizedDescription] UTF8String] : "none")
                         << "  sz=" << sz3);
                    if (sz3 > 0)
                        WARN("*** BINARY via imm_client! ***");
                }
            }
        }

        // Also try: method swizzle _ANEClient.compileModel: to log options dict
        // and inject output path before the sandbox extension check
        WARN("\n--- 9c: compileWithQoS with output path options (fresh model) ---");
        {
            // Use fresh model (no cached compile) with different MIL
            const char* kMIL_fresh =
                "program(1.3)\n[buildInfo = dict<string, string>({"
                "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, "
                "{\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n    func main<ios18>(tensor<fp16, [1,16,1,32]> x) {\n"
                "        tensor<fp16, [1,16,1,32]> y = relu(x=x)[name=string(\"fresh\")];\n"
                "    } -> (y);\n}\n";

            NSData* md2 = [NSData dataWithBytes:kMIL_fresh length:strlen(kMIL_fresh)];
            id desc2 = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
                cls_Desc, sel_registerName("modelWithMILText:weights:optionsPlist:"),
                md2, @{}, nil);
            id model2 = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc2);
            if (!model2) { WARN("9c: model2 nil"); return; }

            NSString* hex2 = ((NSString*(*)(id,SEL))objc_msgSend)(
                model2, sel_registerName("hexStringIdentifier"));
            NSString* md2_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:hex2];
            [[NSFileManager defaultManager] createDirectoryAtPath:md2_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            [md2 writeToFile:[md2_dir stringByAppendingPathComponent:@"model.mil"]
                atomically:YES];
            WARN("9c: fresh model hexID=" << [hex2 UTF8String]);

            NSString* out_fresh = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc9c_fresh.llir"];
            [@"" writeToFile:out_fresh atomically:NO encoding:NSUTF8StringEncoding error:nil];
            NSURL* out_fresh_url = [NSURL fileURLWithPath:out_fresh];

            // Snapshot model_dir before compile
            NSString* fresh_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:hex2];
            NSArray* fresh_before = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:fresh_dir error:nil];

            // Try output path = file within the model_dir (daemon already has
            // access to this dir from reading the MIL source)
            NSString* out_in_dir = [fresh_dir
                stringByAppendingPathComponent:@"compiled.llir"];
            NSURL* out_in_dir_url = [NSURL fileURLWithPath:out_in_dir];
            [@"" writeToFile:out_in_dir atomically:NO
                    encoding:NSUTF8StringEncoding error:nil];

            for (NSString* key in @[@"aotModelBinaryPath", @"ANEFAOTBinaryPath",
                                    @"outputPath", @"compiledModelBinaryPath"]) {
                [[NSFileManager defaultManager] removeItemAtPath:out_in_dir error:nil];
                [@"" writeToFile:out_in_dir atomically:NO
                        encoding:NSUTF8StringEncoding error:nil];
                NSError* ef = nil;
                BOOL okf = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
                    model2, sel_registerName("compileWithQoS:options:error:"),
                    21u, @{key: out_in_dir_url}, &ef);
                NSDictionary* af = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:out_in_dir error:nil];
                long long szf = [af[NSFileSize] longLongValue];
                WARN("9c fresh key=" << [key UTF8String]
                     << "  ok=" << (int)okf
                     << "  err=" << (ef ? [[ef localizedDescription] UTF8String] : "none")
                     << "  sz=" << szf);
                if (szf > 0)
                    WARN("*** BINARY via fresh compileWithQoS! key=" << [key UTF8String]
                         << " (" << szf << " bytes) ***");
            }

            // Scan model_dir for new files after compile
            NSArray* fresh_after = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:fresh_dir error:nil];
            NSSet* fresh_before_set = [NSSet setWithArray:fresh_before];
            WARN("9c: model_dir contents after compile-with-opts:");
            for (NSString* f in fresh_after) {
                NSString* fp = [fresh_dir stringByAppendingPathComponent:f];
                NSDictionary* attrs = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:fp error:nil];
                BOOL is_new = ![fresh_before_set containsObject:f];
                WARN("9c:   " << [f UTF8String]
                     << "  (" << [attrs[NSFileSize] longLongValue] << "b)"
                     << (is_new ? "  *** NEW ***" : ""));
            }
        }

        // After compile: scan model_dir for new files
        WARN("\n--- 9c: model_dir after compile ---");
        NSArray* md_contents = [fm contentsOfDirectoryAtPath:model_dir error:nil];
        for (NSString* f in md_contents) {
            NSString* fp = [model_dir stringByAppendingPathComponent:f];
            BOOL isd = NO; [fm fileExistsAtPath:fp isDirectory:&isd];
            NSDictionary* attrs = [fm attributesOfItemAtPath:fp error:nil];
            WARN("9c:   " << [f UTF8String]
                 << "  (" << [attrs[NSFileSize] longLongValue] << "b, dir=" << (int)isd << ")");
        }

        // Scan for new /tmp entries
        WARN("\n--- 9c: new /tmp entries ---");
        NSArray* after_tmp = [fm contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
        for (NSString* f in after_tmp) {
            if ([before_set containsObject:f]) continue;
            NSString* fp = [NSTemporaryDirectory() stringByAppendingPathComponent:f];
            BOOL isd = NO; [fm fileExistsAtPath:fp isDirectory:&isd];
            NSDictionary* attrs = [fm attributesOfItemAtPath:fp error:nil];
            WARN("9c: NEW " << [f UTF8String]
                 << "  (" << [attrs[NSFileSize] longLongValue] << "b, dir=" << (int)isd << ")");
        }
        if ([after_tmp count] == [before_tmp count])
            WARN("9c: no new /tmp entries after compile");

        // Scan ANECompilerService cache
        WARN("\n--- 9c: ANECompilerService cache ---");
        NSString* ane_cache = [NSHomeDirectory()
            stringByAppendingPathComponent:
                @"Library/Caches/com.apple.ANECompilerService"];
        NSArray* ane_contents = [fm contentsOfDirectoryAtPath:ane_cache error:nil];
        if ([ane_contents count] == 0)
            WARN("9c: ANECompilerService cache empty");
        for (NSString* f in ane_contents)
            WARN("9c: cache: " << [f UTF8String]);

        // Broader scan for any .llir, .hwx, .anecir files in all temp locations
        WARN("\n--- 9c: broad binary suffix scan ---");
        NSArray* scan_dirs = @[
            NSTemporaryDirectory(),
            [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"],
            @"/Library/Caches",
        ];
        for (NSString* dir in scan_dirs) {
            NSArray* items = [fm contentsOfDirectoryAtPath:dir error:nil];
            for (NSString* f in items) {
                if ([f hasSuffix:@".llir"] || [f hasSuffix:@".llir.bundle"] ||
                    [f hasSuffix:@".hwx"]  || [f hasSuffix:@".anecir"] ||
                    [f hasSuffix:@".anec"] || [f hasSuffix:@".espresso.net"])
                    WARN("9c: " << [dir UTF8String] << "/" << [f UTF8String]);
            }
        }
    }
}

// ── XPC-9d: swizzle _ANEClient.compileModel: to check if compileWithQoS: calls it ─
//
// If compileWithQoS: calls compileModel:, our swizzle will fire and log the
// options dict passed.  This tells us:
//   (a) whether the code path goes through _ANEClient at all
//   (b) what options dict arrives at _ANEClient (before sandbox extension handling)
//
// Strategy: add an output path option in compileWithQoS:, which our swizzle
// then injects into the XPC message AFTER the sandbox extension step.

#include <objc/runtime.h>

static BOOL (*orig_compile_model)(id,SEL,id,id,unsigned,NSError**) = nullptr;
static NSString* g_inject_key   = nil;
static NSURL*    g_inject_url   = nil;
static BOOL      g_swizzle_fired = NO;

static BOOL swizzled_compile_model(id self, SEL _cmd,
                                   id model, id opts,
                                   unsigned qos, NSError** err) {
    g_swizzle_fired = YES;
    // Log what's arriving
    NSMutableDictionary* mutable_opts = opts
        ? [NSMutableDictionary dictionaryWithDictionary:opts]
        : [NSMutableDictionary dictionary];

    WARN("SWIZZLE: compileModel:options:qos:error: called");
    WARN("SWIZZLE: opts = " << [[mutable_opts description] UTF8String]);

    // Inject output path AFTER sandbox extension would be issued for model dir
    if (g_inject_key && g_inject_url)
        mutable_opts[g_inject_key] = g_inject_url;

    return orig_compile_model(self, _cmd, model, mutable_opts, qos, err);
}

TEST_CASE("XPC-9d: swizzle _ANEClient.compileModel to observe/intercept call",
          "[pathc][xpc][xpc9d]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls_client = NSClassFromString(@"_ANEClient");
        Class cls_Desc   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model  = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_client || !cls_Desc || !cls_Model) {
            WARN("9d: missing classes"); return;
        }

        // Swizzle _ANEClient.compileModel:options:qos:error:
        SEL sel_cm = sel_registerName("compileModel:options:qos:error:");
        Method orig_m = class_getInstanceMethod(cls_client, sel_cm);
        if (!orig_m) { WARN("9d: compileModel: method not found"); return; }

        orig_compile_model = (BOOL(*)(id,SEL,id,id,unsigned,NSError**))
            method_getImplementation(orig_m);
        method_setImplementation(orig_m,
            (IMP)swizzled_compile_model);
        WARN("9d: swizzled _ANEClient.compileModel:options:qos:error:");

        // Build model
        const char* kMIL_d =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,4,1,48]> x) {\n"
            "        tensor<fp16, [1,4,1,48]> y = relu(x=x)[name=string(\"d\")];\n"
            "    } -> (y);\n}\n";

        NSData* md = [NSData dataWithBytes:kMIL_d length:strlen(kMIL_d)];
        id desc = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
            cls_Desc, sel_registerName("modelWithMILText:weights:optionsPlist:"),
            md, @{}, nil);
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc);
        if (!model) { WARN("9d: model nil"); return; }

        NSString* hex_id_d = ((NSString*(*)(id,SEL))objc_msgSend)(
            model, sel_registerName("hexStringIdentifier"));
        WARN("9d: hexID=" << [hex_id_d UTF8String]);

        // Write model_dir
        NSString* md_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:hex_id_d];
        [[NSFileManager defaultManager] createDirectoryAtPath:md_dir
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[md_dir stringByAppendingPathComponent:@"model.mil"]
            atomically:YES];

        // Prepare output file — pass as NSString NOT NSURL so _ANEClient
        // doesn't try to issue a sandbox extension for it.
        NSString* out_path = [md_dir
            stringByAppendingPathComponent:@"compiled_out.llir"];
        [@"" writeToFile:out_path atomically:NO encoding:NSUTF8StringEncoding error:nil];
        // We set g_inject_url as NSString (repurposing the NSURL slot as id)
        g_inject_key = @"aotModelBinaryPath";
        g_inject_url = (NSURL*)out_path;  // Pass NSString, not NSURL

        // Call compileWithQoS: — will go through swizzled compileModel: if connected
        g_swizzle_fired = NO;
        NSError* err = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(
            model, sel_registerName("compileWithQoS:options:error:"),
            21u, @{}, &err);

        WARN("9d: compileWithQoS ok=" << (int)ok
             << "  err=" << (err ? [[err localizedDescription] UTF8String] : "none"));
        WARN("9d: swizzle_fired=" << (int)g_swizzle_fired);

        // Check for binary
        NSDictionary* attrs = [[NSFileManager defaultManager]
            attributesOfItemAtPath:out_path error:nil];
        long long sz = [attrs[NSFileSize] longLongValue];
        WARN("9d: compiled_out.llir size=" << sz);
        if (sz > 0)
            WARN("*** BINARY CAPTURED! " << sz << " bytes — cross-op patching unlocked! ***");

        // Scan model_dir
        WARN("9d: model_dir contents:");
        NSArray* mdc = [[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:md_dir error:nil];
        for (NSString* f in mdc) {
            NSString* fp = [md_dir stringByAppendingPathComponent:f];
            NSDictionary* a = [[NSFileManager defaultManager]
                attributesOfItemAtPath:fp error:nil];
            WARN("9d:   " << [f UTF8String]
                 << "  (" << [a[NSFileSize] longLongValue] << "b)");
        }

        // Restore original implementation
        method_setImplementation(orig_m, (IMP)orig_compile_model);
        g_inject_key = nil; g_inject_url = nil;
    }
}

// ── XPC-9: enumerate _ANEClient methods to find compile wrapper ──────────────
//
// _ANEClient wraps the XPC call in its own ObjC method (e.g. compileModel:...).
// We want the exact method name so we can swizzle it in XPC-10 and inject an
// output path key into the options dict _ANEClient already builds and sends.

TEST_CASE("XPC-9: enumerate _ANEClient methods — find compile wrapper",
          "[pathc][xpc][xpc9]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls = NSClassFromString(@"_ANEClient");
        if (!cls) { WARN("XPC-9: _ANEClient not found"); return; }

        WARN("XPC-9: instance methods of _ANEClient:");

        Class c = cls;
        while (c && c != [NSObject class]) {
            unsigned int count = 0;
            Method* methods = class_copyMethodList(c, &count);
            for (unsigned int i = 0; i < count; ++i) {
                const char* sel_name = sel_getName(method_getName(methods[i]));
                const char* types    = method_getTypeEncoding(methods[i]);
                WARN("  [-" << sel_name << "]  " << (types ? types : "?"));
            }
            free(methods);

            WARN("  --- class methods ---");
            unsigned int ccnt = 0;
            Method* cmethods = class_copyMethodList(object_getClass(c), &ccnt);
            for (unsigned int i = 0; i < ccnt; ++i) {
                const char* sel_name = sel_getName(method_getName(cmethods[i]));
                const char* types    = method_getTypeEncoding(cmethods[i]);
                WARN("  [+" << sel_name << "]  " << (types ? types : "?"));
            }
            free(cmethods);

            c = class_getSuperclass(c);
        }
    }
}

// ── XPC-10: call _ANEClient methods directly ─────────────────────────────────
//
// _ANEClient.compileModel:options:qos:error: is the synchronous wrapper that
// _ANEInMemoryModel.compileWithQoS:options: calls internally.  We bypass all
// NSXPCInterface proxy issues by calling _ANEClient's own ObjC method directly —
// it handles XPC encoding.  We inject output-path keys into the options dict to
// attempt binary capture.
//
// Sequence:
//   10a  echo: on sharedConnection — confirms the channel is live
//   10b  compileModel:options:qos:error: with output-path candidates
//   10c  scan /tmp for any newly written binary

TEST_CASE("XPC-10: _ANEClient.compileModel:options:qos:error: with output path",
          "[pathc][xpc][compile][xpc10]") {
    @autoreleasepool {
        libane_available();
        libane_set_log_level(LIBANE_LOG_SILENT);

        Class cls_client = NSClassFromString(@"_ANEClient");
        if (!cls_client) { WARN("XPC-10: _ANEClient not found"); return; }

        id client = ((id(*)(Class,SEL))objc_msgSend)(
            cls_client, sel_registerName("sharedConnection"));
        if (!client) { WARN("XPC-10: sharedConnection nil"); return; }
        WARN("XPC-10: client = " << class_getName([client class]));

        // ── 10a: echo: — confirm channel ─────────────────────────────────────
        WARN("\n--- 10a: echo: ---");
        {
            SEL sel_echo = sel_registerName("echo:");
            NSError* err = nil;
            BOOL ok = NO;
            @try {
                ok = ((BOOL(*)(id,SEL,NSString*))objc_msgSend)(
                    client, sel_echo, @"ping-libane");
            } @catch (NSException* ex) {
                WARN("10a: exception: " << [[ex reason] UTF8String]);
            }
            WARN("10a: echo result = " << (int)ok
                 << (ok ? "  *** CHANNEL LIVE ***" : "  (failed or nil)"));
        }

        // ── 10b: build _ANEInMemoryModel + call compileModel:options:qos:error: ─
        WARN("\n--- 10b: compileModel:options:qos:error: with output path ---");
        {
            // Build _ANEInMemoryModel using the same path as ane_compile()
            Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
            if (!cls_Desc || !cls_Model) {
                WARN("10b: missing _ANEInMemoryModelDescriptor or _ANEInMemoryModel");
                WARN("10b: cls_Desc=" << (cls_Desc ? "ok" : "nil")
                     << " cls_Model=" << (cls_Model ? "ok" : "nil"));
                goto xpc10_done;
            }

            // Create descriptor: modelWithMILText:weights:optionsPlist:
            NSData* mil_data = [NSData dataWithBytes:kReluMIL
                                              length:strlen(kReluMIL)];
            SEL sel_desc = sel_registerName("modelWithMILText:weights:optionsPlist:");
            id desc = nil;
            if ([cls_Desc respondsToSelector:sel_desc]) {
                typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
                desc = ((DescFn)objc_msgSend)(cls_Desc, sel_desc, mil_data, @{}, nil);
            }
            if (!desc) { WARN("10b: descriptor creation failed"); goto xpc10_done; }

            // Create model: inMemoryModelWithDescriptor:
            SEL sel_imm = sel_registerName("inMemoryModelWithDescriptor:");
            id model = nil;
            if ([cls_Model respondsToSelector:sel_imm]) {
                typedef id (*ModelFn)(Class, SEL, id);
                model = ((ModelFn)objc_msgSend)(cls_Model, sel_imm, desc);
            }
            if (!model) { WARN("10b: model creation failed"); goto xpc10_done; }
            WARN("10b: model = " << class_getName([model class]));

            // Log the hex ID so we can match binaries to this compile
            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                model, sel_registerName("hexStringIdentifier"));
            WARN("10b: hexID = " << (hex_id ? [hex_id UTF8String] : "nil"));

            // Write temp dir so the daemon can resolve paths (required by compileWithQoS:)
            if (hex_id) {
                NSFileManager* fm = [NSFileManager defaultManager];
                NSString* model_dir = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:hex_id];
                [fm createDirectoryAtPath:model_dir
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSString* mil_path = [model_dir
                    stringByAppendingPathComponent:@"model.mil"];
                if (![fm fileExistsAtPath:mil_path])
                    [mil_data writeToFile:mil_path atomically:YES];
            }

            // The daemon calls issueSandboxExtensionForPath: client-side for any
            // path found in options.  For the sandbox extension to succeed, the
            // file must pre-exist (we create it as an empty placeholder) and be
            // accessible to this process.
            // Try three candidate locations: /tmp directly, model_dir (daemon
            // already writes there), and the user Caches dir.
            NSFileManager* fm10 = [NSFileManager defaultManager];
            NSString* tmp_file = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc10_out.llir"];
            NSString* model_dir_str2 = hex_id
                ? [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id]
                : nil;
            NSString* model_dir_out = model_dir_str2
                ? [model_dir_str2 stringByAppendingPathComponent:@"xpc10_out.llir"]
                : nil;
            NSString* cache_out = [NSHomeDirectory()
                stringByAppendingPathComponent:@"Library/Caches/xpc10_out.llir"];

            // Pre-create placeholder files so sandbox extension issuance succeeds
            for (NSString* p in @[tmp_file,
                                   model_dir_out ?: @"/dev/null",
                                   cache_out]) {
                if (![fm10 fileExistsAtPath:p])
                    [@"" writeToFile:p atomically:NO
                            encoding:NSUTF8StringEncoding error:nil];
            }

            NSMutableArray* path_candidates = [NSMutableArray
                arrayWithObjects:tmp_file, cache_out, nil];
            if (model_dir_out) [path_candidates insertObject:model_dir_out atIndex:0];

            NSArray* key_candidates = @[
                @"aotModelBinaryPath",
                @"ANEFAOTBinaryPath",
                @"outputPath",
                @"compiledModelBinaryPath",
            ];

            SEL sel_compile = sel_registerName("compileModel:options:qos:error:");

            for (NSString* out_path in path_candidates) {
                WARN("10b: output_path = " << [out_path UTF8String]);
                for (NSString* key in key_candidates) {
                    // Reset placeholder
                    [fm10 removeItemAtPath:out_path error:nil];
                    [@"" writeToFile:out_path atomically:NO
                            encoding:NSUTF8StringEncoding error:nil];

                    NSURL* out_url = [NSURL fileURLWithPath:out_path];
                    NSMutableDictionary* opts = [NSMutableDictionary dictionary];
                    opts[key] = out_url;

                    NSError* err = nil;
                    BOOL ok = NO;
                    @try {
                        ok = ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(
                            client, sel_compile, model, opts, 21u, &err);
                    } @catch (NSException* ex) {
                        WARN("10b[" << [key UTF8String] << "]: exception — "
                             << [[ex reason] UTF8String]);
                        continue;
                    }

                    NSDictionary* attrs = [fm10 attributesOfItemAtPath:out_path error:nil];
                    long long sz = [attrs[NSFileSize] longLongValue];
                    WARN("10b key=" << [key UTF8String]
                         << "  ok=" << (int)ok
                         << "  err=" << (err ? [[err localizedDescription] UTF8String] : "none")
                         << "  file_sz=" << sz);

                    if (sz > 0)
                        WARN("*** BINARY WRITTEN (" << sz << " bytes) key=" << [key UTF8String]
                             << " — cross-op patching unlocked! ***");
                }
            }

            // ── 10c: scan /tmp broadly ─────────────────────────────────────────
            WARN("\n--- 10c: /tmp broad scan ---");
            NSArray* tmp_all = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString* f in tmp_all) {
                if ([f hasSuffix:@".llir"]         || [f hasSuffix:@".llir.bundle"] ||
                    [f hasSuffix:@".hwx"]           || [f hasSuffix:@".anecir"] ||
                    [f hasSuffix:@".anec"]           || [f hasSuffix:@".espresso.net"] ||
                    [f hasSuffix:@".mlmodelc"]       || [f hasSuffix:@".mlpackage"])
                    WARN("10c: " << [f UTF8String]);
            }
        }

        xpc10_done:;
    }
}

// ── XPC-10v: _ANEVirtualClient probe — ParavirtualizedANE path ───────────────
//
// ParavirtualizedANE.framework exposes _ANEVirtualClient, which has
// direct IOKit access (callIOUserClient:) and model-file-copy methods
// (copyAllModelFiles:dictionary:ioSurfaceRefs:).  The compile path may
// differ from _ANEClient and could write to user-accessible locations.
//
// 10v-a: init + hasANE + validateEnvironmentForPrecompiledBinarySupport
// 10v-b: compileModel:options:qos:error: — see where output lands
// 10v-c: validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:
//         returns CFDictionary — probe for compiled binary
// 10v-d: copyAllModelFiles:dictionary:ioSurfaceRefs: after compile
//
// Tags: [pathc][xpc][xpc10v]

TEST_CASE("XPC-10v: _ANEVirtualClient compile + binary extraction probe",
          "[pathc][xpc][xpc10v]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);

        // Require ParavirtualizedANE.framework
        void* paravirt_h = dlopen(
            "/System/Library/PrivateFrameworks/ParavirtualizedANE.framework/"
            "ParavirtualizedANE", RTLD_LAZY);
        if (!paravirt_h) {
            WARN("10v: ParavirtualizedANE.framework not available");
            return;
        }
        void* ane_h = dlopen(
            "/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices",
            RTLD_LAZY);
        (void)ane_h;

        Class cls_vc = NSClassFromString(@"_ANEVirtualClient");
        if (!cls_vc) { WARN("10v: _ANEVirtualClient not found"); return; }
        WARN("10v: _ANEVirtualClient found: " << [cls_vc description].UTF8String);

        // ── 10v-a: init ────────────────────────────────────────────────────────
        WARN("\n--- 10v-a: init ---");
        id vc = nil;
        // Try initWithSingletonAccess first (instance init, not factory)
        SEL sel_singleton = sel_registerName("initWithSingletonAccess");
        if (class_getInstanceMethod(cls_vc, sel_singleton)) {
            id alloc_vc = [cls_vc alloc];
            vc = ((id(*)(id,SEL))objc_msgSend)(alloc_vc, sel_singleton);
            WARN("10v-a: initWithSingletonAccess → " << (vc ? "ok" : "nil"));
        }
        if (!vc) {
            vc = [[cls_vc alloc] init];
            WARN("10v-a: alloc/init → " << (vc ? "ok" : "nil"));
        }
        if (!vc) { WARN("10v-a: cannot create _ANEVirtualClient"); return; }

        BOOL has_ane = ((BOOL(*)(id,SEL))objc_msgSend)(vc, sel_registerName("hasANE"));
        WARN("10v-a: hasANE=" << (int)has_ane);

        SEL sel_precomp = sel_registerName("validateEnvironmentForPrecompiledBinarySupport");
        if (class_getInstanceMethod(cls_vc, sel_precomp)) {
            BOOL ok_env = ((BOOL(*)(id,SEL))objc_msgSend)(vc, sel_precomp);
            WARN("10v-a: validateEnvironmentForPrecompiledBinarySupport=" << (int)ok_env);
        }

        SEL sel_bt = sel_registerName("aneBoardtype");
        if (class_getInstanceMethod(cls_vc, sel_bt)) {
            long long bt = ((long long(*)(id,SEL))objc_msgSend)(vc, sel_bt);
            WARN("10v-a: aneBoardtype=" << bt);
        }

        // ── 10v-b: compileModel: via VirtualClient ─────────────────────────────
        WARN("\n--- 10v-b: compileModel via _ANEVirtualClient ---");

        // Build _ANEInMemoryModel
        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_Desc || !cls_Model) { WARN("10v-b: model classes nil"); return; }

        const char* kMIL_v =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,4,1,32]> x) {\n"
            "        tensor<fp16, [1,4,1,32]> y = relu(x=x)[name=string(\"v\")];\n"
            "    } -> (y);\n}\n";

        {
            NSData* md = [NSData dataWithBytes:kMIL_v length:strlen(kMIL_v)];
            id desc = ((id(*)(Class,SEL,NSData*,NSDictionary*,id))objc_msgSend)(
                cls_Desc,
                sel_registerName("modelWithMILText:weights:optionsPlist:"),
                md, @{}, nil);
            id model = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), desc);
            if (!model) { WARN("10v-b: model nil"); goto xpc10v_done; }

            NSString* hex_id = ((NSString*(*)(id,SEL))objc_msgSend)(
                model, sel_registerName("hexStringIdentifier"));
            WARN("10v-b: hexID=" << [hex_id UTF8String]);

            // Write model_dir for VirtualClient compile
            NSString* vc_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:[@"vc_" stringByAppendingString:hex_id]];
            [[NSFileManager defaultManager] createDirectoryAtPath:vc_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSData* md_data = [NSData dataWithBytes:kMIL_v length:strlen(kMIL_v)];
            [md_data writeToFile:[vc_dir stringByAppendingPathComponent:@"model.mil"]
                      atomically:YES];

            WARN("10v-b: model_dir=" << [vc_dir UTF8String]);

            // Compile via VirtualClient
            SEL sel_c = sel_registerName("compileModel:options:qos:error:");
            NSError* c_err = nil;
            BOOL c_ok = NO;
            if (class_getInstanceMethod(cls_vc, sel_c)) {
                @try {
                    c_ok = ((BOOL(*)(id,SEL,id,id,unsigned,NSError**))objc_msgSend)(
                        vc, sel_c, model, @{}, 21u, &c_err);
                } @catch (NSException* ex) {
                    WARN("10v-b: exception: " << [[ex reason] UTF8String]);
                }
                WARN("10v-b: compileModel ok=" << (int)c_ok
                     << "  err=" << (c_err ? [[c_err localizedDescription] UTF8String] : "none"));
            }

            // Scan model_dir after compile
            NSArray* vc_files = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:vc_dir error:nil];
            WARN("10v-b: vc_dir contents after compile:");
            for (NSString* f in vc_files) {
                NSString* fp = [vc_dir stringByAppendingPathComponent:f];
                NSDictionary* a = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:fp error:nil];
                WARN("10v-b:   " << [f UTF8String]
                     << "  (" << [a[NSFileSize] longLongValue] << "b)");
            }

            // ── 10v-c: validateNetworkCreate: with MIL text ────────────────────
            WARN("\n--- 10v-c: validateNetworkCreate ---");
            SEL sel_vnc = sel_registerName(
                "validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:");
            if (class_getInstanceMethod(cls_vc, sel_vnc)) {
                NSString* scratch_dir = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:@"vc_scratch"];
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:scratch_dir
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSString* uuid_str = [[NSUUID UUID] UUIDString];

                CFDictionaryRef result_dict = nil;
                @try {
                    result_dict = ((CFDictionaryRef(*)(id,SEL,uint64_t,id,id,id,id,id))objc_msgSend)(
                        vc, sel_vnc,
                        (uint64_t)0, uuid_str, @"main",
                        vc_dir, scratch_dir, md_data);
                } @catch (NSException* ex) {
                    WARN("10v-c: exception: " << [[ex reason] UTF8String]);
                }
                if (result_dict) {
                    WARN("10v-c: result dict keys="
                         << [(__bridge NSDictionary*)result_dict description].UTF8String);
                    CFRelease(result_dict);
                } else {
                    WARN("10v-c: result_dict=nil");
                }
            } else {
                WARN("10v-c: validateNetworkCreate: not found on VirtualClient");
            }

            // ── 10v-d: copyAllModelFiles after compile ─────────────────────────
            WARN("\n--- 10v-d: copyAllModelFiles ---");
            SEL sel_cam = sel_registerName("copyAllModelFiles:dictionary:ioSurfaceRefs:");
            if (class_getInstanceMethod(cls_vc, sel_cam) && c_ok) {
                CFDictionaryRef out_dict = nil;
                CFArrayRef out_surfs = nil;
                BOOL cam_ok = NO;
                @try {
                    cam_ok = ((BOOL(*)(id,SEL,id,CFDictionaryRef*,CFArrayRef*))objc_msgSend)(
                        vc, sel_cam, model, &out_dict, &out_surfs);
                } @catch (NSException* ex) {
                    WARN("10v-d: exception: " << [[ex reason] UTF8String]);
                }
                WARN("10v-d: copyAllModelFiles ok=" << (int)cam_ok);
                if (out_dict) {
                    WARN("10v-d: dict=" << [(__bridge NSDictionary*)out_dict description].UTF8String);
                    CFRelease(out_dict);
                }
                if (out_surfs) {
                    WARN("10v-d: ioSurfaceRefs count=" << CFArrayGetCount(out_surfs));
                    CFRelease(out_surfs);
                }
            }

            // ── 10v-e: scan /tmp for new binary artifacts ──────────────────────
            WARN("\n--- 10v-e: /tmp scan for new binaries ---");
            NSArray* tmp_files = [[NSFileManager defaultManager]
                contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString* f in tmp_files) {
                if ([f hasSuffix:@".llir"] || [f hasSuffix:@".hwx"] ||
                    [f hasSuffix:@".anecir"] || [f hasSuffix:@".anec"] ||
                    [f hasSuffix:@".mlpackage"] || [f hasSuffix:@".mlmodelc"]) {
                    NSString* fp = [NSTemporaryDirectory() stringByAppendingPathComponent:f];
                    NSDictionary* a = [[NSFileManager defaultManager]
                        attributesOfItemAtPath:fp error:nil];
                    WARN("10v-e: " << [f UTF8String]
                         << "  (" << [a[NSFileSize] longLongValue] << "b)");
                }
            }
        }

        xpc10v_done:;
    }
}

// ── XPC-11: Direct _ANEMILCompiler.compileModelAt:aotModelBinaryPath: ────────
//
// ANECompilerService.xpc can be dlopen'd.  Its compiler classes are then
// accessible as normal ObjC class objects in our process — WITHOUT going through
// the XPC entitlement gate.
//
// _ANEMILCompiler (class method):
//   +compileModelAt:modelName:csIdentity:optionsFilename:outputURL:
//            saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:
//
// We provide:
//   modelAt      = NSURL to model_dir/model.mil (MIL source)
//   modelName    = "model"
//   csIdentity   = "" (or whatever our process reports)
//   optionsFilename = nil  (may need an actual options plist)
//   outputURL    = a user-writable output dir
//   saveSourceURL= a user-writable source dir
//   aotModelBinaryPath = OUR FILE — the compiled binary lands here
//   isEncryptedModel = NO
//   options      = @{}
//
// If this succeeds and writes bytes to aotModelBinaryPath, we have binary
// capture on macOS 26 and can restart cross-op patching.
//
// Tags: [pathc][xpc][xpc11]

static const char kCS_PATH[] =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
    "XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService";

TEST_CASE("XPC-11: direct _ANEMILCompiler.compileModelAt: — binary capture",
          "[pathc][xpc][xpc11]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);

        void* cs_h = dlopen(kCS_PATH, RTLD_LAZY | RTLD_LOCAL);
        if (!cs_h) { WARN("11: dlopen ANECompilerService failed"); return; }
        WARN("11: ANECompilerService loaded: " << cs_h);

        Class cls_mil = NSClassFromString(@"_ANEMILCompiler");
        if (!cls_mil) { WARN("11: _ANEMILCompiler not found"); return; }
        WARN("11: _ANEMILCompiler found");

        // MIL for a relu (simple, proven to compile)
        static const char kMIL_11[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,4,1,16]> x) {\n"
            "        tensor<fp16, [1,4,1,16]> y = relu(x=x)"
            "[name=string(\"xpc11\")];\n"
            "    } -> (y);\n}\n";

        // Write model source to temp dir
        NSString* src_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc11_src"];
        [[NSFileManager defaultManager] createDirectoryAtPath:src_dir
            withIntermediateDirectories:YES attributes:nil error:nil];
        NSString* mil_path = [src_dir stringByAppendingPathComponent:@"model.mil"];
        NSData* mil_data = [NSData dataWithBytes:kMIL_11 length:strlen(kMIL_11)];
        [mil_data writeToFile:mil_path atomically:YES];
        WARN("11: model.mil written to " << [src_dir UTF8String]);

        // Prepare output dirs
        NSString* out_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc11_out"];
        [[NSFileManager defaultManager] createDirectoryAtPath:out_dir
            withIntermediateDirectories:YES attributes:nil error:nil];

        NSString* aot_path = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc11_aot.llir.bundle"];
        // Create empty placeholder (so we can check size delta)
        [@"" writeToFile:aot_path atomically:NO
                encoding:NSUTF8StringEncoding error:nil];

        // Call _ANEMILCompiler.compileModelAt:...
        SEL sel_compile = sel_registerName(
            "compileModelAt:modelName:csIdentity:optionsFilename:outputURL:"
            "saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:");

        NSURL* mil_url  = [NSURL fileURLWithPath:mil_path];
        NSURL* out_url  = [NSURL fileURLWithPath:out_dir];
        NSURL* src_url  = [NSURL fileURLWithPath:src_dir];
        NSURL* aot_url  = [NSURL fileURLWithPath:aot_path];

        BOOL ok11 = NO;
        NSError* err11 = nil;
        id result = nil;

        @try {
            typedef id (*CompileFn)(Class,SEL,NSURL*,NSString*,NSString*,
                                    NSString*,NSURL*,NSURL*,NSURL*,BOOL,
                                    NSDictionary*,BOOL*,NSError**);
            result = ((CompileFn)objc_msgSend)(
                cls_mil, sel_compile,
                src_url,        // modelAt — DIRECTORY containing model source
                @"model.mil",   // modelName — file inside that directory
                @"",            // csIdentity (empty — no signing for ad-hoc)
                nil,            // optionsFilename
                out_url,        // outputURL
                src_url,        // saveSourceURL
                aot_url,        // aotModelBinaryPath  ← OUR PATH
                NO,             // isEncryptedModel
                @{},            // options
                &ok11,          // ok (out)
                &err11);        // error (out)
        } @catch (NSException* ex) {
            WARN("11: EXCEPTION: " << [[ex reason] UTF8String]);
        }

        WARN("11: compile ok=" << (int)ok11
             << "  result=" << (result ? [[result description] UTF8String] : "nil")
             << "  err=" << (err11 ? [[err11 localizedDescription] UTF8String] : "none"));

        // Check aot_path for binary
        NSDictionary* aot_attrs = [[NSFileManager defaultManager]
            attributesOfItemAtPath:aot_path error:nil];
        long long aot_sz = [aot_attrs[NSFileSize] longLongValue];
        WARN("11: aot_path size=" << aot_sz);

        if (aot_sz > 4) {
            // Read magic bytes
            NSData* aot_data = [NSData dataWithContentsOfFile:aot_path];
            uint8_t* b = (uint8_t*)[aot_data bytes];
            char magic_hex[32];
            snprintf(magic_hex, sizeof(magic_hex), "%02x%02x%02x%02x",
                     b[0], b[1], b[2], b[3]);
            WARN("*** BINARY CAPTURED! " << aot_sz << " bytes  magic=" << magic_hex
                 << " — cross-op patching unlocked on macOS 26! ***");
        }

        // Scan out_dir for any artifacts
        WARN("11: out_dir contents:");
        NSDirectoryEnumerator* en = [[NSFileManager defaultManager]
            enumeratorAtPath:out_dir];
        for (NSString* f in en) {
            NSString* fp = [out_dir stringByAppendingPathComponent:f];
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:fp isDirectory:&isDir];
            NSDictionary* a = [[NSFileManager defaultManager]
                attributesOfItemAtPath:fp error:nil];
            if (!isDir)
                WARN("11:   " << [f UTF8String]
                     << "  (" << [a[NSFileSize] longLongValue] << "b)");
        }

        // Also try _ANEMLIRCompiler variant
        WARN("\n--- 11b: _ANEMLIRCompiler variant ---");
        Class cls_mlir = NSClassFromString(@"_ANEMLIRCompiler");
        if (cls_mlir) {
            SEL sel_mlir = sel_registerName(
                "compileModelAt:modelName:csIdentity:optionsFilename:outputURL:"
                "saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:"
                "mpsConstants:ok:error:");
            NSString* aot_mlir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc11b_aot.llir.bundle"];
            [@"" writeToFile:aot_mlir atomically:NO
                    encoding:NSUTF8StringEncoding error:nil];
            NSURL* aot_mlir_url = [NSURL fileURLWithPath:aot_mlir];
            BOOL ok_mlir = NO; NSError* err_mlir = nil;
            id res_mlir = nil;
            @try {
                typedef id (*MLIRFn)(Class,SEL,NSURL*,NSString*,NSString*,
                                     NSString*,NSURL*,NSURL*,NSURL*,BOOL,
                                     NSDictionary*,NSDictionary*,BOOL*,NSError**);
                res_mlir = ((MLIRFn)objc_msgSend)(
                    cls_mlir, sel_mlir,
                    mil_url, @"model", @"", nil,
                    out_url, src_url, aot_mlir_url,
                    NO, @{}, nil, &ok_mlir, &err_mlir);
            } @catch (NSException* ex) {
                WARN("11b: EXCEPTION: " << [[ex reason] UTF8String]);
            }
            WARN("11b: ok=" << (int)ok_mlir
                 << "  err=" << (err_mlir ? [[err_mlir localizedDescription] UTF8String] : "none"));
            NSDictionary* ma = [[NSFileManager defaultManager]
                attributesOfItemAtPath:aot_mlir error:nil];
            long long msz = [ma[NSFileSize] longLongValue];
            WARN("11b: aot size=" << msz);
            if (msz > 4) WARN("*** MLIR binary captured! " << msz << "b ***");
        }
    }
}

// ── XPC-12: loadWithQoS: path vs hexID — does aned read bytes from the URL? ──
//
// The question: when setModelURL: is called with a user-controlled file path
// and then loadWithQoS: is called, does aned:
//   outcome 1 — read the binary from the URL path (win: we can load patched bytes)
//   outcome 2 — reject because the URL is not in aned's Data Vault cache
//   outcome 3 — ignore URL, load by hexID from its internal cache (silent wrong-op)
//
// The LUT fingerprint check distinguishes outcome 1 from outcome 3:
//   TANH ANE LUT: x=0.1 → 0x2E5D (0.099426)
//   RELU pass-through: x=0.1 → fp16(0.1) ≈ 0x2E66 (0.099670)
//
// Tags: [pathc][xpc][xpc12]

#include "../src/graph/hwx_inline_capture.hpp"
#include "../src/graph/hwx_emitter.hpp"
#include "../src/core/buffer_manager.hpp"

static std::string write_mil_dir(const char* subdir, const char* mil_text) {
    NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
                     [NSString stringWithUTF8String:subdir]];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
        withIntermediateDirectories:YES attributes:nil error:nil];
    NSString* mil = [dir stringByAppendingPathComponent:@"model.mil"];
    [[NSData dataWithBytes:mil_text length:strlen(mil_text)] writeToFile:mil atomically:YES];
    return [dir UTF8String];
}

TEST_CASE("XPC-12: loadWithQoS: reads from URL path vs hexID lookup — cross-op patch",
          "[pathc][xpc][xpc12]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("12: ANE unavailable"); return; }

        // C=64, S=512: 65536 bytes > 49 KB IOSurface minimum; same shape as
        // test_hwx_backend.cpp so we can reuse the known LUT fingerprint values.
        static const int  C = 64, S = 512;
        static const size_t N = (size_t)C * S;

        static const char kReluMIL_12[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"xpc12_relu\")];\n    } -> (y);\n}\n";

        static const char kTanhMIL_12[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"xpc12_tanh\")];\n    } -> (y);\n}\n";

        // ── Phase 0: inline-compile RELU and TANH to capture HWX bytes ────────

        std::string relu_src = write_mil_dir("xpc12_relu_src", kReluMIL_12);
        std::string tanh_src = write_mil_dir("xpc12_tanh_src", kTanhMIL_12);

        auto relu_hwx = libane::graph::hwx_capture_inline(relu_src);
        auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);

        if (relu_hwx.empty() || tanh_hwx.empty()) {
            WARN("12: inline capture failed (relu=" << relu_hwx.size()
                 << "b  tanh=" << tanh_hwx.size() << "b) — skipping");
            return;
        }
        WARN("12: relu_hwx=" << relu_hwx.size() << "b  tanh_hwx=" << tanh_hwx.size() << "b");

        // ── Phase 1: patch RELU template with TANH op config ──────────────────

        libane::graph::HwxEmitter emitter;
        REQUIRE(emitter.capture_from_bytes(relu_hwx, C, S, LIBANE_OP_RELU));
        REQUIRE(emitter.capture_from_bytes(tanh_hwx, C, S, LIBANE_OP_TANH));
        REQUIRE(emitter.can_emit(C, S, LIBANE_OP_TANH));

        auto patched_hwx = emitter.emit(C, S, LIBANE_OP_TANH);
        REQUIRE_FALSE(patched_hwx.empty());
        WARN("12: patched TANH HWX = " << patched_hwx.size() << "b");

        // Write patched bytes to a user-controlled temp path
        NSString* patch_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"xpc12_tanh_patch"];
        [[NSFileManager defaultManager] createDirectoryAtPath:patch_dir
            withIntermediateDirectories:YES attributes:nil error:nil];
        NSString* patch_hwx_path = [patch_dir stringByAppendingPathComponent:@"model.hwx"];
        [[NSData dataWithBytes:patched_hwx.data() length:patched_hwx.size()]
            writeToFile:patch_hwx_path atomically:YES];

        std::string patch_file_url =
            "file://" + std::string([patch_hwx_path UTF8String]);
        WARN("12: user URL = " << patch_file_url);

        // ── Phase 2: cold-compile RELU via aned (seeds hexID in aned's table) ─

        auto* relu_h = libane_mil_compile(kReluMIL_12, nullptr, nullptr, nullptr, 0);
        if (!relu_h) {
            WARN("12: aned RELU compile failed: " << libane_last_error());
            return;
        }
        WARN("12: aned RELU cold compile OK  url=" << relu_h->prog->model_url);

        // ── Phase 3b: ane_reconnect with user URL pointing to TANH-patched bytes ─
        //   RELU hexID in descriptor → compiledModelExists=YES (slot from Phase 2).
        //   URL points to TANH-patched HWX bytes at user-controlled path.
        //   Run BEFORE any ane_unload to keep the RELU slot alive.
        //   Execute with x=0.1:
        //     0x2E5D → outcome 1: aned read from path, TANH executed (full win!)
        //     ~0x2E66 → outcome 3: aned used hexID, RELU from cache (silent wrong-op)
        //     load fail → outcome 2: aned rejected user URL

        {
            auto* tp = libane::runtime::ane_reconnect(
                kReluMIL_12, {}, patch_file_url, "xpc12-3b-tanh-patch-url");
            WARN("12 Phase 3b (cross-op TANH patch, RELU hexID): "
                 << (tp ? "LOAD SUCCEEDED" : "LOAD FAILED")
                 << (tp ? "" : std::string(" — ") + libane::runtime::ane_last_error()));

            if (tp) {
                std::vector<uint16_t> in_data2(N, 0x2E66u);  // fp16(0.1) for all elements
                auto in_buf2  = libane::global_buffer_pool().acquire_tensor_with_data(
                    in_data2.data(), C, S);
                auto out_buf2 = libane::global_buffer_pool().acquire_tensor(C, S);
                if (in_buf2 && out_buf2) {
                    bool exec_ok = libane::runtime::ane_execute(
                        tp, in_buf2->iosurface(), out_buf2->iosurface());
                    WARN("12 Phase 3b execute: " << (exec_ok ? "ok" : "FAILED"));
                    if (exec_ok) {
                        std::vector<uint16_t> out_data2(N);
                        out_buf2->copy_to(out_data2.data(), N * sizeof(uint16_t));
                        uint16_t out0 = out_data2[0];
                        if (out0 == 0x2E5Du)
                            WARN("*** OUTCOME 1: TANH LUT fingerprint confirmed "
                                 "(0x2E5D) — aned reads binary from URL path. "
                                 "ane_load_hwx() via path injection is viable! ***");
                        else if (out0 == 0x2E66u || out0 == 0x2E67u)
                            WARN("OUTCOME 3 (silent wrong-op): RELU executed "
                                 "(out=0x" << std::hex << out0 << ") despite TANH "
                                 "bytes at URL — aned used hexID cache, ignored path.");
                        else
                            WARN("OUTCOME ?: unexpected out=0x" << std::hex << out0
                                 << " (not TANH 0x2E5D, not RELU ~0x2E66)");
                    }
                }
                libane::runtime::ane_unload(tp);
            }
        }

        libane_mil_release(relu_h);
    }
}

// ── XPC-13: does _ANEMILCompiler inline compile register in aned's table? ─────
//
// When _ANEMILCompiler.compileModelAt: runs in-process, it might (as a side
// effect) register the compiled result in aned's per-process compile table —
// the same table that compiledModelExists checks.  If it does, we can load the
// result via loadWithQoS: without ever calling compileWithQoS: through aned.
//
// That would give us a clean path for ANY op: inline-compile TANH MIL via
// _ANEMILCompiler, loadWithQoS:, execute.  Cross-op patching would then be
// handled entirely by HwxEmitter + hwx_capture_inline with no XPC gate.
//
// Probe sequence:
//   A. _ANEMILCompiler inline compile of TANH MIL (no compileWithQoS:)
//   B. Create _ANEInMemoryModel from TANH descriptor
//   C. compiledModelExists → YES means inline registered; NO means it didn't
//   D. If YES: loadWithQoS: (no compile) → execute TANH fingerprint
//   E. loadWithQoS: regardless of compiledModelExists (skip guard) → observe
//
// Tags: [pathc][xpc][xpc13]

TEST_CASE("XPC-13: _ANEMILCompiler inline compile — does it register in aned table?",
          "[pathc][xpc][xpc13]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("13: ANE unavailable"); return; }

        const int C = 64, S = 512;
        const size_t N = (size_t)C * S;

        static const char kTanhMIL_13[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"xpc13_tanh\")];\n    } -> (y);\n}\n";

        // ── Phase A: _ANEMILCompiler inline compile of TANH (no aned involvement) ─

        std::string tanh_src = write_mil_dir("xpc13_tanh_src", kTanhMIL_13);
        auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);
        if (tanh_hwx.empty()) {
            WARN("13: inline TANH compile failed — skipping");
            return;
        }
        WARN("13: inline TANH HWX captured: " << tanh_hwx.size() << "b");

        // ── Phase B+C: create _ANEInMemoryModel from TANH descriptor, check table ─

        // Resolve classes (runtime init must have run — ensured by libane_available())
        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        if (!cls_Desc || !cls_Model) {
            WARN("13: ObjC classes not found — skipping");
            return;
        }

        NSData* mil_data = [NSData dataWithBytes:kTanhMIL_13 length:strlen(kTanhMIL_13)];
        typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
        id descriptor = ((DescFn)objc_msgSend)(
            cls_Desc, sel_registerName("modelWithMILText:weights:optionsPlist:"),
            mil_data, @{}, nil);
        if (!descriptor) {
            WARN("13: TANH descriptor creation failed — skipping");
            return;
        }

        typedef id (*ModelFn)(Class, SEL, id);
        id model = ((ModelFn)objc_msgSend)(
            cls_Model, sel_registerName("inMemoryModelWithDescriptor:"), descriptor);
        if (!model) {
            WARN("13: TANH model creation failed — skipping");
            return;
        }

        // hexID
        typedef NSString* (*StrFn)(id, SEL);
        NSString* hex_id = ((StrFn)objc_msgSend)(model, sel_registerName("hexStringIdentifier"));
        WARN("13: TANH hexID = " << (hex_id ? [hex_id UTF8String] : "(nil)"));

        // compiledModelExists: does aned have this hexID after ONLY inline compile?
        SEL sel_cme = sel_registerName("compiledModelExists");
        BOOL model_exists = NO;
        if ([model respondsToSelector:sel_cme]) {
            model_exists = ((BOOL(*)(id,SEL))objc_msgSend)(model, sel_cme);
            WARN("13: compiledModelExists (after inline only) = " << (model_exists ? "YES" : "NO"));
        } else {
            WARN("13: compiledModelExists not available");
        }

        // ── Phase D: if table has it, try loadWithQoS: (no compile) ──────────────

        if (model_exists) {
            WARN("13: *** inline compile DID register in aned table — attempting load ***");
            NSError* load_err = nil;
            typedef BOOL (*QoSFn)(id, SEL, unsigned int, id, NSError**);
            BOOL loaded = ((QoSFn)objc_msgSend)(
                model, sel_registerName("loadWithQoS:options:error:"),
                QOS_CLASS_DEFAULT, @{}, &load_err);
            WARN("13 loadWithQoS: (D, table hit): "
                 << (loaded ? "OK" : "FAILED")
                 << (load_err ? std::string(" — ") + [[load_err localizedDescription] UTF8String] : ""));
        } else {
            WARN("13: inline compile did NOT register in aned table");
        }

        // ── Phase E: try loadWithQoS: unconditionally (skip compiledModelExists) ─
        // Even if compiledModelExists=NO, maybe loadWithQoS: can load from the URL
        // that _ANEMILCompiler wrote to — try injecting that path first.
        WARN("13 Phase E: attempting loadWithQoS: regardless of compiledModelExists");
        {
            // Set modelURL to the inline-compiled output dir (where model.hwx lives)
            NSString* out_path = [[NSString stringWithUTF8String:tanh_src.c_str()]
                stringByAppendingPathComponent:@"../xpc13_tanh_inline_out"];
            // Actually, hwx_capture_inline writes to a temp subdir then cleans up.
            // Re-run _ANEMILCompiler to a persistent dir for this phase.
            NSString* out_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc13_tanh_out"];
            [[NSFileManager defaultManager] createDirectoryAtPath:out_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSURL* src_url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:tanh_src.c_str()]];
            NSURL* out_url = [NSURL fileURLWithPath:out_dir];

            void* cs_h = dlopen(kCS_PATH, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
            if (!cs_h) cs_h = dlopen(kCS_PATH, RTLD_LAZY | RTLD_LOCAL);
            Class cls_mil = cs_h ? NSClassFromString(@"_ANEMILCompiler") : nil;

            if (cls_mil) {
                SEL sel_c = sel_registerName(
                    "compileModelAt:modelName:csIdentity:optionsFilename:outputURL:"
                    "saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:");
                BOOL ok13 = NO; NSError* err13 = nil;
                typedef id (*CmpFn)(Class,SEL,NSURL*,NSString*,NSString*,NSString*,
                                    NSURL*,NSURL*,NSURL*,BOOL,NSDictionary*,BOOL*,NSError**);
                @try {
                    ((CmpFn)objc_msgSend)(cls_mil, sel_c,
                        src_url, @"model.mil", @"", nil,
                        out_url, src_url, nil, NO, @{}, &ok13, &err13);
                } @catch (...) {}
                WARN("13 Phase E: _ANEMILCompiler to persistent dir: ok=" << (int)ok13);
            }

            // Now try setModelURL: with the output dir URL + loadWithQoS:
            NSString* hwx_path = [out_dir stringByAppendingPathComponent:@"model.hwx"];
            BOOL hwx_exists = [[NSFileManager defaultManager] fileExistsAtPath:hwx_path];
            WARN("13 Phase E: model.hwx at out_dir exists=" << (int)hwx_exists);

            if (hwx_exists) {
                std::string out_url_str = "file://" + std::string([out_dir UTF8String]) + "/";
                SEL sel_smu = sel_registerName("setModelURL:");
                if ([model respondsToSelector:sel_smu]) {
                    NSURL* nsurl = [NSURL URLWithString:
                        [NSString stringWithUTF8String:out_url_str.c_str()]];
                    ((void(*)(id,SEL,NSURL*))objc_msgSend)(model, sel_smu, nsurl);
                    WARN("13 Phase E: setModelURL: to inline output dir");
                }
                NSError* load_err_e = nil;
                typedef BOOL (*QoSFn)(id, SEL, unsigned int, id, NSError**);
                BOOL loaded_e = ((QoSFn)objc_msgSend)(
                    model, sel_registerName("loadWithQoS:options:error:"),
                    QOS_CLASS_DEFAULT, @{}, &load_err_e);
                WARN("13 Phase E loadWithQoS: = "
                     << (loaded_e ? "OK" : "FAILED")
                     << (load_err_e ? std::string(" — ") + [[load_err_e localizedDescription] UTF8String] : ""));

                // ── Fingerprint check if load succeeded ───────────────────────────
                if (loaded_e) {
                    [model retain];
                    auto* prog13 = new libane::runtime::AneProgram{};
                    prog13->objc_model = (void*)model;
                    prog13->debug_name = "xpc13-tanh";
                    prog13->mil_text   = kTanhMIL_13;

                    std::vector<uint16_t> in_d(N, 0x2E66u);  // fp16(0.1)
                    auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(in_d.data(), C, S);
                    auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
                    if (in_b && out_b) {
                        bool exec_ok = libane::runtime::ane_execute(
                            prog13, in_b->iosurface(), out_b->iosurface());
                        WARN("13 Phase E execute: " << (exec_ok ? "ok" : "FAILED"));
                        if (exec_ok) {
                            std::vector<uint16_t> out_d(N);
                            out_b->copy_to(out_d.data(), N * sizeof(uint16_t));
                            uint16_t v = out_d[0];
                            if (v == 0x2E5Du)
                                WARN("*** XPC-13 OUTCOME 1: TANH LUT 0x2E5D — "
                                     "inline compile + loadWithQoS: works! ***");
                            else
                                WARN("13 Phase E out[0]=0x" << std::hex << v
                                     << " (TANH LUT=0x2E5D, RELU=~0x2E66)");
                        }
                    }
                    // Clean up without unloading since we don't own the full session
                    delete prog13;
                }
            }
        }
    }
}

// ── XPC-14: ane_load_mlmodelc with pre-staged HWX — does aned read it? ────────
//
// aned has "Loading pre-compiled model modelFilePath=%@" code path.
// _ANEClient.compileModel: takes a directory-based _ANEModel.  If aned reads a
// pre-compiled HWX from that directory directly, we can inject patched bytes.
//
// Three sub-probes:
//   A. Dir with ONLY model.hwx (TANH patched) — no model.mil, no fallback
//      → if load ok + TANH fingerprint: aned reads HWX directly
//      → if compile error: aned needs source
//   B. Dir with model.mil (RELU) + model.hwx (TANH patched) — source present
//      → if TANH fingerprint: aned used the HWX, ignored MIL
//      → if RELU fingerprint: aned recompiled from MIL, ignored HWX
//   C. Same as B but HWX named as aned's actual compiled file might be
//      (try model.mlmodelc, model.anec, etc.)
//
// Tags: [pathc][xpc][xpc14]

TEST_CASE("XPC-14: ane_load_mlmodelc with pre-staged HWX — does aned use it?",
          "[pathc][xpc][xpc14]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("14: ANE unavailable"); return; }
        if (!libane::runtime::path_b_available()) {
            WARN("14: Path B symbols unavailable — skipping");
            return;
        }

        const int C = 64, S = 512;
        const size_t N = (size_t)C * S;

        // MIL sources
        static const char kReluMIL_14[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"xpc14_relu\")];\n    } -> (y);\n}\n";
        static const char kTanhMIL_14[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"xpc14_tanh\")];\n    } -> (y);\n}\n";

        // Get RELU and TANH HWX bytes via inline compile
        auto relu_hwx = libane::graph::hwx_capture_inline(
            write_mil_dir("xpc14_relu_src", kReluMIL_14));
        auto tanh_hwx = libane::graph::hwx_capture_inline(
            write_mil_dir("xpc14_tanh_src", kTanhMIL_14));
        if (relu_hwx.empty() || tanh_hwx.empty()) {
            WARN("14: inline capture failed — skipping"); return;
        }

        // Produce TANH-patched RELU HWX via HwxEmitter (cross-op patch)
        libane::graph::HwxEmitter em14;
        em14.capture_from_bytes(relu_hwx, C, S, LIBANE_OP_RELU);
        em14.capture_from_bytes(tanh_hwx, C, S, LIBANE_OP_TANH);
        auto patched = em14.emit(C, S, LIBANE_OP_TANH);
        WARN("14: relu_hwx=" << relu_hwx.size() << "b  tanh_hwx=" << tanh_hwx.size()
             << "b  patched=" << patched.size() << "b");

        // Helper: run a probe dir through ane_load_mlmodelc, execute, report
        auto probe = [&](const std::string& label, const std::string& dir_path) {
            auto* prog = libane::runtime::ane_load_mlmodelc(
                dir_path, C, S, C, S, "xpc14-" + label);
            if (!prog) {
                WARN("14 " << label << ": FAILED — " << libane::runtime::ane_last_error());
                return;
            }
            WARN("14 " << label << ": LOAD OK");
            std::vector<uint16_t> in_d(N, 0x2E66u);  // fp16(0.1)
            auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(in_d.data(), C, S);
            auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
            if (!in_b || !out_b) {
                WARN("14 " << label << ": buffer alloc failed");
                libane::runtime::ane_unload(prog);
                return;
            }
            // Path B uses ane_execute_client (different dispatch)
            bool exec_ok = libane::runtime::ane_execute(
                prog, in_b->iosurface(), out_b->iosurface());
            WARN("14 " << label << ": execute " << (exec_ok ? "ok" : "FAILED"));
            if (exec_ok) {
                std::vector<uint16_t> out_d(N);
                out_b->copy_to(out_d.data(), N * sizeof(uint16_t));
                uint16_t v = out_d[0];
                if (v == 0x2E5Du)
                    WARN("14 " << label
                         << ": *** TANH LUT 0x2E5D — aned loaded pre-compiled HWX! ***");
                else if (v >= 0x2E64u && v <= 0x2E68u)
                    WARN("14 " << label << ": RELU ~0x2E66 (out=0x" << std::hex << v
                         << ") — aned recompiled from MIL, ignored HWX");
                else
                    WARN("14 " << label << ": out=0x" << std::hex << v
                         << " (unexpected — not TANH 0x2E5D, not RELU ~0x2E66)");
            }
            libane::runtime::ane_unload(prog);
        };

        NSFileManager* fm = [NSFileManager defaultManager];

        // ── Probe A: dir with ONLY model.hwx (TANH patched) ──────────────────
        {
            NSString* dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc14_probe_a"];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                attributes:nil error:nil];
            NSString* hwx = [dir stringByAppendingPathComponent:@"model.hwx"];
            [[NSData dataWithBytes:patched.data() length:patched.size()]
                writeToFile:hwx atomically:YES];
            probe("A(hwx-only)", [dir UTF8String]);
        }

        // ── Probe B: RELU model.mil + TANH-patched model.hwx ─────────────────
        {
            NSString* dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc14_probe_b"];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                attributes:nil error:nil];
            [[NSData dataWithBytes:kReluMIL_14 length:strlen(kReluMIL_14)]
                writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [[NSData dataWithBytes:patched.data() length:patched.size()]
                writeToFile:[dir stringByAppendingPathComponent:@"model.hwx"] atomically:YES];
            probe("B(mil+hwx)", [dir UTF8String]);
        }

        // ── Probe C: TANH model.mil + TANH model.hwx (same op, simpler baseline)
        {
            NSString* dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:@"xpc14_probe_c"];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                attributes:nil error:nil];
            [[NSData dataWithBytes:kTanhMIL_14 length:strlen(kTanhMIL_14)]
                writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [[NSData dataWithBytes:tanh_hwx.data() length:tanh_hwx.size()]
                writeToFile:[dir stringByAppendingPathComponent:@"model.hwx"] atomically:YES];
            probe("C(tanh-mil+hwx)", [dir UTF8String]);
        }
    }
}

// ── XPC-15: _ANEMILCompiler output inventory → ane_load_mlmodelc on real dir ──
//
// XPC-14 failed because _ANEClient.compileModel: expects model.espresso.net,
// not model.mil or model.hwx.  hwx_capture_inline deletes the output dir after
// reading model.hwx — we never saw what else _ANEMILCompiler wrote there.
//
// This test duplicates the inline-compile call, keeps the output dir, lists all
// files written there, then calls ane_load_mlmodelc on it directly.
//
// If _ANEMILCompiler writes model.espresso.net + model.hwx, the real output dir
// should satisfy _ANEClient.compileModel: without recompilation — and if aned
// reads model.hwx from it we get cross-op injection.
//
// Tags: [pathc][xpc][xpc15]

TEST_CASE("XPC-15: _ANEMILCompiler output dir contents + ane_load_mlmodelc probe",
          "[pathc][xpc][xpc15]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("15: ANE unavailable"); return; }
        if (!libane::runtime::path_b_available()) {
            WARN("15: Path B unavailable — skipping"); return;
        }

        static const char* kCS_BINARY =
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"
            "/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService";
        static const char* kCompileSel =
            "compileModelAt:modelName:csIdentity:optionsFilename:outputURL:"
            "saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:";

        void* handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if (!handle) handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL);
        REQUIRE_FALSE(!handle);

        Class cls = NSClassFromString(@"_ANEMILCompiler");
        REQUIRE(cls);
        SEL sel = sel_registerName(kCompileSel);
        REQUIRE([cls respondsToSelector:sel]);

        const int C = 64, S = 512;
        const size_t N = (size_t)C * S;

        static const char kReluMIL_15[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"xpc15_relu\")];\n    } -> (y);\n}\n";
        static const char kTanhMIL_15[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"xpc15_tanh\")];\n    } -> (y);\n}\n";

        // Write MIL source dirs
        std::string relu_src = write_mil_dir("xpc15_relu_src", kReluMIL_15);
        std::string tanh_src = write_mil_dir("xpc15_tanh_src", kTanhMIL_15);

        // Compile RELU and TANH — keep output dirs, list contents
        typedef id (*CompileFn)(Class, SEL, NSURL*, NSString*, NSString*, NSString*,
                                NSURL*, NSURL*, NSURL*, BOOL, NSDictionary*,
                                BOOL*, NSError**);

        auto do_compile = [&](const std::string& src_dir,
                              const std::string& tag) -> std::string {
            NSString* src_ns = [NSString stringWithUTF8String:src_dir.c_str()];
            NSURL*    src_url = [NSURL fileURLWithPath:src_ns];

            NSString* out_dir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:
                    [NSString stringWithUTF8String:("xpc15_out_" + tag).c_str()]];
            [[NSFileManager defaultManager]
                createDirectoryAtPath:out_dir
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSURL* out_url = [NSURL fileURLWithPath:out_dir];

            BOOL ok = NO; NSError* err = nil;
            @try {
                ((CompileFn)objc_msgSend)(cls, sel,
                    src_url, @"model.mil", @"", nil,
                    out_url, src_url, nil, NO, @{}, &ok, &err);
            } @catch (...) { return ""; }

            if (!ok) {
                WARN("15 " << tag << ": compile FAILED — "
                     << (err ? [[err localizedDescription] UTF8String] : "?"));
                return "";
            }
            WARN("15 " << tag << ": compile OK → " << [out_dir UTF8String]);

            // List all files in output dir
            NSArray<NSString*>* items = [[NSFileManager defaultManager]
                subpathsAtPath:out_dir];
            for (NSString* item in items) {
                NSString* full = [out_dir stringByAppendingPathComponent:item];
                NSDictionary* attrs = [[NSFileManager defaultManager]
                    attributesOfItemAtPath:full error:nil];
                unsigned long long sz = [[attrs objectForKey:NSFileSize]
                    unsignedLongLongValue];
                WARN("15 " << tag << ":   " << [item UTF8String]
                     << " (" << sz << " bytes)");
            }

            return [out_dir UTF8String];
        };

        std::string relu_out = do_compile(relu_src, "relu");
        std::string tanh_out = do_compile(tanh_src, "tanh");
        if (relu_out.empty() || tanh_out.empty()) {
            WARN("15: compile step failed — aborting"); return;
        }

        // Now call ane_load_mlmodelc on the actual _ANEMILCompiler output dir.
        // This dir was NOT registered via aned's normal compile path, so if
        // compileModel: succeeds, aned must be reading from disk (model.espresso.net
        // or similar) rather than its in-process compile table.

        auto probe_dir = [&](const std::string& label, const std::string& dir) {
            auto* prog = libane::runtime::ane_load_mlmodelc(
                dir, C, S, C, S, "xpc15-" + label);
            if (!prog) {
                WARN("15 " << label << ": load FAILED — "
                     << libane::runtime::ane_last_error());
                return;
            }
            WARN("15 " << label << ": LOAD OK");

            std::vector<uint16_t> in_d(N, 0x2E66u); // fp16(0.1)
            auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(
                in_d.data(), C, S);
            auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
            if (!in_b || !out_b) {
                WARN("15 " << label << ": buffer alloc failed");
                libane::runtime::ane_unload(prog);
                return;
            }
            bool exec_ok = libane::runtime::ane_execute(
                prog, in_b->iosurface(), out_b->iosurface());
            WARN("15 " << label << ": execute " << (exec_ok ? "ok" : "FAILED"));
            if (exec_ok) {
                std::vector<uint16_t> out_d(N);
                out_b->copy_to(out_d.data(), N * sizeof(uint16_t));
                uint16_t v = out_d[0];
                if (v == 0x2E5Du)
                    WARN("15 " << label << ": *** TANH 0x2E5D ***");
                else if (v >= 0x2E64u && v <= 0x2E68u)
                    WARN("15 " << label << ": RELU ~0x2E66 (0x"
                         << std::hex << v << ")");
                else
                    WARN("15 " << label << ": out=0x" << std::hex << v);
            }
            libane::runtime::ane_unload(prog);
        };

        probe_dir("relu_out", relu_out);
        probe_dir("tanh_out", tanh_out);

        // ── Phase 2: patch model.hwx in relu_out with tanh op bytes, retry ──
        // If RELU output dir loaded and executed correctly above, now swap its
        // model.hwx (if present) with TANH bytes and call ane_load_mlmodelc again.
        // A TANH fingerprint here = aned reads model.hwx from the dir on load.

        // Get TANH HWX bytes via capture
        auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);
        if (tanh_hwx.empty()) { WARN("15 phase2: tanh hwx capture failed"); return; }

        auto relu_hwx = libane::graph::hwx_capture_inline(relu_src);
        if (relu_hwx.empty()) { WARN("15 phase2: relu hwx capture failed"); return; }

        // Build TANH-patched RELU HWX via HwxEmitter
        libane::graph::HwxEmitter em15;
        em15.capture_from_bytes(relu_hwx, C, S, LIBANE_OP_RELU);
        em15.capture_from_bytes(tanh_hwx, C, S, LIBANE_OP_TANH);
        auto patched = em15.emit(C, S, LIBANE_OP_TANH);
        WARN("15 phase2: patched=" << patched.size() << "b");
        if (patched.empty()) { WARN("15 phase2: HwxEmitter emit failed"); return; }

        // Write patched bytes over model.hwx in relu_out (if it exists)
        NSString* relu_out_ns = [NSString stringWithUTF8String:relu_out.c_str()];
        NSString* hwx_path    = [relu_out_ns stringByAppendingPathComponent:@"model.hwx"];
        BOOL hwx_exists = [[NSFileManager defaultManager] fileExistsAtPath:hwx_path];
        WARN("15 phase2: model.hwx in relu_out exists=" << (int)hwx_exists);
        if (hwx_exists) {
            [[NSData dataWithBytes:patched.data() length:patched.size()]
                writeToFile:hwx_path atomically:YES];
            WARN("15 phase2: wrote " << patched.size() << "b tanh-patched bytes over relu_out/model.hwx");
            probe_dir("relu_out_patched", relu_out);
        }
    }
}

// ── XPC-16: _ANECoreMLModelCompiler probe ─────────────────────────────────────
//
// XPC-15 confirmed _ANEMILCompiler only writes model.hwx + model.src.
// _ANEClient.compileModel: requires model.espresso.net (Espresso stack).
//
// New angles discovered from class enumeration:
//   A. _ANECoreMLModelCompiler.pathsForModelURL: — what file layout does aned
//      expect for a CoreML model dir?  Call with a known model URL (post-compile)
//      to see what paths it resolves.
//   B. _ANECoreMLModelCompiler.compileModelAt:csIdentity:key:... — the key:
//      parameter may be what aned uses for hexID lookup.  Passing a known-good
//      hexID key from a prior compile may reuse the aned slot.
//   C. _ANECompilerService.compileModelAt:...cloneDirectory:...withReply: —
//      async XPC variant.  cloneDirectory: may be where aned writes the binary.
//
// Tags: [pathc][xpc][xpc16]

TEST_CASE("XPC-16: _ANECoreMLModelCompiler + _ANECompilerService probe",
          "[pathc][xpc][xpc16]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("16: ANE unavailable"); return; }

        static const char* kCS_BINARY =
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"
            "/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService";

        void* handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if (!handle) handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) { WARN("16: dlopen failed"); return; }

        const int C = 64, S = 512;

        static const char kReluMIL_16[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"xpc16_relu\")];\n    } -> (y);\n}\n";

        // Cold-compile RELU to get a real model_url and hexID
        auto* relu_prog = libane::runtime::ane_compile(
            kReluMIL_16, {}, "xpc16_relu");
        REQUIRE(relu_prog);
        WARN("16: relu_prog model_url=" << relu_prog->model_url);
        WARN("16: relu_prog model_dir=" << relu_prog->model_dir);

        // ── Phase A: pathsForModelURL: on the real model URL ──────────────────
        {
            Class cls = NSClassFromString(@"_ANECoreMLModelCompiler");
            if (!cls) { WARN("16A: _ANECoreMLModelCompiler not found"); }
            else {
                SEL sel = sel_registerName("pathsForModelURL:");
                if (![cls respondsToSelector:sel]) {
                    WARN("16A: pathsForModelURL: not found");
                } else {
                    // Try with model_dir URL (source dir, has model.mil)
                    if (!relu_prog->model_dir.empty()) {
                        NSString* s = [NSString stringWithUTF8String:
                            relu_prog->model_dir.c_str()];
                        NSURL* u = [NSURL fileURLWithPath:s];
                        typedef id (*PathsFn)(Class, SEL, NSURL*);
                        id paths = ((PathsFn)objc_msgSend)(cls, sel, u);
                        WARN("16A model_dir paths: " << (paths
                            ? [[[paths description] componentsSeparatedByString:@"\n"]
                               componentsJoinedByString:@" | "] .UTF8String
                            : "nil"));
                    }

                    // Try with model_url (aned's URL, captured post-compile)
                    if (!relu_prog->model_url.empty()) {
                        NSURL* u = [NSURL URLWithString:
                            [NSString stringWithUTF8String:relu_prog->model_url.c_str()]];
                        typedef id (*PathsFn)(Class, SEL, NSURL*);
                        id paths = ((PathsFn)objc_msgSend)(cls, sel, u);
                        WARN("16A model_url paths: " << (paths
                            ? [[[paths description] componentsSeparatedByString:@"\n"]
                               componentsJoinedByString:@" | "] .UTF8String
                            : "nil"));
                    }
                }
            }
        }

        // ── Phase B: _ANECoreMLModelCompiler.compileModelAt:...key:... ────────
        // key: is the hexID aned uses for compile-table lookup.
        // Hypothesis: passing the RELU hexID key with TANH source might
        // piggyback the existing aned slot → skipPreparePhase.
        {
            Class cls = NSClassFromString(@"_ANECoreMLModelCompiler");
            if (!cls) { WARN("16B: class not found"); goto phase_c; }

            SEL sel = sel_registerName(
                "compileModelAt:csIdentity:key:optionsFilename:tempDirectory:"
                "outputURL:saveSourceModelPath:aotModelBinaryPath:"
                "isEncryptedModel:options:ok:error:");
            if (![cls respondsToSelector:sel]) {
                WARN("16B: selector not found"); goto phase_c;
            }

            // Extract hexID from the relu_prog model_url
            // model_url looks like "file:///var/folders/.../T/{hexID}/" or similar
            // Try to get hexStringIdentifier via the model
            std::string hex_key;
            if (!relu_prog->model_url.empty()) {
                // The hexID is the last path component of the model_url
                NSURL* mu = [NSURL URLWithString:
                    [NSString stringWithUTF8String:relu_prog->model_url.c_str()]];
                hex_key = [[[mu.path lastPathComponent]
                    stringByDeletingPathExtension] UTF8String] ?: "";
                WARN("16B: extracted hex_key=" << hex_key);
            }

            {
                std::string tanh_src = write_mil_dir("xpc16_tanh_src",
                    "program(1.3)\n[buildInfo = dict<string, string>({"
                    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                    "{\"coremlc-version\", \"3505.4.1\"}, "
                    "{\"coremltools-component-milinternal\", \"\"}, "
                    "{\"coremltools-version\", \"9.0\"}})]\n"
                    "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
                    "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
                    "[name=string(\"xpc16_tanh\")];\n    } -> (y);\n}\n");

                NSString* src_ns = [NSString stringWithUTF8String:tanh_src.c_str()];
                NSURL*    src_url = [NSURL fileURLWithPath:src_ns];

                NSString* out_dir_s = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:@"xpc16_coreml_out"];
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:out_dir_s
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSURL* out_url  = [NSURL fileURLWithPath:out_dir_s];
                NSURL* tmp_url  = [NSURL fileURLWithPath:out_dir_s]; // reuse as temp

                NSString* key_ns = [NSString stringWithUTF8String:hex_key.c_str()];
                BOOL ok16 = NO; NSError* err16 = nil;

                typedef id (*CMFn)(Class, SEL,
                                   NSURL*,      // compileModelAt:
                                   NSString*,   // csIdentity:
                                   NSString*,   // key:
                                   NSString*,   // optionsFilename:
                                   NSURL*,      // tempDirectory:
                                   NSURL*,      // outputURL:
                                   NSString*,   // saveSourceModelPath:
                                   NSURL*,      // aotModelBinaryPath:
                                   BOOL,        // isEncryptedModel:
                                   NSDictionary*, // options:
                                   BOOL*,       // ok:
                                   NSError**);  // error:
                @try {
                    ((CMFn)objc_msgSend)(cls, sel,
                        src_url,                    // source dir
                        @"",                        // csIdentity
                        key_ns.length > 0 ? key_ns : @"xpc16_tanh_key",
                        nil,                        // optionsFilename
                        tmp_url,                    // tempDirectory
                        out_url,                    // outputURL
                        [src_ns stringByAppendingPathComponent:@"model.mil"], // saveSourceModelPath
                        nil,                        // aotModelBinaryPath
                        NO,                         // isEncryptedModel
                        @{},                        // options
                        &ok16, &err16);
                } @catch (...) { WARN("16B: exception in compileModelAt:"); goto phase_c; }

                WARN("16B: compile ok=" << (int)ok16
                     << " err=" << (err16 ? [[err16 localizedDescription] UTF8String] : "nil"));
                if (ok16) {
                    NSArray<NSString*>* items = [[NSFileManager defaultManager]
                        subpathsAtPath:out_dir_s];
                    for (NSString* item in items) {
                        NSString* full = [out_dir_s stringByAppendingPathComponent:item];
                        NSDictionary* attrs = [[NSFileManager defaultManager]
                            attributesOfItemAtPath:full error:nil];
                        unsigned long long sz =
                            [[attrs objectForKey:NSFileSize] unsignedLongLongValue];
                        WARN("16B out:   " << [item UTF8String] << " (" << sz << "b)");
                    }
                }
            }
        }
        phase_c:;

        libane::runtime::ane_unload(relu_prog);
    }
}

// ── XPC-17: _ANEInMemoryModel.setModelAttributes: + localModelPath probe ────────
//
// _ANEInMemoryModel has:
//   setModelAttributes: / modelAttributes  — inject attributes before compile
//   localModelPath                         — where aned caches the compiled binary?
//   compilerOptionsWithOptions:isCompiledModelCached:  — what options are generated?
//
// Hypothesis 1: setModelAttributes: can inject isPreCompiledModel=YES +
//   preCompiledModelFilePath=<patched.hwx> → aned loads binary from file.
// Hypothesis 2: localModelPath reveals where aned writes the binary → we can
//   pre-stage our patched bytes there before compileWithQoS:.
// Hypothesis 3: compilerOptionsWithOptions:isCompiledModelCached: returns YES
//   after a cold compile → options contain cache-hit flags we can reproduce.
//
// Tags: [pathc][xpc][xpc17]

TEST_CASE("XPC-17: _ANEInMemoryModel modelAttributes + localModelPath probe",
          "[pathc][xpc][xpc17]") {
    @autoreleasepool {
        libane_set_log_level(LIBANE_LOG_SILENT);
        if (!libane_available()) { WARN("17: ANE unavailable"); return; }

        Class cls_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class cls_Model = NSClassFromString(@"_ANEInMemoryModel");
        REQUIRE(cls_Desc);
        REQUIRE(cls_Model);

        const int C = 64, S = 512;

        static const char kReluMIL_17[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = relu(x=x)"
            "[name=string(\"xpc17_relu\")];\n    } -> (y);\n}\n";

        static const char kTanhMIL_17[] =
            "program(1.3)\n[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n    func main<ios18>(tensor<fp16, [1,64,1,512]> x) {\n"
            "        tensor<fp16, [1,64,1,512]> y = tanh(x=x)"
            "[name=string(\"xpc17_tanh\")];\n    } -> (y);\n}\n";

        // Helper: create model + descriptor, no compile
        auto make_model = [&](const char* mil) -> id {
            NSData* mil_data = [NSData dataWithBytes:mil length:strlen(mil)];
            typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
            id desc = ((DescFn)objc_msgSend)(cls_Desc,
                sel_registerName("modelWithMILText:weights:optionsPlist:"),
                mil_data, @{}, nil);
            if (!desc) return nil;
            typedef id (*ModelFn)(Class, SEL, id);
            return ((ModelFn)objc_msgSend)(cls_Model,
                sel_registerName("inMemoryModelWithDescriptor:"), desc);
        };

        // ── Phase A: inspect localModelPath on a fresh uncompiled model ────────
        {
            id model = make_model(kReluMIL_17);
            if (!model) { WARN("17A: model creation failed"); goto phase_b; }

            SEL sel_lmp = sel_registerName("localModelPath");
            if ([model respondsToSelector:sel_lmp]) {
                id lmp = ((id(*)(id,SEL))objc_msgSend)(model, sel_lmp);
                WARN("17A localModelPath (pre-compile): " << (lmp
                    ? [[lmp description] UTF8String] : "nil"));
            } else {
                WARN("17A: localModelPath not available");
            }

            SEL sel_ma = sel_registerName("modelAttributes");
            if ([model respondsToSelector:sel_ma]) {
                id attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_ma);
                WARN("17A modelAttributes (pre-compile): " << (attrs
                    ? [[attrs description] UTF8String] : "nil"));
            }

            // Try compilerOptionsWithOptions:isCompiledModelCached:
            SEL sel_co = sel_registerName("compilerOptionsWithOptions:isCompiledModelCached:");
            if ([model respondsToSelector:sel_co]) {
                BOOL cached = NO;
                typedef id (*COFn)(id, SEL, id, BOOL*);
                id opts = ((COFn)objc_msgSend)(model, sel_co, @{}, &cached);
                WARN("17A compilerOptions (pre-compile): "
                     << "cached=" << (int)cached
                     << " opts=" << (opts ? [[opts description] UTF8String] : "nil"));
            }
        }
        phase_b:;

        // ── Phase B: cold-compile RELU, then inspect localModelPath + attrs ────
        {
            id model = make_model(kReluMIL_17);
            if (!model) { WARN("17B: model creation failed"); goto phase_c; }

            // Get hexID for logging
            SEL sel_hex = sel_registerName("hexStringIdentifier");
            id hex = [model respondsToSelector:sel_hex]
                ? ((id(*)(id,SEL))objc_msgSend)(model, sel_hex) : nil;
            WARN("17B: relu hexID=" << (hex ? [hex UTF8String] : "?"));

            // compileWithQoS:
            SEL sel_compile = sel_registerName("compileWithQoS:options:error:");
            if (![model respondsToSelector:sel_compile]) {
                WARN("17B: compileWithQoS: not found"); goto phase_c;
            }
            NSError* err = nil;
            typedef BOOL (*CompFn)(id, SEL, unsigned int, id, NSError**);
            BOOL ok = ((CompFn)objc_msgSend)(model, sel_compile, 21u, @{}, &err);
            WARN("17B: compile ok=" << (int)ok
                 << " err=" << (err ? [[err localizedDescription] UTF8String] : "nil"));
            if (!ok) goto phase_c;

            // Read localModelPath post-compile
            SEL sel_lmp = sel_registerName("localModelPath");
            if ([model respondsToSelector:sel_lmp]) {
                id lmp = ((id(*)(id,SEL))objc_msgSend)(model, sel_lmp);
                WARN("17B localModelPath (post-compile): " << (lmp
                    ? [[lmp description] UTF8String] : "nil"));
                // List files at that path if it's a dir
                if (lmp) {
                    NSString* lmp_s = [lmp description];
                    NSArray* items = [[NSFileManager defaultManager]
                        subpathsAtPath:lmp_s];
                    for (NSString* item in items) {
                        NSString* full = [lmp_s stringByAppendingPathComponent:item];
                        NSDictionary* attrs = [[NSFileManager defaultManager]
                            attributesOfItemAtPath:full error:nil];
                        WARN("17B   " << [item UTF8String]
                             << " (" << [[attrs objectForKey:NSFileSize]
                                          unsignedLongLongValue] << "b)");
                    }
                }
            }

            // modelAttributes post-compile
            SEL sel_ma = sel_registerName("modelAttributes");
            if ([model respondsToSelector:sel_ma]) {
                id attrs = ((id(*)(id,SEL))objc_msgSend)(model, sel_ma);
                WARN("17B modelAttributes (post-compile): " << (attrs
                    ? [[attrs description] UTF8String] : "nil"));
            }

            // compilerOptionsWithOptions:isCompiledModelCached: post-compile
            SEL sel_co = sel_registerName("compilerOptionsWithOptions:isCompiledModelCached:");
            if ([model respondsToSelector:sel_co]) {
                BOOL cached = NO;
                typedef id (*COFn)(id, SEL, id, BOOL*);
                id opts = ((COFn)objc_msgSend)(model, sel_co, @{}, &cached);
                WARN("17B compilerOptions (post-compile): "
                     << "cached=" << (int)cached
                     << " opts=" << (opts ? [[opts description] UTF8String] : "nil"));
            }

            // Unload
            SEL sel_unload = sel_registerName("unloadWithQoS:error:");
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, sel_unload, 21u, nil);
        }
        phase_c:;

        // ── Phase C: setModelAttributes: with isPreCompiledModel flag ─────────
        // Set isPreCompiledModel=YES + localModelPath on TANH model before compile.
        // First get the RELU localModelPath (from a real compile) and pre-stage
        // patched TANH bytes there.
        {
            // Get patched TANH HWX bytes
            std::string relu_src = write_mil_dir("xpc17_relu_src", kReluMIL_17);
            std::string tanh_src = write_mil_dir("xpc17_tanh_src", kTanhMIL_17);
            auto relu_hwx = libane::graph::hwx_capture_inline(relu_src);
            auto tanh_hwx = libane::graph::hwx_capture_inline(tanh_src);
            if (relu_hwx.empty() || tanh_hwx.empty()) {
                WARN("17C: capture failed"); return;
            }
            libane::graph::HwxEmitter em17;
            em17.capture_from_bytes(relu_hwx, C, S, LIBANE_OP_RELU);
            em17.capture_from_bytes(tanh_hwx, C, S, LIBANE_OP_TANH);
            auto patched = em17.emit(C, S, LIBANE_OP_TANH);
            WARN("17C: patched=" << patched.size() << "b");

            // Get the local model path aned would use for TANH
            id tanh_model = make_model(kTanhMIL_17);
            if (!tanh_model) { WARN("17C: tanh model creation failed"); return; }

            SEL sel_hex = sel_registerName("hexStringIdentifier");
            id tanh_hex = [tanh_model respondsToSelector:sel_hex]
                ? ((id(*)(id,SEL))objc_msgSend)(tanh_model, sel_hex) : nil;
            WARN("17C: tanh hexID=" << (tanh_hex ? [tanh_hex UTF8String] : "?"));

            // Compute what localModelPath would be for tanh
            // (it's probably TempDir/{hexID} based on what we saw in Phase B)
            if (tanh_hex) {
                NSString* predicted_path = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:tanh_hex];
                WARN("17C: predicted tanh localModelPath=" << [predicted_path UTF8String]);

                // Pre-stage patched HWX at predicted path
                [[NSFileManager defaultManager]
                    createDirectoryAtPath:predicted_path
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSString* hwx_dst = [predicted_path stringByAppendingPathComponent:@"model.hwx"];
                [[NSData dataWithBytes:patched.data() length:patched.size()]
                    writeToFile:hwx_dst atomically:YES];
                WARN("17C: pre-staged patched HWX at " << [hwx_dst UTF8String]);

                // Try setModelAttributes: with pre-compiled flags
                SEL sel_sma = sel_registerName("setModelAttributes:");
                if ([tanh_model respondsToSelector:sel_sma]) {
                    NSDictionary* attrs = @{
                        @"isPrecompiledModel":  @YES,
                        @"isPreCompiled":       @YES,
                        @"skipPreparePhase":    @YES,
                    };
                    ((void(*)(id,SEL,id))objc_msgSend)(tanh_model, sel_sma, attrs);
                    WARN("17C: setModelAttributes: done");
                }

                // Now compile TANH model — does aned skip compilation?
                SEL sel_compile = sel_registerName("compileWithQoS:options:error:");
                NSError* err = nil;
                typedef BOOL (*CompFn)(id, SEL, unsigned int, id, NSError**);
                BOOL ok = ((CompFn)objc_msgSend)(tanh_model, sel_compile, 21u, @{}, &err);
                WARN("17C: tanh compile ok=" << (int)ok
                     << " err=" << (err ? [[err localizedDescription] UTF8String] : "nil"));

                // Check what localModelPath contains after compile
                SEL sel_lmp = sel_registerName("localModelPath");
                if ([tanh_model respondsToSelector:sel_lmp]) {
                    id lmp = ((id(*)(id,SEL))objc_msgSend)(tanh_model, sel_lmp);
                    WARN("17C localModelPath post-compile: " << (lmp
                        ? [[lmp description] UTF8String] : "nil"));
                }

                if (ok) {
                    // Load and execute
                    SEL sel_load = sel_registerName("loadWithQoS:options:error:");
                    NSError* lerr = nil;
                    typedef BOOL (*LoadFn)(id, SEL, unsigned int, id, NSError**);
                    BOOL loaded = ((LoadFn)objc_msgSend)(tanh_model, sel_load, 21u, @{}, &lerr);
                    WARN("17C: tanh load ok=" << (int)loaded
                         << " err=" << (lerr ? [[lerr localizedDescription] UTF8String] : "nil"));
                    if (loaded) {
                        WARN("17C: *** LOAD SUCCEEDED — checking execution fingerprint ***");
                        // Use ane_reconnect-style AneProgram to execute
                        [tanh_model retain];
                        auto* prog     = new libane::runtime::AneProgram{};
                        prog->objc_model = (void*)tanh_model;
                        prog->debug_name = "xpc17_tanh_precomp";

                        const size_t N = (size_t)C * S;
                        std::vector<uint16_t> in_d(N, 0x2E66u);
                        auto in_b  = libane::global_buffer_pool().acquire_tensor_with_data(
                            in_d.data(), C, S);
                        auto out_b = libane::global_buffer_pool().acquire_tensor(C, S);
                        if (in_b && out_b) {
                            bool ex = libane::runtime::ane_execute(
                                prog, in_b->iosurface(), out_b->iosurface());
                            WARN("17C: execute " << (ex ? "ok" : "FAILED"));
                            if (ex) {
                                std::vector<uint16_t> out_d(N);
                                out_b->copy_to(out_d.data(), N * sizeof(uint16_t));
                                uint16_t v = out_d[0];
                                if (v == 0x2E5Du)
                                    WARN("17C: *** TANH LUT 0x2E5D — PRE-COMPILED HWX LOADED! ***");
                                else if (v >= 0x2E64u && v <= 0x2E68u)
                                    WARN("17C: RELU ~0x2E66 (aned recompiled, ignored pre-staged HWX)");
                                else
                                    WARN("17C: out=0x" << std::hex << v);
                            }
                        }
                        libane::runtime::ane_unload(prog);
                    }
                }
            }
        }
    }
}

#else
TEST_CASE("XPC protocol probe: Apple-only", "[pathc][xpc]") { WARN("Apple-only"); }
#endif
