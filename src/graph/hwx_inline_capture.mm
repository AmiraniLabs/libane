/**
 * hwx_inline_capture.mm — in-process HWX binary capture via _ANEMILCompiler.
 *
 * On macOS 26 aned no longer writes the compiled binary to the model temp
 * dir.  ANECompilerService.xpc can be dlopen'd into the caller, making
 * _ANEMILCompiler's class method callable directly — same class method the
 * XPC service invokes internally, just via a shorter code path that avoids
 * the XPC round-trip.  We call it with the existing model.mil in model_dir
 * as source, and read model.hwx from a local output directory.
 */
#include "hwx_inline_capture.hpp"

#import <Foundation/Foundation.h>
#include <objc/message.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

static const char kCS_BINARY[] =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"
    "/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService";

// Selector for _ANEMILCompiler class method (confirmed via runtime introspection)
static const char kCompileSel[] =
    "compileModelAt:modelName:csIdentity:optionsFilename:outputURL:"
    "saveSourceURL:aotModelBinaryPath:isEncryptedModel:options:ok:error:";

namespace libane {
namespace graph {

std::vector<uint8_t> hwx_capture_inline(const std::string& model_dir) {
    @autoreleasepool {
        // dlopen ANECompilerService.xpc into this process so the class method
        // is callable directly via objc_msgSend.  RTLD_NOLOAD first: if
        // ANECompilerService is already loaded (e.g. from a prior call)
        // reuse it without re-executing its +loads.
        void* handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if (!handle)
            handle = dlopen(kCS_BINARY, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) return {};

        Class cls = NSClassFromString(@"_ANEMILCompiler");
        if (!cls) return {};

        SEL sel = sel_registerName(kCompileSel);
        if (![cls respondsToSelector:sel]) return {};

        NSString* src_dir_ns = [NSString stringWithUTF8String:model_dir.c_str()];
        NSURL*    src_url    = [NSURL fileURLWithPath:src_dir_ns];

        // Create a unique temp output directory
        NSString* out_dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:
                [NSString stringWithFormat:@"libane_hwx_capture_%lld",
                 (long long)[[NSDate date] timeIntervalSince1970] * 1000]];
        [[NSFileManager defaultManager]
            createDirectoryAtPath:out_dir
            withIntermediateDirectories:YES attributes:nil error:nil];
        NSURL* out_url = [NSURL fileURLWithPath:out_dir];

        BOOL ok = NO;
        NSError* err = nil;

        typedef id (*CompileFn)(Class, SEL,
                                NSURL*,      // compileModelAt: (source dir)
                                NSString*,   // modelName:
                                NSString*,   // csIdentity:
                                NSString*,   // optionsFilename:
                                NSURL*,      // outputURL:
                                NSURL*,      // saveSourceURL:
                                NSURL*,      // aotModelBinaryPath:
                                BOOL,        // isEncryptedModel:
                                NSDictionary*, // options:
                                BOOL*,       // ok:
                                NSError**);  // error:

        @try {
            ((CompileFn)objc_msgSend)(
                cls, sel,
                src_url,      // model source directory (contains model.mil)
                @"model.mil", // model filename inside that directory
                @"",          // csIdentity
                nil,          // optionsFilename
                out_url,      // outputURL — model.hwx written here
                src_url,      // saveSourceURL
                nil,          // aotModelBinaryPath (unused — binary goes to outputURL)
                NO,           // isEncryptedModel
                @{},          // options
                &ok, &err);
        } @catch (...) {
            return {};
        }

        if (!ok) return {};

        // Read model.hwx from the output dir
        NSString* hwx_path = [out_dir stringByAppendingPathComponent:@"model.hwx"];
        NSData*   hwx_data = [NSData dataWithContentsOfFile:hwx_path];
        if (!hwx_data || hwx_data.length < 0x4060) return {};

        // Verify BEEFFACE magic
        uint32_t magic = 0;
        [hwx_data getBytes:&magic length:4];
        if (magic != 0xBEEFFACEu) return {};

        std::vector<uint8_t> result((const uint8_t*)hwx_data.bytes,
                                    (const uint8_t*)hwx_data.bytes + hwx_data.length);

        // Clean up temp output directory
        [[NSFileManager defaultManager]
            removeItemAtPath:out_dir error:nil];

        return result;
    }
}

} // namespace graph
} // namespace libane
