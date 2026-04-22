/**
 * probe_device_info.m — _ANEDeviceInfo interface probe
 *
 * Discovers the exact method names and return types that _ANEDeviceInfo
 * exposes on the current firmware.  Run this before integrating new device
 * info methods into the library.
 *
 * Build:
 *   clang -fobjc-arc -fmodules -framework Foundation \
 *         -o probe_device_info probe_device_info.m && ./probe_device_info
 *
 * Expected output on M3 (h15g):
 *   _ANEDeviceInfo: found
 *   architectureType  → "h15g"
 *   totalNeuralEngineCoreCount → 16  (or similar)
 *   ...
 *
 * Expected output on M4 (h16g):
 *   _ANEDeviceInfo: found
 *   architectureType  → "h16g"
 *   ...
 */

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

// ── Helpers ────────────────────────────────────────────────────────────────

static void dump_methods(Class cls, const char* label) {
    printf("\n[%s instance methods]\n", label);
    unsigned int count = 0;
    Method* methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; ++i) {
        printf("  - %s  (enc: %s)\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);

    printf("\n[%s class methods]\n", label);
    methods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; ++i) {
        printf("  + %s  (enc: %s)\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(void) {
    @autoreleasepool {
        void* fh = dlopen(kFrameworkPath, RTLD_NOW | RTLD_LOCAL);
        if (!fh) {
            fprintf(stderr, "dlopen failed: %s\n", dlerror());
            return 1;
        }

        // ── Step 1: Resolve _ANEDeviceInfo ─────────────────────────────────
        Class cls = NSClassFromString(@"_ANEDeviceInfo");
        if (!cls) {
            printf("_ANEDeviceInfo: NOT FOUND on this firmware\n");
            return 0;
        }
        printf("_ANEDeviceInfo: found\n\n");

        // Dump all methods for discovery
        dump_methods(cls, "_ANEDeviceInfo");
        printf("\n");

        // ── Step 2: Query class methods directly ───────────────────────────
        // _ANEDeviceInfo has NO instance methods — all methods are class methods.
        // Confirmed via probe on M3 Pro (2026-04-16).

        printf("--- Confirmed class method values ---\n");

        // Architecture type
        printf("\n+aneArchitectureType: ");
        SEL s_arch = sel_registerName("aneArchitectureType");
        if ([cls respondsToSelector:s_arch]) {
            @try {
                NSString* val = ((NSString*(*)(Class,SEL))objc_msgSend)(cls, s_arch);
                printf("%s\n", val ? [val UTF8String] : "(nil)");
            } @catch (NSException* e) {
                printf("EXCEPTION: %s\n", [[e reason] UTF8String]);
            }
        } else {
            printf("(not found)\n");
        }

        printf("+numANECores: ");
        SEL s_cores = sel_registerName("numANECores");
        if ([cls respondsToSelector:s_cores]) {
            unsigned int val = ((unsigned int(*)(Class,SEL))objc_msgSend)(cls, s_cores);
            printf("%u\n", val);
        } else {
            printf("(not found)\n");
        }

        printf("+numANEs: ");
        SEL s_nanes = sel_registerName("numANEs");
        if ([cls respondsToSelector:s_nanes]) {
            unsigned int val = ((unsigned int(*)(Class,SEL))objc_msgSend)(cls, s_nanes);
            printf("%u\n", val);
        } else {
            printf("(not found)\n");
        }

        printf("+aneBoardType: ");
        SEL s_board = sel_registerName("aneBoardType");
        if ([cls respondsToSelector:s_board]) {
            int64_t val = ((int64_t(*)(Class,SEL))objc_msgSend)(cls, s_board);
            printf("%lld\n", (long long)val);
        } else {
            printf("(not found)\n");
        }

        printf("+aneSubType: ");
        SEL s_sub = sel_registerName("aneSubType");
        if ([cls respondsToSelector:s_sub]) {
            @try {
                NSString* val = ((NSString*(*)(Class,SEL))objc_msgSend)(cls, s_sub);
                printf("%s\n", val ? [val UTF8String] : "(nil)");
            } @catch (...) { printf("(exception)\n"); }
        } else {
            printf("(not found)\n");
        }

        printf("+productName: ");
        SEL s_prod = sel_registerName("productName");
        if ([cls respondsToSelector:s_prod]) {
            @try {
                NSString* val = ((NSString*(*)(Class,SEL))objc_msgSend)(cls, s_prod);
                printf("%s\n", val ? [val UTF8String] : "(nil)");
            } @catch (...) { printf("(exception)\n"); }
        } else {
            printf("(not found)\n");
        }

        printf("+hasANE: ");
        SEL s_has = sel_registerName("hasANE");
        if ([cls respondsToSelector:s_has]) {
            BOOL val = ((BOOL(*)(Class,SEL))objc_msgSend)(cls, s_has);
            printf("%s\n", val ? "YES" : "NO");
        } else {
            printf("(not found)\n");
        }

        // Probe all other class methods for undiscovered SRAM/shape-limit fields
        printf("\n--- All class methods (raw values) ---\n");
        unsigned int count = 0;
        Method* methods = class_copyMethodList(object_getClass(cls), &count);
        for (unsigned int i = 0; i < count; ++i) {
            SEL s = method_getName(methods[i]);
            const char* enc = method_getTypeEncoding(methods[i]);
            // Only call zero-argument methods (no ':' in name)
            if (strchr(sel_getName(s), ':')) continue;
            @try {
                // Dispatch based on return type encoding
                char ret = enc[0];
                if (ret == '@') {
                    id val = ((id(*)(Class,SEL))objc_msgSend)(cls, s);
                    printf("  +%s → %s\n", sel_getName(s),
                           val ? [[val description] UTF8String] : "(nil)");
                } else if (ret == 'q' || ret == 'Q' || ret == 'i' || ret == 'I' ||
                           ret == 'l' || ret == 'L') {
                    int64_t val = ((int64_t(*)(Class,SEL))objc_msgSend)(cls, s);
                    printf("  +%s → %lld\n", sel_getName(s), (long long)val);
                } else if (ret == 'B') {
                    BOOL val = ((BOOL(*)(Class,SEL))objc_msgSend)(cls, s);
                    printf("  +%s → %s\n", sel_getName(s), val ? "YES" : "NO");
                } else {
                    printf("  +%s  (enc: %s — skipped)\n", sel_getName(s), enc);
                }
            } @catch (...) {
                printf("  +%s  (exception)\n", sel_getName(s));
            }
        }
        free(methods);

        printf("\nDone.\n");
        return 0;
    }
}
