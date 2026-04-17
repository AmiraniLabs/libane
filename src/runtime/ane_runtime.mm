/**
 * ANE Runtime Wrapper — ObjC++ implementation.
 *
 * Implements the maderix/ANE inmem_basic.m dispatch sequence:
 *
 *   descriptor = [_ANEInMemoryModelDescriptor
 *                     modelWithMILText:mil_data
 *                             weights:weights_dict
 *                        optionsPlist:nil]
 *
 *   model = [_ANEInMemoryModel inMemoryModelWithDescriptor:descriptor]
 *
 *   hexID = [model hexStringIdentifier]
 *   // write model.mil + weights/ to /tmp/<hexID>/
 *
 *   [model compileWithQoS:QOS_CLASS_DEFAULT options:@{} error:&err]
 *   [model loadWithQoS:QOS_CLASS_DEFAULT options:@{} error:&err]
 *
 *   input_obj  = [_ANEIOSurfaceObject objectWithIOSurface:input_surface]
 *   output_obj = [_ANEIOSurfaceObject objectWithIOSurface:output_surface]
 *
 *   request = [_ANERequest
 *               requestWithInputs:@[input_obj]
 *                    inputIndices:@[@0]
 *                         outputs:@[output_obj]
 *                   outputIndices:@[@0]
 *                   weightsBuffer:nil
 *                       perfStats:nil
 *                  procedureIndex:0]
 *
 *   [model evaluateWithQoS:QOS_CLASS_DEFAULT
 *                  options:@{} request:request error:&err]
 *
 * Private class names and selectors are resolved lazily via
 * NSClassFromString / sel_registerName so no link-time dependency on the
 * private framework is introduced.
 *
 * References:
 *   github.com/maderix/ANE  (inmem_basic.m, ane_bridge.m)
 *   Orion: Characterizing and Programming Apple's Neural Engine (arXiv:2603.06728)
 */
#include "ane_runtime.hpp"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include <atomic>
#include <algorithm>

// QOS_CLASS_DEFAULT = 0x15 = 21  (matches Orion's proven value)
static constexpr unsigned int kQoS = 21;

static thread_local char tl_error[512] = "";
static void set_error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(tl_error, sizeof(tl_error), fmt, ap);
    va_end(ap);
}

namespace libane {
namespace runtime {

/* ── Private framework symbols ───────────────────────────────────────────── */

static constexpr const char* kFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

struct AneSymbols {
    // ObjC classes (required)
    Class cls_Descriptor    = nil;   // _ANEInMemoryModelDescriptor
    Class cls_Model         = nil;   // _ANEInMemoryModel
    Class cls_IOSurfaceObj  = nil;   // _ANEIOSurfaceObject
    Class cls_Request       = nil;   // _ANERequest

    // ObjC classes (optional — non-nil if successfully resolved)
    Class cls_DeviceInfo    = nil;   // _ANEDeviceInfo

    // Selectors (resolved lazily once)
    SEL sel_modelWithMILText   = nullptr;  // modelWithMILText:weights:optionsPlist:
    SEL sel_inMemoryModel      = nullptr;  // inMemoryModelWithDescriptor:
    SEL sel_hexID              = nullptr;  // hexStringIdentifier
    SEL sel_compile            = nullptr;  // compileWithQoS:options:error:
    SEL sel_load               = nullptr;  // loadWithQoS:options:error:
    SEL sel_unload             = nullptr;  // unloadWithQoS:error:
    SEL sel_evaluate           = nullptr;  // evaluateWithQoS:options:request:error:
    SEL sel_objectWithSurface  = nullptr;  // objectWithIOSurface:
    SEL sel_buildRequest       = nullptr;  // requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:
    SEL sel_processRequest     = nullptr;  // processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:

    bool loaded = false;
};

static AneSymbols     g_syms;
static std::once_flag g_init_flag;
static std::atomic<AneState> g_state{AneState::Uninitialized};
static char g_fallback_reason[512] = "";
static AneDeviceInfo  g_device_info;

/* ── IOReport perf sampler ───────────────────────────────────────────────── */
// Translates ane-perf's Python Sampler to C. Uses libIOReport.dylib which
// requires no entitlements and no root — bandwidth histograms and energy
// counters are available to any userspace process on Apple Silicon.

typedef CFTypeRef (*IOReportCopyChannelsInGroupFn)(CFStringRef, CFStringRef, CFDictionaryRef);
typedef CFTypeRef (*IOReportCreateSubscriptionFn)(void*, CFTypeRef, CFTypeRef*, uint64_t, CFTypeRef);
typedef CFTypeRef (*IOReportCreateSamplesFn)(CFTypeRef, CFTypeRef, CFTypeRef);
typedef CFTypeRef (*IOReportCreateSamplesDeltaFn)(CFTypeRef, CFTypeRef, CFTypeRef);
typedef int       (*IOReportChannelGetFormatFn)(CFTypeRef);
typedef CFStringRef (*IOReportChannelGetChannelNameFn)(CFTypeRef);
typedef CFStringRef (*IOReportChannelGetSubGroupFn)(CFTypeRef);
typedef long      (*IOReportSimpleGetIntegerValueFn)(CFTypeRef);
typedef int       (*IOReportStateGetCountFn)(CFTypeRef);
typedef uint64_t  (*IOReportStateGetResidencyFn)(CFTypeRef, int);

struct IOReportSyms {
    IOReportCopyChannelsInGroupFn    CopyChannelsInGroup    = nullptr;
    IOReportCreateSubscriptionFn     CreateSubscription     = nullptr;
    IOReportCreateSamplesFn          CreateSamples          = nullptr;
    IOReportCreateSamplesDeltaFn     CreateSamplesDelta     = nullptr;
    IOReportChannelGetFormatFn       ChannelGetFormat       = nullptr;
    IOReportChannelGetChannelNameFn  ChannelGetChannelName  = nullptr;
    IOReportChannelGetSubGroupFn     ChannelGetSubGroup     = nullptr;
    IOReportSimpleGetIntegerValueFn  SimpleGetIntegerValue  = nullptr;
    IOReportStateGetCountFn          StateGetCount          = nullptr;
    IOReportStateGetResidencyFn      StateGetResidency      = nullptr;
    bool available = false;
};

struct IOReportSub {
    CFTypeRef subscription = nullptr;
    CFTypeRef channels     = nullptr;
};

static IOReportSyms              g_ior;
static std::vector<IOReportSub>  g_ior_subs;
static std::once_flag            g_ior_init_flag;

static void init_ioreport() {
    void* lib = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return;

#define LOAD_IOR(fn) \
    g_ior.fn = (IOReport##fn##Fn)dlsym(lib, "IOReport" #fn); \
    if (!g_ior.fn) { dlclose(lib); return; }
    LOAD_IOR(CopyChannelsInGroup)
    LOAD_IOR(CreateSubscription)
    LOAD_IOR(CreateSamples)
    LOAD_IOR(CreateSamplesDelta)
    LOAD_IOR(ChannelGetFormat)
    LOAD_IOR(ChannelGetChannelName)
    LOAD_IOR(ChannelGetSubGroup)
    LOAD_IOR(SimpleGetIntegerValue)
    LOAD_IOR(StateGetCount)
    LOAD_IOR(StateGetResidency)
#undef LOAD_IOR

    const char* groups[] = {"Energy Model", "SoC Stats", "PMP"};
    for (const char* grp : groups) {
        CFStringRef gstr = CFStringCreateWithCString(nullptr, grp, kCFStringEncodingUTF8);
        CFTypeRef ch = g_ior.CopyChannelsInGroup(gstr, nullptr, nullptr);
        CFRelease(gstr);
        if (!ch) continue;

        CFMutableDictionaryRef mch = CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0,
                                                                    (CFDictionaryRef)ch);
        CFRelease(ch);

        CFTypeRef out_sub = nullptr;
        CFTypeRef sub = g_ior.CreateSubscription(nullptr, mch, &out_sub, 0, nullptr);
        if (sub) {
            IOReportSub s;
            s.subscription = sub;
            s.channels     = out_sub ? out_sub : (CFTypeRef)mch;
            g_ior_subs.push_back(s);
        }
    }

    g_ior.available = !g_ior_subs.empty();
}

struct IOReportSnapshot {
    std::vector<CFTypeRef> samples;
};

static IOReportSnapshot ioreport_capture() {
    IOReportSnapshot snap;
    for (auto& s : g_ior_subs) {
        CFTypeRef samp = g_ior.CreateSamples(s.subscription, s.channels, nullptr);
        snap.samples.push_back(samp);
    }
    return snap;
}

static void ioreport_delta(const IOReportSnapshot& before,
                           const IOReportSnapshot& after,
                           AnePerfStats* out) {
    if (before.samples.size() != after.samples.size()) return;

    CFStringRef key_ch = CFSTR("IOReportChannels");
    bool found_bw = false;

    for (size_t i = 0; i < before.samples.size(); i++) {
        CFTypeRef s1 = before.samples[i];
        CFTypeRef s2 = after.samples[i];
        if (!s1 || !s2) continue;

        CFTypeRef delta = g_ior.CreateSamplesDelta(s1, s2, nullptr);
        if (!delta) continue;

        CFArrayRef arr = (CFArrayRef)CFDictionaryGetValue((CFDictionaryRef)delta, key_ch);
        if (!arr) { CFRelease(delta); continue; }

        CFIndex count = CFArrayGetCount(arr);
        for (CFIndex j = 0; j < count; j++) {
            CFTypeRef ch  = (CFTypeRef)CFArrayGetValueAtIndex(arr, j);
            int fmt       = g_ior.ChannelGetFormat(ch);
            CFStringRef cfname = g_ior.ChannelGetChannelName(ch);
            CFStringRef cfsg   = g_ior.ChannelGetSubGroup(ch);

            char name[64] = "", sg[64] = "";
            if (cfname) CFStringGetCString(cfname, name, sizeof(name), kCFStringEncodingUTF8);
            if (cfsg)   CFStringGetCString(cfsg,   sg,   sizeof(sg),   kCFStringEncodingUTF8);

            // Energy counter (fmt == 1, integer)
            if (fmt == 1 && strcmp(name, "ANE") == 0)
                out->ane_energy_units = g_ior.SimpleGetIntegerValue(ch);

            // Bandwidth histogram (fmt == 2, state histogram)
            if (fmt == 2 && strcmp(name, "ANE0 RD+WR") == 0 && strcmp(sg, "DCS BW") == 0) {
                int nstates = g_ior.StateGetCount(ch);
                uint64_t total = 0, active = 0;
                double weighted = 0.0;
                int peak = 0;
                for (int s = 0; s < nstates; s++) {
                    uint64_t r = g_ior.StateGetResidency(ch, s);
                    total += r;
                    if (s > 0) { active += r; if (r > 0) peak = s; }
                    weighted += (double)s * r;
                }
                if (total > 0) {
                    out->ane_bw_utilization = (float)active / (float)total;
                    out->avg_bw_state       = (float)(weighted / (double)total);
                    out->peak_bw_state      = peak;
                    found_bw = true;
                }
            }

            // Throttle residency
            if (fmt == 2 && strstr(name, "THROTTLE")) {
                int nstates = g_ior.StateGetCount(ch);
                for (int s = 1; s < nstates; s++)
                    out->throttle_ns += (long)g_ior.StateGetResidency(ch, s);
            }
        }
        CFRelease(delta);
    }

    // Release snapshots
    for (auto p : before.samples) if (p) CFRelease(p);
    for (auto p : after.samples)  if (p) CFRelease(p);

    out->available = found_bw ? 1 : 0;
}

/* ── Initialize ──────────────────────────────────────────────────────────── */

static void do_initialize() {
    g_state = AneState::Fallback;

    // Load the private framework
    void* fh = dlopen(kFrameworkPath, RTLD_NOW | RTLD_LOCAL);
    if (!fh) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "dlopen failed: %s", dlerror());
        return;
    }

    // Resolve classes
    g_syms.cls_Descriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    if (!g_syms.cls_Descriptor) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "_ANEInMemoryModelDescriptor class not found");
        return;
    }

    g_syms.cls_Model = NSClassFromString(@"_ANEInMemoryModel");
    if (!g_syms.cls_Model) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "_ANEInMemoryModel class not found");
        return;
    }

    g_syms.cls_IOSurfaceObj = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_syms.cls_IOSurfaceObj) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "_ANEIOSurfaceObject class not found");
        return;
    }

    g_syms.cls_Request = NSClassFromString(@"_ANERequest");
    if (!g_syms.cls_Request) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "_ANERequest class not found");
        return;
    }

    // Resolve selectors
    g_syms.sel_modelWithMILText  = sel_registerName("modelWithMILText:weights:optionsPlist:");
    g_syms.sel_inMemoryModel     = sel_registerName("inMemoryModelWithDescriptor:");
    g_syms.sel_hexID             = sel_registerName("hexStringIdentifier");
    g_syms.sel_compile           = sel_registerName("compileWithQoS:options:error:");
    g_syms.sel_load              = sel_registerName("loadWithQoS:options:error:");
    g_syms.sel_unload            = sel_registerName("unloadWithQoS:error:");
    g_syms.sel_evaluate          = sel_registerName("evaluateWithQoS:options:request:error:");
    g_syms.sel_objectWithSurface = sel_registerName("objectWithIOSurface:");
    g_syms.sel_buildRequest      = sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
    g_syms.sel_processRequest    = sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:");

    // Verify the model class responds to compile
    if (![g_syms.cls_Model instancesRespondToSelector:g_syms.sel_compile]) {
        snprintf(g_fallback_reason, sizeof(g_fallback_reason),
                 "_ANEInMemoryModel does not respond to compileWithQoS:options:error:");
        return;
    }

    g_syms.loaded = true;

    // ── Optional: _ANEDeviceInfo ───────────────────────────────────────────
    // _ANEDeviceInfo exposes ONLY class methods (no instance methods).
    // Call them directly on the Class object — do NOT alloc/init an instance.
    //
    // Confirmed class method inventory on M3 (probe_device_info, 2026-04-16):
    //   +aneArchitectureType   → NSString  ("h15g", "h16g", …)
    //   +numANECores           → unsigned int  (number of inference cores)
    //   +numANEs               → unsigned int  (number of ANE units)
    //   +aneBoardType          → int64_t
    //   +aneSubType            → NSString
    //   +productName           → NSString
    //   +hasANE                → BOOL
    @try {
        g_syms.cls_DeviceInfo = NSClassFromString(@"_ANEDeviceInfo");
        if (g_syms.cls_DeviceInfo) {
            Class cls = g_syms.cls_DeviceInfo;

            // Architecture type: "h15g" (M3), "h16g" (M4), etc.
            SEL sel_arch = sel_registerName("aneArchitectureType");
            if ([cls respondsToSelector:sel_arch]) {
                NSString* arch = ((NSString*(*)(Class,SEL))objc_msgSend)(cls, sel_arch);
                if (arch && arch.length > 0 && arch.length < 32) {
                    strncpy(g_device_info.architecture, [arch UTF8String], 31);
                    g_device_info.architecture[31] = '\0';
                }
            }

            // Core count — returns unsigned int (type encoding 'I')
            SEL sel_cores = sel_registerName("numANECores");
            if ([cls respondsToSelector:sel_cores]) {
                g_device_info.core_count =
                    ((unsigned int(*)(Class,SEL))objc_msgSend)(cls, sel_cores);
            }

            // Number of ANE units (usually 1, but exposed for completeness)
            SEL sel_num_anes = sel_registerName("numANEs");
            if ([cls respondsToSelector:sel_num_anes]) {
                g_device_info.num_anes =
                    ((unsigned int(*)(Class,SEL))objc_msgSend)(cls, sel_num_anes);
            }

            g_device_info.available = true;
        }
    } @catch (...) {
        // Non-fatal: leave g_device_info in its zero-initialized state
        g_device_info = AneDeviceInfo{};
    }

    g_state = AneState::Available;
}

AneState initialize() {
    std::call_once(g_init_flag, do_initialize);
    return g_state;
}

AneState state() {
    return g_state.load();
}

const char* fallback_reason() {
    return g_fallback_reason;
}

AneDeviceInfo device_info() {
    return g_device_info;
}

const char* ane_last_error() {
    return tl_error;
}

/* ── MIL Parameter Extraction ────────────────────────────────────────────── */

/**
 * Extract input parameter names from MIL function signature.
 * Parses: func main<...>(param1: ..., param2: ..., ...) -> (...)
 * Returns names in declaration order.
 */
static std::vector<std::string> extract_input_params(const std::string& mil_text) {
    std::vector<std::string> params;
    size_t func_pos = mil_text.find("func main");
    if (func_pos == std::string::npos) return params;

    size_t paren_open = mil_text.find('(', func_pos);
    if (paren_open == std::string::npos) return params;

    size_t arrow_pos = mil_text.find("->", paren_open);
    if (arrow_pos == std::string::npos) return params;

    // Extract substring between ( and )
    std::string sig = mil_text.substr(paren_open + 1, arrow_pos - paren_open - 1);

    // Split by ',' and extract parameter names (before ':')
    size_t pos = 0;
    while (pos < sig.length()) {
        // Find next comma or end
        size_t comma = sig.find(',', pos);
        if (comma == std::string::npos) comma = sig.length();

        std::string param = sig.substr(pos, comma - pos);

        // Trim whitespace
        size_t start = param.find_first_not_of(" \t\n\r");
        size_t colon = param.find(':', start);
        if (start != std::string::npos && colon != std::string::npos) {
            std::string name = param.substr(start, colon - start);
            // Trim trailing whitespace
            size_t end = name.find_last_not_of(" \t\n\r");
            if (end != std::string::npos) {
                name = name.substr(0, end + 1);
                params.push_back(name);
            }
        }

        pos = comma + 1;
    }

    return params;
}

/**
 * Extract output variable names from MIL return type.
 * Parses: -> (var1: ..., var2: ..., ...)
 * Returns names in declaration order.
 *
 * Robust against nested parentheses in tensor types: finds the matching closing
 * paren by depth-counting, not just the first ')' character.
 */
static std::vector<std::string> extract_output_vars(const std::string& mil_text) {
    std::vector<std::string> outputs;
    size_t arrow_pos = mil_text.find("->");
    if (arrow_pos == std::string::npos) return outputs;

    size_t paren_open = mil_text.find('(', arrow_pos);
    if (paren_open == std::string::npos) return outputs;

    // Find matching closing paren by depth counting (robust against nested parens in types)
    int depth = 0;
    size_t paren_close = std::string::npos;
    for (size_t i = paren_open; i < mil_text.length(); ++i) {
        if (mil_text[i] == '(') depth++;
        else if (mil_text[i] == ')') {
            depth--;
            if (depth == 0) {
                paren_close = i;
                break;
            }
        }
    }

    if (paren_close == std::string::npos) return outputs;

    // Extract substring between ( and )
    std::string ret_sig = mil_text.substr(paren_open + 1, paren_close - paren_open - 1);

    // Split by ',' and extract variable names (before ':')
    size_t pos = 0;
    while (pos < ret_sig.length()) {
        size_t comma = ret_sig.find(',', pos);
        if (comma == std::string::npos) comma = ret_sig.length();

        std::string var = ret_sig.substr(pos, comma - pos);

        // Trim whitespace
        size_t start = var.find_first_not_of(" \t\n\r");
        size_t colon = var.find(':', start);
        if (start != std::string::npos && colon != std::string::npos) {
            std::string name = var.substr(start, colon - start);
            // Trim trailing whitespace
            size_t end = name.find_last_not_of(" \t\n\r");
            if (end != std::string::npos) {
                name = name.substr(0, end + 1);
                outputs.push_back(name);
            }
        }

        pos = comma + 1;
    }

    return outputs;
}

/* ── Compile ─────────────────────────────────────────────────────────────── */

AneProgram* ane_compile(const std::string& mil_text,
                        const std::vector<WeightEntry>& weights,
                        const std::string& debug_name) {
    if (g_state != AneState::Available) {
        set_error("ANE not available: %s", g_fallback_reason);
        return nullptr;
    }

    @autoreleasepool {
        NSError* error = nil;

        // --- Step 1: Create model descriptor ---
        // MIL text → NSData (UTF-8 bytes)
        NSData* mil_data = [NSData dataWithBytes:mil_text.data()
                                          length:mil_text.size()];

        // Build weight dict in Orion format:
        //   key   = "@model_path/weights/<filename>"  (full path as in BLOBFILE)
        //   value = @{@"offset": @0, @"data": <NSData blob>}
        // This must match the BLOBFILE path in the MIL program exactly.
        // CRITICAL: must be @{} (empty dict), never nil, for weight-free programs.
        NSMutableDictionary* weights_dict = [NSMutableDictionary dictionary];
        for (const auto& w : weights) {
            NSString* full_path = [NSString stringWithFormat:@"@model_path/weights/%s",
                                   w.filename.c_str()];
            NSData*   blob = [NSData dataWithBytes:w.data.data() length:w.data.size()];
            weights_dict[full_path] = @{@"offset": @0, @"data": blob};
        }
        NSDictionary* final_weights = (weights_dict.count > 0) ? weights_dict : @{};

        typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
        id descriptor = ((DescFn)objc_msgSend)(
            g_syms.cls_Descriptor,
            g_syms.sel_modelWithMILText,
            mil_data, final_weights, nil);

        if (!descriptor) {
            set_error("_ANEInMemoryModelDescriptor creation failed");
            return nullptr;
        }

        // --- Step 2: Create in-memory model ---
        typedef id (*ModelFn)(Class, SEL, id);
        id model = ((ModelFn)objc_msgSend)(
            g_syms.cls_Model,
            g_syms.sel_inMemoryModel,
            descriptor);

        if (!model) {
            set_error("_ANEInMemoryModel creation failed");
            return nullptr;
        }

        // --- Step 3: Get hex ID and set up temp dir ---
        typedef NSString* (*StrFn)(id, SEL);
        NSString* hex_id = ((StrFn)objc_msgSend)(model, g_syms.sel_hexID);
        if (!hex_id || hex_id.length == 0) {
            set_error("hexStringIdentifier returned empty string");
            return nullptr;
        }

        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* model_dir = [NSTemporaryDirectory()
                               stringByAppendingPathComponent:hex_id];
        NSString* weights_dir = [model_dir stringByAppendingPathComponent:@"weights"];

        // Create directories
        [fm createDirectoryAtPath:model_dir
          withIntermediateDirectories:YES attributes:nil error:nil];
        [fm createDirectoryAtPath:weights_dir
          withIntermediateDirectories:YES attributes:nil error:nil];

        // Write model.mil
        NSString* mil_path = [model_dir stringByAppendingPathComponent:@"model.mil"];
        [mil_data writeToFile:mil_path atomically:YES];

        // Write weight files to disk.
        // Key format is "@model_path/weights/<name>" — strip "@model_path/" to get
        // relative path, then append to model_dir (matching Orion's approach).
        for (NSString* path in final_weights) {
            NSDictionary* entry = final_weights[path];
            NSData* data = entry[@"data"];
            if (!data) continue;
            NSString* rel = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                            withString:@""];
            NSString* full = [model_dir stringByAppendingPathComponent:rel];
            NSString* dir  = [full stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                           attributes:nil error:nil];
            [data writeToFile:full atomically:YES];
        }

        // --- Step 4: Compile to E5 FlatBuffer ---
        // CRITICAL: QoS must be 21 (DEFAULT), options must be @{} not nil
        // (matches Orion's proven compileWithQoS:options:error: call)
        typedef BOOL (*QoSFn3)(id, SEL, unsigned int, id, NSError**);
        BOOL ok = ((QoSFn3)objc_msgSend)(
            model, g_syms.sel_compile, kQoS, @{}, &error);
        if (!ok || error) {
            set_error("ANE compile failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        // --- Step 5: Load into ANE SRAM ---
        error = nil;
        ok = ((QoSFn3)objc_msgSend)(
            model, g_syms.sel_load, kQoS, @{}, &error);
        if (!ok || error) {
            set_error("ANE load failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        // --- Step 5b: Detect SRAM spill via intermediateBufferHandle ---
        // After loadWithQoS: the firmware sets intermediateBufferHandle on
        // _ANEInMemoryModel to a non-zero IOSurface handle if the model's
        // intermediate activations exceeded ~32 MB on-chip SRAM and were
        // spilled to DRAM.  DRAM-backed intermediates incur ~30% throughput
        // penalty.  Read before retain so the field is available before first
        // execute.
        bool sram_spill = false;
        @try {
            SEL sel_ibh = sel_registerName("intermediateBufferHandle");
            if ([model respondsToSelector:sel_ibh]) {
                uint64_t ibh = ((uint64_t(*)(id,SEL))objc_msgSend)(model, sel_ibh);
                if (ibh != 0) {
                    sram_spill = true;
                    fprintf(stderr,
                            "libane: WARNING: SRAM spill detected "
                            "(intermediateBufferHandle=%llu) — model '%s' "
                            "exceeds ~32 MB SRAM; expect ~30%% throughput drop\n",
                            (unsigned long long)ibh,
                            debug_name.empty() ? "(unnamed)" : debug_name.c_str());
                }
            }
        } @catch (...) {
            // Non-fatal: intermediateBufferHandle unavailable on this firmware
        }

        // Retain the model (manual retain — MRC, not ARC)
        [model retain];

        auto* prog       = new AneProgram{};
        prog->objc_model = (void*)model;
        prog->sram_spill = sram_spill;

        // Extract _ANEProgramForEvaluation for fast-path dispatch (processRequest:).
        // Accessed via KVC: model → inner _ANEModel → _ANEProgramForEvaluation.
        // Non-fatal if unavailable — falls back to evaluateWithQoS:.
        @try {
            id inner = [model valueForKey:@"model"];
            if (inner) {
                id prog_eval = [inner valueForKey:@"program"];
                if (prog_eval) {
                    [inner retain];
                    [prog_eval retain];
                    prog->objc_inner_model = (void*)inner;
                    prog->objc_program     = (void*)prog_eval;
                    prog->model_string_id  =
                        ((uint64_t(*)(id,SEL))objc_msgSend)(inner, sel_registerName("string_id"));
                }
            }
        } @catch (...) {
            // Non-fatal: fast path unavailable, evaluateWithQoS: will be used
            prog->objc_program    = nullptr;
            prog->objc_inner_model = nullptr;
        }
        prog->size_bytes = mil_text.size();
        for (const auto& w : weights) prog->size_bytes += w.data.size();
        prog->debug_name = debug_name;
        prog->model_dir  = [model_dir UTF8String];
        prog->weights    = weights;  // copy for delta reload

        // Extract parameter and output names for constraint validation
        prog->input_param_names = extract_input_params(mil_text);
        prog->output_var_names  = extract_output_vars(mil_text);

        return prog;
    }
}

/* ── Execute ─────────────────────────────────────────────────────────────── */

#ifdef __APPLE__
bool ane_execute_multi(AneProgram* program,
                       const std::vector<IOSurfaceRef>& inputs,
                       const std::vector<IOSurfaceRef>& outputs,
                       AnePerfStats* stats_out) {
    if (!program || !program->objc_model || g_state != AneState::Available) {
        set_error("ane_execute_multi: invalid program or ANE unavailable");
        return false;
    }
    if (inputs.empty() || outputs.empty()) {
        set_error("ane_execute_multi: empty inputs or outputs");
        return false;
    }
    for (auto s : inputs)  { if (!s) { set_error("ane_execute_multi: null input IOSurface"); return false; } }
    for (auto s : outputs) { if (!s) { set_error("ane_execute_multi: null output IOSurface"); return false; } }

    // Constraint #18: all input IOSurfaces must have the same allocation size.
    // The ANE reads flat packed [1,C,1,S] data starting at byte 0 of each surface;
    // mismatched alloc sizes cause silent wrong-data reads at runtime.
    if (inputs.size() > 1) {
        size_t ref_size = IOSurfaceGetAllocSize(inputs[0]);
        for (size_t i = 1; i < inputs.size(); ++i) {
            size_t sz = IOSurfaceGetAllocSize(inputs[i]);
            if (sz != ref_size) {
                set_error("ane_execute_multi: constraint #18 violated — "
                          "input[0] alloc size %zu != input[%zu] alloc size %zu. "
                          "Pad all inputs to the largest size.",
                          ref_size, i, sz);
                return false;
            }
        }
    }

    // Constraint #2: all output IOSurfaces must have the same allocation size.
    // Multi-output programs silently produce garbage if sizes differ.
    if (outputs.size() > 1) {
        size_t ref_size = IOSurfaceGetAllocSize(outputs[0]);
        for (size_t i = 1; i < outputs.size(); ++i) {
            size_t sz = IOSurfaceGetAllocSize(outputs[i]);
            if (sz != ref_size) {
                set_error("ane_execute_multi: constraint #2 violated — "
                          "output[0] alloc size %zu != output[%zu] alloc size %zu. "
                          "Pad all outputs to the largest size.",
                          ref_size, i, sz);
                return false;
            }
        }
    }

    // Constraint #13: Multi-input surfaces must be in alphabetical order of MIL parameter names.
    // ANE reads inputs in alphabetical order regardless of how they are provided.
    std::vector<IOSurfaceRef> inputs_reordered = inputs;
    if (inputs.size() > 1 && !program->input_param_names.empty()) {
        if (program->input_param_names.size() != inputs.size()) {
            set_error("ane_execute_multi: constraint #13 check failed — "
                      "program has %zu input params but %zu inputs provided",
                      program->input_param_names.size(), inputs.size());
            return false;
        }

        // Create sorted index mapping: indices of params in alphabetical order
        std::vector<size_t> alpha_indices(program->input_param_names.size());
        for (size_t i = 0; i < alpha_indices.size(); ++i) alpha_indices[i] = i;

        std::sort(alpha_indices.begin(), alpha_indices.end(),
                  [&](size_t a, size_t b) {
                      return program->input_param_names[a] < program->input_param_names[b];
                  });

        // Reorder inputs to alphabetical order
        std::vector<IOSurfaceRef> temp(inputs.size());
        for (size_t i = 0; i < alpha_indices.size(); ++i) {
            temp[i] = inputs[alpha_indices[i]];
        }
        inputs_reordered = temp;
    }

    // Constraint #3: Multi-output surfaces must be in alphabetical order of MIL variable names.
    // ANE writes outputs in alphabetical order regardless of how they are provided.
    std::vector<IOSurfaceRef> outputs_reordered = outputs;
    if (outputs.size() > 1 && !program->output_var_names.empty()) {
        if (program->output_var_names.size() != outputs.size()) {
            set_error("ane_execute_multi: constraint #3 check failed — "
                      "program has %zu outputs but %zu output surfaces provided",
                      program->output_var_names.size(), outputs.size());
            return false;
        }

        // Create sorted index mapping
        std::vector<size_t> alpha_indices(program->output_var_names.size());
        for (size_t i = 0; i < alpha_indices.size(); ++i) alpha_indices[i] = i;

        std::sort(alpha_indices.begin(), alpha_indices.end(),
                  [&](size_t a, size_t b) {
                      return program->output_var_names[a] < program->output_var_names[b];
                  });

        // Reorder outputs to alphabetical order
        std::vector<IOSurfaceRef> temp(outputs.size());
        for (size_t i = 0; i < alpha_indices.size(); ++i) {
            temp[i] = outputs[alpha_indices[i]];
        }
        outputs_reordered = temp;
    }

    @autoreleasepool {
        id model = (id)program->objc_model;
        NSError* error = nil;

        typedef id (*SurfFn)(Class, SEL, IOSurfaceRef);

        // --- Step 7: Wrap input IOSurfaces in _ANEIOSurfaceObject ---
        NSMutableArray* input_objs   = [NSMutableArray arrayWithCapacity:inputs_reordered.size()];
        NSMutableArray* input_idxs   = [NSMutableArray arrayWithCapacity:inputs_reordered.size()];
        for (NSUInteger i = 0; i < inputs_reordered.size(); ++i) {
            id obj = ((SurfFn)objc_msgSend)(
                g_syms.cls_IOSurfaceObj, g_syms.sel_objectWithSurface, inputs_reordered[i]);
            if (!obj) { set_error("_ANEIOSurfaceObject creation failed for input %lu", (unsigned long)i); return false; }
            [input_objs addObject:obj];
            [input_idxs addObject:@(i)];
        }

        // --- Wrap output IOSurfaces ---
        NSMutableArray* output_objs  = [NSMutableArray arrayWithCapacity:outputs_reordered.size()];
        NSMutableArray* output_idxs  = [NSMutableArray arrayWithCapacity:outputs_reordered.size()];
        for (NSUInteger i = 0; i < outputs_reordered.size(); ++i) {
            id obj = ((SurfFn)objc_msgSend)(
                g_syms.cls_IOSurfaceObj, g_syms.sel_objectWithSurface, outputs_reordered[i]);
            if (!obj) { set_error("_ANEIOSurfaceObject creation failed for output %lu", (unsigned long)i); return false; }
            [output_objs addObject:obj];
            [output_idxs addObject:@(i)];
        }

        // --- Step 8: Build _ANERequest ---
        typedef id (*ReqFn)(Class, SEL,
                            NSArray*, NSArray*,   // inputs, inputIndices
                            NSArray*, NSArray*,   // outputs, outputIndices
                            id, id,               // weightsBuffer, perfStats
                            NSUInteger);          // procedureIndex
        id request = ((ReqFn)objc_msgSend)(
            g_syms.cls_Request,
            g_syms.sel_buildRequest,
            input_objs,    // inputs
            input_idxs,    // inputIndices
            output_objs,   // outputs
            output_idxs,   // outputIndices
            nil,           // weightsBuffer
            nil,           // perfStats (unused — IOReport samples externally)
            (NSUInteger)0);

        if (!request) { set_error("_ANERequest creation failed"); return false; }

        // --- Step 9: IOReport pre-sample + Evaluate ---
        // Sample IOReport before dispatch so we get a clean per-execute delta.
        std::call_once(g_ior_init_flag, init_ioreport);
        IOReportSnapshot before_snap;
        if (stats_out && g_ior.available)
            before_snap = ioreport_capture();

        // Fast path: processRequest: on _ANEProgramForEvaluation — ~13% lower latency
        // by bypassing _ANEInMemoryModel dispatch overhead.
        // Falls back to evaluateWithQoS: if fast-path objects are unavailable.
        if (program->objc_program && program->objc_inner_model) {
            id prog_eval   = (id)program->objc_program;
            id inner_model = (id)program->objc_inner_model;
            uint32_t ret_val = 0;
            typedef BOOL (*ProcReqFn)(id, SEL, id, id, unsigned int,
                                      uint64_t, uint64_t, id, uint32_t*, NSError**);
            BOOL ok = ((ProcReqFn)objc_msgSend)(
                prog_eval,
                g_syms.sel_processRequest,
                request,
                inner_model,
                kQoS,
                (uint64_t)0,
                program->model_string_id,
                @{},
                &ret_val,
                &error);
            if (!ok || error) {
                set_error("ANE processRequest failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
                return false;
            }
        } else {
            typedef BOOL (*EvalFn)(id, SEL, unsigned int, id, id, NSError**);
            BOOL ok = ((EvalFn)objc_msgSend)(
                model, g_syms.sel_evaluate, kQoS, @{}, request, &error);
            if (!ok || error) {
                set_error("ANE evaluate failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
                return false;
            }
        }

        // --- Step 10 (optional): IOReport post-sample and delta ---
        if (stats_out) {
            if (g_ior.available) {
                IOReportSnapshot after_snap = ioreport_capture();
                ioreport_delta(before_snap, after_snap, stats_out);
            } else {
                stats_out->available = 0;
            }
        }

        return true;
    }
}

bool ane_execute(AneProgram* program,
                 IOSurfaceRef input,
                 IOSurfaceRef output) {
    return ane_execute_multi(program, {input}, {output});
}
#else
bool ane_execute(AneProgram*, void*, void*) {
    set_error("ANE not available on this platform");
    return false;
}
#endif

/* ── Delta reload ────────────────────────────────────────────────────────── */

// NOTE ON WEIGHT UPDATES — confirmed via probe_delta_reload (2026-04-16):
// ANE bakes weights into the compiled HWX at compileWithQoS: time.
// Writing new weight blobs to disk before loadWithQoS: has zero effect
// on execution. Weight values cannot be changed without a full recompile.
// This function therefore only performs unload + load (no disk writes).
bool ane_delta_reload(AneProgram* program) {
    if (!program || !program->objc_model || g_state != AneState::Available) {
        set_error("ane_delta_reload: invalid program or ANE unavailable");
        return false;
    }
    if (program->model_dir.empty()) {
        set_error("ane_delta_reload: model_dir not set");
        return false;
    }

    @autoreleasepool {
        id model = (id)program->objc_model;

        // Unload from SRAM
        if (g_syms.loaded) {
            NSError* err = nil;
            typedef BOOL (*UnloadFn)(id, SEL, unsigned int, NSError**);
            ((UnloadFn)objc_msgSend)(model, g_syms.sel_unload, kQoS, &err);
            // Ignore errors on unload — proceed to reload
        }

        // Reload (no compile — reuses the existing compiled HWX bytecode)
        NSError* err = nil;
        typedef BOOL (*QoSFn3)(id, SEL, unsigned int, id, NSError**);
        BOOL ok = ((QoSFn3)objc_msgSend)(model, g_syms.sel_load, kQoS, @{}, &err);
        if (!ok || err) {
            set_error("ane_delta_reload: load failed: %s",
                      err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        return true;
    }
}

/* ── Unload ──────────────────────────────────────────────────────────────── */

void ane_unload(AneProgram* program) {
    if (!program) return;

    if (program->objc_model) {
        @autoreleasepool {
            id model = (id)program->objc_model;

            // Step 10: unload from SRAM (enables delta compilation)
            if (g_syms.loaded) {
                NSError* error = nil;
                typedef BOOL (*UnloadFn)(id, SEL, unsigned int, NSError**);
                ((UnloadFn)objc_msgSend)(model, g_syms.sel_unload, kQoS, &error);
                // Ignore errors — best effort unload
            }

            // Release fast-path objects before releasing the parent model
            if (program->objc_program)     [(id)program->objc_program release];
            if (program->objc_inner_model) [(id)program->objc_inner_model release];

            [model release];
        }
        program->objc_model = nullptr;
    }

    delete program;
}

} // namespace runtime
} // namespace libane
