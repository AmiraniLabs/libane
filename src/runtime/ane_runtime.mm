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
#include <unordered_map>

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

    // ── Path B (_ANEClient + standalone _ANEModel) — optional ─────────────
    Class cls_ANEClient       = nil;
    Class cls_ANEModel_b      = nil;  // standalone _ANEModel (not _ANEInMemoryModel)
    SEL sel_sharedConnection  = nullptr;
    SEL sel_compileModelOpts  = nullptr; // compileModel:options:qos:error:
    SEL sel_loadModel         = nullptr;
    SEL sel_doUnloadModel     = nullptr;
    SEL sel_doEvalDirect      = nullptr; // doEvaluateDirectWithModel:options:request:qos:error:
    SEL sel_mapIOSurfaces     = nullptr; // mapIOSurfacesWithModel:request:cacheInference:error:
    SEL sel_unmapIOSurfaces   = nullptr; // unmapIOSurfacesWithModel:request:
    SEL sel_modelAtURLKey     = nullptr; // modelAtURL:key:
    SEL sel_initWithSurface_b = nullptr; // initWithIOSurface:startOffset:shouldRetain:
    SEL sel_initRequest_b     = nullptr; // initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:
    SEL sel_purgeMatchingHash    = nullptr; // purgeCompiledModelMatchingHash: (Path A slot release)
    SEL sel_compiledModelExists  = nullptr; // compiledModelExists (Path C warm-path skip)
    SEL sel_modelURL             = nullptr; // modelURL (getter — Path C URL reconnect)
    SEL sel_setModelURL          = nullptr; // setModelURL: (setter — Path C URL reconnect)
    bool path_b_loaded           = false;
};

static AneSymbols     g_syms;
static std::once_flag g_init_flag;
static std::atomic<AneState> g_state{AneState::Uninitialized};
static char g_fallback_reason[512] = "";
static AneDeviceInfo  g_device_info;

// ── Compile-slot budget ───────────────────────────────────────────────────────
// aned enforces a hard limit of ~119 unique compilations per process lifetime.
// Exceeding it produces silent failures then a hard crash (SIGSEGV).
// Empirically confirmed by test_qos_sweep.mm Probe 4 (2026-04-22).
//
// Slots consumed by `compileWithQoS:` calls where compiledModelExists=NO.
// `purgeCompiledModel` / `purgeCompiledModelMatchingHash:` may free a slot
// in aned's table but the kernel-side limit appears to be per-process and
// does NOT reset on purge; treat this counter as monotonically increasing.
//
// Hard limit:  kCompileHardLimit  — refuse with a descriptive error
// Warn limit:  kCompileWarnAt     — print to stderr, allow compile
static constexpr int         kCompileHardLimit = 115; // safe margin below 119
static constexpr int         kCompileWarnAt    = 100;
static std::atomic<int>      g_compile_count{0};

// model_dir paths are not guaranteed to be unique per live program (Path A uses
// hexStringIdentifier-derived temp dirs), so teardown must only remove a path
// once the final live program referencing it is gone.
static std::mutex g_model_dir_refs_mu;
static std::unordered_map<std::string, size_t> g_model_dir_refs;

static void retain_model_dir(const std::string& model_dir) {
    if (model_dir.empty()) return;
    std::lock_guard<std::mutex> lock(g_model_dir_refs_mu);
    ++g_model_dir_refs[model_dir];
}

static bool release_model_dir_ref(const std::string& model_dir) {
    if (model_dir.empty()) return false;
    std::lock_guard<std::mutex> lock(g_model_dir_refs_mu);
    auto it = g_model_dir_refs.find(model_dir);
    if (it == g_model_dir_refs.end()) return true; // fail-safe cleanup
    if (it->second > 1) {
        --it->second;
        return false;
    }
    g_model_dir_refs.erase(it);
    return true;
}

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
    g_syms.sel_modelWithMILText     = sel_registerName("modelWithMILText:weights:optionsPlist:");
    g_syms.sel_inMemoryModel        = sel_registerName("inMemoryModelWithDescriptor:");
    g_syms.sel_hexID                = sel_registerName("hexStringIdentifier");
    g_syms.sel_compile              = sel_registerName("compileWithQoS:options:error:");
    g_syms.sel_load                 = sel_registerName("loadWithQoS:options:error:");
    g_syms.sel_unload               = sel_registerName("unloadWithQoS:error:");
    g_syms.sel_evaluate             = sel_registerName("evaluateWithQoS:options:request:error:");
    g_syms.sel_objectWithSurface    = sel_registerName("objectWithIOSurface:");
    g_syms.sel_buildRequest         = sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
    g_syms.sel_processRequest       = sel_registerName("processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:");
    g_syms.sel_compiledModelExists  = sel_registerName("compiledModelExists");

    // Optional: setModelURL: / modelURL for Path C URL reconnect (macOS 26+).
    // Discovered in test_compiler_options_probe.mm Opts-3 (2026-04-22):
    // setModelURL: + loadWithQoS: (no compileWithQoS:) = ~0.722ms when aned slot alive.
    @try {
        SEL sel_mu  = sel_registerName("modelURL");
        SEL sel_smu = sel_registerName("setModelURL:");
        if ([g_syms.cls_Model instancesRespondToSelector:sel_mu] &&
            [g_syms.cls_Model instancesRespondToSelector:sel_smu]) {
            g_syms.sel_modelURL    = sel_mu;
            g_syms.sel_setModelURL = sel_smu;
        }
    } @catch (...) {}

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

    // ── Optional: Path B (_ANEClient / Espresso .mlmodelc) ───────────────
    @try {
        g_syms.cls_ANEClient  = NSClassFromString(@"_ANEClient");
        g_syms.cls_ANEModel_b = NSClassFromString(@"_ANEModel");
        if (g_syms.cls_ANEClient && g_syms.cls_ANEModel_b) {
            g_syms.sel_sharedConnection  = sel_registerName("sharedConnection");
            g_syms.sel_compileModelOpts  = sel_registerName("compileModel:options:qos:error:");
            g_syms.sel_loadModel         = sel_registerName("loadModel:options:qos:error:");
            g_syms.sel_doUnloadModel     = sel_registerName("doUnloadModel:options:qos:error:");
            g_syms.sel_doEvalDirect      = sel_registerName("doEvaluateDirectWithModel:options:request:qos:error:");
            g_syms.sel_mapIOSurfaces     = sel_registerName("mapIOSurfacesWithModel:request:cacheInference:error:");
            g_syms.sel_unmapIOSurfaces   = sel_registerName("unmapIOSurfacesWithModel:request:");
            g_syms.sel_modelAtURLKey     = sel_registerName("modelAtURL:key:");
            g_syms.sel_initWithSurface_b = sel_registerName("initWithIOSurface:startOffset:shouldRetain:");
            g_syms.sel_initRequest_b     = sel_registerName("initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:");
            g_syms.sel_purgeMatchingHash = sel_registerName("purgeCompiledModelMatchingHash:");
            g_syms.path_b_loaded = true;
        }
    } @catch (...) {
        g_syms.path_b_loaded = false;
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

bool path_b_available() {
    return g_syms.path_b_loaded;
}

int ane_compile_count() {
    return g_compile_count.load(std::memory_order_relaxed);
}

int ane_compile_slots_remaining() {
    int used = g_compile_count.load(std::memory_order_relaxed);
    int remaining = kCompileHardLimit - used;
    return remaining < 0 ? 0 : remaining;
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

        // --- Step 3b: Path C warm path — skip compile if aned already has this hexID ---
        // aned deduplicates compiled programs by content hash (hexID).
        // compiledModelExists returns YES if aned's compile table has this hexID,
        // even if we haven't called compileWithQoS: on this particular model object.
        // Skipping compile saves the full ~4200ms ANECompilerService round-trip.
        //
        // Confirmed by test_pathc3_reconnect.mm Probes 1-3 (2026-04-21):
        //   Probe 1:  loadWithQoS: after initWithModelIdentifier inject  → 2.8ms
        //   Probe 2:  loadWithQoS: after setProgramHandle: inject        → 1.0ms
        //   Probe 3:  loadWithQoS: on fresh model, same hexID alive      → 0.7ms
        bool already_compiled = false;
        if (g_syms.sel_compiledModelExists &&
            [g_syms.cls_Model instancesRespondToSelector:g_syms.sel_compiledModelExists]) {
            already_compiled = ((BOOL(*)(id,SEL))objc_msgSend)(
                model, g_syms.sel_compiledModelExists);
        }

        typedef BOOL (*QoSFn3)(id, SEL, unsigned int, id, NSError**);
        BOOL ok = YES;

        if (!already_compiled) {
            // --- Step 4: Compile to E5 FlatBuffer ---
            // CRITICAL: QoS must be 21 (DEFAULT), options must be @{} not nil
            // (matches Orion's proven compileWithQoS:options:error: call)

            // Guard against the per-process compile slot limit (~119).
            // Exceeding it causes silent failures then SIGSEGV.
            // Confirmed empirically: test_qos_sweep Probe 4 (2026-04-22).
            int current_count = g_compile_count.load(std::memory_order_relaxed);
            if (current_count >= kCompileHardLimit) {
                set_error(
                    "ANE compile slot budget exhausted (%d/%d). "
                    "aned enforces a per-process limit of ~119 compilations. "
                    "Use ane_compile_count() to monitor usage. "
                    "Restart the process to reset the budget.",
                    current_count, kCompileHardLimit);
                return nullptr;
            }
            if (current_count >= kCompileWarnAt) {
                fprintf(stderr,
                    "libane: WARNING: ANE compile slot budget at %d/%d. "
                    "Approaching per-process limit (~119). "
                    "Consider reusing compiled programs.\n",
                    current_count, kCompileHardLimit);
            }

            ok = ((QoSFn3)objc_msgSend)(
                model, g_syms.sel_compile, kQoS, @{}, &error);
            if (!ok || error) {
                set_error("ANE compile failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
                return nullptr;
            }
            g_compile_count.fetch_add(1, std::memory_order_relaxed);
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

        // --- Step 5b: Capture model URL for Path C reconnect ---
        // After loadWithQoS:, modelURL holds the URL aned uses to key the compile
        // slot.  Injecting it via setModelURL: on a fresh model + calling
        // loadWithQoS: (no compile) reconnects in ~0.722ms when the slot is alive.
        // Confirmed: test_compiler_options_probe.mm Opts-3 (2026-04-22).
        std::string captured_model_url;
        if (g_syms.sel_modelURL) {
            @try {
                typedef NSURL* (*URLFn)(id, SEL);
                NSURL* url = ((URLFn)objc_msgSend)(model, g_syms.sel_modelURL);
                if (url) captured_model_url = [[url absoluteString] UTF8String];
            } @catch (...) {}
        }

        // --- Step 5c: Detect SRAM spill via intermediateBufferHandle ---
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
        retain_model_dir(prog->model_dir);
        prog->mil_text   = mil_text;
        prog->weights    = weights;
        prog->model_url  = captured_model_url;

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

        // Dispatch priority (fastest → slowest):
        //
        //   Tier 0 — processRequest: on _ANEProgramForEvaluation
        //            Bypasses _ANEInMemoryModel dispatch overhead. ~13% faster
        //            than evaluateWithQoS:. Measured baseline: 346 µs p50.
        //
        //            NOTE: doEvaluateDirectWithModel: on _ANEClient.sharedConnection
        //            was benchmarked (2026-04-17) and found to be 18% SLOWER (401 µs
        //            p50) — the "37% faster" claim in ane-dispatch compares against
        //            CoreML's evaluateWithModel:, not processRequest:. On Path A
        //            (MIL / _ANEInMemoryModel), _ANEClient adds connection-table
        //            lookup overhead that processRequest: avoids.
        //
        //   Tier 1 — evaluateWithQoS: on _ANEInMemoryModel (fallback)
        //
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

        // Invalidate cached _ANEProgramForEvaluation after reload.
        //
        // loadWithQoS: assigns a NEW kernel handle to the outer _ANEInMemoryModel.
        // The previously-retained _ANEProgramForEvaluation (prog->objc_program) still
        // references the OLD kernel handle and causes "Program Inference error" when
        // processRequest: is called with it.
        //
        // The outer model's evaluateWithQoS: always uses the current programHandle
        // (updated by loadWithQoS:), so falling back to Tier 1 is correct.
        // The performance difference vs processRequest: is ~13% — acceptable for
        // a reload scenario.
        //
        // TODO: investigate whether inner._ANEModel.program is refreshed after
        // loadWithQoS: so the fast path can be re-enabled post-reload.
        if (program->objc_program) {
            [(id)program->objc_program release];
            program->objc_program = nullptr;
        }
        if (program->objc_inner_model) {
            [(id)program->objc_inner_model release];
            program->objc_inner_model = nullptr;
        }

        return true;
    }
}

/* ── Program serialization ───────────────────────────────────────────────── */

// HOW ANE COMPILATION WORKS (confirmed by probe_save_files, 2026-04-22):
//
// After compileWithQoS: the model_dir contains only:
//   weights/   — weight blobs written before compile
//   model.mil  — the MIL source text
//   net.plist  — byte-for-byte copy of model.mil (written by aned as a receipt)
//
// No .hwx binary, no compiled flatbuffer, no artifact of any kind.
// The compiled ANE binary lives exclusively inside aned (Apple Neural Engine
// daemon) in its per-process in-memory compile table, keyed by hexID.
//
// saveModelFiles — discovered via ObjC introspection — returns an NSURL
// pointing to the existing model_dir but writes nothing new.  It is a
// read-only accessor, not a serialization mechanism.
//
// Consequence: ane_serialize_program stores only the MIL source + weights.
// ane_restore_program recompiles from scratch.  If aned's per-process cache
// still holds the hexID (Path C warm path), recompile completes in ~1 ms
// instead of the full ~4200 ms cold round-trip.

bool ane_serialize_program(const AneProgram* program, SerializedProgram& out) {
    if (!program) {
        set_error("ane_serialize_program: null program");
        return false;
    }
    if (program->mil_text.empty()) {
        set_error("ane_serialize_program: mil_text not stored "
                  "(program created via ane_load_mlmodelc cannot be serialized)");
        return false;
    }

    out.mil_text          = program->mil_text;
    out.weights           = program->weights;
    out.hwx_bytes.clear();        // always empty — no compiled binary on disk
    out.hwx_rel_path.clear();     // reserved; unused
    out.debug_name        = program->debug_name;
    out.input_param_names = program->input_param_names;
    out.output_var_names  = program->output_var_names;
    return true;
}

AneProgram* ane_restore_program(const SerializedProgram& sp) {
    if (g_state != AneState::Available) {
        set_error("ane_restore_program: ANE not available: %s", g_fallback_reason);
        return nullptr;
    }
    if (sp.mil_text.empty()) {
        set_error("ane_restore_program: empty MIL text");
        return nullptr;
    }

    // The compiled ANE binary lives exclusively inside aned and cannot be
    // restored from a serialized blob — ane_compile() is the only path to a
    // live AneProgram.  sp.hwx_bytes is intentionally ignored: it is always
    // empty in files written by the current ane_serialize_program(); any
    // non-empty value from a hypothetical older file is harmless dead data.
    //
    // Performance: if aned's per-process in-memory cache still holds the
    // hexID for this MIL+weights, ane_compile() takes ~1 ms (Path C warm
    // path); otherwise the full ~4200 ms cold compile round-trip is required.
    return ane_compile(sp.mil_text, sp.weights, sp.debug_name);
}

/* ── Path C URL reconnect ────────────────────────────────────────────────── */

AneProgram* ane_reconnect(const std::string&              mil_text,
                           const std::vector<WeightEntry>& weights,
                           const std::string&              model_url,
                           const std::string&              debug_name) {
    if (g_state != AneState::Available) {
        set_error("ane_reconnect: ANE not available: %s", g_fallback_reason);
        return nullptr;
    }
    if (model_url.empty()) {
        set_error("ane_reconnect: empty model_url");
        return nullptr;
    }
    if (!g_syms.sel_setModelURL || !g_syms.sel_modelURL) {
        set_error("ane_reconnect: setModelURL: not available on this OS");
        return nullptr;
    }

    @autoreleasepool {
        // --- Steps 1-2: Descriptor + model (same as ane_compile) ---
        NSData* mil_data = [NSData dataWithBytes:mil_text.data() length:mil_text.size()];
        NSMutableDictionary* weights_dict = [NSMutableDictionary dictionary];
        for (const auto& w : weights) {
            NSString* full_path = [NSString stringWithFormat:@"@model_path/weights/%s",
                                   w.filename.c_str()];
            NSData* blob = [NSData dataWithBytes:w.data.data() length:w.data.size()];
            weights_dict[full_path] = @{@"offset": @0, @"data": blob};
        }
        NSDictionary* final_weights = (weights_dict.count > 0) ? weights_dict : @{};

        typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
        id descriptor = ((DescFn)objc_msgSend)(
            g_syms.cls_Descriptor, g_syms.sel_modelWithMILText,
            mil_data, final_weights, nil);
        if (!descriptor) {
            set_error("ane_reconnect: descriptor creation failed");
            return nullptr;
        }

        typedef id (*ModelFn)(Class, SEL, id);
        id model = ((ModelFn)objc_msgSend)(
            g_syms.cls_Model, g_syms.sel_inMemoryModel, descriptor);
        if (!model) {
            set_error("ane_reconnect: model creation failed");
            return nullptr;
        }

        // --- Step 3: Write temp dir (idempotent — hexID dir may already exist) ---
        typedef NSString* (*StrFn)(id, SEL);
        NSString* hex_id = ((StrFn)objc_msgSend)(model, g_syms.sel_hexID);
        if (!hex_id || hex_id.length == 0) {
            set_error("ane_reconnect: hexStringIdentifier empty");
            return nullptr;
        }
        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* model_dir_ns = [NSTemporaryDirectory() stringByAppendingPathComponent:hex_id];
        NSString* weights_dir  = [model_dir_ns stringByAppendingPathComponent:@"weights"];
        [fm createDirectoryAtPath:model_dir_ns withIntermediateDirectories:YES
                       attributes:nil error:nil];
        [fm createDirectoryAtPath:weights_dir withIntermediateDirectories:YES
                       attributes:nil error:nil];
        NSString* mil_path = [model_dir_ns stringByAppendingPathComponent:@"model.mil"];
        [mil_data writeToFile:mil_path atomically:YES];
        for (NSString* path in final_weights) {
            NSDictionary* entry = final_weights[path];
            NSData* data = entry[@"data"];
            if (!data) continue;
            NSString* rel  = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                             withString:@""];
            NSString* full = [model_dir_ns stringByAppendingPathComponent:rel];
            NSString* dir  = [full stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                           attributes:nil error:nil];
            [data writeToFile:full atomically:YES];
        }

        // --- Step 4: Inject the stored model URL ---
        NSURL* nsurl = [NSURL URLWithString:[NSString stringWithUTF8String:model_url.c_str()]];
        if (!nsurl) {
            set_error("ane_reconnect: malformed model_url: %s", model_url.c_str());
            return nullptr;
        }
        typedef void (*SetURLFn)(id, SEL, NSURL*);
        ((SetURLFn)objc_msgSend)(model, g_syms.sel_setModelURL, nsurl);

        // --- Step 5: Verify aned still holds the compile slot ---
        if (g_syms.sel_compiledModelExists) {
            BOOL exists = ((BOOL(*)(id,SEL))objc_msgSend)(model, g_syms.sel_compiledModelExists);
            if (!exists) {
                set_error("ane_reconnect: aned slot purged (compiledModelExists=NO) "
                          "for hexID %s — caller must fall back to ane_compile()",
                          [hex_id UTF8String]);
                return nullptr;
            }
        }

        // --- Step 6: Load directly (no compileWithQoS:, no slot consumed) ---
        NSError* error = nil;
        typedef BOOL (*QoSFn3)(id, SEL, unsigned int, id, NSError**);
        BOOL ok = ((QoSFn3)objc_msgSend)(model, g_syms.sel_load, kQoS, @{}, &error);
        if (!ok || error) {
            set_error("ane_reconnect: loadWithQoS: failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        // --- Step 7: Build AneProgram ---
        [model retain];
        auto* prog        = new AneProgram{};
        prog->objc_model  = (void*)model;
        prog->debug_name  = debug_name;
        prog->model_dir   = [model_dir_ns UTF8String];
        retain_model_dir(prog->model_dir);
        prog->mil_text    = mil_text;
        prog->weights     = weights;
        prog->model_url   = model_url;
        prog->size_bytes  = mil_text.size();
        for (const auto& w : weights) prog->size_bytes += w.data.size();

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
        } @catch (...) {}

        prog->input_param_names = extract_input_params(mil_text);
        prog->output_var_names  = extract_output_vars(mil_text);

        return prog;
    }
}

/* ── Load HWX (not supported) ────────────────────────────────────────────── */

// ane_load_hwx is not supported and cannot be made to work.
//
// The premise was: compile a stub model, find its .hwx on disk, overwrite it
// with the caller's bytes, then reload.  This premise is false.  After
// compileWithQoS: the model_dir contains only model.mil + net.plist (a copy of
// model.mil) + an empty weights/ directory.  No .hwx file is written to disk
// at any point in the compile pipeline — confirmed by probe_save_files
// (2026-04-22), which also tested saveModelFiles and found it writes nothing.
//
// The compiled binary lives exclusively inside aned's per-process IPC table.
// There is no client-side filesystem path to intercept.
//
// For program persistence use ane_serialize_program() / ane_restore_program(),
// which serialize the source MIL text and weights instead.
AneProgram* ane_load_hwx(const std::vector<uint8_t>& hwx_bytes,
                          const std::string&          mil_text,
                          int                         /*channels*/,
                          int                         /*seq*/,
                          const std::string&          /*input_name*/,
                          const std::string&          /*output_name*/,
                          const std::string&          debug_name) {
    if (g_state != AneState::Available) {
        set_error("ane_load_hwx: ANE not available: %s", g_fallback_reason);
        return nullptr;
    }
    if (hwx_bytes.empty()) {
        set_error("ane_load_hwx: hwx_bytes is empty");
        return nullptr;
    }
    if (mil_text.empty()) {
        set_error("ane_load_hwx: mil_text is empty");
        return nullptr;
    }

    @autoreleasepool {
        NSError* error = nil;

        // Steps 1-3: same as ane_compile — create descriptor, model, get hexID.
        NSData* mil_data = [NSData dataWithBytes:mil_text.data()
                                          length:mil_text.size()];
        typedef id (*DescFn)(Class, SEL, NSData*, NSDictionary*, id);
        id descriptor = ((DescFn)objc_msgSend)(
            g_syms.cls_Descriptor, g_syms.sel_modelWithMILText,
            mil_data, @{}, nil);
        if (!descriptor) {
            set_error("ane_load_hwx: _ANEInMemoryModelDescriptor creation failed");
            return nullptr;
        }

        typedef id (*ModelFn)(Class, SEL, id);
        id model = ((ModelFn)objc_msgSend)(
            g_syms.cls_Model, g_syms.sel_inMemoryModel, descriptor);
        if (!model) {
            set_error("ane_load_hwx: _ANEInMemoryModel creation failed");
            return nullptr;
        }

        typedef NSString* (*StrFn)(id, SEL);
        NSString* hex_id = ((StrFn)objc_msgSend)(model, g_syms.sel_hexID);
        if (!hex_id || hex_id.length == 0) {
            set_error("ane_load_hwx: hexStringIdentifier returned empty string");
            return nullptr;
        }

        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* model_dir = [NSTemporaryDirectory()
                               stringByAppendingPathComponent:hex_id];
        [fm createDirectoryAtPath:model_dir
          withIntermediateDirectories:YES attributes:nil error:nil];

        [mil_data writeToFile:[model_dir stringByAppendingPathComponent:@"model.mil"]
                   atomically:YES];

        // Pre-stage the HWX binary before compileWithQoS:.
        // aned's "compileAsNeeded" path uses the existing model.hwx rather than
        // running ANECCompile().  Confirmed by XPC-18 (2026-04-22).
        [[NSData dataWithBytes:hwx_bytes.data() length:hwx_bytes.size()]
            writeToFile:[model_dir stringByAppendingPathComponent:@"model.hwx"]
            atomically:YES];

        // Check if aned already has this hexID cached (compiledModelExists).
        // If so, skip compileWithQoS: — aned's cached binary will be used at load.
        bool already_compiled = false;
        if (g_syms.sel_compiledModelExists &&
            [g_syms.cls_Model instancesRespondToSelector:g_syms.sel_compiledModelExists]) {
            already_compiled = ((BOOL(*)(id,SEL))objc_msgSend)(
                model, g_syms.sel_compiledModelExists);
        }

        typedef BOOL (*QoSFn3)(id, SEL, unsigned int, id, NSError**);

        if (!already_compiled) {
            int current_count = g_compile_count.load(std::memory_order_relaxed);
            if (current_count >= kCompileHardLimit) {
                set_error("ane_load_hwx: compile slot budget exhausted (%d/%d)",
                          current_count, kCompileHardLimit);
                return nullptr;
            }
            BOOL ok = ((QoSFn3)objc_msgSend)(
                model, g_syms.sel_compile, kQoS, @{}, &error);
            if (!ok || error) {
                set_error("ane_load_hwx: compileWithQoS: failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "unknown");
                return nullptr;
            }
            g_compile_count.fetch_add(1, std::memory_order_relaxed);
        }

        error = nil;
        BOOL ok = ((QoSFn3)objc_msgSend)(
            model, g_syms.sel_load, kQoS, @{}, &error);
        if (!ok || error) {
            set_error("ane_load_hwx: loadWithQoS: failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        std::string captured_model_url;
        if (g_syms.sel_modelURL) {
            @try {
                typedef NSURL* (*URLFn)(id, SEL);
                NSURL* url = ((URLFn)objc_msgSend)(model, g_syms.sel_modelURL);
                if (url) captured_model_url = [[url absoluteString] UTF8String];
            } @catch (...) {}
        }

        [model retain];
        auto* prog        = new AneProgram{};
        prog->objc_model  = (void*)model;
        prog->debug_name  = debug_name;
        prog->model_dir   = [model_dir UTF8String];
        retain_model_dir(prog->model_dir);
        prog->mil_text    = mil_text;
        prog->model_url   = captured_model_url;
        prog->size_bytes  = mil_text.size() + hwx_bytes.size();

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
                        ((uint64_t(*)(id,SEL))objc_msgSend)(
                            inner, sel_registerName("string_id"));
                }
            }
        } @catch (...) {}

        prog->input_param_names = extract_input_params(mil_text);
        prog->output_var_names  = extract_output_vars(mil_text);

        return prog;
    }
}

/* ── ane_load_mlmodelc (Path B) ──────────────────────────────────────────── */

AneProgram* ane_load_mlmodelc(const std::string& model_dir_path,
                               int in_channels, int in_seq,
                               int out_channels, int out_seq,
                               const std::string& debug_name) {
    if (g_state != AneState::Available) {
        set_error("ane_load_mlmodelc: ANE not available: %s", g_fallback_reason);
        return nullptr;
    }
    if (!g_syms.path_b_loaded) {
        set_error("ane_load_mlmodelc: Path B symbols not available");
        return nullptr;
    }

    @autoreleasepool {
        // 1. _ANEClient.sharedConnection — retained so it survives the autorelease pool
        typedef id (*ClassMsgFn)(Class, SEL);
        id client = ((ClassMsgFn)objc_msgSend)(
            g_syms.cls_ANEClient, g_syms.sel_sharedConnection);
        if (!client) {
            set_error("ane_load_mlmodelc: _ANEClient.sharedConnection returned nil");
            return nullptr;
        }
        [client retain];

        // 2. Create _ANEModel from directory
        NSString* dir_ns  = [NSString stringWithUTF8String:model_dir_path.c_str()];
        NSURL*    dir_url = [NSURL fileURLWithPath:dir_ns];

        typedef id (*ModelAtURLFn)(Class, SEL, NSURL*, NSString*);
        id model_b = ((ModelAtURLFn)objc_msgSend)(
            g_syms.cls_ANEModel_b, g_syms.sel_modelAtURLKey, dir_url, dir_ns);
        if (!model_b) {
            [client release];
            set_error("ane_load_mlmodelc: _ANEModel modelAtURL:key: returned nil");
            return nullptr;
        }
        [model_b retain];

        // 3. Compile via _ANEClient
        NSError* error = nil;
        typedef BOOL (*CompileFn)(id, SEL, id, id, unsigned int, NSError**);
        BOOL ok = ((CompileFn)objc_msgSend)(
            client, g_syms.sel_compileModelOpts, model_b, @{}, kQoS, &error);
        if (!ok || error) {
            [model_b release]; [client release];
            set_error("ane_load_mlmodelc: compileModel failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return nullptr;
        }

        // 4. Load via _ANEClient
        {
            NSError* load_err = nil;
            typedef BOOL (*LoadFn)(id, SEL, id, id, unsigned int, NSError**);
            BOOL loaded = [client respondsToSelector:g_syms.sel_loadModel]
                ? ((LoadFn)objc_msgSend)(client, g_syms.sel_loadModel,
                                         model_b, @{}, kQoS, &load_err)
                : YES;  // assume loaded if method absent (some firmware versions)
            if (!loaded || load_err) {
                [model_b release]; [client release];
                set_error("ane_load_mlmodelc: loadModel failed: %s",
                          load_err ? [[load_err localizedDescription] UTF8String] : "NO");
                return nullptr;
            }
        }

        // 5. Compute stride fallbacks (used when modelAttributes KVC fails)
        auto round64 = [](size_t x) -> size_t {
            return ((x + 63) / 64) * 64;
        };

        auto* prog = new AneProgram{};
        prog->objc_ane_client     = (void*)client;
        prog->objc_client_model   = (void*)model_b;
        prog->client_in_channels  = in_channels;
        prog->client_out_channels = out_channels;
        prog->client_in_seq       = in_seq;
        prog->client_out_seq      = out_seq;
        prog->debug_name          = debug_name;
        // Store the .mlmodelc bundle path so ane_unload() can delete it.
        // ane_delta_reload() is Path A only (it checks objc_model, which is
        // null for Path B), so reusing model_dir here is safe.
        prog->model_dir           = model_dir_path;
        retain_model_dir(prog->model_dir);

        // ANE requires 64-byte plane alignment; plane stride = round64(seq * sizeof(fp16)).
        const size_t in_ps_fallback  = round64((size_t)in_seq  * 2);
        const size_t out_ps_fallback = round64((size_t)out_seq * 2);

        prog->client_in_plane_stride  = in_ps_fallback;
        prog->client_out_plane_stride = out_ps_fallback;
        prog->client_in_batch_stride  = round64(
            std::max((size_t)in_channels  * in_ps_fallback,  (size_t)49152));
        prog->client_out_batch_stride = round64(
            std::max((size_t)out_channels * out_ps_fallback, (size_t)49152));

        // 6. Read authoritative strides + SRAM spill from _ANEModel (KVC)
        @try {
            // intermediateBufferHandle: non-zero = SRAM spill to DRAM
            SEL sel_ibh = sel_registerName("intermediateBufferHandle");
            if ([model_b respondsToSelector:sel_ibh]) {
                uint64_t ibh =
                    ((uint64_t(*)(id,SEL))objc_msgSend)(model_b, sel_ibh);
                if (ibh != 0) {
                    prog->sram_spill = true;
                    fprintf(stderr,
                            "libane: WARNING: Path B SRAM spill "
                            "(intermediateBufferHandle=%llu) — model '%s' "
                            "exceeds SRAM; expect throughput drop\n",
                            (unsigned long long)ibh,
                            debug_name.empty() ? "(unnamed)" : debug_name.c_str());
                }
            }

            // modelAttributes → networkStatusList[0] → liveInputList/liveOutputList
            id attrs = [model_b valueForKey:@"modelAttributes"];
            if (attrs) {
                NSArray* net_list = [attrs valueForKey:@"networkStatusList"];
                if (net_list && [net_list count] > 0) {
                    id net = [net_list objectAtIndex:0];
                    NSArray* in_list  = [net valueForKey:@"liveInputList"];
                    NSArray* out_list = [net valueForKey:@"liveOutputList"];
                    if (in_list && [in_list count] > 0) {
                        id in0 = [in_list objectAtIndex:0];
                        size_t bs = [[in0 valueForKey:@"batchStride"] unsignedLongValue];
                        size_t ps = [[in0 valueForKey:@"planeStride"] unsignedLongValue];
                        if (bs > 0) prog->client_in_batch_stride = bs;
                        if (ps > 0) prog->client_in_plane_stride = ps;
                    }
                    if (out_list && [out_list count] > 0) {
                        id out0 = [out_list objectAtIndex:0];
                        size_t bs = [[out0 valueForKey:@"batchStride"] unsignedLongValue];
                        size_t ps = [[out0 valueForKey:@"planeStride"] unsignedLongValue];
                        if (bs > 0) prog->client_out_batch_stride = bs;
                        if (ps > 0) prog->client_out_plane_stride = ps;
                    }
                }
            }
        } @catch (...) {
            // Non-fatal: computed fallback strides remain in effect
        }

        // 7. Pre-allocate persistent IOSurfaces — reused on every ane_execute_client call.
        //    On first execute they are mapped with cacheInference:YES so that subsequent
        //    doEvaluateDirectWithModel: calls skip IOSurface marshal/unmarshal overhead.
        NSDictionary* in_props = @{
            (__bridge NSString*)kIOSurfaceAllocSize: @(prog->client_in_batch_stride),
        };
        NSDictionary* out_props = @{
            (__bridge NSString*)kIOSurfaceAllocSize: @(prog->client_out_batch_stride),
        };

        IOSurfaceRef in_surf  = IOSurfaceCreate((__bridge CFDictionaryRef)in_props);
        IOSurfaceRef out_surf = IOSurfaceCreate((__bridge CFDictionaryRef)out_props);

        if (!in_surf || !out_surf) {
            if (in_surf)  CFRelease(in_surf);
            if (out_surf) CFRelease(out_surf);
            delete prog;
            [model_b release]; [client release];
            set_error("ane_load_mlmodelc: IOSurface creation failed");
            return nullptr;
        }

        prog->client_in_surf  = (void*)in_surf;   // CFRetained by IOSurfaceCreate
        prog->client_out_surf = (void*)out_surf;

        return prog;
    }
}

/* ── ane_execute_client (Path B) ─────────────────────────────────────────── */

#ifdef __APPLE__
bool ane_execute_client(AneProgram* program,
                        const void* input_fp16,
                        void*       output_fp16) {
    if (!program || !program->objc_ane_client || !program->objc_client_model
        || !program->client_in_surf || !program->client_out_surf) {
        set_error("ane_execute_client: invalid program");
        return false;
    }
    if (g_state != AneState::Available) {
        set_error("ane_execute_client: ANE not available");
        return false;
    }

    @autoreleasepool {
        id client  = (id)program->objc_ane_client;
        id model_b = (id)program->objc_client_model;

        IOSurfaceRef in_surf  = (IOSurfaceRef)program->client_in_surf;
        IOSurfaceRef out_surf = (IOSurfaceRef)program->client_out_surf;

        const size_t in_ps  = program->client_in_plane_stride;
        const int    in_ch  = program->client_in_channels;
        const int    in_seq = program->client_in_seq;

        const size_t out_ps  = program->client_out_plane_stride;
        const int    out_ch  = program->client_out_channels;
        const int    out_seq = program->client_out_seq;

        // Fill input surface: scatter fp16 channels into PlaneStride-spaced slots.
        // Channel j at base + j*PlaneStride; each channel is in_seq fp16 values.
        IOSurfaceLock(in_surf, 0, nullptr);
        {
            uint8_t*       dst     = (uint8_t*)IOSurfaceGetBaseAddress(in_surf);
            const uint8_t* src     = (const uint8_t*)input_fp16;
            const size_t   ch_bytes = (size_t)in_seq * 2;
            for (int j = 0; j < in_ch; ++j)
                memcpy(dst + (size_t)j * in_ps, src + (size_t)j * ch_bytes, ch_bytes);
        }
        IOSurfaceUnlock(in_surf, 0, nullptr);

        // Lazy warm-path setup: done once per AneProgram lifetime.
        if (!program->client_mapped) {
            // Wrap persistent IOSurfaces in _ANEIOSurfaceObject (Path B alloc+init)
            typedef id (*InitSurfFn)(id, SEL, IOSurfaceRef, NSUInteger, BOOL);

            id in_obj  = [g_syms.cls_IOSurfaceObj alloc];
            in_obj  = ((InitSurfFn)objc_msgSend)(
                in_obj,  g_syms.sel_initWithSurface_b, in_surf,  (NSUInteger)0, (BOOL)YES);

            id out_obj = [g_syms.cls_IOSurfaceObj alloc];
            out_obj = ((InitSurfFn)objc_msgSend)(
                out_obj, g_syms.sel_initWithSurface_b, out_surf, (NSUInteger)0, (BOOL)YES);

            if (!in_obj || !out_obj) {
                if (in_obj)  [in_obj release];
                if (out_obj) [out_obj release];
                set_error("ane_execute_client: _ANEIOSurfaceObject init failed");
                return false;
            }

            // Build _ANERequest (9-arg Path B init: sharedEvents + transactionHandle = nil)
            typedef id (*InitReqFn)(id, SEL,
                                    NSArray*, NSArray*,
                                    NSArray*, NSArray*,
                                    id, id, NSUInteger, id, id);
            id request = [g_syms.cls_Request alloc];
            request = ((InitReqFn)objc_msgSend)(
                request, g_syms.sel_initRequest_b,
                @[in_obj],  @[@0],
                @[out_obj], @[@0],
                nil, nil,
                (NSUInteger)0,
                nil, nil);

            if (!request) {
                [in_obj release]; [out_obj release];
                set_error("ane_execute_client: _ANERequest init failed");
                return false;
            }

            // mapIOSurfaces:cacheInference:YES — registers handles with the ANE kernel
            // once.  doEvaluateDirectWithModel: will then use fastConn on all subsequent
            // calls, bypassing the XPC marshal/unmarshal overhead entirely.
            NSError* map_err = nil;
            typedef BOOL (*MapFn)(id, SEL, id, id, BOOL, NSError**);
            BOOL ok = ((MapFn)objc_msgSend)(
                client, g_syms.sel_mapIOSurfaces,
                model_b, request, (BOOL)YES, &map_err);
            if (!ok || map_err) {
                [request release]; [in_obj release]; [out_obj release];
                set_error("ane_execute_client: mapIOSurfaces failed: %s",
                          map_err ? [[map_err localizedDescription] UTF8String] : "NO");
                return false;
            }

            program->client_in_surf_obj  = (void*)in_obj;
            program->client_out_surf_obj = (void*)out_obj;
            program->client_request      = (void*)request;
            program->client_mapped       = true;
        }

        // Execute via fastConn warm path
        id request = (id)program->client_request;
        NSError* error = nil;
        typedef BOOL (*EvalDirectFn)(id, SEL, id, id, id, unsigned int, NSError**);
        BOOL ok = ((EvalDirectFn)objc_msgSend)(
            client, g_syms.sel_doEvalDirect, model_b, @{}, request, kQoS, &error);
        if (!ok || error) {
            set_error("ane_execute_client: doEvaluateDirect failed: %s",
                      error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        // Drain output: gather PlaneStride-spaced channels into flat fp16
        IOSurfaceLock(out_surf, kIOSurfaceLockReadOnly, nullptr);
        {
            const uint8_t* src     = (const uint8_t*)IOSurfaceGetBaseAddress(out_surf);
            uint8_t*       dst     = (uint8_t*)output_fp16;
            const size_t   ch_bytes = (size_t)out_seq * 2;
            for (int j = 0; j < out_ch; ++j)
                memcpy(dst + (size_t)j * ch_bytes, src + (size_t)j * out_ps, ch_bytes);
        }
        IOSurfaceUnlock(out_surf, kIOSurfaceLockReadOnly, nullptr);

        return true;
    }
}
#endif

/* ── Unload ──────────────────────────────────────────────────────────────── */

void ane_unload_sram(AneProgram* program) {
    if (!program || !program->objc_model || g_state != AneState::Available) return;

    @autoreleasepool {
        id model = (id)program->objc_model;
        if (g_syms.loaded) {
            NSError* err = nil;
            typedef BOOL (*UnloadFn)(id, SEL, unsigned int, NSError**);
            ((UnloadFn)objc_msgSend)(model, g_syms.sel_unload, kQoS, &err);
            // Compile slot remains alive — compiledModelExists=YES on next
            // ane_compile() / ane_delta_reload() for the same hexID.
        }
    }
}

void ane_unload(AneProgram* program) {
    if (!program) return;

    // Path B cleanup
    if (program->objc_client_model) {
        @autoreleasepool {
            id client  = (id)program->objc_ane_client;
            id model_b = (id)program->objc_client_model;

            // Unmap cached IOSurfaces from the ANE kernel before releasing
            if (program->client_mapped && client && g_syms.path_b_loaded
                && [client respondsToSelector:g_syms.sel_unmapIOSurfaces]) {
                id request = (id)program->client_request;
                typedef void (*UnmapFn)(id, SEL, id, id);
                ((UnmapFn)objc_msgSend)(
                    client, g_syms.sel_unmapIOSurfaces, model_b, request);
            }

            if (program->client_request)      [(id)program->client_request release];
            if (program->client_in_surf_obj)  [(id)program->client_in_surf_obj release];
            if (program->client_out_surf_obj) [(id)program->client_out_surf_obj release];

            if (program->client_in_surf)  CFRelease((IOSurfaceRef)program->client_in_surf);
            if (program->client_out_surf) CFRelease((IOSurfaceRef)program->client_out_surf);

            if (client && g_syms.path_b_loaded
                && [client respondsToSelector:g_syms.sel_doUnloadModel]) {
                typedef BOOL (*UnloadFn)(id, SEL, id, id, unsigned int, NSError**);
                ((UnloadFn)objc_msgSend)(client, g_syms.sel_doUnloadModel,
                                          model_b, @{}, kQoS, nil);
            }
            [model_b release];
            if (client) [client release];
        }
        program->objc_client_model = nullptr;
        program->objc_ane_client   = nullptr;
    }

    // Path A cleanup
    if (program->objc_model) {
        @autoreleasepool {
            id model = (id)program->objc_model;
            if (g_syms.loaded) {
                // Step 1: Grab hex ID before unload (needed for compile-slot purge below)
                NSString* hex_id = nil;
                @try {
                    typedef NSString* (*StrFn)(id, SEL);
                    hex_id = ((StrFn)objc_msgSend)(model, g_syms.sel_hexID);
                } @catch (...) {}

                // Step 2: Deregister from ANE SRAM (unloadWithQoS:)
                NSError* error = nil;
                typedef BOOL (*UnloadFn)(id, SEL, unsigned int, NSError**);
                ((UnloadFn)objc_msgSend)(model, g_syms.sel_unload, kQoS, &error);

                // Step 3: Purge the compile slot from aned's table.
                //
                // _ANEClient.purgeCompiledModelMatchingHash: was confirmed via
                // test_pathc_probe (Probe C, macOS 26.3.1) to deregister the
                // compile slot immediately, freeing it for reuse without requiring
                // a reboot.  This is belt-and-suspenders on top of unloadWithQoS:
                // and model_dir deletion; together the three mechanisms ensure
                // no slot accumulation across repeated compile/unload cycles.
                if (hex_id && g_syms.path_b_loaded && g_syms.sel_purgeMatchingHash) {
                    @try {
                        id client = ((id(*)(Class,SEL))objc_msgSend)(
                            g_syms.cls_ANEClient, g_syms.sel_sharedConnection);
                        if (client &&
                            [client respondsToSelector:g_syms.sel_purgeMatchingHash]) {
                            typedef BOOL (*PurgeFn)(id, SEL, NSString*, NSError**);
                            NSError* perr = nil;
                            ((PurgeFn)objc_msgSend)(
                                client, g_syms.sel_purgeMatchingHash, hex_id, &perr);
                        }
                    } @catch (...) {}
                }
            }
            if (program->objc_program)     [(id)program->objc_program release];
            if (program->objc_inner_model) [(id)program->objc_inner_model release];
            [model release];
        }
        program->objc_model = nullptr;
    }

    // Delete on-disk model assets only when no live program still references
    // this path.
    if (!program->model_dir.empty()) {
        const std::string model_dir = program->model_dir;
        const bool should_delete = release_model_dir_ref(model_dir);
        if (should_delete) {
            @autoreleasepool {
                NSString* dir = [NSString stringWithUTF8String:model_dir.c_str()];
                [[NSFileManager defaultManager] removeItemAtPath:dir error:nil];
            }
        }
        program->model_dir.clear();
    }

    delete program;
}

} // namespace runtime
} // namespace libane
