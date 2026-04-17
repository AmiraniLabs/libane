/**
 * ANE Runtime Wrapper (ObjC++ interface)
 *
 * Wraps AppleNeuralEngine.framework via dlopen.
 * Implements the correct 10-step dispatch sequence from maderix/ANE:
 *
 *  1.  modelWithMILText:weights:optionsPlist:  → _ANEInMemoryModelDescriptor
 *  2.  inMemoryModelWithDescriptor:            → _ANEInMemoryModel
 *  3.  hexStringIdentifier + write temp dir    (model.mil + weights/)
 *  4.  compileWithQoS:options:error:           → E5 FlatBuffer on disk
 *  5.  loadWithQoS:options:error:              → SRAM resident
 *  6.  Create IOSurfaces (≥49 KB, format=0, bpe=1)
 *  7.  Wrap in _ANEIOSurfaceObject
 *  8.  requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:
 *  9.  evaluateWithQoS:options:request:error:
 *  10. (optional) unloadWithQoS:error:  for delta compilation
 *
 * Delta compilation: overwrite weight files in model_dir → load (no recompile).
 *   Cold: ~4200 ms  |  Delta: ~494 ms  |  Warm cache: <1 ms
 *
 * Private API symbols (all resolved via NSClassFromString / sel_registerName):
 *   _ANEInMemoryModelDescriptor, _ANEInMemoryModel,
 *   _ANEIOSurfaceObject, _ANERequest
 *
 * If any symbol lookup fails, sets state = Fallback and all operations are no-ops.
 *
 * Implementation: src/runtime/ane_runtime.mm (ObjC++)
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

#ifdef __APPLE__
#  include <IOSurface/IOSurface.h>
#endif

namespace libane {
namespace runtime {

/* ── Runtime state ───────────────────────────────────────────────────────── */

enum class AneState {
    Uninitialized,
    Available,      ///< ANE accessible, private API loaded
    Fallback,       ///< ANE unavailable, all ops route to CPU
};

/**
 * Initialize the ANE runtime.
 * Thread-safe; idempotent (safe to call multiple times).
 *
 * @return AneState::Available if ANE is accessible, AneState::Fallback otherwise.
 */
AneState initialize();

/** Return current runtime state without re-initializing. */
AneState state();

/** Human-readable description of why fallback mode was entered (if applicable). */
const char* fallback_reason();

/* ── Weight entry ────────────────────────────────────────────────────────── */

/**
 * One entry in the ANE weight dictionary.
 * filename = the key used in the MIL file() reference (e.g., "weight.bin").
 * data     = full weight blob including the 128-byte ANE header.
 */
struct WeightEntry {
    std::string          filename;  ///< e.g., "weight.bin" or "wq.bin"
    std::vector<uint8_t> data;      ///< blob[0..127]=ANE header, blob[128..]=fp16 weights
};

/* ── Compiled program handle ─────────────────────────────────────────────── */

/**
 * Opaque handle to a compiled ANE model.
 * Created by ane_compile(), destroyed by ane_unload().
 */
struct AneProgram {
    void*  objc_model        = nullptr;  ///< ObjC _ANEInMemoryModel* (retained)
    void*  objc_program      = nullptr;  ///< ObjC _ANEProgramForEvaluation* (retained), nil if unavailable
    void*  objc_inner_model  = nullptr;  ///< ObjC _ANEModel* (retained), for processRequest:
    uint64_t model_string_id = 0;        ///< string_id of the inner _ANEModel
    size_t size_bytes        = 0;        ///< Approximate memory footprint
    bool   sram_spill        = false;    ///< true if model exceeded SRAM (intermediateBufferHandle != 0)
    std::string debug_name;
    std::string model_dir;          ///< Temp dir path for delta compilation
    std::vector<WeightEntry> weights;  ///< stored for delta reload

    /// Parameter names extracted from MIL function signature (inputs in declaration order).
    /// Used for constraint #13 validation (alphabetical input ordering).
    std::vector<std::string> input_param_names;

    /// Output variable names extracted from MIL return type (outputs in declaration order).
    /// Used for constraint #3 validation (alphabetical output ordering).
    std::vector<std::string> output_var_names;
};

/* ── Device info ─────────────────────────────────────────────────────────── */

/**
 * Hardware capabilities queried once at initialization from _ANEDeviceInfo.
 * All fields are zero/empty if _ANEDeviceInfo is unavailable (fallback or
 * older firmware that doesn't expose the class).
 */
struct AneDeviceInfo {
    char     architecture[32] = "";   ///< e.g. "h15g" (M3), "h16g" (M4); "" if unavailable
    uint32_t core_count       = 0;    ///< number of ANE inference cores (+numANECores); 0 if unavailable
    uint32_t num_anes         = 0;    ///< number of ANE units (+numANEs); 0 if unavailable
    bool     available        = false;///< true if _ANEDeviceInfo was successfully queried
};

/* ── Performance stats ───────────────────────────────────────────────────── */

/**
 * Per-execution hardware counters from IOReport (libIOReport.dylib).
 * Populated by ane_execute_multi when stats_out != nullptr.
 * available == 0 if IOReport could not be sampled.
 */
struct AnePerfStats {
    float ane_bw_utilization = 0.0f;
    float avg_bw_state       = 0.0f;
    int   peak_bw_state      = 0;
    long  ane_energy_units   = 0;
    long  throttle_ns        = 0;
    int   available          = 0;
};

/* ── Core primitives ─────────────────────────────────────────────────────── */

/**
 * Compile a MIL text program to an ANE program.
 *
 * @param mil_text   UTF-8 MIL source (passed as NSData* to _ANEInMemoryModelDescriptor).
 * @param weights    Weight files to embed (each has filename + blob-with-header).
 * @param debug_name Optional label.
 *
 * @return Non-null AneProgram* on success; caller must call ane_unload().
 *         Returns nullptr on failure (check ane_last_error()).
 *         Always returns nullptr in Fallback mode.
 */
AneProgram* ane_compile(const std::string& mil_text,
                        const std::vector<WeightEntry>& weights,
                        const std::string& debug_name = "");

/**
 * Execute a compiled ANE program.
 *
 * Wraps the IOSurfaces in _ANEIOSurfaceObject and dispatches via _ANERequest.
 *
 * @param program  Handle from ane_compile().
 * @param input    IOSurface-backed input buffer (≥49 KB, bpe=1, format=0).
 * @param output   IOSurface-backed output buffer (caller-allocated, same spec).
 *
 * @return true on success, false on failure.
 */
#ifdef __APPLE__
bool ane_execute(AneProgram* program,
                 IOSurfaceRef input,
                 IOSurfaceRef output);

/**
 * Execute a compiled ANE program with multiple inputs and outputs.
 *
 * Automatically enforces Orion constraints #2, #3, #13, #18:
 *
 * - #2:  All output IOSurfaces have uniform alloc size (enforced; caller must provide)
 * - #3:  Output IOSurfaces are in alphabetical order of MIL var names
 *        (enforced; automatically reorders if caller provides them out of order)
 * - #13: Input IOSurfaces are in alphabetical order of MIL param names
 *        (enforced; automatically reorders if caller provides them out of order)
 * - #18: All input IOSurfaces have uniform alloc size (enforced; caller must provide)
 *
 * Caller may provide inputs/outputs in any order; the runtime automatically
 * reorders them to alphabetical order before passing to ANE.
 *
 * @param program  Handle from ane_compile().
 * @param inputs   IOSurface-backed input buffers (will be reordered if needed).
 * @param outputs  IOSurface-backed output buffers (will be reordered if needed).
 *
 * @return true on success, false on failure.
 */
bool ane_execute_multi(AneProgram* program,
                       const std::vector<IOSurfaceRef>& inputs,
                       const std::vector<IOSurfaceRef>& outputs,
                       AnePerfStats* stats_out = nullptr);
#else
bool ane_execute(AneProgram* program, void* input, void* output);
#endif

/**
 * Re-load a compiled program into SRAM without recompiling.
 *
 * Useful after ane_unload() to restore a program to SRAM quickly.
 * Load-only is ~8.5x faster than a full ane_compile() (494 ms vs 4,200 ms).
 *
 * NOTE: Weight values cannot be changed after compilation.
 * Confirmed via probe_delta_reload (2026-04-16, M3 Pro / macOS 26.3.1):
 * overwriting the on-disk weight file has zero effect on execution —
 * the ANE bakes weights into the compiled HWX at compileWithQoS: time.
 * Any caller expecting ane_delta_reload() to update weights will silently
 * execute stale values. Use ane_compile() to change weights.
 *
 * @param program  Handle from ane_compile() — model_dir must still exist.
 * @return true on success. On failure call ane_unload() + ane_compile().
 */
bool ane_delta_reload(AneProgram* program);

/**
 * Unload a compiled ANE program and release its resources.
 * Calls unloadWithQoS: on the model before releasing, enabling delta
 * compilation on the next ane_compile() call for the same model_dir.
 * Safe to call with nullptr.
 */
void ane_unload(AneProgram* program);

/** Last error string from the ANE runtime (thread-local). */
const char* ane_last_error();

/**
 * Return cached device info (populated during initialize()).
 * Thread-safe after initialize() completes.
 */
AneDeviceInfo device_info();

} // namespace runtime
} // namespace libane
