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
    void*  objc_model   = nullptr;  ///< ObjC _ANEInMemoryModel* (retained)
    size_t size_bytes   = 0;        ///< Approximate memory footprint
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
                       const std::vector<IOSurfaceRef>& outputs);
#else
bool ane_execute(AneProgram* program, void* input, void* output);
#endif

/**
 * Reload a compiled program with new weights WITHOUT recompiling.
 * 8.5x faster than ane_compile() per Orion Table 6 (494ms vs 4,200ms).
 *
 * @param program    Handle from ane_compile() — model_dir must still exist.
 * @param new_weights  New weight entries. Must match the filenames in program->weights.
 * @return true on success. On failure the program is in an undefined state — caller must ane_unload() and ane_compile() fresh.
 */
bool ane_delta_reload(AneProgram* program,
                      const std::vector<WeightEntry>& new_weights);

/**
 * Unload a compiled ANE program and release its resources.
 * Calls unloadWithQoS: on the model before releasing, enabling delta
 * compilation on the next ane_compile() call for the same model_dir.
 * Safe to call with nullptr.
 */
void ane_unload(AneProgram* program);

/** Last error string from the ANE runtime (thread-local). */
const char* ane_last_error();

} // namespace runtime
} // namespace libane
