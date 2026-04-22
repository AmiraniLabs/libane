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
 * Created by ane_compile() or ane_load_mlmodelc(), destroyed by ane_unload().
 *
 * Path A fields (objc_model*): populated by ane_compile() / ane_load_hwx().
 * Path B fields (objc_client_model / objc_ane_client): populated by ane_load_mlmodelc().
 * Exactly one path is active per AneProgram instance.
 */
struct AneProgram {
    // ── Path A (_ANEInMemoryModel) ─────────────────────────────────────────
    void*    objc_model        = nullptr;  ///< ObjC _ANEInMemoryModel* (retained)
    void*    objc_program      = nullptr;  ///< ObjC _ANEProgramForEvaluation* (retained)
    void*    objc_inner_model  = nullptr;  ///< ObjC _ANEModel* (retained), for processRequest:
    uint64_t model_string_id   = 0;        ///< string_id of the inner _ANEModel
    size_t   size_bytes        = 0;        ///< Approximate memory footprint
    bool     sram_spill        = false;    ///< true if intermediateBufferHandle != 0 after load
    std::string debug_name;
    std::string model_dir;               ///< Temp dir path for delta compilation
    std::string mil_text;                ///< Original MIL source (stored for serialization)
    std::string model_url;               ///< URL from _ANEInMemoryModel.modelURL after compile; inject via setModelURL: + loadWithQoS: for ~0.722ms Path C reconnect
    std::string hex_id;                  ///< hexStringIdentifier — aned's cache key for this (mil_text, weights) pair
    std::vector<WeightEntry> weights;    ///< stored for delta reload
    std::vector<std::string> input_param_names;
    std::vector<std::string> output_var_names;

    // ── Path B (_ANEClient / Espresso .mlmodelc) ───────────────────────────
    // IOSurfaces and ObjC wrappers are created once in ane_load_mlmodelc and
    // reused across all ane_execute_client calls.  On first execute, they are
    // mapped with cacheInference:YES so that subsequent doEvaluateDirect: calls
    // use the fastConn warm path and skip IOSurface marshal/unmarshal overhead.
    void*    objc_ane_client        = nullptr; ///< _ANEClient* (retained)
    void*    objc_client_model      = nullptr; ///< _ANEModel* (retained)
    void*    client_in_surf         = nullptr; ///< IOSurfaceRef (CFRetained) — input surface
    void*    client_out_surf        = nullptr; ///< IOSurfaceRef (CFRetained) — output surface
    void*    client_in_surf_obj     = nullptr; ///< _ANEIOSurfaceObject* (retained)
    void*    client_out_surf_obj    = nullptr; ///< _ANEIOSurfaceObject* (retained)
    void*    client_request         = nullptr; ///< _ANERequest* (retained)
    bool     client_mapped          = false;   ///< true after mapIOSurfaces:cacheInference:YES
    size_t   client_in_batch_stride  = 0;
    size_t   client_out_batch_stride = 0;
    size_t   client_in_plane_stride  = 0;
    size_t   client_out_plane_stride = 0;
    int      client_in_channels      = 0;
    int      client_out_channels     = 0;
    int      client_in_seq           = 0;
    int      client_out_seq          = 0;
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
 * Compute aned's hexID for a (mil_text, weights) pair without compiling.
 *
 * Creates an _ANEInMemoryModelDescriptor and reads its hexStringIdentifier,
 * then discards both.  This is the same key aned uses to deduplicate compiled
 * programs in its in-process cache, so it can be used to look up warm-path
 * entries before a program exists.
 *
 * @return Non-empty hexID string on success.  Empty string on failure or in
 *         Fallback mode (check ane_last_error()).
 */
std::string ane_compute_hex_id(const std::string& mil_text,
                                const std::vector<WeightEntry>& weights);

/**
 * Reconnect to an existing aned compile slot without triggering compileWithQoS:.
 *
 * Creates a fresh _ANEInMemoryModel from the given MIL text and weights,
 * injects model_url via setModelURL:, then calls loadWithQoS: directly.
 * If aned still holds the compiled slot for this hexID, the load completes
 * in ~0.722ms and no compile slot is consumed.
 *
 * This is the Path C warm path for macOS 26+, where no HWX binary is ever
 * written to disk.  Discovered via test_compiler_options_probe.mm Opts-3
 * (2026-04-22, M3 Pro / macOS 26.3.1).
 *
 * @param mil_text   UTF-8 MIL source — must produce the same hexID as the
 *                   original compile (i.e., identical text + identical weights).
 * @param weights    Same weight blobs as the original compile.
 * @param model_url  AneProgram::model_url from a previous ane_compile() call.
 * @param debug_name Optional label.
 *
 * @return Non-null AneProgram* on success (caller calls ane_unload()).
 *         Returns nullptr if aned slot is dead (caller must call ane_compile()).
 */
AneProgram* ane_reconnect(const std::string&              mil_text,
                           const std::vector<WeightEntry>& weights,
                           const std::string&              model_url,
                           const std::string&              debug_name = "");

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
 * Remove a compiled program from ANE SRAM without freeing the compile slot.
 *
 * Releases SRAM (unloadWithQoS:) but keeps the ObjC model object alive and
 * aned's compile-slot entry intact.  A subsequent ane_delta_reload() on the
 * same program will skip compileWithQoS: and reload in ~1 ms (Layer 3 cache).
 *
 * Use this for LRU SRAM management when the slot limit (~16) is not yet
 * reached and you want to swap models in/out of SRAM cheaply:
 *
 *   ane_unload_sram(prog_a);       // free SRAM, keep slot
 *   ane_delta_reload(prog_b);      // reload b into SRAM — ~1 ms if slot alive
 *   ane_delta_reload(prog_a);      // reload a into SRAM — ~1 ms (slot intact)
 *
 * To fully release a program (SRAM + compile slot + memory), use ane_unload().
 * Safe to call with nullptr.
 */
void ane_unload_sram(AneProgram* program);

/**
 * Unload a compiled ANE program and release all resources.
 *
 * Calls unloadWithQoS: (SRAM) + purgeCompiledModelMatchingHash: (compile slot)
 * + releases ObjC objects + deletes the model_dir temp directory.
 * A subsequent ane_compile() with the same MIL text will pay the full
 * ANECompilerService cost (Layer 2 ~40 ms from disk cache, or Layer 1
 * ~4000 ms on first-ever compile).
 *
 * Safe to call with nullptr.
 */
void ane_unload(AneProgram* program);

/** Last error string from the ANE runtime (thread-local). */
const char* ane_last_error();

/* ── Program serialization ───────────────────────────────────────────────── */

/**
 * All data needed to restore a compiled AneProgram.
 * Produced by ane_serialize_program(); consumed by ane_restore_program().
 *
 * NOTE: hwx_bytes and hwx_rel_path are always empty.  The compiled ANE binary
 * lives exclusively inside aned (Apple Neural Engine daemon) and cannot be
 * extracted to the client filesystem.  These fields are retained in the struct
 * for binary compatibility with any previously serialized files; they are
 * ignored on restore.  Serialization stores only mil_text + weights, which are
 * sufficient for ane_restore_program() to fully reconstruct the program.
 */
struct SerializedProgram {
    std::string              mil_text;          ///< MIL source text
    std::vector<WeightEntry> weights;           ///< weight blobs (with 128-byte headers)
    std::vector<uint8_t>     hwx_bytes;         ///< always empty (reserved, unused)
    std::string              hwx_rel_path;      ///< always empty (reserved, unused)
    std::string              debug_name;
    std::vector<std::string> input_param_names;
    std::vector<std::string> output_var_names;
};

/**
 * Extract serializable data from a compiled program.
 *
 * Copies mil_text, weights, and I/O names from the program.  hwx_bytes is
 * always left empty — the compiled ANE binary cannot be extracted from aned.
 *
 * @param program  Compiled program (must have mil_text set).
 * @param out      Populated on success.
 * @return         true on success; false if mil_text is absent.
 */
bool ane_serialize_program(const AneProgram* program, SerializedProgram& out);

/**
 * Restore a compiled program from serialized data.
 *
 * Calls ane_compile() with the stored MIL text and weights.  If aned's
 * per-process in-memory cache still holds the hexID for this model (Path C
 * warm path), load completes in ~1 ms; otherwise the full ~4200 ms cold
 * compile round-trip is required.
 *
 * sp.hwx_bytes is intentionally ignored — see SerializedProgram above.
 *
 * @param sp   Data from ane_serialize_program().
 * @return     Heap-allocated AneProgram* on success (caller calls ane_unload()).
 *             Returns nullptr on failure.
 */
AneProgram* ane_restore_program(const SerializedProgram& sp);

/**
 * Return cached device info (populated during initialize()).
 * Thread-safe after initialize() completes.
 */
AneDeviceInfo device_info();

/**
 * Load a pre-compiled HWX binary by pre-staging it in localModelPath before
 * compileWithQoS:.  On macOS 26+, aned's "compileAsNeeded" logic uses an
 * existing model.hwx in the model directory rather than running ANECCompile().
 * The caller must supply the matching MIL text for the target op — aned checks
 * that the staged binary produces correct output for that op before registering
 * it (confirmed by XPC-18, 2026-04-22).
 *
 * On success, the returned AneProgram executes the op described by mil_text
 * but uses hwx_bytes as the compiled binary (enabling cross-op patching).
 *
 * @param hwx_bytes   BEEFFACE-magic HWX binary to stage as model.hwx.
 * @param mil_text    MIL source for the target op.  Must be consistent with
 *                    the computation hwx_bytes performs (aned validates this).
 * @param channels    Output tensor channel count.
 * @param seq         Output tensor sequence length.
 * @param input_name  MIL variable name for the input tensor.
 * @param output_name MIL variable name for the output tensor.
 * @param debug_name  Optional label.
 *
 * @return Non-null AneProgram* on success; caller must call ane_unload().
 *         Returns nullptr on failure (check ane_last_error()).
 *         Returns nullptr in Fallback mode or if hwx_bytes is empty.
 */
AneProgram* ane_load_hwx(const std::vector<uint8_t>& hwx_bytes,
                          const std::string&          mil_text,
                          int                         channels,
                          int                         seq,
                          const std::string&          input_name  = "x",
                          const std::string&          output_name = "y",
                          const std::string&          debug_name  = "");

/**
 * Return true if Path B (_ANEClient / Espresso .mlmodelc) symbols were
 * successfully resolved during initialize().  Always false in Fallback mode.
 */
bool path_b_available();

/**
 * Return the number of `compileWithQoS:` calls consumed in this process.
 *
 * aned enforces a hard per-process limit of ~119 unique compilations.
 * Exceeding it causes silent failures followed by SIGSEGV (confirmed
 * empirically, 2026-04-22).  libane refuses new compiles at
 * kCompileHardLimit (115) with a descriptive error, and warns to stderr
 * at kCompileWarnAt (100).
 *
 * The count is NOT decremented by purge / unload — slots appear to be
 * per-process-lifetime in the kernel.
 *
 * Path C warm-path hits (compiledModelExists=YES) do NOT consume a slot
 * and are not counted here.
 */
int ane_compile_count();

/**
 * Return the number of compile slots remaining before the hard limit.
 * Returns 0 once the budget is exhausted (further compiles will fail).
 */
int ane_compile_slots_remaining();

/**
 * Load a pre-built Espresso .mlmodelc bundle via _ANEClient (Path B).
 *
 * Calls _ANEClient.sharedConnection, creates _ANEModel from the directory,
 * compiles and loads it, reads BatchStride/PlaneStride from modelAttributes,
 * checks intermediateBufferHandle for SRAM spill, and pre-allocates the
 * persistent IOSurfaces that ane_execute_client reuses on every call.
 *
 * @param model_dir_path  Path to the .mlmodelc directory.
 * @param in_channels     Input tensor channels.
 * @param in_seq          Input tensor seq.
 * @param out_channels    Output tensor channels.
 * @param out_seq         Output tensor seq.
 * @param debug_name      Optional label.
 * @return                Loaded AneProgram* (caller calls ane_unload()), or nullptr.
 */
AneProgram* ane_load_mlmodelc(const std::string& model_dir_path,
                               int in_channels, int in_seq,
                               int out_channels, int out_seq,
                               const std::string& debug_name = "");

#ifdef __APPLE__
/**
 * Execute a Path B program loaded via ane_load_mlmodelc().
 *
 * On the first call: creates _ANEIOSurfaceObject wrappers, builds _ANERequest,
 * calls mapIOSurfacesWithModel:cacheInference:YES to register the IOSurface
 * handles with the ANE kernel once.  All subsequent calls use the cached
 * mapping via fastConn — no IOSurface marshal/unmarshal overhead.
 *
 * Every call: fills client_in_surf with a PlaneStride-aware scatter of the
 * input fp16 data, fires doEvaluateDirectWithModel:, drains client_out_surf
 * with the inverse gather into the output buffer.
 *
 * @param program     Handle from ane_load_mlmodelc().
 * @param input_fp16  Flat fp16 input [C, S] channel-first row-major.
 * @param output_fp16 Output buffer for flat fp16 result (same layout).
 * @return            true on success.
 */
bool ane_execute_client(AneProgram* program,
                        const void* input_fp16,
                        void*       output_fp16);
#endif

} // namespace runtime
} // namespace libane
