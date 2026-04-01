/**
 * MIL Text Builder — generates Model Intermediate Language programs for ANE.
 *
 * Produces MIL *text* (the string-based IR format used by _ANEInMemoryModelDescriptor),
 * NOT CoreML protobuf. The distinction matters: the private ANE API takes the text
 * format directly, compiled internally by _ANECompiler to E5 FlatBuffer microcode.
 *
 * MIL text format (from maderix/ANE and arXiv:2603.06728):
 *
 *   program(1.3)
 *   [buildInfo = dict<string, string>({
 *       {"coremlc-component-MIL", "3510.2.1"},
 *       {"coremltools-version", "9.0"}
 *   })]
 *   {
 *       func main<ios18>(tensor<fp16, [1, IC, 1, SP]> x) {
 *           %weight = const()[val = tensor<fp16, [OC,1,1,IC]>(
 *                       file("@model_path/weights/weight.bin", offset=64))];
 *           %result = conv(x = %x, weight = %weight, bias = nothing,
 *                          strides=[1,1], pad_type="valid", pad=[0,0,0,0],
 *                          dilations=[1,1], groups=1);
 *           return %result;
 *       } -> (tensor<fp16, [1, OC, 1, SP]>);
 *   }
 *
 * Key facts from Orion §4 and maderix/ANE:
 *  - ANE tensors are ALWAYS [1, C, 1, S] (batch=1, height=1, no exceptions)
 *  - conv 1×1 is 3× faster than matmul on ANE — use conv for all linear projections
 *  - concat is banned (compiles but crashes at runtime)
 *  - S must be a multiple of 8, ≤ 65536; C ≤ 16384
 *  - conv bias is NOT supported — use a separate add op
 *  - GELU must use tanh approximation only
 *  - matmul transpose flags require named const nodes, not inline literals
 *  - Output variable must reference a live post-DCE node
 *  - Weight dict must NOT be nil (use @{} for weight-free kernels)
 *  - MIL text must be NSData*, not NSString* (the bridge handles this)
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>

namespace libane {
namespace mil {

/* ── ANE tensor shape ────────────────────────────────────────────────────── */

struct TensorShape {
    int32_t batch    = 1;    // must be 1 (ANE constraint)
    int32_t channels = 0;    // C — number of channels
    int32_t height   = 1;    // must be 1 (ANE constraint)
    int32_t seq      = 0;    // S — spatial / sequence dimension

    /** Validate all ANE constraints. Throws std::invalid_argument on violation. */
    void validate() const;

    /** Total fp16 elements. */
    size_t numel() const {
        return static_cast<size_t>(batch) * channels * height * seq;
    }
    size_t bytes()  const { return numel() * 2; }

    bool operator==(const TensorShape& o) const {
        return batch == o.batch && channels == o.channels &&
               height == o.height && seq == o.seq;
    }
};

/* ── Weight blob ─────────────────────────────────────────────────────────── */

/**
 * ANE weight blob format (from maderix/ANE ane_bridge.m):
 *
 *  [0..63]   64-byte file header   (buf[0]=0x01, buf[4]=0x02)
 *  [64..127] 64-byte chunk header  (magic=0xDEADBEEF LE at offset 0,
 *                                   buf[4]=0x01, buf[8..11]=fp16_size uint32 LE)
 *  [128..]   raw fp16 weight data  (row-major)
 *
 * The weight dict @"offset" key is 64 (Orion constraint #8 — NOT 128).
 * ANE reads actual weights starting at blob[64], skipping the file header.
 * The chunk header at [64..127] is consumed internally.
 */
struct WeightBlob {
    std::vector<uint8_t> data;   // full blob (header + weights)
    uint64_t             hash = 0;

    /** Compute xxhash64 of the weight data section only (offset 128 onwards). */
    void compute_hash();

    /**
     * Build blob from raw fp16 weight bytes.
     * src points to the weight matrix data (row-major fp16).
     */
    static WeightBlob from_fp16(const void* src, size_t weight_bytes);

    /**
     * Build blob from fp16 [rows, cols] matrix, transposed to [cols, rows].
     * Use for MATMUL weights: user provides W[IC, OC], conv1x1 needs [OC, IC].
     */
    static WeightBlob from_fp16_transposed(const void* src, int rows, int cols);

    /**
     * Build blob from fp32 weights; internally casts to fp16.
     * Supports optional column-major transposition for conv1x1 weight layout.
     * Use transpose=true when building conv1x1 weights from a row-major
     * [IC, OC] matrix (packs as [OC, IC]).
     */
    static WeightBlob from_fp32(const float* src, int rows, int cols,
                                 bool transpose = false);

    /** Offset into this blob where the ANE weight dict @"offset" should point. */
    static constexpr uint64_t kWeightDictOffset = 64;

    /** Byte offset where fp16 data starts within the blob. */
    static constexpr size_t kDataOffset = 128;
};

/* ── MIL text builder ────────────────────────────────────────────────────── */

/**
 * Builds MIL text programs for ANE dispatch.
 *
 * All operations that map to a linear projection (matmul, layernorm affine,
 * attention QKV) use conv1x1 internally for the 3× throughput advantage
 * documented in the Orion paper (§5.2).
 *
 * Usage:
 *   auto mil = MilBuilder::matmul(IC, OC, SP, "weight.bin");
 *   // mil.text contains the MIL program string
 *   // mil.weight_name is "weight.bin" (key into weight dict)
 */
struct MilProgram {
    std::string              text;             // MIL source text (pass as NSData*)
    std::string              weight_name;      // primary weight filename (may be empty)
    std::vector<std::string> all_weight_names; // all weight files referenced by this program
    TensorShape              input_shape;
    TensorShape              output_shape;
};

/* ── Fragment types for fused multi-op programs ──────────────────────────── */

/**
 * The inner body of one op's contribution to a fused MIL program.
 *
 * body        — MIL statements only (no program/func wrapper), 8-space indent.
 * input_name  — primary chain input variable name.
 * output_name — output variable name (unique across all fragments in the group).
 * side_input_name — second input variable for binary ops; empty for unary ops.
 * weight_file — BLOBFILE filename referenced in body; empty if weight-free.
 * output_shape — shape of the output tensor.
 *
 * All internal (intermediate) variable names are prefixed with output_name+"_"
 * to guarantee uniqueness when multiple fragments share a MIL program scope.
 */
struct MilFragment {
    std::string      body;
    std::string      input_name;
    std::string      output_name;
    std::string      side_input_name;  // empty for single-input ops
    std::string      weight_file;      // empty for weight-free ops
    TensorShape      output_shape;
};

/**
 * A named, typed input to a fused MIL program.
 * build_fused sorts these alphabetically before emitting the func signature
 * (ANE constraint #13 — IOSurface assignment order is alphabetical).
 */
struct FusedInput {
    std::string  var_name;
    TensorShape  shape;
};

/** Weight file names for QKV projection. */
struct QKVWeights {
    std::string q_file = "wq.bin";
    std::string k_file = "wk.bin";
    std::string v_file = "wv.bin";
};

class MilBuilder {
public:
    /**
     * Linear projection via conv1x1: Y = X * W^T
     *
     * X:      [1, IC, 1, SP]  (input activations)
     * W:      [OC, IC, 1, 1]  (stored transposed in blob as [OC, IC])
     * Output: [1, OC, 1, SP]
     *
     * 3× faster than matmul on ANE (Orion §5.2).
     * SP must be multiple of 8.
     */
    static MilProgram matmul_conv1x1(int IC, int OC, int SP,
                                      const std::string& weight_file = "weight.bin");

    /**
     * Attention QKV projection via three fused conv1x1 ops.
     *
     * Produces three outputs: Q, K, V each [1, head_dim*n_heads, 1, SP].
     * All three weight matrices are packed into a single weight blob
     * with separate file names.
     */
    static MilProgram qkv_proj(int IC, int head_dim, int n_heads, int SP,
                                 const struct QKVWeights& w);

    /**
     * RMSNorm: y = (x / rms(x)) * scale
     *
     * Uses: rsqrt, mul operations on ANE.
     * scale: [1, C, 1, 1] broadcast constant.
     */
    static MilProgram rmsnorm(int C, int SP,
                               const std::string& scale_file = "scale.bin");

    /**
     * GELU activation (tanh approximation — only variant supported by ANE).
     * No weights.
     */
    static MilProgram gelu(int C, int SP);

    /**
     * Softmax over the spatial (S) dimension.
     * No weights.
     */
    static MilProgram softmax(int C, int SP);

    /**
     * avg_pool lowering path for graph IR.
     *
     * Current graph usage is identity-equivalent kernel/stride (1x1/1x1), so
     * we lower to identity to avoid ANE standalone compile rejects.
     */
    static MilProgram avg_pool(int C, int SP);

    /**
     * max_pool lowering path for graph IR.
     *
     * Current graph usage is identity-equivalent kernel/stride (1x1/1x1), so
     * we lower to identity to avoid ANE standalone compile rejects.
     */
    static MilProgram max_pool(int C, int SP);

    /**
     * Logical AND lowering path.
     * Semantics: out = fp16(bool(x) && bool(y)).
     */
    static MilProgram logical_and(int C, int SP);

    /**
     * Logical OR lowering path.
     * Semantics: out = fp16(bool(x) || bool(y)).
     */
    static MilProgram logical_or(int C, int SP);

    /**
     * Logical XOR lowering path.
     * Semantics: out = fp16(bool(x) xor bool(y)).
     */
    static MilProgram logical_xor(int C, int SP);

    /**
     * Elementwise add. No weights.
     */
    static MilProgram add(int C, int SP);

    /**
     * Elementwise multiply. No weights.
     * Inputs named "x" and "y" (alphabetical order per ANE constraint #13).
     */
    static MilProgram mul(int C, int SP);

    /**
     * SiLU activation: y = x * sigmoid(x). No weights.
     */
    static MilProgram silu(int C, int SP);

    /**
     * LayerNorm: normalizes across channels (axis=1), then applies gamma/beta.
     * gamma_file and beta_file are weight blob filenames.
     */
    static MilProgram layernorm(int C, int SP,
                                 const std::string& gamma_file = "gamma.bin",
                                 const std::string& beta_file  = "beta.bin",
                                 float eps = 1e-5f);

    /**
     * Transpose [0,3,2,1]: input [1,C,1,S] → output [1,S,1,C].
     * (The only transpose reliably supported on ANE per Orion.)
     */
    static MilProgram transpose_cssc(int C, int SP);

    /**
     * Output projection + residual add in one kernel:
     *   out = X * W_proj^T + residual
     */
    static MilProgram out_proj_add(int IC, int OC, int SP,
                                    const std::string& weight_file = "weight_proj.bin");

    /* ── Fragment emitters ───────────────────────────────────────────────── */
    //
    // Each emitter returns a MilFragment containing only the inner MIL
    // statements (no program/func wrapper).  The caller specifies:
    //
    //   in_var    — variable name carrying the chain input into this fragment.
    //   out_var   — variable name to assign the fragment's output to.
    //               Must be unique across all fragments in the fused group.
    //               All internal intermediate vars are named out_var+"_<suffix>".
    //
    // For binary ops (add_fragment, mul_fragment), a second input is given via
    // side_var.  Both in_var and side_var appear as parameters of the fused
    // MIL function.

    static MilFragment matmul_fragment(int IC, int OC, int SP,
                                        const std::string& in_var,
                                        const std::string& out_var,
                                        const std::string& weight_file = "weight.bin");

    static MilFragment rmsnorm_fragment(int C, int SP,
                                         const std::string& in_var,
                                         const std::string& out_var,
                                         const std::string& scale_file = "scale.bin");

    static MilFragment layernorm_fragment(int C, int SP,
                                           const std::string& in_var,
                                           const std::string& out_var,
                                           const std::string& gamma_file = "gamma.bin",
                                           const std::string& beta_file  = "beta.bin",
                                           float eps = 1e-5f);

    static MilFragment gelu_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& out_var);

    static MilFragment silu_fragment(int C, int SP,
                                      const std::string& in_var,
                                      const std::string& out_var);

    static MilFragment softmax_fragment(int C, int SP,
                                         const std::string& in_var,
                                         const std::string& out_var);

    static MilFragment avg_pool_fragment(int C, int SP,
                                          const std::string& in_var,
                                          const std::string& out_var);

    static MilFragment max_pool_fragment(int C, int SP,
                                          const std::string& in_var,
                                          const std::string& out_var);

    static MilFragment add_fragment(int C, int SP,
                                     const std::string& in_var,
                                     const std::string& side_var,
                                     const std::string& out_var);

    static MilFragment mul_fragment(int C, int SP,
                                     const std::string& in_var,
                                     const std::string& side_var,
                                     const std::string& out_var);

    static MilFragment logical_and_fragment(int C, int SP,
                                             const std::string& in_var,
                                             const std::string& side_var,
                                             const std::string& out_var);

    static MilFragment logical_or_fragment(int C, int SP,
                                            const std::string& in_var,
                                            const std::string& side_var,
                                            const std::string& out_var);

    static MilFragment logical_xor_fragment(int C, int SP,
                                             const std::string& in_var,
                                             const std::string& side_var,
                                             const std::string& out_var);

    static MilFragment transpose_fragment(int C, int SP,
                                           const std::string& in_var,
                                           const std::string& out_var);

    /* ── Fused program assembly ──────────────────────────────────────────── */

    /**
     * Assemble a complete MIL program from an ordered sequence of fragments.
     *
     * inputs    — ALL inputs to the fused program (chain input + any side inputs).
     *             Sorted alphabetically before being emitted as func parameters
     *             (ANE constraint #13).
     * fragments — Op fragments in execution order.  Each fragment's output_name
     *             must match the next fragment's input_name or side_input_name.
     *
     * Returns a MilProgram whose all_weight_names lists every BLOBFILE referenced.
     */
    static MilProgram build_fused(const std::vector<FusedInput>&  inputs,
                                   const std::vector<MilFragment>& fragments);

    /** Convenience overload for the common single-input case. */
    static MilProgram build_fused(const std::string& input_name,
                                   TensorShape        input_shape,
                                   const std::vector<MilFragment>& fragments);

private:
    /** Standard MIL program header with build info. */
    static std::string header();

    /** Render a tensor<fp16, [1,C,1,S]> type string. */
    static std::string tensor_type(const TensorShape& s);

    /** Render a file(...) const reference in MIL. */
    static std::string file_ref(const std::string& filename, uint64_t offset,
                                 const TensorShape& shape);
};

} // namespace mil
} // namespace libane
