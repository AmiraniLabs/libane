/**
 * EspressoBuilder — constructs Apple Neural Engine .mlmodelc bundles.
 *
 * Emits raw espresso-format bundles that aned can compile to HWX without
 * going through coremltools or the CoreML framework.
 *
 * Bundle layout written by write():
 *   <dir>/
 *     model.espresso.net         JSON layer graph
 *     model.espresso.shape       JSON tensor dimensions (NHWK)
 *     model.espresso.weights     Binary weight data (v2 or empty)
 *     metadata.json              CoreML schema stub
 *     coremldata.bin             Protobuf stub (0x08 0x04)
 *     model/coremldata.bin       64-byte zero stub
 *     analytics/coremldata.bin
 *     neural_network_optionals/coremldata.bin
 *
 * Reference: reference/ane-compiler/src/compiler.py
 *            reference/beefface/docs/ESPRESSO_VOCABULARY.md
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace libane {
namespace graph {

class EspressoBuilder {
public:
    /* ── Tensor shape (NHWK order) ─────────────────────────────────────────── */

    struct Shape { int n = 1, h = 1, w = 1, k = 1; };

    /* ── Metadata I/O schema ───────────────────────────────────────────────── */

    struct IOTensor {
        std::string      name;
        std::vector<int> shape;  ///< [C, H, W] or [C, 1, 1] for 1-D
    };

    /* ── Activation mode constants (model.espresso.net "mode" field) ───────── */

    static constexpr int RELU        =  0;
    static constexpr int TANH        =  1;
    static constexpr int LEAKY_RELU  =  2;
    static constexpr int SIGMOID     =  3;
    static constexpr int ELU         =  8;
    static constexpr int GELU_EXACT  = 19;
    static constexpr int GELU_TANH   = 21;
    static constexpr int GELU_SIGMA  = 22;  ///< sigmoid approx: x*sigmoid(1.702*x)
    static constexpr int SILU        = 25;
    static constexpr int HARDSWISH   = 26;

    /* ── Layer builders ────────────────────────────────────────────────────── */

    /**
     * Add an activation layer.
     * @param mode   Activation mode (use constants above or raw int).
     * @param alpha  Scale/slope for mode 2 (leaky_relu) and mode 8 (elu).
     * @param beta   Offset for mode 7 (hard sigmoid) and mode 6 (linear).
     */
    EspressoBuilder& add_activation(std::string name,
                                    std::string bottom, std::string top,
                                    int mode,
                                    float alpha     = 0.0f,
                                    float beta      = 0.0f,
                                    bool  is_output = false);

    /**
     * Add a fully-connected (inner_product) layer.
     * @param nB          Input channels.
     * @param nC          Output channels.
     * @param blob_weights  Blob index for weights (1 = no-bias, 3 = with-bias).
     * @param has_biases  Whether a bias blob is present (blob_biases=1).
     * @param has_relu    Whether relu is fused after the projection.
     */
    EspressoBuilder& add_inner_product(std::string name,
                                       std::string bottom, std::string top,
                                       int  nB, int nC,
                                       int  blob_weights = 1,
                                       bool has_biases   = false,
                                       bool has_relu     = false,
                                       bool is_output    = false);

    /** Add a softmax layer. */
    EspressoBuilder& add_softmax(std::string name,
                                 std::string bottom, std::string top,
                                 bool is_output = false);

    /**
     * Add an l2_normalize layer (layer normalization — MVN pass).
     * @param axis  Normalisation axis (usually 2).
     * @param eps   Epsilon for numerical stability.
     */
    EspressoBuilder& add_l2_normalize(std::string name,
                                      std::string bottom, std::string top,
                                      int axis = 2, float eps = 1e-5f,
                                      bool is_output = false);

    /**
     * Add a batchnorm layer.
     * When paired with l2_normalize this implements affine layer norm
     * (gamma * x + beta).
     * @param C   Channel count; the batchnorm weights blob has 4*C floats
     *            interleaved as [gamma, beta, mean, var] per channel.
     */
    EspressoBuilder& add_batchnorm(std::string name,
                                   std::string bottom, std::string top,
                                   int C, bool is_output = false);

    /* ── Shape registry ────────────────────────────────────────────────────── */

    /** Register tensor shape in NHWK order. */
    EspressoBuilder& add_shape(std::string name, Shape s);

    /** Shorthand: add_shape(name, {n, h, w, k}). */
    EspressoBuilder& add_shape(std::string name, int n, int h, int w, int k);

    /* ── Weight data ───────────────────────────────────────────────────────── */

    /**
     * Append FP32 weight floats to the weight blob.
     * For inner_product: caller provides weights in [out_ch × in_ch] row-major
     * order (ANE performs: out = weights @ input).
     */
    EspressoBuilder& add_weight_blob(const float* data, size_t count);

    /** Convenience overload. */
    EspressoBuilder& add_weight_blob(const std::vector<float>& data);

    /* ── Metadata ──────────────────────────────────────────────────────────── */

    EspressoBuilder& set_inputs (std::vector<IOTensor> inputs);
    EspressoBuilder& set_outputs(std::vector<IOTensor> outputs);

    /* ── Emit ──────────────────────────────────────────────────────────────── */

    /**
     * Write the complete .mlmodelc bundle to @p dir.
     * The directory is created if it does not exist.
     * Returns true on success.
     */
    bool write(const std::string& dir) const;

    /* ── Factory helpers ───────────────────────────────────────────────────── */

    /**
     * Build a single-layer activation graph for a tensor with
     * @p channels channels and @p seq spatial/sequence elements.
     *
     * Shape in Espresso NHWK: n=1, h=1, w=seq, k=channels.
     * Metadata shape: [channels, 1, seq].
     */
    static EspressoBuilder activation(const std::string& input_name,
                                      const std::string& output_name,
                                      int   channels, int seq,
                                      int   mode,
                                      float alpha = 0.0f,
                                      float beta  = 0.0f);

    /**
     * Build an inner_product graph: [1, 1, 1, in_ch] → [1, 1, 1, out_ch].
     * @param weights  Row-major FP32 matrix [out_ch × in_ch].
     * @param bias     Optional FP32 bias [out_ch]; if non-null, v4 weight format.
     */
    static EspressoBuilder inner_product(int         in_ch,
                                          int         out_ch,
                                          const float* weights,
                                          const float* bias       = nullptr,
                                          bool         fused_relu = false);

private:
    std::vector<std::string>      layer_jsons_;
    std::map<std::string, Shape>  shapes_;
    std::vector<float>            weights_;
    std::vector<IOTensor>         inputs_;
    std::vector<IOTensor>         outputs_;

    bool write_net     (const std::string& path) const;
    bool write_shape   (const std::string& path) const;
    bool write_weights (const std::string& path) const;
    bool write_metadata(const std::string& path) const;

    static bool write_coremldata(const std::string& path);
    static bool make_stub       (const std::string& path);
    static bool ensure_dir      (const std::string& path);
};

} // namespace graph
} // namespace libane
