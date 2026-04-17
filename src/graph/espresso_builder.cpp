/**
 * EspressoBuilder implementation.
 */
#include "espresso_builder.hpp"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace libane {
namespace graph {

/* ── Minimal JSON helpers ────────────────────────────────────────────────── */

static std::string jstr(const std::string& s) {
    return "\"" + s + "\"";
}

static std::string jkv_str(const std::string& k, const std::string& v) {
    return jstr(k) + ":" + jstr(v);
}

static std::string jkv_int(const std::string& k, int v) {
    return jstr(k) + ":" + std::to_string(v);
}

static std::string jkv_float(const std::string& k, float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return jstr(k) + ":" + buf;
}

static std::string jkv_bool(const std::string& k, bool v) {
    return jstr(k) + ":" + (v ? "true" : "false");
}

/* ── Layer builders ──────────────────────────────────────────────────────── */

EspressoBuilder& EspressoBuilder::add_activation(std::string name,
                                                  std::string bottom,
                                                  std::string top,
                                                  int  mode,
                                                  float alpha,
                                                  float beta,
                                                  bool  is_output) {
    std::string j = "{";
    j += jkv_str("type",   "activation") + ",";
    j += jkv_int("mode",    mode)        + ",";
    j += jkv_str("name",    name)        + ",";
    j += jkv_str("bottom",  bottom)      + ",";
    j += jkv_str("top",     top)         + ",";
    if (alpha != 0.0f) j += jkv_float("alpha", alpha) + ",";
    if (beta  != 0.0f) j += jkv_float("beta",  beta)  + ",";
    j += jstr("weights") + ":{}";
    if (is_output) j += "," + jstr("attributes") + ":{" + jkv_int("is_output", 1) + "}";
    j += "}";
    layer_jsons_.push_back(std::move(j));
    return *this;
}

EspressoBuilder& EspressoBuilder::add_inner_product(std::string name,
                                                     std::string bottom,
                                                     std::string top,
                                                     int  nB, int nC,
                                                     int  blob_weights,
                                                     bool has_biases,
                                                     bool has_relu,
                                                     bool is_output) {
    std::string j = "{";
    j += jkv_str("type",         "inner_product") + ",";
    j += jkv_str("name",          name)           + ",";
    j += jkv_str("debug_info",    name)           + ",";
    j += jkv_str("bottom",        bottom)         + ",";
    j += jkv_str("top",           top)            + ",";
    j += jkv_int("nB",            nB)             + ",";
    j += jkv_int("nC",            nC)             + ",";
    j += jkv_int("has_biases",    has_biases ? 1 : 0) + ",";
    j += jkv_int("blob_weights",  blob_weights)   + ",";
    if (has_biases) j += jkv_int("blob_biases", 1) + ",";
    j += jkv_int("has_relu",      has_relu  ? 1 : 0) + ",";
    j += jkv_int("has_tanh",      0) + ",";
    j += jkv_int("has_prelu",     0) + ",";
    j += jstr("weights") + ":{}";
    if (is_output) j += "," + jstr("attributes") + ":{" + jkv_int("is_output", 1) + "}";
    j += "}";
    layer_jsons_.push_back(std::move(j));
    return *this;
}

EspressoBuilder& EspressoBuilder::add_softmax(std::string name,
                                               std::string bottom,
                                               std::string top,
                                               bool is_output) {
    std::string j = "{";
    j += jkv_str("type",   "softmax")  + ",";
    j += jkv_str("name",    name)      + ",";
    j += jkv_str("bottom",  bottom)    + ",";
    j += jkv_str("top",     top)       + ",";
    j += jstr("weights") + ":{}";
    if (is_output) j += "," + jstr("attributes") + ":{" + jkv_int("is_output", 1) + "}";
    j += "}";
    layer_jsons_.push_back(std::move(j));
    return *this;
}

EspressoBuilder& EspressoBuilder::add_l2_normalize(std::string name,
                                                    std::string bottom,
                                                    std::string top,
                                                    int axis, float eps,
                                                    bool is_output) {
    std::string j = "{";
    j += jkv_str("type",               "l2_normalize") + ",";
    j += jkv_str("name",                name)          + ",";
    j += jkv_str("debug_info",          name)          + ",";
    j += jkv_str("bottom",              bottom)        + ",";
    j += jkv_str("top",                 top)           + ",";
    j += jkv_int("normalization_mode",  1)             + ",";
    j += jkv_int("axis",                axis)          + ",";
    j += jkv_float("eps",               eps)           + ",";
    j += jstr("weights") + ":{}";
    if (is_output) j += "," + jstr("attributes") + ":{" + jkv_int("is_output", 1) + "}";
    j += "}";
    layer_jsons_.push_back(std::move(j));
    return *this;
}

EspressoBuilder& EspressoBuilder::add_batchnorm(std::string name,
                                                 std::string bottom,
                                                 std::string top,
                                                 int C, bool is_output) {
    std::string j = "{";
    j += jkv_str("type",                "batchnorm")  + ",";
    j += jkv_str("name",                 name)        + ",";
    j += jkv_str("debug_info",           name)        + ",";
    j += jkv_str("bottom",               bottom)      + ",";
    j += jkv_str("top",                  top)         + ",";
    j += jkv_int("blob_batchnorm_params", 1)          + ",";
    j += jkv_int("C",                    C)           + ",";
    j += jstr("weights") + ":{}";
    if (is_output) j += "," + jstr("attributes") + ":{" + jkv_int("is_output", 1) + "}";
    j += "}";
    layer_jsons_.push_back(std::move(j));
    return *this;
}

/* ── Shape registry ──────────────────────────────────────────────────────── */

EspressoBuilder& EspressoBuilder::add_shape(std::string name, Shape s) {
    shapes_[std::move(name)] = s;
    return *this;
}

EspressoBuilder& EspressoBuilder::add_shape(std::string name,
                                             int n, int h, int w, int k) {
    return add_shape(std::move(name), Shape{n, h, w, k});
}

/* ── Weight data ─────────────────────────────────────────────────────────── */

EspressoBuilder& EspressoBuilder::add_weight_blob(const float* data, size_t count) {
    weights_.insert(weights_.end(), data, data + count);
    return *this;
}

EspressoBuilder& EspressoBuilder::add_weight_blob(const std::vector<float>& data) {
    return add_weight_blob(data.data(), data.size());
}

/* ── Metadata ────────────────────────────────────────────────────────────── */

EspressoBuilder& EspressoBuilder::set_inputs(std::vector<IOTensor> inputs) {
    inputs_ = std::move(inputs);
    return *this;
}

EspressoBuilder& EspressoBuilder::set_outputs(std::vector<IOTensor> outputs) {
    outputs_ = std::move(outputs);
    return *this;
}

/* ── File writing helpers ────────────────────────────────────────────────── */

bool EspressoBuilder::ensure_dir(const std::string& path) {
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

bool EspressoBuilder::write_coremldata(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint8_t stub[2] = {0x08, 0x04};
    f.write(reinterpret_cast<const char*>(stub), 2);
    return f.good();
}

bool EspressoBuilder::make_stub(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint8_t zeros[64] = {};
    f.write(reinterpret_cast<const char*>(zeros), 64);
    return f.good();
}

/* ── write_net ───────────────────────────────────────────────────────────── */

bool EspressoBuilder::write_net(const std::string& path) const {
    std::string s;
    s += "{";
    s += jkv_str("storage", "model.espresso.weights") + ",";
    s += jstr("analyses")          + ":{},";
    s += jstr("properties")        + ":{},";
    s += jkv_int("format_version", 200) + ",";
    s += jstr("metadata_in_weights") + ":[],";
    s += jstr("layers") + ":[";
    for (size_t i = 0; i < layer_jsons_.size(); ++i) {
        if (i) s += ",";
        s += layer_jsons_[i];
    }
    s += "]}";

    std::ofstream f(path);
    if (!f) return false;
    f << s;
    return f.good();
}

/* ── write_shape ─────────────────────────────────────────────────────────── */

bool EspressoBuilder::write_shape(const std::string& path) const {
    std::string s;
    s += "{" + jstr("layer_shapes") + ":{";
    bool first = true;
    for (const auto& kv : shapes_) {
        if (!first) s += ",";
        first = false;
        const Shape& sh = kv.second;
        s += jstr(kv.first) + ":{";
        s += jkv_int("n", sh.n) + ",";
        s += jkv_int("h", sh.h) + ",";
        s += jkv_int("w", sh.w) + ",";
        s += jkv_int("k", sh.k);
        s += "}";
    }
    s += "}}";

    std::ofstream f(path);
    if (!f) return false;
    f << s;
    return f.good();
}

/* ── write_weights ───────────────────────────────────────────────────────── */

bool EspressoBuilder::write_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    if (weights_.empty()) {
        // No weights: 8 zero bytes
        uint8_t zero[8] = {};
        f.write(reinterpret_cast<const char*>(zero), 8);
        return f.good();
    }

    // v2 format: 0x40-byte header + FP32 weight data
    const uint32_t weight_bytes =
        static_cast<uint32_t>(weights_.size() * sizeof(float));

    uint8_t header[0x40] = {};
    auto put_u32 = [&](size_t off, uint32_t v) {
        std::memcpy(header + off, &v, 4);
    };
    put_u32(0x00, 2);            // version
    put_u32(0x10, 0x18);         // blob_table_offset
    put_u32(0x18, 1);            // num_blobs
    put_u32(0x20, weight_bytes); // total weight bytes

    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(weights_.data()), weight_bytes);
    return f.good();
}

/* ── write_metadata ──────────────────────────────────────────────────────── */

bool EspressoBuilder::write_metadata(const std::string& path) const {
    auto make_schema = [](const IOTensor& t) -> std::string {
        std::string s = "{";
        s += jkv_str("name",     t.name)         + ",";
        s += jkv_str("type",     "MultiArray")   + ",";
        s += jstr("shape") + ":[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(t.shape[i]);
        }
        s += "],";
        s += jkv_str("dataType", "Float16");
        s += "}";
        return s;
    };

    std::string s;
    s += "{";
    s += jkv_int("specificationVersion", 4) + ",";
    s += jkv_bool("isUpdatable", false) + ",";
    s += jstr("modelType") + ":{" + jkv_str("name", "MLModelType_neuralNetwork") + "},";
    s += jkv_str("computePrecision", "Float16") + ",";
    s += jstr("inputSchema") + ":[";
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (i) s += ",";
        s += make_schema(inputs_[i]);
    }
    s += "],";
    s += jstr("outputSchema") + ":[";
    for (size_t i = 0; i < outputs_.size(); ++i) {
        if (i) s += ",";
        s += make_schema(outputs_[i]);
    }
    s += "]}";

    std::ofstream f(path);
    if (!f) return false;
    f << s;
    return f.good();
}

/* ── write ───────────────────────────────────────────────────────────────── */

bool EspressoBuilder::write(const std::string& dir) const {
    if (!ensure_dir(dir)) return false;

    for (const char* sub : {"model", "analytics", "neural_network_optionals"}) {
        if (!ensure_dir(dir + "/" + sub)) return false;
        if (!make_stub(dir + "/" + sub + "/coremldata.bin")) return false;
    }

    if (!write_net     (dir + "/model.espresso.net"))     return false;
    if (!write_shape   (dir + "/model.espresso.shape"))   return false;
    if (!write_weights (dir + "/model.espresso.weights")) return false;
    if (!write_metadata(dir + "/metadata.json"))          return false;
    if (!write_coremldata(dir + "/coremldata.bin"))       return false;

    return true;
}

/* ── Factory: activation ─────────────────────────────────────────────────── */

EspressoBuilder EspressoBuilder::activation(const std::string& input_name,
                                             const std::string& output_name,
                                             int channels, int seq,
                                             int mode,
                                             float alpha, float beta) {
    EspressoBuilder b;
    b.add_activation("act", input_name, output_name, mode, alpha, beta,
                     /*is_output=*/true);

    // NHWK: n=1, h=1, w=seq, k=channels
    b.add_shape(input_name,  {1, 1, seq, channels});
    b.add_shape(output_name, {1, 1, seq, channels});

    // Metadata shape: [channels, 1, seq]
    b.set_inputs ({std::vector<IOTensor>{{input_name,  {channels, 1, seq}}}});
    b.set_outputs({std::vector<IOTensor>{{output_name, {channels, 1, seq}}}});

    return b;
}

/* ── Factory: inner_product ──────────────────────────────────────────────── */

EspressoBuilder EspressoBuilder::inner_product(int in_ch, int out_ch,
                                                const float* weights,
                                                const float* bias,
                                                bool fused_relu) {
    EspressoBuilder b;

    const bool has_bias     = (bias != nullptr);
    const int  blob_weights = has_bias ? 3 : 1;

    b.add_inner_product("fc", "input", "output",
                        in_ch, out_ch,
                        blob_weights, has_bias, fused_relu,
                        /*is_output=*/true);

    b.add_shape("input",  {1, 1, 1, in_ch});
    b.add_shape("output", {1, 1, 1, out_ch});

    b.set_inputs ({std::vector<IOTensor>{{"input",  {in_ch,  1, 1}}}});
    b.set_outputs({std::vector<IOTensor>{{"output", {out_ch, 1, 1}}}});

    if (weights) b.add_weight_blob(weights, static_cast<size_t>(in_ch) * out_ch);

    // TODO: v4 weight format (bias support) — implement when EspressoBackend
    //       adds bias-bearing layers.  For now the blob layout is v2 (weights only).
    (void)bias;

    return b;
}

} // namespace graph
} // namespace libane
