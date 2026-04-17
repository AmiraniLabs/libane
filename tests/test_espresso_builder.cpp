/**
 * EspressoBuilder unit tests.
 *
 * Verifies that write() produces correctly structured .mlmodelc bundles
 * without requiring ANE hardware (pure file I/O tests).
 */
#include <catch2/catch_test_macros.hpp>
#include "graph/espresso_builder.hpp"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

using namespace libane::graph;

/* ── helpers ─────────────────────────────────────────────────────────────── */

static bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

static std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<uint8_t> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

static std::string tmpdir() {
    char tmpl[] = "/tmp/esp_test_XXXXXX";
    return ::mkdtemp(tmpl);
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

TEST_CASE("EspressoBuilder activation factory writes valid bundle", "[espresso]") {
    const std::string dir = tmpdir() + "/tanh.mlmodelc";

    auto b = EspressoBuilder::activation("input", "output", 64, 512,
                                          EspressoBuilder::TANH);
    REQUIRE(b.write(dir));

    // Required files exist
    CHECK(file_exists(dir + "/model.espresso.net"));
    CHECK(file_exists(dir + "/model.espresso.shape"));
    CHECK(file_exists(dir + "/model.espresso.weights"));
    CHECK(file_exists(dir + "/metadata.json"));
    CHECK(file_exists(dir + "/coremldata.bin"));
    CHECK(file_exists(dir + "/model/coremldata.bin"));
    CHECK(file_exists(dir + "/analytics/coremldata.bin"));
    CHECK(file_exists(dir + "/neural_network_optionals/coremldata.bin"));

    // coremldata.bin has 2-byte protobuf stub
    auto cd = read_bin(dir + "/coremldata.bin");
    REQUIRE(cd.size() == 2);
    CHECK(cd[0] == 0x08);
    CHECK(cd[1] == 0x04);

    // Subdirectory stubs are 64 zero bytes
    auto stub = read_bin(dir + "/model/coremldata.bin");
    REQUIRE(stub.size() == 64);
    for (uint8_t b2 : stub) CHECK(b2 == 0);

    // model.espresso.net is valid JSON with expected keys
    std::string net = read_file(dir + "/model.espresso.net");
    CHECK(net.find("\"format_version\":200") != std::string::npos);
    CHECK(net.find("\"type\":\"activation\"") != std::string::npos);
    CHECK(net.find("\"mode\":1")             != std::string::npos);  // TANH = 1
    CHECK(net.find("\"is_output\":1")        != std::string::npos);
    CHECK(net.find("\"bottom\":\"input\"")   != std::string::npos);
    CHECK(net.find("\"top\":\"output\"")     != std::string::npos);

    // model.espresso.shape has NHWK shapes for both tensors
    std::string shp = read_file(dir + "/model.espresso.shape");
    CHECK(shp.find("\"layer_shapes\"") != std::string::npos);
    CHECK(shp.find("\"input\"")  != std::string::npos);
    CHECK(shp.find("\"output\"") != std::string::npos);
    // channels=64 → k=64, seq=512 → w=512
    CHECK(shp.find("\"k\":64")  != std::string::npos);
    CHECK(shp.find("\"w\":512") != std::string::npos);

    // model.espresso.weights is 8 zero bytes (no weights for activation)
    auto wt = read_bin(dir + "/model.espresso.weights");
    REQUIRE(wt.size() == 8);
    for (uint8_t b2 : wt) CHECK(b2 == 0);

    // metadata.json has expected fields
    std::string meta = read_file(dir + "/metadata.json");
    CHECK(meta.find("\"specificationVersion\":4")          != std::string::npos);
    CHECK(meta.find("MLModelType_neuralNetwork")           != std::string::npos);
    CHECK(meta.find("\"computePrecision\":\"Float16\"")    != std::string::npos);
    CHECK(meta.find("\"dataType\":\"Float16\"")            != std::string::npos);
}

TEST_CASE("EspressoBuilder activation modes: relu/sigmoid/leaky_relu/elu", "[espresso]") {
    struct Case { int mode; const char* mode_str; float alpha; };
    Case cases[] = {
        { EspressoBuilder::RELU,       "0",  0.0f  },
        { EspressoBuilder::SIGMOID,    "3",  0.0f  },
        { EspressoBuilder::LEAKY_RELU, "2",  0.01f },
        { EspressoBuilder::ELU,        "8",  1.0f  },
        { EspressoBuilder::HARDSWISH,  "26", 0.0f  },
        { EspressoBuilder::SILU,       "25", 0.0f  },
        { EspressoBuilder::GELU_EXACT, "19", 0.0f  },
    };

    for (const auto& c : cases) {
        const std::string dir = tmpdir() + "/act.mlmodelc";
        auto b = EspressoBuilder::activation("x", "y", 32, 128, c.mode, c.alpha);
        REQUIRE(b.write(dir));

        std::string net = read_file(dir + "/model.espresso.net");
        std::string mode_field = std::string("\"mode\":") + c.mode_str;
        CHECK(net.find(mode_field) != std::string::npos);

        if (c.alpha != 0.0f) {
            CHECK(net.find("\"alpha\"") != std::string::npos);
        }
    }
}

TEST_CASE("EspressoBuilder inner_product writes v2 weight format", "[espresso]") {
    const int in_ch = 8, out_ch = 4;
    std::vector<float> weights(in_ch * out_ch, 1.0f);
    // Make unique values so we can verify data integrity
    for (int i = 0; i < in_ch * out_ch; ++i) weights[i] = static_cast<float>(i);

    const std::string dir = tmpdir() + "/fc.mlmodelc";
    auto b = EspressoBuilder::inner_product(in_ch, out_ch, weights.data());
    REQUIRE(b.write(dir));

    // model.espresso.net
    std::string net = read_file(dir + "/model.espresso.net");
    CHECK(net.find("\"type\":\"inner_product\"") != std::string::npos);
    CHECK(net.find("\"nB\":8")  != std::string::npos);
    CHECK(net.find("\"nC\":4")  != std::string::npos);
    CHECK(net.find("\"has_biases\":0") != std::string::npos);

    // model.espresso.weights: v2 format
    auto wt = read_bin(dir + "/model.espresso.weights");
    // header (0x40) + weight data (8*4*4 = 128 bytes)
    REQUIRE(wt.size() == 0x40 + in_ch * out_ch * sizeof(float));

    // version field at [0x00] = 2
    uint32_t version;
    std::memcpy(&version, wt.data() + 0x00, 4);
    CHECK(version == 2u);

    // blob_table_offset at [0x10] = 0x18
    uint32_t bto;
    std::memcpy(&bto, wt.data() + 0x10, 4);
    CHECK(bto == 0x18u);

    // total_weight_bytes at [0x20]
    uint32_t twb;
    std::memcpy(&twb, wt.data() + 0x20, 4);
    CHECK(twb == static_cast<uint32_t>(in_ch * out_ch * sizeof(float)));

    // Data at 0x40: verify first few float values
    float f0;
    std::memcpy(&f0, wt.data() + 0x40, 4);
    CHECK(f0 == 0.0f);
    float f1;
    std::memcpy(&f1, wt.data() + 0x44, 4);
    CHECK(f1 == 1.0f);
}

TEST_CASE("EspressoBuilder multi-layer graph: inner_product + activation", "[espresso]") {
    const std::string dir = tmpdir() + "/ip_act.mlmodelc";

    EspressoBuilder b;
    b.add_inner_product("fc", "input", "hidden", 16, 8);
    b.add_activation("relu", "hidden", "output", EspressoBuilder::RELU,
                     0.0f, 0.0f, /*is_output=*/true);
    b.add_shape("input",  {1, 1, 1, 16});
    b.add_shape("hidden", {1, 1, 1, 8});
    b.add_shape("output", {1, 1, 1, 8});
    b.set_inputs ({std::vector<EspressoBuilder::IOTensor>{{"input",  {16, 1, 1}}}});
    b.set_outputs({std::vector<EspressoBuilder::IOTensor>{{"output", {8,  1, 1}}}});

    std::vector<float> weights(16 * 8, 0.1f);
    b.add_weight_blob(weights);

    REQUIRE(b.write(dir));

    std::string net = read_file(dir + "/model.espresso.net");
    // Two layers
    CHECK(net.find("\"inner_product\"") != std::string::npos);
    CHECK(net.find("\"activation\"")    != std::string::npos);
    // is_output only on the final layer
    auto pos = net.rfind("\"is_output\":1");
    REQUIRE(pos != std::string::npos);
    // The activation is the last entry in the JSON
    CHECK(net.rfind("\"activation\"") < pos);
}

TEST_CASE("EspressoBuilder write is idempotent (overwrites existing dir)", "[espresso]") {
    const std::string dir = tmpdir() + "/idem.mlmodelc";

    auto b = EspressoBuilder::activation("input", "output", 16, 32,
                                          EspressoBuilder::SIGMOID);
    REQUIRE(b.write(dir));
    // Second write to same dir should succeed
    REQUIRE(b.write(dir));
    CHECK(file_exists(dir + "/model.espresso.net"));
}
