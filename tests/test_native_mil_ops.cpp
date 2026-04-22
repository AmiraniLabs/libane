/**
 * Empirical verification: do "lowered" ops have native MIL counterparts?
 *
 * The library currently decomposes several ops into multi-instruction MIL chains
 * (e.g. neg → mul(x,-1), sinh → 0.5*(exp(x)-exp(-x))).  This test suite probes
 * whether each of those ops has a direct single-instruction MIL form that the
 * ANE compiler accepts and executes correctly on hardware.
 *
 * For each candidate op we:
 *   1. Compile a tiny MIL program using the direct instruction.
 *   2. Execute it with a sweep of known input values.
 *   3. Compare output against a float32 reference (allowing fp16 tolerance).
 *   4. A PASS means the op is native — mil_builder.cpp should be upgraded.
 *   5. A compile failure means the ANE compiler rejects the direct form — keep
 *      the existing decomposition.
 *
 * Run: ctest -R native_mil --output-on-failure
 *
 * Tags: [native_mil][tier3] — requires Apple Silicon with ANE available.
 */
#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <functional>

// ── fp16 helpers (same as test_c_api.cpp) ─────────────────────────────────

#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
#  include <arm_fp16.h>
static __fp16 f16(float v) { return static_cast<__fp16>(v); }
static float  f32(__fp16 v) { return static_cast<float>(v); }
#else
using fp16_test = uint16_t;
static fp16_test f16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t s = (fb >> 16) & 0x8000;
    int32_t  e = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FF;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00);
    else              h = static_cast<uint16_t>(s | (e << 10) | m);
    return h;
}
static float f32(fp16_test h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t fb;
    if (e == 0)       fb = s | (m << 13);
    else if (e == 31) fb = s | 0x7F800000 | (m << 13);
    else              fb = s | ((e + 112) << 23) | (m << 13);
    float fv; std::memcpy(&fv, &fb, 4); return fv;
}
#  define __fp16 fp16_test
#endif

// ── Test shape: [1, 4, 1, 32] = 128 fp16 elements ─────────────────────────

static constexpr int kC = 4;
static constexpr int kS = 32;
static constexpr int kN = kC * kS;   // 128 elements
static constexpr int kBytes = kN * 2; // 256 bytes

// ── Tolerance for fp16 transcendentals ────────────────────────────────────
//
// ANE fp16 ALU: ~3 decimal digits, plus rounding in transcendentals.
// We use a generous 2% relative error (or 0.005 absolute for near-zero values).

static bool fp16_close(float got, float ref, float rtol = 0.02f, float atol = 0.005f) {
    float diff = std::abs(got - ref);
    return diff <= atol || diff <= rtol * std::abs(ref);
}

// ── MIL program builder helpers ───────────────────────────────────────────

// Standard MIL program header (program(1.3) format required by ANE compiler).
static std::string mil_header() {
    return
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n";
}

// Typed tensor type string: tensor<fp16, [1,C,1,S]>
static std::string tt(int C, int S) {
    return "tensor<fp16, [1," + std::to_string(C) + ",1," + std::to_string(S) + "]>";
}

// Unary op: func main<ios18>(tensor<fp16,...> x) { tensor<fp16,...> y = op(x=x)[...]; } -> (y);
static std::string unary_mil(const std::string& op, int C, int S) {
    return mil_header() +
        "    func main<ios18>(" + tt(C,S) + " x) {\n"
        "        " + tt(C,S) + " y = " + op + "(x=x)[name=string(\"" + op + "\")];\n"
        "    } -> (y);\n"
        "}\n";
}

// Binary op with two inputs named "x" and "y" (alphabetical = natural).
static std::string binary_mil(const std::string& op, const std::string& x_param,
                               const std::string& y_param, int C, int S) {
    return mil_header() +
        "    func main<ios18>(" + tt(C,S) + " x, " + tt(C,S) + " y) {\n"
        "        " + tt(C,S) + " z = " + op + "(" + x_param + "=x, " + y_param +
        "=y)[name=string(\"" + op + "\")];\n"
        "    } -> (z);\n"
        "}\n";
}

// ── Sentinel: verify the ANE compile slot budget isn't exhausted ──────────
//
// The ANE kernel daemon (aned) maintains a per-machine limit of ~119
// simultaneously registered model slots.  Slots are deregistered by
// unloadWithQoS: (called in ane_unload / libane_mil_release).  If previous
// test runs exited without releasing their programs the slots accumulate in
// aned's in-memory table and cannot be freed without a reboot (aned is
// SIP-protected and cannot be restarted at runtime).
//
// We probe with a trivial relu program first; if that fails ALL tests in this
// file skip with a diagnostic rather than reporting false negatives.
//
// If this fires: reboot the machine and re-run.  The libane fix in ane_unload()
// ensures slots are released going forward so this should not recur.

static bool ane_compile_slots_available() {
    // program(1.3) format — same as MilBuilder::header() + func main<ios18>
    static const char kSentinel[] =
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1,32,1,32]> x) {\n"
        "        tensor<fp16, [1,32,1,32]> y = relu(x=x)[name=string(\"sentinel\")];\n"
        "    } -> (y);\n"
        "}\n";
    libane_set_log_level(LIBANE_LOG_SILENT);
    auto* h = libane_mil_compile(kSentinel, nullptr, nullptr, nullptr, 0);
    if (!h) return false;
    libane_mil_release(h);
    return true;
}

// ── Core probe: compile + execute + check ─────────────────────────────────

struct ProbeResult {
    bool compiled   = false;
    bool executed   = false;
    bool numerics   = false;
    int  mismatches = 0;
};

static ProbeResult probe_unary(
    const std::string& mil_text,
    const std::vector<float>& input_f32,
    std::function<float(float)> ref_fn,
    float rtol = 0.02f,
    float atol = 0.005f)
{
    ProbeResult r;
    int N = static_cast<int>(input_f32.size());

    // Build fp16 input
    std::vector<__fp16> in(N), out(N, f16(0.0f));
    for (int i = 0; i < N; ++i) in[i] = f16(input_f32[i]);

    auto* h = libane_mil_compile(mil_text.c_str(), nullptr, nullptr, nullptr, 0);
    if (!h) return r;
    r.compiled = true;

    const void* in_ptrs[]  = { in.data() };
    void*       out_ptrs[] = { out.data() };
    size_t      in_bytes[] = { static_cast<size_t>(N) * 2 };
    size_t      out_bytes[]= { static_cast<size_t>(N) * 2 };

    auto st = libane_mil_execute(h, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    libane_mil_release(h);
    if (st != LIBANE_OK) return r;
    r.executed = true;

    for (int i = 0; i < N; ++i) {
        float got = f32(out[i]);
        float ref = ref_fn(input_f32[i]);
        if (!fp16_close(got, ref, rtol, atol)) ++r.mismatches;
    }
    r.numerics = (r.mismatches == 0);
    return r;
}

static ProbeResult probe_binary(
    const std::string& mil_text,
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    std::function<float(float, float)> ref_fn,
    float rtol = 0.02f,
    float atol = 0.005f)
{
    ProbeResult r;
    int N = static_cast<int>(xs.size());

    std::vector<__fp16> ix(N), iy(N), out(N, f16(0.0f));
    for (int i = 0; i < N; ++i) { ix[i] = f16(xs[i]); iy[i] = f16(ys[i]); }

    auto* h = libane_mil_compile(mil_text.c_str(), nullptr, nullptr, nullptr, 0);
    if (!h) return r;
    r.compiled = true;

    const void* in_ptrs[]  = { ix.data(), iy.data() };
    void*       out_ptrs[] = { out.data() };
    size_t      in_bytes[] = { static_cast<size_t>(N) * 2, static_cast<size_t>(N) * 2 };
    size_t      out_bytes[]= { static_cast<size_t>(N) * 2 };

    auto st = libane_mil_execute(h, in_ptrs, in_bytes, 2, out_ptrs, out_bytes, 1);
    libane_mil_release(h);
    if (st != LIBANE_OK) return r;
    r.executed = true;

    for (int i = 0; i < N; ++i) {
        float got = f32(out[i]);
        float ref = ref_fn(xs[i], ys[i]);
        if (!fp16_close(got, ref, rtol, atol)) ++r.mismatches;
    }
    r.numerics = (r.mismatches == 0);
    return r;
}

// ── Build a representative input sweep ────────────────────────────────────

// For ops defined on all reals, use a sweep of small values.
static std::vector<float> sweep_general() {
    std::vector<float> v(kN);
    // Values chosen to stay well within fp16 range and avoid
    // transcendental singularities (no exactly ±π/2 for tan, etc.)
    float xs[] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f,
                    1.0f,  0.1f,  -0.1f,  0.3f,  -0.3f, 0.7f, -0.7f, 0.9f };
    for (int i = 0; i < kN; ++i) v[i] = xs[i % 16];
    return v;
}

// For ops that need input in (-1, 1): asin, acos.
static std::vector<float> sweep_unit() {
    std::vector<float> v(kN);
    float xs[] = { -0.9f, -0.7f, -0.5f, -0.3f, -0.1f, 0.0f, 0.1f, 0.3f,
                    0.5f,  0.7f,  0.9f,  0.6f, -0.6f, 0.4f, -0.4f, 0.2f };
    for (int i = 0; i < kN; ++i) v[i] = xs[i % 16];
    return v;
}

// ── Individual op probes ───────────────────────────────────────────────────
//
// These are discovery probes, not correctness assertions.
//
// If the op compiles:    REQUIRE execution is OK, CHECK numerics.  A PASS
//                        means the op is native — upgrade mil_builder.cpp.
// If the op does NOT compile: WARN and return.  This is an informational
//                        PASS: the ANE compiler rejects the op, confirming
//                        that the existing decomposition in mil_builder.cpp
//                        is the right approach.

#define PROBE_PREAMBLE() \
    do { \
        libane_set_backend(nullptr); \
        libane_set_log_level(LIBANE_LOG_SILENT); \
        if (!libane_available())             { WARN("ANE not available — skipping"); return; } \
        if (!ane_compile_slots_available())  { WARN("ANE slots exhausted — reboot required, then re-run"); return; } \
    } while (0)

#define CHECK_PROBE(r, op_name, advice) \
    do { \
        INFO("compiled=" << (r).compiled << " executed=" << (r).executed \
             << " mismatches=" << (r).mismatches); \
        if (!(r).compiled) { \
            WARN(op_name ": ANE compiler rejects direct form (InvalidMILProgram) — " advice); \
            return; \
        } \
        REQUIRE((r).executed); \
        CHECK((r).mismatches == 0); \
    } while (0)

TEST_CASE("native MIL: neg(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    auto r = probe_unary(unary_mil("neg", kC, kS), sweep_general(),
                         [](float x) { return -x; });
    CHECK_PROBE(r, "neg", "keep mul(x,-1) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: sinh(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    auto r = probe_unary(unary_mil("sinh", kC, kS), sweep_general(),
                         [](float x) { return std::sinh(x); }, 0.03f, 0.005f);
    CHECK_PROBE(r, "sinh", "keep 0.5*(exp(x)-exp(-x)) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: cosh(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    auto r = probe_unary(unary_mil("cosh", kC, kS), sweep_general(),
                         [](float x) { return std::cosh(x); }, 0.03f, 0.005f);
    CHECK_PROBE(r, "cosh", "keep 0.5*(exp(x)+exp(-x)) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: tan(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    // Avoid values near ±π/2 where tan diverges; sweep_general stays in (-1, 1)
    auto r = probe_unary(unary_mil("tan", kC, kS), sweep_general(),
                         [](float x) { return std::tan(x); }, 0.03f, 0.005f);
    CHECK_PROBE(r, "tan", "keep sin(x)/cos(x) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: asin(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    auto r = probe_unary(unary_mil("asin", kC, kS), sweep_unit(),
                         [](float x) { return std::asin(x); }, 0.03f, 0.005f);
    CHECK_PROBE(r, "asin", "keep atan2(x, sqrt(1-x^2)) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: acos(x=x) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    auto r = probe_unary(unary_mil("acos", kC, kS), sweep_unit(),
                         [](float x) { return std::acos(x); }, 0.03f, 0.008f);
    CHECK_PROBE(r, "acos", "keep pi/2 - asin(x) decomposition in mil_builder.cpp");
}

TEST_CASE("native MIL: mod(x=x, y=y) compiles and is numerically correct", "[native_mil][tier3]") {
    PROBE_PREAMBLE();
    // Positive divisors only — floor-mod semantics match std::fmod for x>0, y>0.
    std::vector<float> xs(kN), ys(kN);
    float xvals[] = { 3.5f, 2.0f, 5.0f, 1.75f, 4.0f, 0.5f, 3.0f, 2.5f,
                      7.0f, 1.0f, 6.5f, 0.25f, 4.5f, 3.25f, 2.75f, 1.5f };
    float yvals[] = { 2.0f, 1.5f, 3.0f, 1.0f, 2.5f, 0.4f, 1.75f, 2.0f,
                      4.0f, 0.6f, 4.0f, 0.2f, 3.0f, 2.0f, 1.5f, 1.0f };
    for (int i = 0; i < kN; ++i) { xs[i] = xvals[i % 16]; ys[i] = yvals[i % 16]; }
    // CoreML MIL mod uses floor semantics: x - floor(x/y)*y.
    auto r = probe_binary(binary_mil("mod", "x", "y", kC, kS), xs, ys,
                          [](float x, float y) { return x - std::floor(x / y) * y; });
    INFO("compiled=" << r.compiled << " executed=" << r.executed
         << " mismatches=" << r.mismatches);
    if (!r.compiled) {
        WARN("mod: ANE compiler rejects direct form (InvalidMILProgram) — keep floor_div decomposition in mil_builder.cpp");
        return;
    }
    REQUIRE(r.executed);
    CHECK(r.mismatches == 0);
}

// ── Summary test: print a table of which ops are native ───────────────────

TEST_CASE("native MIL: summary — print which ops have direct ANE support",
          "[native_mil][tier3][summary]") {
    libane_set_backend(nullptr);
    libane_set_log_level(LIBANE_LOG_SILENT);
    if (!libane_available())           { WARN("ANE not available — skipping"); return; }
    if (!ane_compile_slots_available()) { WARN("ANE slots exhausted — reboot required, then re-run"); return; }

    struct Entry { const char* name; bool unary; std::string mil; };

    auto sw_gen  = sweep_general();
    auto sw_unit = sweep_unit();

    // pairs: {name, input_sweep, expected_fn, mil_text}
    struct Probe {
        const char* name;
        std::string mil;
        std::vector<float> inp;
        std::function<float(float)> ref;
        float rtol, atol;
    };

    std::vector<Probe> probes = {
        { "neg",  unary_mil("neg",  kC,kS), sw_gen,  [](float x){return -x;},         0.02f, 0.005f },
        { "sinh", unary_mil("sinh", kC,kS), sw_gen,  [](float x){return std::sinh(x);},0.03f, 0.005f },
        { "cosh", unary_mil("cosh", kC,kS), sw_gen,  [](float x){return std::cosh(x);},0.03f, 0.005f },
        { "tan",  unary_mil("tan",  kC,kS), sw_gen,  [](float x){return std::tan(x);}, 0.03f, 0.005f },
        { "asin", unary_mil("asin", kC,kS), sw_unit, [](float x){return std::asin(x);},0.03f, 0.005f },
        { "acos", unary_mil("acos", kC,kS), sw_unit, [](float x){return std::acos(x);},0.03f, 0.008f },
    };

    WARN("=== Native MIL Op Support ===");
    WARN("  op       | compiled | executed | numerics");
    WARN("  ---------+----------+----------+---------");

    for (auto& p : probes) {
        auto r = probe_unary(p.mil, p.inp, p.ref, p.rtol, p.atol);
        std::string row = std::string("  ") + std::string(p.name);
        while (row.size() < 9) row += ' ';
        row += "| " + std::string(r.compiled ? "YES      " : "NO       ");
        row += "| " + std::string(r.executed ? "YES      " : "NO       ");
        row += "| " + std::string(r.numerics ? "PASS" : ("FAIL (" + std::to_string(r.mismatches) + ")"));
        WARN(row);
    }
    WARN("  (Ops with YES/YES/PASS can replace their decomposition in mil_builder.cpp)");
}
