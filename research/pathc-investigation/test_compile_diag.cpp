#include <catch2/catch_test_macros.hpp>
#include "libane.h"
#include "../src/libane_internal.hpp"
#include <string>

TEST_CASE("compile diagnostic — with logging", "[diag]") {
    libane_set_log_level(LIBANE_LOG_DEBUG);
    // program(1.3) format — same as MilBuilder::header() + func main<ios18>
    static const char kMIL[] =
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1,32,1,32]> x) {\n"
        "        tensor<fp16, [1,32,1,32]> y = relu(x=x)[name=string(\"diag\")];\n"
        "    } -> (y);\n"
        "}\n";

    auto* h = libane_mil_compile(kMIL, nullptr, nullptr, nullptr, 0);
    INFO("libane_last_error: " << libane_last_error());
    if (!h) FAIL("compile returned nullptr — see libane_last_error above");
    std::string mdir = h->prog ? h->prog->model_dir : "(none)";
    INFO("model_dir: " << mdir);
    libane_mil_release(h);
    SUCCEED("compiled and released");
}
