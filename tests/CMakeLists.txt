cmake_minimum_required(VERSION 3.24)

# ── Fetch Catch2 ──────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ── Helper to add a test ──────────────────────────────────────────────────────
function(add_libane_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name}
        PRIVATE
            libane_static
            Catch2::Catch2WithMain
    )
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    if(APPLE)
        target_link_libraries(${name} PRIVATE
            "-framework Foundation"
            "-framework IOSurface"
            "-framework CoreFoundation"
            "-framework Accelerate"
        )
    endif()
    catch_discover_tests(${name})
endfunction()

# ── Test targets ──────────────────────────────────────────────────────────────
add_libane_test(test_mil_builder    test_mil_builder.cpp)
add_libane_test(test_compile_cache  test_compile_cache.cpp)
add_libane_test(test_buffer_manager test_buffer_manager.cpp)
add_libane_test(test_fallback       test_fallback.cpp)
add_libane_test(test_c_api          test_c_api.cpp)
add_libane_test(test_graph          test_graph.cpp)
add_libane_test(test_graph_validator test_graph_validator.cpp)
add_libane_test(test_fusion          test_fusion.cpp)
add_libane_test(test_graph_compiler    test_graph_compiler.cpp)
add_libane_test(test_graph_integration test_graph_integration.cpp)
add_libane_test(test_hwx_backend       test_hwx_backend.cpp)
add_libane_test(test_espresso_builder  test_espresso_builder.cpp)
add_libane_test(test_espresso_backend  test_espresso_backend.cpp)
add_libane_test(test_native_mil_ops    test_native_mil_ops.cpp)
add_libane_test(test_pathc_warmpath    test_pathc_warmpath.cpp)
add_libane_test(test_compile_budget    test_compile_budget.cpp)
add_libane_test(test_ane_load_hwx      test_ane_load_hwx.mm)
