cmake_minimum_required(VERSION 3.24)
project(libane VERSION 0.7.0 LANGUAGES CXX OBJCXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ── Platform guards ───────────────────────────────────────────────────────────
if(NOT APPLE)
  message(WARNING "libane is macOS/Apple Silicon only. Building fallback-only library.")
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(LIBANE_SOURCES
    src/core/mil_builder.cpp
    src/core/compile_cache.cpp
    src/core/buffer_manager.cpp
    src/fallback/fallback.cpp
    src/graph/ane_graph.cpp
    src/graph/graph_validator.cpp
    src/graph/fusion_rules.cpp
    src/graph/graph_compiler.cpp
    src/graph/graph_executor.cpp
    src/libane.cpp
)

if(APPLE)
  list(APPEND LIBANE_SOURCES src/runtime/ane_runtime.mm)
endif()

# ── Library target ────────────────────────────────────────────────────────────
add_library(libane SHARED ${LIBANE_SOURCES})
add_library(libane_static STATIC ${LIBANE_SOURCES})

foreach(target libane libane_static)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_options(${target} PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -fvisibility=hidden
    -DACCELERATE_NEW_LAPACK
    -DACCELERATE_LAPACK_ILP64
  )
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "ane"
    VERSION     ${PROJECT_VERSION}
    SOVERSION   ${PROJECT_VERSION_MAJOR}
  )
endforeach()

# ── Apple-specific frameworks ─────────────────────────────────────────────────
if(APPLE)
  foreach(target libane libane_static)
    target_link_libraries(${target}
      PUBLIC
        "-framework Foundation"
        "-framework IOSurface"
        "-framework CoreFoundation"
      PRIVATE
        "-framework Accelerate"
    )
  endforeach()
else()
  target_link_libraries(libane PRIVATE m)
  target_link_libraries(libane_static PRIVATE m)
endif()

# ── find_package support ──────────────────────────────────────────────────────
install(TARGETS libane libane_static
        EXPORT LibaneTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(FILES include/libane.h DESTINATION include)
install(EXPORT LibaneTargets
        FILE LibaneTargets.cmake
        NAMESPACE Libane::
        DESTINATION lib/cmake/libane)
install(FILES cmake/FindLibane.cmake DESTINATION lib/cmake/libane)

# ── Tests ─────────────────────────────────────────────────────────────────────
option(LIBANE_BUILD_TESTS "Build test suite" ON)
if(LIBANE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Python bindings ───────────────────────────────────────────────────────────
option(LIBANE_BUILD_PYTHON "Build Python bindings" OFF)
if(LIBANE_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "libane ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform:      ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Tests:         ${LIBANE_BUILD_TESTS}")
message(STATUS "  Python:        ${LIBANE_BUILD_PYTHON}")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  message(STATUS "  ANE backend:   ENABLED (Apple Silicon)")
else()
  message(STATUS "  ANE backend:   DISABLED (CPU fallback only)")
endif()
