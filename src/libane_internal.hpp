/**
 * Internal C++ types for libane — not part of the public C API.
 * Only included from src/libane.cpp.
 */
#pragma once

#include "../include/libane.h"
#include "core/compile_cache.hpp"
#include "core/mil_builder.hpp"
#include "core/buffer_manager.hpp"
#include "runtime/ane_runtime.hpp"
#include "fallback/fallback.hpp"
#include "graph/ane_graph.hpp"
#include "graph/graph_compiler.hpp"
#include "graph/graph_executor.hpp"

#include <memory>

// Convenience type alias used throughout libane.cpp
using fp16_t = libane::fallback::fp16_t;

/* ── Graph opaque handle definitions (global namespace, per libane.h) ─────── */

struct libane_graph_s {
    libane::graph::AneGraph graph;
};

struct libane_compiled_graph_s {
    std::unique_ptr<libane::graph::CompiledGraph> cg;
};
