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

/**
 * RAII guard that returns a buffer to the pool on scope exit.
 * Eliminates duplicated release calls on success/error paths.
 */
class PooledBuffer {
public:
    PooledBuffer(std::unique_ptr<libane::AneBuffer> buf, libane::BufferPool& pool)
        : buf_(std::move(buf)), pool_(pool) {}
    ~PooledBuffer() { if (buf_) pool_.release(std::move(buf_)); }

    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    PooledBuffer(PooledBuffer&&) = delete;
    PooledBuffer& operator=(PooledBuffer&&) = delete;

    libane::AneBuffer* operator->() const { return buf_.get(); }
    libane::AneBuffer& operator*()  const { return *buf_; }
    explicit operator bool()        const { return buf_ != nullptr; }

private:
    std::unique_ptr<libane::AneBuffer> buf_;
    libane::BufferPool& pool_;
};

/* ── Graph opaque handle definitions (global namespace, per libane.h) ─────── */

struct libane_graph_s {
    libane::graph::AneGraph graph;
};

struct libane_compiled_graph_s {
    std::unique_ptr<libane::graph::CompiledGraph> cg;
};

/* ── KV cache handle (global namespace, per libane.h) ────────────────────── */

struct libane_kv_cache_s {
    int  num_heads;
    int  head_dim;
    int  max_seq;
    int  pos;   ///< Number of valid token positions written (0..max_seq)

    /** K buffer: num_heads × max_seq × head_dim fp16 (zeros beyond pos). */
    std::vector<uint16_t> k_buf;
    /** V buffer: num_heads × max_seq × head_dim fp16 (zeros beyond pos). */
    std::vector<uint16_t> v_buf;

    /** Total elements per buffer. */
    size_t total_elems() const {
        return static_cast<size_t>(num_heads) * max_seq * head_dim;
    }
    /** Elements per head per token (= head_dim). */
    size_t elems_per_token() const {
        return static_cast<size_t>(num_heads) * head_dim;
    }
};

/* ── Raw MIL program handle (global namespace, per libane.h) ─────────────── */

struct libane_mil_program_s {
    libane::runtime::AneProgram* prog = nullptr;
};
