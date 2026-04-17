/**
 * bench_dispatch — latency benchmark for ANE dispatch path.
 *
 * Compiles a small matmul graph (IC=512, OC=512, SP=128) and executes it
 * N times, recording per-call wall-clock latency. Run before and after
 * swapping evaluateWithQoS: → doEvaluateDirectWithModel: to measure the
 * improvement.
 *
 * Usage:
 *   ./build/tools/bench_dispatch [iterations]   (default: 1000)
 *
 * Output: mean, median, p95, p99, min, max in microseconds.
 */
#include "libane.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::duration<double, std::micro>;

/* ── fp16 helpers ────────────────────────────────────────────────────────── */

static uint16_t f32_to_f16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    uint32_t s = (fb >> 16) & 0x8000u;
    int32_t  e = static_cast<int32_t>((fb >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = (fb >> 13) & 0x3FFu;
    uint16_t h;
    if (e <= 0)       h = static_cast<uint16_t>(s);
    else if (e >= 31) h = static_cast<uint16_t>(s | 0x7C00u);
    else              h = static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | m);
    return h;
}

/* ── Percentile helper ───────────────────────────────────────────────────── */

static double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double idx = p * static_cast<double>(sorted.size() - 1);
    size_t lo  = static_cast<size_t>(idx);
    size_t hi  = lo + 1 < sorted.size() ? lo + 1 : lo;
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int iterations = 1000;
    if (argc >= 2) iterations = std::atoi(argv[1]);
    if (iterations < 1) iterations = 1;

    printf("bench_dispatch: %d iterations\n\n", iterations);

    if (!libane_available()) {
        fprintf(stderr, "ANE not available on this machine.\n");
        return 1;
    }

    /* ── Build and compile the graph ─────────────────────────────────────── */

    const int IC = 512, OC = 512, SP = 128;

    // Weights: IC × OC fp16, all 1/IC so matmul output ≈ 1.0
    const size_t w_elems = static_cast<size_t>(IC) * OC;
    std::vector<uint16_t> weights(w_elems, f32_to_f16(1.0f / IC));

    libane_shape_t s_in;
    s_in.ndim = 4;
    s_in.dims[0] = 1; s_in.dims[1] = IC; s_in.dims[2] = 1; s_in.dims[3] = SP;

    libane_shape_t s_out;
    s_out.ndim = 4;
    s_out.dims[0] = 1; s_out.dims[1] = OC; s_out.dims[2] = 1; s_out.dims[3] = SP;

    libane_graph_t g = libane_graph_create();
    if (!g) { fprintf(stderr, "libane_graph_create failed\n"); return 1; }

    uint32_t x_id = libane_graph_add_input(g, "x", s_in);
    uint32_t y_id = libane_graph_add_op(g, LIBANE_OP_MATMUL, &x_id, 1, s_out,
                                         weights.data(), w_elems * 2);

    if (x_id == LIBANE_INVALID_TENSOR_ID || y_id == LIBANE_INVALID_TENSOR_ID) {
        fprintf(stderr, "graph construction failed\n");
        libane_graph_release(g);
        return 1;
    }
    libane_graph_mark_output(g, y_id, "out");

    libane_compiled_graph_t cg = libane_graph_compile(g);
    libane_graph_release(g);

    if (!cg) {
        fprintf(stderr, "libane_graph_compile failed\n");
        return 1;
    }

    printf("Graph: matmul IC=%d OC=%d SP=%d  (%.1f KB weights)\n",
           IC, OC, SP, static_cast<double>(w_elems * 2) / 1024.0);

    /* ── Allocate I/O buffers ─────────────────────────────────────────────── */

    const size_t in_elems  = static_cast<size_t>(IC) * SP;
    const size_t out_elems = static_cast<size_t>(OC) * SP;

    std::vector<uint16_t> in_buf(in_elems,  f32_to_f16(1.0f));
    std::vector<uint16_t> out_buf(out_elems, 0);

    const void* in_ptrs[1]  = { in_buf.data() };
    size_t      in_bytes[1] = { in_elems * 2 };
    void*       out_ptrs[1] = { out_buf.data() };
    size_t      out_bytes[1]= { out_elems * 2 };

    /* ── Warmup ──────────────────────────────────────────────────────────── */

    printf("Warming up (10 iterations)...\n");
    for (int i = 0; i < 10; ++i) {
        libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
    }

    /* ── Timed runs ──────────────────────────────────────────────────────── */

    printf("Timing %d iterations...\n\n", iterations);
    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        libane_graph_execute(cg, in_ptrs, in_bytes, 1, out_ptrs, out_bytes, 1);
        auto t1 = Clock::now();
        latencies.push_back(Us(t1 - t0).count());
    }

    libane_compiled_graph_release(cg);

    /* ── Report ──────────────────────────────────────────────────────────── */

    std::sort(latencies.begin(), latencies.end());

    double sum  = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / static_cast<double>(latencies.size());
    double p50  = percentile(latencies, 0.50);
    double p95  = percentile(latencies, 0.95);
    double p99  = percentile(latencies, 0.99);
    double mn   = latencies.front();
    double mx   = latencies.back();

    printf("Dispatch latency (%d iters, matmul %dx%d SP=%d):\n", iterations, IC, OC, SP);
    printf("  mean  %8.1f us\n", mean);
    printf("  p50   %8.1f us\n", p50);
    printf("  p95   %8.1f us\n", p95);
    printf("  p99   %8.1f us\n", p99);
    printf("  min   %8.1f us\n", mn);
    printf("  max   %8.1f us\n", mx);

    return 0;
}
