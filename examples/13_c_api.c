/**
 * 13 — Minimal C API example
 *
 * The complete flow using only the stable C ABI:
 *   1. Check ANE availability
 *   2. Build a graph (RMSNorm → Matmul → GELU)
 *   3. Compile
 *   4. Execute with fp16 input/output
 *   5. Clean up
 *
 * Build:
 *   cmake --build build          # builds libane.dylib
 *   cc -std=c17 examples/13_c_api.c -I include -L build -lane -o 13_c_api
 *   DYLD_LIBRARY_PATH=build ./13_c_api
 *
 * Note: weights and activations are all fp16 (uint16_t on this platform).
 * ANE shape is always [1, C, 1, S] — dims[0]=1 batch, dims[1]=channels,
 * dims[2]=1 height, dims[3]=sequence.  S must be a multiple of 8.
 */
#include "libane.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define D   256    /* model dim (channels) */
#define SEQ 128    /* sequence length — must be multiple of 8 */

/* ── Simple fp32 → fp16 cast ─────────────────────────────────────────────── */
static uint16_t f32_to_f16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int32_t  e = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FF;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | m);
}

int main(void) {
    printf("libane %s\n", libane_version());
    printf("ANE available: %s\n\n", libane_available() ? "yes" : "no");

    if (!libane_available()) {
        fprintf(stderr, "ANE not available — exiting.\n");
        return 1;
    }

    /* ── Allocate and fill weights (fp16) ─────────────────────────────────
     * RMSNorm scale: D elements (all 1.0)
     * Matmul weight: D×D elements (identity-ish for easy verification)
     */
    size_t rn_len = D * sizeof(uint16_t);
    size_t mm_len = (size_t)D * D * sizeof(uint16_t);

    uint16_t* rn_scale = calloc(D, sizeof(uint16_t));
    uint16_t* mm_w     = calloc((size_t)D * D, sizeof(uint16_t));

    for (int i = 0; i < D; i++)
        rn_scale[i] = f32_to_f16(1.0f);

    /* Diagonal weights: W[i][i] = 1.0 (identity matrix) */
    for (int i = 0; i < D; i++)
        mm_w[i * D + i] = f32_to_f16(1.0f);

    /* ── Build graph ──────────────────────────────────────────────────────
     * x[D×SEQ]  →  RMSNorm  →  Matmul[D×D]  →  GELU  →  output
     */
    libane_graph_t g = libane_graph_create();
    if (!g) { fprintf(stderr, "graph_create failed\n"); return 1; }

    libane_shape_t shape = { .dims = {1, D, 1, SEQ}, .ndim = 4 };

    uint32_t x   = libane_graph_add_input(g, "x", shape);
    uint32_t rn  = libane_graph_add_op(g, LIBANE_OP_RMSNORM,  &x,  1, shape,
                                        rn_scale, rn_len);
    uint32_t mm  = libane_graph_add_op(g, LIBANE_OP_MATMUL,   &rn, 1, shape,
                                        mm_w, mm_len);
    uint32_t act = libane_graph_add_op(g, LIBANE_OP_GELU,     &mm, 1, shape,
                                        NULL, 0);

    libane_status_t st = libane_graph_mark_output(g, act, "out");
    if (st != LIBANE_OK) {
        fprintf(stderr, "mark_output failed: %s\n", libane_last_error());
        return 1;
    }

    /* ── Compile ──────────────────────────────────────────────────────────*/
    printf("Compiling (first time ~4 s)…\n");
    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) {
        fprintf(stderr, "compile failed: %s\n", libane_last_error());
        return 1;
    }
    printf("Compiled.\n\n");

    /* ── Allocate fp16 input (all 1.0) ───────────────────────────────────*/
    size_t     n_elem  = (size_t)D * SEQ;
    uint16_t*  input   = calloc(n_elem, sizeof(uint16_t));
    uint16_t*  output  = calloc(n_elem, sizeof(uint16_t));

    for (size_t i = 0; i < n_elem; i++)
        input[i] = f32_to_f16(1.0f);

    /* ── Execute ──────────────────────────────────────────────────────────*/
    const void* in_ptrs[]   = { input };
    size_t      in_bytes[]  = { n_elem * sizeof(uint16_t) };
    void*       out_ptrs[]  = { output };
    size_t      out_bytes[] = { n_elem * sizeof(uint16_t) };

    st = libane_graph_execute(cg, in_ptrs, in_bytes, 1,
                                   out_ptrs, out_bytes, 1);
    if (st != LIBANE_OK) {
        fprintf(stderr, "execute failed: %s\n", libane_last_error());
        return 1;
    }
    printf("Execute succeeded.\n");
    printf("First 4 output values (should be GELU(1.0) ≈ 0.841):\n");
    for (int i = 0; i < 4; i++) {
        uint32_t h = output[i];
        uint32_t s = (h & 0x8000u) << 16;
        uint32_t e = (h >> 10) & 0x1Fu;
        uint32_t m = h & 0x3FFu;
        uint32_t f32b = s | ((e + 112) << 23) | (m << 13);
        float v; memcpy(&v, &f32b, 4);
        printf("  output[%d] = %.4f\n", i, v);
    }

    /* ── Cleanup ──────────────────────────────────────────────────────────*/
    libane_compiled_graph_release(cg);
    libane_graph_release(g);
    free(rn_scale); free(mm_w);
    free(input);    free(output);

    return 0;
}
