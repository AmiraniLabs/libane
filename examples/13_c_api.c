/**
 * 13 — C API example
 *
 * Demonstrates three parts of the stable C ABI:
 *
 *   Part 1 — Graph API (RMSNorm → Matmul → GELU)
 *   Part 2 — libane_compile_batch(): compile multiple ops at once
 *   Part 3 — libane_execute2(): two-input elementwise dispatch (ADD)
 *
 * Build:
 *   cmake --build build          # builds libane.dylib
 *   cc -std=c17 examples/13_c_api.c -I include -L build -lane -o 13_c_api \
 *      -Wl,-rpath,build
 *   ./13_c_api
 */
#include "libane.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define D   256    /* model dim (channels) */
#define SEQ 128    /* sequence length — must be multiple of 16 */

/* ── fp32 → fp16 ─────────────────────────────────────────────────────────── */
static uint16_t f32_to_f16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 16) & 0x8000;
    int32_t  e = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FF;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | m);
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0)  return 0.0f;
    if (e == 31) return s ? -__builtin_inff() : __builtin_inff();
    uint32_t b = s | ((e + 112) << 23) | (m << 13);
    float v; memcpy(&v, &b, 4); return v;
}

int main(void) {
    printf("libane %s\n", libane_version());
    printf("ANE available: %s\n\n", libane_available() ? "yes" : "no");

    if (!libane_available()) {
        fprintf(stderr, "ANE not available — exiting.\n");
        return 1;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Part 1 — Graph API: RMSNorm → Matmul → GELU
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("── Part 1: Graph API (RMSNorm → Matmul → GELU) ────────────────────\n");

    size_t    rn_len = D * sizeof(uint16_t);
    size_t    mm_len = (size_t)D * D * sizeof(uint16_t);
    uint16_t* rn_scale = calloc(D, sizeof(uint16_t));
    uint16_t* mm_w     = calloc((size_t)D * D, sizeof(uint16_t));

    for (int i = 0; i < D; i++)
        rn_scale[i] = f32_to_f16(1.0f);                  /* unit scale */
    for (int i = 0; i < D; i++)
        mm_w[i * D + i] = f32_to_f16(1.0f);              /* identity matrix */

    libane_graph_t g = libane_graph_create();
    libane_shape_t shape = { .dims = {1, D, 1, SEQ}, .ndim = 4 };

    uint32_t x   = libane_graph_add_input(g, "x", shape);
    uint32_t rn  = libane_graph_add_op(g, LIBANE_OP_RMSNORM, &x,  1, shape,
                                        rn_scale, rn_len);
    uint32_t mm  = libane_graph_add_op(g, LIBANE_OP_MATMUL,  &rn, 1, shape,
                                        mm_w, mm_len);
    uint32_t act = libane_graph_add_op(g, LIBANE_OP_GELU,    &mm, 1, shape,
                                        NULL, 0);
    libane_graph_mark_output(g, act, "out");

    printf("Compiling graph (first time may take a few seconds)…\n");
    libane_compiled_graph_t cg = libane_graph_compile(g);
    if (!cg) { fprintf(stderr, "compile failed: %s\n", libane_last_error()); return 1; }

    size_t    n_elem = (size_t)D * SEQ;
    uint16_t* input  = calloc(n_elem, sizeof(uint16_t));
    uint16_t* output = calloc(n_elem, sizeof(uint16_t));
    for (size_t i = 0; i < n_elem; i++) input[i] = f32_to_f16(1.0f);

    const void* in_ptrs[]   = { input };
    size_t      in_bytes[]  = { n_elem * sizeof(uint16_t) };
    void*       out_ptrs[]  = { output };
    size_t      out_bytes[] = { n_elem * sizeof(uint16_t) };

    libane_status_t st = libane_graph_execute(cg, in_ptrs, in_bytes, 1,
                                                   out_ptrs, out_bytes, 1);
    if (st != LIBANE_OK) { fprintf(stderr, "execute failed: %s\n", libane_last_error()); return 1; }

    printf("Execute ok. First 4 outputs (GELU(1.0) ≈ 0.841):\n");
    for (int i = 0; i < 4; i++)
        printf("  output[%d] = %.4f\n", i, f16_to_f32(output[i]));
    printf("\n");

    libane_compiled_graph_release(cg);
    libane_graph_release(g);
    free(rn_scale); free(mm_w); free(input); free(output);

    /* ═══════════════════════════════════════════════════════════════════════
     * Part 2 — libane_compile_batch(): compile multiple ops at once
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("── Part 2: compile_batch (GELU + RELU + SILU, no weights) ─────────\n");

    libane_compile_request_t reqs[3] = {
        { LIBANE_OP_GELU, shape, NULL, 0 },
        { LIBANE_OP_RELU, shape, NULL, 0 },
        { LIBANE_OP_SILU, shape, NULL, 0 },
    };
    libane_handle_t handles[3] = { NULL, NULL, NULL };
    const char*     names[3]   = { "GELU", "RELU", "SILU" };

    st = libane_compile_batch(reqs, 3, handles);
    printf("compile_batch status: %s\n", st == LIBANE_OK ? "LIBANE_OK" : "partial failure");
    for (int i = 0; i < 3; i++)
        printf("  %s: %s\n", names[i], handles[i] ? "compiled" : "failed");
    printf("\n");

    for (int i = 0; i < 3; i++)
        if (handles[i]) libane_release(handles[i]);

    /* ═══════════════════════════════════════════════════════════════════════
     * Part 3 — libane_execute2(): two-input ADD dispatch
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("── Part 3: execute2 — elementwise ADD of two tensors ───────────────\n");

    libane_handle_t add_h = libane_compile(LIBANE_OP_ADD, shape, NULL, 0);
    if (!add_h) { fprintf(stderr, "ADD compile failed: %s\n", libane_last_error()); return 1; }

    n_elem = (size_t)D * SEQ;
    uint16_t* a   = calloc(n_elem, sizeof(uint16_t));
    uint16_t* b   = calloc(n_elem, sizeof(uint16_t));
    uint16_t* out = calloc(n_elem, sizeof(uint16_t));

    for (size_t i = 0; i < n_elem; i++) a[i] = f32_to_f16(3.0f);
    for (size_t i = 0; i < n_elem; i++) b[i] = f32_to_f16(4.0f);

    /* libane_execute2: inputs in alphabetical MIL parameter order */
    st = libane_execute2(add_h, a, b, out, shape);
    if (st != LIBANE_OK) { fprintf(stderr, "execute2 failed: %s\n", libane_last_error()); return 1; }

    printf("ADD(3.0, 4.0) → first 4 outputs (expected 7.0):\n");
    for (int i = 0; i < 4; i++)
        printf("  out[%d] = %.1f\n", i, f16_to_f32(out[i]));

    libane_release(add_h);
    free(a); free(b); free(out);

    return 0;
}
