#!/usr/bin/env python3
"""
09 — Attention scores: what runs on ANE, what runs on CPU

Not everything fits on the ANE.  This example shows the realistic split for
scaled dot-product attention in a transformer decoder:

    ON ANE:
        Q = RMSNorm(x) @ W_Q          [fused, 1 dispatch]
        K = RMSNorm(x) @ W_K          [fused, 1 dispatch]
        V = RMSNorm(x) @ W_V          [fused, 1 dispatch]
        out_proj = context @ W_O      [1 dispatch]

    ON CPU (numpy / Accelerate):
        scores = (Q.T @ K) / sqrt(head_dim)   — seq×seq matrix, wrong shape for ANE
        weights = softmax(scores)
        context = weights @ V.T

The ANE handles the bulk of FLOPs (all 4 projections).  The attention
matrix multiply (seq×seq) goes to Accelerate BLAS since its shape doesn't
map cleanly to [1, C, 1, S].

Total ANE coverage for a typical decoder layer: ~80% of FLOPs.
"""
import time
import numpy as np
import ane

D     = 1024   # model dim
H     = 64     # head dim
N_H   = 16     # heads
QKV   = H * N_H
SEQ   = 128

rng = np.random.default_rng(7)
def rand(r, c): return (rng.standard_normal((r, c)) * 0.02).astype(np.float16)

W_Q   = rand(D, QKV)
W_K   = rand(D, QKV)
W_V   = rand(D, QKV)
W_O   = rand(QKV, D)
scale = np.ones(D, dtype=np.float16)

# ── Build two ANE graphs ──────────────────────────────────────────────────────
# Graph 1: QKV projections (three outputs from one input)
g_qkv = ane.Graph()
x     = g_qkv.add_input("x", [1, D, 1, SEQ])
rn    = g_qkv.add_op(ane.RMSNORM, [x], [1, D, 1, SEQ], weights=scale)
q     = g_qkv.add_op(ane.MATMUL,  [rn],[1, QKV, 1, SEQ], weights=W_Q)
k     = g_qkv.add_op(ane.MATMUL,  [rn],[1, QKV, 1, SEQ], weights=W_K)
v     = g_qkv.add_op(ane.MATMUL,  [rn],[1, QKV, 1, SEQ], weights=W_V)
g_qkv.mark_output(q); g_qkv.mark_output(k); g_qkv.mark_output(v)

cg_qkv = g_qkv.compile()
cg_qkv.set_output_shapes([[1,QKV,1,SEQ]]*3)

# Graph 2: output projection
g_out = ane.Graph()
ctx   = g_out.add_input("ctx", [1, QKV, 1, SEQ])
out   = g_out.add_op(ane.MATMUL, [ctx], [1, D, 1, SEQ], weights=W_O)
g_out.mark_output(out)

cg_out = g_out.compile()
cg_out.set_output_shapes([[1, D, 1, SEQ]])

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)

# Warm-up
for _ in range(3):
    Q, K, V = cg_qkv(x_data)
    cg_out(Q)

# ── Timed full attention forward pass ────────────────────────────────────────
N = 30
times_ane = []; times_cpu = []; times_total = []

for _ in range(N):
    t_start = time.perf_counter()

    # ANE: QKV projections
    t0 = time.perf_counter()
    Q, K, V = cg_qkv(x_data)
    t_ane1 = (time.perf_counter() - t0) * 1000

    # Reshape for multi-head: [N_H, H, SEQ]
    Q32 = Q.reshape(N_H, H, SEQ).astype(np.float32)
    K32 = K.reshape(N_H, H, SEQ).astype(np.float32)
    V32 = V.reshape(N_H, H, SEQ).astype(np.float32)

    # CPU: attention scores + softmax + context
    t0 = time.perf_counter()
    scores  = np.einsum("hds,hdk->hsk", Q32, K32) / (H ** 0.5)   # [N_H, SEQ, SEQ]
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    context = (weights @ V32.transpose(0, 2, 1)).transpose(0, 2, 1)   # [N_H, H, SEQ]
    context_f16 = context.reshape(QKV, SEQ).astype(np.float16)
    t_cpu = (time.perf_counter() - t0) * 1000

    # ANE: output projection
    t0 = time.perf_counter()
    output = cg_out(context_f16)
    t_ane2 = (time.perf_counter() - t0) * 1000

    times_ane.append(t_ane1 + t_ane2)
    times_cpu.append(t_cpu)
    times_total.append((time.perf_counter() - t_start) * 1000)

times_ane.sort(); times_cpu.sort(); times_total.sort()

print(f"Attention  D={D}  heads={N_H}  head_dim={H}  seq={SEQ}")
print()
print(f"  ANE (QKV + out_proj) p50: {times_ane[N//2]:.2f} ms")
print(f"  CPU (scores + softmax + context) p50: {times_cpu[N//2]:.2f} ms")
print(f"  Total p50: {times_total[N//2]:.2f} ms")
print()
ane_pct = times_ane[N//2] / times_total[N//2] * 100
flops_ane  = 4 * 2 * D * QKV * SEQ   # 4 matmuls × 2 (mul+add) × D×QKV×SEQ
flops_total= flops_ane + 2 * N_H * SEQ * SEQ * H
print(f"  ANE share of latency : {ane_pct:.0f}%")
print(f"  ANE share of FLOPs   : {flops_ane/flops_total*100:.0f}%")
print()
print("  The seq×seq attention matrix (QK^T) stays on CPU — its shape")
print("  doesn't map to [1,C,1,S].  Everything else runs on ANE.")
