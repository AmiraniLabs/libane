#!/usr/bin/env python3
"""
12 — Full pre-norm transformer layer

Combines attention projections (example 09) and SwiGLU FFN (example 06)
into a complete pre-norm transformer layer as used in LLaMA 3 / Mistral:

    # Attention sub-layer
    h    = RMSNorm(x)
    Q, K, V = h @ W_Q, h @ W_K, h @ W_V    [ANE, 3 groups]
    attn = softmax(Q^T K / √d) @ V^T        [CPU — seq×seq doesn't fit ANE]
    h    = x + attn @ W_O                   [ANE — proj + residual]

    # FFN sub-layer
    h2   = RMSNorm(h)
    gate = SiLU(h2 @ W_gate)                [ANE, fused]
    up   = h2 @ W_up                        [ANE]
    h    = h + (gate ⊙ up) @ W_down        [ANE, fused]

Six ANE graphs (some single-group, some fused) + two small CPU matmuls.
The residual adds are absorbed into the ANE graphs as side inputs.

This is the pattern for building a full LLM inference stack on top of libane.
"""
import time
import numpy as np
import ane

# ── Dimensions (Llama 3 8B scaled down for quick test) ───────────────────────
D    = 1024   # model dim
H    = 64     # head dim
N_H  = 16     # heads
QKV  = H * N_H
INT  = 3584   # FFN intermediate (~3.5× D)
SEQ  = 64

rng = np.random.default_rng(10)
def rand(r, c): return (rng.standard_normal((r, c)) * 0.02).astype(np.float16)
ones = np.ones(D, dtype=np.float16)

W_Q    = rand(D, QKV)
W_K    = rand(D, QKV)
W_V    = rand(D, QKV)
W_O    = rand(QKV, D)
W_gate = rand(D, INT)
W_up   = rand(D, INT)
W_down = rand(INT, D)


# ── Graph 1: attn RMSNorm + QKV projections ──────────────────────────────────
g1 = ane.Graph()
x1 = g1.add_input("x", [1, D, 1, SEQ])
rn1= g1.add_op(ane.RMSNORM, [x1], [1, D, 1, SEQ], weights=ones)
q  = g1.add_op(ane.MATMUL,  [rn1],[1, QKV, 1, SEQ], weights=W_Q)
k  = g1.add_op(ane.MATMUL,  [rn1],[1, QKV, 1, SEQ], weights=W_K)
v  = g1.add_op(ane.MATMUL,  [rn1],[1, QKV, 1, SEQ], weights=W_V)
g1.mark_output(q); g1.mark_output(k); g1.mark_output(v)
cg1 = g1.compile()
cg1.set_output_shapes([[1,QKV,1,SEQ]]*3)

# ── Graph 2: output projection + residual ────────────────────────────────────
g2  = ane.Graph()
ctx = g2.add_input("ctx", [1, QKV, 1, SEQ])
x2  = g2.add_input("x",   [1, D,   1, SEQ])
proj= g2.add_op(ane.MATMUL, [ctx], [1, D, 1, SEQ], weights=W_O)
res1= g2.add_op(ane.ADD,    [proj, x2], [1, D, 1, SEQ])
g2.mark_output(res1)
cg2 = g2.compile()
cg2.set_output_shapes([[1, D, 1, SEQ]])

# ── Graph 3: FFN RMSNorm + SwiGLU + residual ─────────────────────────────────
g3   = ane.Graph()
h_in = g3.add_input("h",  [1, D, 1, SEQ])
rn3  = g3.add_op(ane.RMSNORM, [h_in], [1, D,   1, SEQ], weights=ones)
gate = g3.add_op(ane.MATMUL,  [rn3],  [1, INT, 1, SEQ], weights=W_gate)
silu = g3.add_op(ane.SILU,    [gate], [1, INT, 1, SEQ])
up   = g3.add_op(ane.MATMUL,  [rn3],  [1, INT, 1, SEQ], weights=W_up)
merg = g3.add_op(ane.MUL,     [silu, up], [1, INT, 1, SEQ])
down = g3.add_op(ane.MATMUL,  [merg], [1, D,   1, SEQ], weights=W_down)
res2 = g3.add_op(ane.ADD,     [down, h_in], [1, D, 1, SEQ])
g3.mark_output(res2)
cg3 = g3.compile()
cg3.set_output_shapes([[1, D, 1, SEQ]])

print("Compiled 3 ANE graphs.  Running forward pass…")

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)

def forward(x_np):
    # ── Attention sub-layer ───────────────────────────────────────────────
    Q, K, V = cg1(x_np)

    Q32 = Q.reshape(N_H, H, SEQ).astype(np.float32)
    K32 = K.reshape(N_H, H, SEQ).astype(np.float32)
    V32 = V.reshape(N_H, H, SEQ).astype(np.float32)

    scores  = np.einsum("hds,hdk->hsk", Q32, K32) / (H ** 0.5)
    w       = np.exp(scores - scores.max(-1, keepdims=True))
    w      /= w.sum(-1, keepdims=True)
    ctx32   = (w @ V32.transpose(0,2,1)).transpose(0,2,1).reshape(QKV, SEQ).astype(np.float16)

    h = cg2([ctx32, x_np])   # out_proj + residual

    # ── FFN sub-layer ─────────────────────────────────────────────────────
    h = cg3(h)   # RMSNorm + SwiGLU + residual
    return h

# Warm-up
for _ in range(3): forward(x_data)

# Timed
N = 30
times = []
for _ in range(N):
    t0 = time.perf_counter()
    out = forward(x_data)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
total_params = (D*QKV*3 + QKV*D + D*INT*2 + INT*D) / 1e6

print(f"\nTransformer layer  D={D}  heads={N_H}  FFN={INT}  seq={SEQ}")
print(f"  Parameters : {total_params:.0f} M")
print(f"  Latency p50: {times[N//2]:.2f} ms")
print(f"  Latency p95: {times[int(N*0.95)]:.2f} ms")
print(f"  Output shape: {out.shape}")
print(f"  Output finite: {np.all(np.isfinite(out.astype(np.float32)))}")
print()
print("  3 ANE graphs + 1 CPU einsum for the seq×seq attention matrix.")
print("  Everything else — all projections, norms, activations — runs on ANE.")
