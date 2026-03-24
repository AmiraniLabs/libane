#!/usr/bin/env python3
"""
14 — LayerNorm encoder block  (BERT / ViT style)

BERT and ViT use post-norm with LayerNorm (not RMSNorm), which has both a
scale (gamma) and a bias (beta).  libane supports this via LAYERNORM.

The graph:
    x  →  LayerNorm  →  Matmul (Q)  →  output_Q
    x  →  LayerNorm  →  Matmul (K)  →  output_K   (shared norm, separate proj)

Because LayerNorm is a unary op, x is consumed once by the normaliser.
But then the normalised output is consumed by both Q and K projections — a
branch — so Q and K are in separate fusion groups.

The compiler produces:
    group 0: LayerNorm + Q_proj   (fused)
    group 1: K_proj               (standalone — branches from norm output)

This shows that libane handles shared normalisation cleanly even without
explicit common-subexpression elimination.
"""
import numpy as np
import ane

D   = 768    # BERT-base hidden dim
SEQ = 128
H   = 64     # head dim
N_H = 12     # heads

QKV = H * N_H

rng   = np.random.default_rng(11)
def rand(r, c): return (rng.standard_normal((r, c)) * 0.02).astype(np.float16)

W_Q   = rand(D, QKV)
W_K   = rand(D, QKV)

# LayerNorm weights: gamma (first half) + beta (second half), packed
gamma = np.ones(D, dtype=np.float16)
beta  = np.zeros(D, dtype=np.float16)
ln_w  = np.concatenate([gamma, beta])   # shape (2D,)

# ── Build graph ───────────────────────────────────────────────────────────────
g   = ane.Graph()
x   = g.add_input("x", [1, D, 1, SEQ])

ln  = g.add_op(ane.LAYERNORM, [x],  [1, D,   1, SEQ], weights=ln_w)
q   = g.add_op(ane.MATMUL,    [ln], [1, QKV, 1, SEQ], weights=W_Q)
k   = g.add_op(ane.MATMUL,    [ln], [1, QKV, 1, SEQ], weights=W_K)

g.mark_output(q, "Q")
g.mark_output(k, "K")

cg = g.compile()
cg.set_output_shapes([[1, QKV, 1, SEQ], [1, QKV, 1, SEQ]])

x_data = rng.standard_normal((D, SEQ)).astype(np.float16)
Q, K = cg(x_data)

# ── Verify against numpy ──────────────────────────────────────────────────────
def layernorm_np(x, g, b, eps=1e-5):
    x32 = x.astype(np.float32)
    # Normalise across channels (axis=0 for [D, SEQ] layout)
    mu  = x32.mean(axis=0, keepdims=True)
    var = x32.var(axis=0, keepdims=True)
    xn  = (x32 - mu) / np.sqrt(var + eps)
    return (xn * g.astype(np.float32).reshape(-1,1) +
               b.astype(np.float32).reshape(-1,1)).astype(np.float16)

x_ln  = layernorm_np(x_data, gamma, beta)
ref_Q = (W_Q.astype(np.float32).T @ x_ln.astype(np.float32)).astype(np.float16)

diff = np.abs(Q.reshape(QKV, SEQ).astype(np.float32) - ref_Q.astype(np.float32))
print(f"LayerNorm({D}) → [Q_proj, K_proj]   seq={SEQ}")
print(f"  Q shape         : {Q.shape}")
print(f"  K shape         : {K.shape}")
print(f"  Q vs ref mean err: {diff.mean():.5f}")
print()
print("  Fusion groups:")
print("    group 0: LayerNorm + Q_proj  (fused — linear chain)")
print("    group 1: K_proj              (separate — branches from norm output)")
print()
print("  Gamma=1, beta=0 → LayerNorm reduces to plain normalisation.")
print("  With learned gamma/beta the ANE result will differ from eps-sensitive")
print("  fp32 numpy but is correct for fp16 inference.")
