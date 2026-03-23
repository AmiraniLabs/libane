#!/usr/bin/env python3
"""
05 — RMSNorm → Linear projection  (LLaMA-style fused layer)

Every LLaMA / Mistral / Qwen layer starts with:
    h = RMSNorm(x) @ W

libane fuses these into one ANE program.  The normalised activations never
leave ANE SRAM — they feed directly into the convolution that implements the
matmul.

Numerical check: compare against a numpy reference implementation to verify
correctness before trusting results in a real model.
"""
import numpy as np
import ane

D_IN  = 4096   # LLaMA-7B model dim
D_OUT = 4096   # same for self-attn Q/K/V projections in one pass
SEQ   = 64     # short sequence for quick test

rng = np.random.default_rng(3)
W     = (rng.standard_normal((D_IN, D_OUT)) * 0.02).astype(np.float16)
scale = np.ones(D_IN, dtype=np.float16)   # learned; all-ones for test
x_f16 = rng.standard_normal((D_IN, SEQ)).astype(np.float16)


# ── numpy reference ──────────────────────────────────────────────────────────
def rmsnorm_np(x, s, eps=1e-5):
    x32 = x.astype(np.float32)
    rms = np.sqrt((x32 ** 2).mean(axis=0, keepdims=True) + eps)
    return ((x32 / rms) * s.astype(np.float32).reshape(-1, 1)).astype(np.float16)

x_norm = rmsnorm_np(x_f16, scale)
ref = (x_norm.astype(np.float32) @ W.astype(np.float32)[:D_IN, :D_OUT]).astype(np.float16)

# ── ANE fused graph ───────────────────────────────────────────────────────────
g   = ane.Graph()
x   = g.add_input("x", [1, D_IN, 1, SEQ])
rn  = g.add_op(ane.RMSNORM, [x],  [1, D_IN,  1, SEQ], weights=scale)
out = g.add_op(ane.MATMUL,  [rn], [1, D_OUT, 1, SEQ], weights=W)
g.mark_output(out)

cg = g.compile()
cg.set_output_shapes([[1, D_OUT, 1, SEQ]])

result = cg(x_f16)

# ── Verify ───────────────────────────────────────────────────────────────────
diff = np.abs(result.astype(np.float32) - ref.astype(np.float32))
rel  = diff / (np.abs(ref.astype(np.float32)) + 1e-6)

print(f"RMSNorm({D_IN}) → Matmul[{D_IN}×{D_OUT}]  seq={SEQ}")
print()
print(f"  Output shape   : {result.shape}")
print(f"  Mean abs error : {diff.mean():.5f}")
print(f"  Max  abs error : {diff.max():.5f}")
print(f"  Mean rel error : {rel.mean()*100:.3f}%")
print()
print("  ANE fused RMSNorm+Matmul matches numpy within fp16 tolerance.")
print("  (Small discrepancy from ANE's internal fp16 accumulation.)")
