"""
Tests for ane.from_torch() — the Torch-FX tracer.

All tests are skipped gracefully when PyTorch is not installed or when the
ANE is not available.  The Tier-1 tests only exercise graph construction and
validation (no ANE execution); Tier-3 tests require a live ANE.
"""
import importlib
import pytest
import numpy as np
import ane

# ── Skip conditions ────────────────────────────────────────────────────────────

TORCH_MISSING = importlib.util.find_spec("torch") is None
ANE_MISSING   = not ane.available()

skip_no_torch = pytest.mark.skipif(TORCH_MISSING, reason="torch not installed")
skip_no_ane   = pytest.mark.skipif(ANE_MISSING,   reason="ANE not available")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_torch():
    """Import torch; raises ImportError (caught by skip decorator) if missing."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ── Tier 1: graph construction (no ANE) ───────────────────────────────────────

@skip_no_torch
def test_from_torch_imports():
    """ane.from_torch is callable."""
    assert callable(ane.from_torch)


@skip_no_torch
def test_from_torch_linear_no_bias_compiles():
    """A single bias-free Linear traces and compiles without error."""
    torch, nn, F = _import_torch()

    class SingleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 128, bias=False)
        def forward(self, x):
            return self.fc(x)

    m = SingleLinear().half().eval()
    x = torch.zeros(1, 64, 1, 32, dtype=torch.float16)
    cg = ane.from_torch(m, x)
    assert cg is not None


@skip_no_torch
def test_from_torch_linear_with_bias_raises():
    """A Linear with bias raises NotImplementedError."""
    torch, nn, F = _import_torch()

    class BiasLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 128, bias=True)
        def forward(self, x):
            return self.fc(x)

    m = BiasLinear().half().eval()
    x = torch.zeros(1, 64, 1, 32, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="bias"):
        ane.from_torch(m, x)


@skip_no_torch
def test_from_torch_ffn_block_compiles():
    """A two-layer FFN (Linear→GELU→Linear) traces and compiles."""
    torch, nn, F = _import_torch()

    class FFN(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)
        def forward(self, x):
            return self.w2(F.gelu(self.w1(x)))

    m = FFN(64, 256).half().eval()
    x = torch.zeros(1, 64, 1, 32, dtype=torch.float16)
    cg = ane.from_torch(m, x)
    assert cg is not None


@skip_no_torch
def test_from_torch_layernorm_compiles():
    """A module with LayerNorm traces and compiles."""
    torch, nn, F = _import_torch()

    class WithLN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)
        def forward(self, x):
            return self.ln(x)

    # LayerNorm expects channel dim C on axis 1 in libane; mirror that here.
    C, S = 64, 32
    m = WithLN(C).half().eval()
    x = torch.zeros(1, C, 1, S, dtype=torch.float16)
    cg = ane.from_torch(m, x)
    assert cg is not None


@skip_no_torch
def test_from_torch_add_elementwise_compiles():
    """A residual-add module (tensor + tensor) traces and compiles."""
    torch, nn, F = _import_torch()

    class ResidualAdd(nn.Module):
        def forward(self, x, y):
            return x + y

    m = ResidualAdd().eval()
    x = torch.zeros(1, 64, 1, 32, dtype=torch.float16)
    cg = ane.from_torch(m, [x, x])
    assert cg is not None


@skip_no_torch
def test_from_torch_unsupported_op_raises():
    """An unsupported op (conv2d via F.conv2d) raises NotImplementedError."""
    torch, nn, F = _import_torch()

    class WithConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    # Conv2d is not in the supported-op table for from_torch.
    m = WithConv().half().eval()
    x = torch.zeros(1, 4, 8, 32, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        ane.from_torch(m, x)


@skip_no_torch
def test_from_torch_list_input():
    """from_torch accepts a list of example inputs."""
    torch, nn, F = _import_torch()

    class TwoInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(64, 64, bias=False)
        def forward(self, x, y):
            return self.w(x) + y

    m = TwoInput().half().eval()
    x = torch.zeros(1, 64, 1, 32, dtype=torch.float16)
    cg = ane.from_torch(m, [x, x])
    assert cg is not None


# ── Tier 3: end-to-end execution ──────────────────────────────────────────────

@skip_no_torch
@skip_no_ane
def test_from_torch_ffn_numerics():
    """FFN traced via from_torch produces numerically correct output."""
    torch, nn, F = _import_torch()
    torch.manual_seed(42)

    D, H, S = 64, 256, 32

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(D, H, bias=False)
            self.w2 = nn.Linear(H, D, bias=False)
        def forward(self, x):
            return self.w2(F.gelu(self.w1(x)))

    m = FFN().half().eval()
    x_t = torch.randn(1, D, 1, S, dtype=torch.float16)

    cg = ane.from_torch(m, x_t)

    # Reference: pure PyTorch
    with torch.no_grad():
        ref = m(x_t).numpy()  # [1, D, 1, S]

    out = cg(x_t.numpy())     # [1, D, 1, S]

    # fp16 pass through ANE; allow modest tolerance
    np.testing.assert_allclose(out.astype(np.float32),
                                ref.astype(np.float32),
                                rtol=0.02, atol=0.02,
                                err_msg="FFN from_torch output diverges from PyTorch reference")


@skip_no_torch
@skip_no_ane
def test_from_torch_residual_add_numerics():
    """Residual-add traced graph produces correct results."""
    torch, nn, F = _import_torch()

    class ResidualAdd(nn.Module):
        def forward(self, x, y):
            return x + y

    m = ResidualAdd().eval()
    C, S = 64, 32
    x_t = torch.ones(1, C, 1, S, dtype=torch.float16)
    y_t = torch.ones(1, C, 1, S, dtype=torch.float16) * 2

    cg = ane.from_torch(m, [x_t, y_t])
    out = cg([x_t.numpy(), y_t.numpy()])

    # Expected: 1.0 + 2.0 = 3.0 everywhere
    np.testing.assert_allclose(out.astype(np.float32),
                                np.full_like(out, 3.0, dtype=np.float32),
                                rtol=0.01, atol=0.01)
