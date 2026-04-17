import ane
import numpy as np


def test_matmul_f32_non_contiguous_inputs_match_numpy() -> None:
    # Force CPU backend so this test stays deterministic and independent of ANE kernels.
    ane.set_backend("cpu")
    try:
        rng = np.random.default_rng(0)

        # Non-contiguous view: transpose changes strides without copying.
        a_base = rng.standard_normal((127, 63)).astype(np.float32)
        a = a_base.T  # shape (63, 127), C-contiguous=False
        b = rng.standard_normal((127, 31)).astype(np.float32)

        assert not a.flags["C_CONTIGUOUS"]

        ref = a @ b
        out = ane.matmul_f32(a, b)

        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
    finally:
        ane.set_backend(None)
