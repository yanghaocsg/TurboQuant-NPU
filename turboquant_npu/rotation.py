"""Random rotation utilities for TurboQuant-NPU.

Pi = QR decomposition of a fixed-seed Gaussian matrix (paper Algorithm 1),
S = Gaussian sketching matrix for QJL. Both are generated on CPU for
reproducibility and moved to the target device (npu/cuda/cpu).

Derived from 0xSero/turboquant (GPLv3).
"""

from __future__ import annotations

import torch


def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """Random orthogonal matrix Pi in R^{d x d} via QR (det fixed to +1)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """Gaussian projection matrix S in R^{d x d} for QJL (iid N(0,1))."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """y = x @ Pi^T"""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """x = y @ Pi"""
    return torch.matmul(y, Pi)
