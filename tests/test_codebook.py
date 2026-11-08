import json
import math
import os

import numpy as np
import pytest
import torch

from turboquant_npu.codebook import (
    compute_lloyd_max_codebook,
    get_codebook,
    get_codebook_tensors,
    beta_pdf,
)
from turboquant_npu.rotation import generate_rotation_matrix

CODEBOOK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "turboquant_npu", "codebooks"
)


def test_pdf_normalizes():
    """Beta pdf on [-1,1] integrates to 1."""
    from scipy import integrate

    for d in (8, 64, 128):
        val, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], -1, 1)
        assert abs(val - 1.0) < 1e-8


def test_shipped_codebooks_present():
    for d in (64, 128, 256):
        for bits in (1, 2, 3, 4):
            path = os.path.join(CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
            assert os.path.exists(path), path
            cb = json.load(open(path))
            assert len(cb["centroids"]) == 2**bits
            assert len(cb["boundaries"]) == 2**bits + 1
            # centroids sorted, within [-1, 1]
            cents = np.array(cb["centroids"])
            assert np.all(np.diff(cents) > 0)
            assert np.all(np.abs(cents) <= 1.0)


@pytest.mark.parametrize("d,bits", [(64, 2), (128, 2), (128, 3)])
def test_codebook_optimality(d, bits):
    """Lloyd-Max fixed point: centroids = conditional means of their cells."""
    cb = get_codebook(d, bits)
    cents = np.array(cb["centroids"])
    bounds = np.array(cb["boundaries"])

    from scipy import integrate

    for i in range(len(cents)):
        num, _ = integrate.quad(
            lambda x: x * beta_pdf(np.array([x]), d)[0], bounds[i], bounds[i + 1]
        )
        den, _ = integrate.quad(
            lambda x: beta_pdf(np.array([x]), d)[0], bounds[i], bounds[i + 1]
        )
        cond_mean = num / den
        assert abs(cond_mean - cents[i]) < 1e-4, f"centroid {i} not optimal"


def test_mse_scaling_with_bits():
    """Paper Thm 3: MSE falls roughly 4x per extra bit (1/4^b scaling)."""
    d = 128
    mses = [get_codebook(d, b)["mse_per_coord"] for b in (1, 2, 3, 4)]
    for i in range(3):
        ratio = mses[i] / mses[i + 1]
        # optimum for Gaussian-like sources is ~4x (uniform: exactly 4x)
        assert 3.0 < ratio < 6.0, f"mse ratio b{i+1}->b{i+2} = {ratio}"


def test_rotation_orthogonal():
    for d in (64, 128):
        Pi = generate_rotation_matrix(d, torch.device("cpu"), seed=42)
        err = (Pi.T @ Pi - torch.eye(d)).abs().max().item()
        assert err < 1e-5
        # Q is orthogonal; det is +/-1 (sign of det is irrelevant for the algorithm)
        assert abs(abs(torch.det(Pi).item()) - 1.0) < 1e-4


def test_rotation_reproducible():
    a = generate_rotation_matrix(64, torch.device("cpu"), seed=7)
    b = generate_rotation_matrix(64, torch.device("cpu"), seed=7)
    assert torch.equal(a, b)


def test_get_codebook_tensors(device):
    c, b = get_codebook_tensors(64, 2, device)
    assert c.device == device and b.device == device
    assert c.shape == (4,) and b.shape == (5,)
