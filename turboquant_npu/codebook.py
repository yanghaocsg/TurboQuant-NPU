"""Lloyd-Max codebook computation for TurboQuant-NPU.

After random rotation, each coordinate of a unit-norm d-dimensional vector
follows a scaled Beta distribution on [-1, 1]:

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

We solve continuous 1D k-means (Lloyd-Max) to find optimal centroids.
Computation is pure CPU (scipy numeric integration) and cached to JSON,
independent of the accelerator backend.

Derived from 0xSero/turboquant (GPLv3).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy import integrate, special


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of one coordinate of a uniform random point on S^{d-1}."""
    if d <= 2:
        raise ValueError(f"Dimension d={d} too small, need d >= 3")
    log_const = (
        special.gammaln(d / 2.0)
        - 0.5 * np.log(np.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return np.exp(log_const + exponent * np.log(1 - x**2))


def _conditional_mean(lo: float, hi: float, d: int) -> float:
    """E[X | lo < X < hi] under the Beta PDF on [-1, 1]."""
    num, _ = integrate.quad(lambda x: x * beta_pdf(np.array([x]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], lo, hi)
    if den < 1e-30:
        return (lo + hi) / 2.0
    return num / den


def _mse_cost(centroids: np.ndarray, d: int) -> float:
    """MSE cost for a set of sorted centroids under the Beta PDF."""
    n = len(centroids)
    boundaries = np.zeros(n + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    cost = 0.0
    for i in range(n):
        lo, hi = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(
            lambda x: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0], lo, hi
        )
        cost += val
    return cost


def compute_lloyd_max_codebook(
    d: int, bits: int, max_iter: int = 200, tol: float = 1e-12
) -> dict:
    """Optimal Lloyd-Max codebook for the Beta distribution on [-1, 1].

    Returns dict with 'centroids', 'boundaries', 'mse_per_coord',
    'mse_total', 'd', 'bits'.
    """
    n_clusters = 2**bits

    # Initialize centroids at quantile midpoints of the discretized CDF.
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]

    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_mid = (quantile_edges[i] + quantile_edges[i + 1]) / 2.0
        idx = min(int(np.searchsorted(cdf_vals, q_mid)), len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    prev_cost = float("inf")
    cost = float("inf")
    for _ in range(max_iter):
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        new_centroids = np.array(
            [
                _conditional_mean(boundaries[i], boundaries[i + 1], d)
                for i in range(n_clusters)
            ]
        )

        cost = _mse_cost(new_centroids, d)
        centroids = new_centroids
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    boundaries = np.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(cost),
        "mse_total": float(cost * d),
        "d": d,
        "bits": bits,
    }


# ── Codebook cache (memory + disk) ────────────────────────────────────
_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")


def get_codebook(d: int, bits: int) -> dict:
    """Get or compute a codebook with on-disk caching."""
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    os.makedirs(_CODEBOOK_DIR, exist_ok=True)
    path = os.path.join(_CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
        _CODEBOOK_CACHE[key] = cb
        return cb

    print(f"[TurboQuant-NPU] Computing Lloyd-Max codebook d={d} bits={bits} ...")
    cb = compute_lloyd_max_codebook(d, bits)
    with open(path, "w") as f:
        json.dump(cb, f, indent=2)
    print(
        f"[TurboQuant-NPU] MSE/coord = {cb['mse_per_coord']:.6e}, "
        f"total = {cb['mse_total']:.6f}"
    )
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_tensors(
    d: int,
    bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Codebook as device tensors ready for quantization."""
    cb = get_codebook(d, bits)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    return centroids, boundaries
