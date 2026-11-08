"""TurboQuant quantizers — Algorithm 1 (MSE) and Algorithm 2 (inner product).

Pure-torch implementation that runs on npu (torch_npu) / cuda / cpu.
Operates on tensors of shape (..., d); d is typically head_dim (64/128/256).

Derived from 0xSero/turboquant (GPLv3), adapted for Ascend NPU.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from turboquant_npu._device import resolve_device
from turboquant_npu.bitpack import pack_indices, pack_signs, unpack_indices, unpack_signs
from turboquant_npu.codebook import get_codebook_tensors
from turboquant_npu.rotation import (
    generate_qjl_matrix,
    generate_rotation_matrix,
    rotate_backward,
    rotate_forward,
)


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""

    indices: torch.Tensor  # (..., packed_len) uint8 bit-packed
    norms: torch.Tensor  # (...,) original L2 norms
    bits: int


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""

    mse_indices: torch.Tensor  # (..., packed_len) uint8 bit-packed MSE indices
    qjl_signs: torch.Tensor  # (..., packed_len) uint8 packed sign bits
    residual_norms: torch.Tensor  # (...,) L2 norms of residuals
    norms: torch.Tensor  # (...,) original L2 norms
    mse_bits: int


class TurboQuantMSE(nn.Module):
    """TurboQuant optimized for MSE (Algorithm 1).

    Quantize:   y = Pi @ (x/||x||), nearest centroid per coordinate
    Dequantize: centroid lookup -> rotate back -> rescale by ||x||
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device=None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits
        self.device = resolve_device(device)

        self.register_buffer(
            "Pi", generate_rotation_matrix(dim, self.device, dtype, seed=seed)
        )

        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype)
        self.register_buffer("centroids", centroids)  # (2^b,)
        self.register_buffer("boundaries", boundaries)  # (2^b + 1,)
        # interior decision boundaries for searchsorted
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        norms = x.norm(dim=-1, keepdim=False)
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)

        y = rotate_forward(x_unit.float(), self.Pi)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = pack_indices(indices, self.bits)

        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = rotate_backward(y_hat, self.Pi)
        x_hat = x_hat * q.norms.unsqueeze(-1)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))


class TurboQuantProd(nn.Module):
    """TurboQuant optimized for inner products (Algorithm 2).

    Stage 1: TurboQuantMSE at (b-1) bits -> x_mse
    Stage 2: QJL on residual r = x - x_mse: sign(S r), 1 bit per coordinate

    Unbiased estimator:
        <y, x> ~= <y, x_mse> + sqrt(pi/2)/d * ||r|| * <S^T sign(S r), y>
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device=None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.device = resolve_device(device)

        assert bits >= 2, "Inner-product TurboQuant needs >= 2 bits (1 MSE + 1 QJL)"

        self.mse_quantizer = TurboQuantMSE(
            dim=dim, bits=bits - 1, device=self.device, dtype=dtype, seed=seed
        )

        self.register_buffer(
            "S", generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000)
        )
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)

        residual = x - x_hat
        residual_norms = residual.norm(dim=-1)

        projected = torch.matmul(residual.float(), self.S.T)
        packed_signs = pack_signs(projected > 0)

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_q = MSEQuantized(
            indices=q.mse_indices, norms=q.norms, bits=q.mse_bits
        )
        x_mse = self.mse_quantizer.dequantize(mse_q)

        signs = unpack_signs(q.qjl_signs, self.dim)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))

        return x_mse + x_qjl

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: ProdQuantized,
        backend: str = "torch",
    ) -> torch.Tensor:
        """Attention logits <query, key_j> estimated from compressed keys.

        query: (..., n_q, d); quantized_key: (..., n_k, ...)
        returns: (..., n_q, n_k)
        """
        if backend == "triton":
            return self._attention_score_triton(query, quantized_key)
        return self._attention_score_torch(query, quantized_key)

    def _attention_score_torch(
        self, query: torch.Tensor, quantized_key: ProdQuantized
    ) -> torch.Tensor:
        """Query (B, H_q, n, d) vs keys (B, H_kv, N, ...).

        GQA: when H_q != H_kv (both present), query head h attends kv head
        h // G with G = H_q / H_kv. Lower-dim queries (..., n, d) keep the
        legacy aligned-batch broadcasting.
        """
        qf = query.float()
        G = 1
        B = Hq = n = None
        if qf.dim() == 4 and quantized_key.norms.dim() >= 3:
            B, Hq, n, d = qf.shape
            Hkv = quantized_key.norms.shape[1]
            if Hq != Hkv:
                if Hq % Hkv != 0:
                    raise ValueError(f"H_q={Hq} not divisible by H_kv={Hkv}")
                G = Hq // Hkv
                qf = qf.reshape(B, Hkv, G, n, d)

        # MSE contribution
        mse_q = MSEQuantized(
            indices=quantized_key.mse_indices,
            norms=quantized_key.norms,
            bits=quantized_key.mse_bits,
        )
        k_mse = self.mse_quantizer.dequantize(mse_q).float()  # (..., N, d)
        if G > 1:
            k_mse = k_mse.unsqueeze(2)
        scores_mse = torch.matmul(qf, k_mse.transpose(-2, -1))

        # QJL contribution (asymmetric: sketch the query)
        q_sketched = torch.matmul(qf, self.S.T)
        signs = unpack_signs(quantized_key.qjl_signs, self.dim)  # (..., N, d)
        if G > 1:
            signs = signs.unsqueeze(2)
        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        rn = quantized_key.residual_norms.float()
        if G > 1:
            rn = rn.unsqueeze(2)
        scores_qjl = scores_qjl * (self.qjl_scale * rn.unsqueeze(-2))

        scores = scores_mse + scores_qjl.to(scores_mse.dtype)
        if G > 1:
            scores = scores.reshape(B, Hq, n, -1)
        return scores

    def _attention_score_triton(
        self, query: torch.Tensor, quantized_key: ProdQuantized
    ) -> torch.Tensor:
        from turboquant_npu.backend import triton_attention_score

        return triton_attention_score(self, query, quantized_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
