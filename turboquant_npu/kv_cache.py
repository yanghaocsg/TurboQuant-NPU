"""TurboQuant KV cache for NPU — drop-in compressed KV cache.

Keys:   TurboQuantProd (unbiased inner-product estimator)
Values: group quantization (asymmetric, per-group min/max, bit-packed)
Recent tokens: full-precision rolling buffer for quality.

Derived from 0xSero/turboquant (GPLv3), adapted for Ascend NPU.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch

from turboquant_npu.bitpack import packing_params
from turboquant_npu.quantizer import ProdQuantized, TurboQuantProd


class ValueQuantized(NamedTuple):
    """Quantized value cache (bit-packed)."""

    data: torch.Tensor  # (..., n_tokens, packed_d)
    scales: torch.Tensor  # (..., n_tokens, n_groups)
    zeros: torch.Tensor  # (..., n_tokens, n_groups)
    bits: int = 2


def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element (last dim = d)."""
    bits = vq.bits
    packed = vq.data
    bits_eff, vals_per_byte = packing_params(bits)
    mask = (1 << bits_eff) - 1
    parts = [(packed >> (i * bits_eff)) & mask for i in range(vals_per_byte)]
    unpacked = torch.stack(parts, dim=-1)
    return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * vals_per_byte)


def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """Asymmetric group quantization for value vectors.

    v: (..., seq_len, d)
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)

    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = ((v_max - v_min) / n_levels).clamp(min=1e-10)
    zero = v_min

    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack: 2-bit -> 4 values/byte, 4-bit -> 2 values/byte
    bits_eff, vals_per_byte = packing_params(bits)
    assert d % vals_per_byte == 0
    v_n = v_q_flat.reshape(*orig_shape[:-1], d // vals_per_byte, vals_per_byte)
    parts = [v_n[..., i] << (i * bits_eff) for i in range(vals_per_byte)]
    packed = torch.stack(parts, dim=-1).sum(dim=-1, dtype=torch.uint8)

    return ValueQuantized(
        data=packed,
        scales=scale.squeeze(-1).float(),
        zeros=zero.squeeze(-1).float(),
        bits=bits,
    )


def dequantize_values(vq: ValueQuantized, group_size: int = 32) -> torch.Tensor:
    """Dequantize bit-packed values back to float (..., seq_len, d)."""
    data = unpack_values(vq).float()
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)


class TurboQuantKVCache:
    """Compressed KV cache: TurboQuant keys + group-quantized values + fp buffer."""

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device=None,
        dtype: torch.dtype = torch.float16,
        layer_idx: int = 0,
    ):
        from turboquant_npu._device import resolve_device

        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.device = resolve_device(device)
        self.dtype = dtype
        self.layer_idx = layer_idx

        self.key_quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        self.seq_len: int = 0
        self.key_quantized: Optional[ProdQuantized] = None
        self.value_quantized: Optional[ValueQuantized] = None

        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None

    def prefill(self, keys: torch.Tensor, values: torch.Tensor):
        """keys/values: (batch, n_heads, seq_len, head_dim)."""
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            self.key_buffer = keys
            self.value_buffer = values
            return

        n_quant = seq_len - self.buffer_size
        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]

        self.key_buffer = keys[..., n_quant:, :].contiguous()
        self.value_buffer = values[..., n_quant:, :].contiguous()

        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)
        self.value_quantized = quantize_values(
            values_to_quant, bits=self.value_bits, group_size=self.value_group_size
        )

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """Append one decode token: key/value (batch, n_heads, 1, head_dim)."""
        self.seq_len += 1

        if self.key_buffer is not None:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Move oldest tokens from the buffer into compressed storage."""
        n_flush = self.key_buffer.shape[-2] - self.buffer_size

        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        new_key_q = self.key_quantizer.quantize(keys_flush)
        new_val_q = quantize_values(
            values_flush, bits=self.value_bits, group_size=self.value_group_size
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
        else:
            self.key_quantized = ProdQuantized(
                mse_indices=torch.cat(
                    [self.key_quantized.mse_indices, new_key_q.mse_indices], dim=-2
                ),
                qjl_signs=torch.cat(
                    [self.key_quantized.qjl_signs, new_key_q.qjl_signs], dim=-2
                ),
                residual_norms=torch.cat(
                    [self.key_quantized.residual_norms, new_key_q.residual_norms], dim=-1
                ),
                norms=torch.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
                mse_bits=new_key_q.mse_bits,
            )
            self.value_quantized = ValueQuantized(
                data=torch.cat([self.value_quantized.data, new_val_q.data], dim=-2),
                scales=torch.cat([self.value_quantized.scales, new_val_q.scales], dim=-2),
                zeros=torch.cat([self.value_quantized.zeros, new_val_q.zeros], dim=-2),
                bits=self.value_bits,
            )

    def _num_kv_heads(self) -> Optional[int]:
        """Number of KV heads backing this cache (from whichever state exists)."""
        if self.key_quantized is not None:
            return self.key_quantized.norms.shape[1]
        if self.key_buffer is not None:
            return self.key_buffer.shape[1]
        if self.value_quantized is not None:
            return self.value_quantized.scales.shape[1]
        if self.value_buffer is not None:
            return self.value_buffer.shape[1]
        return None

    def _gqa_view(self, x: torch.Tensor):
        """View a (B, H_q, ...) tensor as (B, H_kv, G, ...); returns (view, G)."""
        Hq = x.shape[1]
        Hkv = self._num_kv_heads()
        if Hkv is None or Hq == Hkv:
            return x, 1
        if Hq % Hkv != 0:
            raise ValueError(f"H_q={Hq} not divisible by H_kv={Hkv}")
        B = x.shape[0]
        rest = list(x.shape[2:])
        return x.reshape(B, Hkv, Hq // Hkv, *rest), Hq // Hkv

    def attention_scores(
        self, query: torch.Tensor, scale: float = None, backend: str = "torch"
    ) -> torch.Tensor:
        """score[i, j] = <query_i, key_j> / sqrt(d).

        query: (batch, H_q, n_q, head_dim) -> scores (batch, H_q, n_q, seq_len)
        Handles GQA (H_q multiple of H_kv).
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []

        if self.key_quantized is not None:
            scores_quant = self.key_quantizer.attention_score(
                query, self.key_quantized, backend=backend
            )
            scores_parts.append(scores_quant * scale)

        if self.key_buffer is not None:
            q_g, G = self._gqa_view(query)
            kb = self.key_buffer.to(query.dtype)
            if G > 1:
                kb = kb.unsqueeze(2)
            scores_buf = torch.matmul(q_g, kb.transpose(-2, -1))
            if G > 1:
                scores_buf = scores_buf.reshape(*query.shape[:-1], -1)
            scores_parts.append(scores_buf * scale)

        return torch.cat(scores_parts, dim=-1)

    def attend(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """out = softmax(scores) @ values. attn_weights: (b, H_q, n_q, seq_len)."""
        B, Hq, n, _ = attn_weights.shape
        Hkv = self._num_kv_heads()

        G = 1
        w = attn_weights
        if Hkv is not None and Hq != Hkv:
            if Hq % Hkv != 0:
                raise ValueError(f"H_q={Hq} not divisible by H_kv={Hkv}")
            G = Hq // Hkv
            w = attn_weights.view(B, Hkv, G, n, -1)

        output_parts = []
        col_offset = 0

        if self.value_quantized is not None:
            n_quant = self.value_quantized.data.shape[-2]
            w_quant = w[..., col_offset : col_offset + n_quant]
            v_dequant = dequantize_values(self.value_quantized, self.value_group_size)
            v = v_dequant.to(w_quant.dtype)
            if G > 1:
                v = v.unsqueeze(2)
            out_q = torch.matmul(w_quant, v)
            if G > 1:
                out_q = out_q.reshape(B, Hq, n, -1)
            output_parts.append(out_q)
            col_offset += n_quant

        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = w[..., col_offset : col_offset + n_buf]
            vb = self.value_buffer.to(w_buf.dtype)
            if G > 1:
                vb = vb.unsqueeze(2)
            out_b = torch.matmul(w_buf, vb)
            if G > 1:
                out_b = out_b.reshape(B, Hq, n, -1)
            output_parts.append(out_b)

        return sum(output_parts)

    def memory_bytes(self) -> dict:
        """Estimated memory usage in bytes."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += self.key_quantized.mse_indices.nelement()
            info["quantized_keys"] += self.key_quantized.qjl_signs.nelement()
            info["quantized_keys"] += self.key_quantized.residual_norms.nelement() * 2
            info["quantized_keys"] += self.key_quantized.norms.nelement() * 2

        if self.value_quantized is not None:
            info["quantized_values"] += self.value_quantized.data.nelement()
            info["quantized_values"] += self.value_quantized.scales.nelement() * 2
            info["quantized_values"] += self.value_quantized.zeros.nelement() * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.nelement() * 2
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.nelement() * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
