"""Bit-packing utilities for TurboQuant-NPU (torch ops, NPU-compatible).

Packing scheme (same as reference):
  bits=1 -> 8 values per byte
  bits=2 -> 4 values per byte
  bits=3 -> stored as 4-bit (2 per byte, rounded up)
  bits=4 -> 2 values per byte

QJL sign bits are always packed 8 per byte.

NPU note: aclnn bit-shift ops on CANN 26.x do NOT broadcast a tensor RHS
(scalar shift works fine), and arange has no uint8 output. All shifts here
are Python-int scalars in small loops (<= 8 iterations), which run on
npu/cuda/cpu alike. Verified on Ascend 910 / CANN 26.1.1.

Derived from 0xSero/turboquant (GPLv3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def packing_params(bits: int) -> tuple[int, int]:
    """(effective_bits, vals_per_byte) matching the packing scheme."""
    if bits == 1:
        return 1, 8
    if bits == 2:
        return 2, 4
    if bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    return 8, 1


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices (0..2^bits-1) into uint8 bytes."""
    bits_eff, vals_per_byte = packing_params(bits)

    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    u = indices.to(torch.uint8)
    if padded_d > d:
        u = F.pad(u, (0, padded_d - d), value=0)

    reshaped = u.reshape(*batch_shape, -1, vals_per_byte)
    parts = [reshaped[..., i] << (i * bits_eff) for i in range(vals_per_byte)]
    return torch.stack(parts, dim=-1).sum(dim=-1, dtype=torch.uint8)


def unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bit-packed bytes to int64 indices of last-dim size d."""
    bits_eff, vals_per_byte = packing_params(bits)
    batch_shape = packed.shape[:-1]

    mask = (1 << bits_eff) - 1
    parts = [(packed >> (i * bits_eff)) & mask for i in range(vals_per_byte)]
    unpacked = torch.stack(parts, dim=-1).reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


def pack_signs(signs_positive: torch.Tensor) -> torch.Tensor:
    """Pack boolean/uint8 sign bits (1 = positive) 8 per byte."""
    signs = signs_positive.to(torch.uint8)
    d = signs.shape[-1]
    if d % 8 != 0:
        signs = F.pad(signs, (0, 8 - d % 8), value=0)
    signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
    parts = [signs_reshaped[..., i] * (1 << i) for i in range(8)]
    return torch.stack(parts, dim=-1).sum(dim=-1, dtype=torch.uint8)


def unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack sign bytes to float in {-1, +1} of last-dim size d."""
    p = packed.long()
    parts = [(p >> i) & 1 for i in range(8)]
    bits = torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], -1)[..., :d]
    return 2.0 * bits.float() - 1.0
