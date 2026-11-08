"""Attention over compressed + exact KV segments (torch path).

Read path:
  - compressed history via TurboQuant dequantize (or fused triton kernels, Phase 2)
  - exact recent buffer via standard matmul attention
  - merged by concatenation

Derived from 0xSero/turboquant (GPLv3).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from turboquant_npu.kv_cache import dequantize_values
from turboquant_npu.store import CompressedKVStore, FlatCache

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: torch.Tensor | None,
    recent_v: torch.Tensor | None,
    num_query_heads: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention output combining compressed history and exact recent buffer.

    query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 decode
    recent_k/v: (recent_len, num_kv_heads, head_dim) or None
    """
    head_dim = store.head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0],
            num_query_heads,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // store.num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(query, flat, store, gqa_ratio, scale)

    if not has_history and has_recent:
        return _attend_exact_only(query, recent_k, recent_v, gqa_ratio, scale)

    return _attend_hybrid(
        query, flat, store, recent_k, recent_v, gqa_ratio, scale
    )


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    store: CompressedKVStore,
    gqa_ratio: int,
    scale: float,
) -> torch.Tensor:
    k_dequant = store.quantizer.dequantize(flat.prod_q)  # (H_kv, N, D)
    v_dequant = dequantize_values(flat.value_q, store.value_group_size)
    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, scale)


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    scale: float,
) -> torch.Tensor:
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1), gqa_ratio, scale
    )


def _attend_hybrid(
    query: torch.Tensor,
    flat: FlatCache,
    store: CompressedKVStore,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    scale: float,
) -> torch.Tensor:
    k_hist = store.quantizer.dequantize(flat.prod_q)
    v_hist = dequantize_values(flat.value_q, store.value_group_size)

    k_recent = recent_k.transpose(0, 1)  # (H_kv, N_recent, D)
    v_recent = recent_v.transpose(0, 1)

    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _matmul_attend(query, k_all, v_all, gqa_ratio, scale)


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    scale: float,
) -> torch.Tensor:
    """Matmul attention with GQA broadcast (no repeat_interleave on KV).

    query: (T, Q_heads, D); kv_keys/values: (H_kv, N, D) -> (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = kv_keys.shape[0]

    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)  # (H, G, T, D)
    k = kv_keys.float().unsqueeze(1)  # (H_kv, 1, N, D)
    v = kv_values.float().unsqueeze(1)

    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
