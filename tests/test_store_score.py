import math

import pytest
import torch
import torch.nn.functional as F

from turboquant_npu.score import compute_hybrid_attention
from turboquant_npu.store import CompressedKVStore


def _ref_attention(query, k, v, scale):
    """query: (T, QH, D); k/v: (N, H_kv, D) -> (T, QH, D)."""
    q = query.float().transpose(0, 1)  # (QH, T, D)
    kk = k.float().transpose(0, 1)  # (H_kv, N, D)
    vv = v.float().transpose(0, 1)
    qh, kvh = q.shape[0], kk.shape[0]
    if qh != kvh:
        g = qh // kvh
        kk = kk.unsqueeze(1).expand(kvh, g, *kk.shape[1:]).reshape(qh, *kk.shape[1:])
        vv = vv.unsqueeze(1).expand(kvh, g, *vv.shape[1:]).reshape(qh, *vv.shape[1:])
    scores = torch.matmul(q, kk.transpose(-2, -1)) * scale
    out = torch.matmul(F.softmax(scores, dim=-1), vv)
    return out.transpose(0, 1)


def test_store_chunking(device):
    d, h = 128, 4
    store = CompressedKVStore(head_dim=d, num_kv_heads=h, key_bits=3,
                              value_bits=2, device=device)

    for i in range(4):
        k = torch.randn(256, h, d, device=device)
        v = torch.randn(256, h, d, device=device)
        store.append_chunk(k, v)

    assert store.num_chunks == 4
    assert store.num_tokens == 1024

    flat = store.get_flat_cache()
    assert flat.num_tokens == 1024
    assert flat.prod_q.norms.shape == (h, 1024)

    # flat cache is cached
    assert store.get_flat_cache() is flat

    # write invalidates
    store.append_chunk(torch.randn(16, h, d, device=device),
                       torch.randn(16, h, d, device=device))
    assert store.get_flat_cache().num_tokens == 1040


def test_hybrid_attention_gqa(device):
    """Compressed history + recent buffer vs fp32 reference with GQA."""
    d, kvh, qh = 64, 2, 8
    gqa = qh // kvh
    hist_n, recent_n = 1024, 64

    store = CompressedKVStore(head_dim=d, num_kv_heads=kvh, key_bits=3,
                              value_bits=2, device=device)
    gcpu = torch.Generator(device="cpu").manual_seed(21)
    k_hist = torch.randn(hist_n, kvh, d, generator=gcpu).to(device)
    v_hist = torch.randn(hist_n, kvh, d, generator=gcpu).to(device)
    store.append_chunk(k_hist, v_hist)

    k_rec = torch.randn(recent_n, kvh, d, generator=gcpu).to(device)
    v_rec = torch.randn(recent_n, kvh, d, generator=gcpu).to(device)

    q = torch.randn(1, qh, d, generator=gcpu).to(device)  # decode step

    out = compute_hybrid_attention(q, store, k_rec, v_rec, num_query_heads=qh)
    assert out.shape == (1, qh, d)

    k_all = torch.cat([k_hist, k_rec], dim=0)
    v_all = torch.cat([v_hist, v_rec], dim=0)
    ref = _ref_attention(q, k_all, v_all, 1.0 / math.sqrt(d))

    cos = F.cosine_similarity(out.float().reshape(-1, d), ref.reshape(-1, d), dim=-1)
    # worst-case synthetic data (random isotropic KV): 0.83-0.88 across seeds;
    # real LLM KV (concentrated logits) measures ~0.94 (reference repo)
    assert cos.mean().item() > 0.83, f"gqa hybrid cos {cos.mean().item()}"


def test_hybrid_attention_exact_only(device):
    """No history -> pure exact attention path."""
    d, kvh, qh = 64, 2, 4
    store = CompressedKVStore(head_dim=d, num_kv_heads=kvh, device=device)
    assert store.get_flat_cache() is None

    k_rec = torch.randn(50, kvh, d, device=device)
    v_rec = torch.randn(50, kvh, d, device=device)
    q = torch.randn(3, qh, d, device=device)

    out = compute_hybrid_attention(q, store, k_rec, v_rec, num_query_heads=qh)
    ref = _ref_attention(q, k_rec, v_rec, 1.0 / math.sqrt(d))
    cos = F.cosine_similarity(out.float().reshape(-1, d), ref.reshape(-1, d), dim=-1)
    assert cos.mean().item() > 0.999
