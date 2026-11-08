"""Alignment tests: triton-ascend kernels vs the torch reference path.

NPU-only (triton-ascend requires an Ascend device).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from turboquant_npu._device import resolve_device

triton_kernels = pytest.importorskip("turboquant_npu.triton_kernels")

from turboquant_npu.kv_cache import dequantize_values, quantize_values
from turboquant_npu.quantizer import TurboQuantProd


def _get_device():
    dev = resolve_device()
    if dev.type != "npu":
        pytest.skip("triton-ascend kernels require an Ascend NPU")
    return dev


def _make_quantized(n, kvh, d, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, kvh, n, d, generator=g).to(device)
    q = TurboQuantProd(dim=d, bits=3, device=device)
    return q, x, q.quantize(x)


@pytest.mark.parametrize("kvh,g", [(4, 1), (4, 2)])
def test_triton_scores_match_torch(kvh, g):
    device = _get_device()
    d, n = 128, 1537  # non-power-of-two N to exercise masking
    q, x, xq = _make_quantized(n, kvh, d, device, seed=1)

    hq = kvh * g
    gq = torch.Generator(device="cpu").manual_seed(2)
    query = torch.randn(1, hq, 1, d, generator=gq).to(device)

    torch_scores = q.attention_score(query, xq, backend="torch")
    triton_scores = triton_kernels.attention_score(q, query, xq)

    assert triton_scores.shape == torch_scores.shape
    assert torch.allclose(triton_scores, torch_scores, atol=2e-2, rtol=2e-2), (
        f"max diff {(triton_scores - torch_scores).abs().max().item()}"
    )


@pytest.mark.parametrize("kvh,g", [(2, 1), (2, 4)])
def test_triton_fused_decode_matches_torch(kvh, g):
    device = _get_device()
    d, n = 64, 1000
    q, x, xq = _make_quantized(n, kvh, d, device, seed=3)

    g2 = torch.Generator(device="cpu").manual_seed(4)
    v = torch.randn(1, kvh, n, d, generator=g2).to(device)
    vq = quantize_values(v, bits=2, group_size=32)

    hq = kvh * g
    query = torch.randn(1, hq, 1, d, generator=g2).to(device)
    sm_scale = 1.0 / math.sqrt(d)

    # torch reference path: scores -> softmax -> attend with dequantized values
    scores = q.attention_score(query, xq, backend="torch") * sm_scale
    weights = F.softmax(scores.float(), dim=-1)

    v_deq = dequantize_values(vq, 32)
    if g > 1:
        B, Hkv, N, D = v_deq.shape
        v_e = v_deq.unsqueeze(2).expand(B, Hkv, g, N, D).reshape(B, hq, N, D)
    else:
        v_e = v_deq
    ref = torch.matmul(weights, v_e.float())  # (B, hq, 1, d)

    out = triton_kernels.fused_decode(q, query, xq, vq, sm_scale)

    assert out.shape == ref.shape
    cos = F.cosine_similarity(out.float().reshape(-1, d), ref.reshape(-1, d), dim=-1)
    assert cos.mean().item() > 0.999, f"fused decode cos {cos.mean().item()}"


def test_triton_scores_noncontiguous_value_bits4():
    """4-bit values through the fused kernel."""
    device = _get_device()
    d, n, kvh = 128, 640, 2
    q, x, xq = _make_quantized(n, kvh, d, device, seed=5)

    g2 = torch.Generator(device="cpu").manual_seed(6)
    v = torch.randn(1, kvh, n, d, generator=g2).to(device)
    vq = quantize_values(v, bits=4, group_size=32)

    query = torch.randn(1, kvh, 1, d, generator=g2).to(device)
    sm_scale = 1.0 / math.sqrt(d)

    scores = q.attention_score(query, xq, backend="torch") * sm_scale
    weights = F.softmax(scores.float(), dim=-1)
    ref = torch.matmul(weights, dequantize_values(vq, 32).float())

    out = triton_kernels.fused_decode(q, query, xq, vq, sm_scale)
    cos = F.cosine_similarity(out.float().reshape(-1, d), ref.reshape(-1, d), dim=-1)
    assert cos.mean().item() > 0.999
