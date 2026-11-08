import math

import pytest
import torch
import torch.nn.functional as F

from turboquant_npu.kv_cache import (
    TurboQuantKVCache,
    dequantize_values,
    quantize_values,
)


def _rand_kv(batch, heads, seq, d, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(batch, heads, seq, d, generator=g).to(device)
    v = torch.randn(batch, heads, seq, d, generator=g).to(device)
    return k, v


@pytest.mark.parametrize("bits", [2, 4])
def test_value_quantize_roundtrip(bits, device):
    d = 128
    v = torch.randn(2, 4, 100, d, device=device) * 3

    vq = quantize_values(v, bits=bits, group_size=32)
    v_back = dequantize_values(vq, group_size=32)

    err = (v_back - v).abs().mean().item() / v.abs().mean().item()
    # 2-bit: heavy error ~20-30%; 4-bit: small ~3-6%
    thr = 0.45 if bits == 2 else 0.10
    assert err < thr, f"bits={bits} rel err {err}"

    # packed storage actually saves memory
    assert vq.data.shape[-1] == d // (4 if bits == 2 else 2)


def test_prefill_and_attention_e2e(device):
    """Full attention through compressed cache vs fp32 baseline."""
    batch, heads, seq, d = 1, 4, 512, 128
    k, v = _rand_kv(batch, heads, seq, d, device, seed=1)
    q = torch.randn(batch, heads, 8, d, device=device)

    cache = TurboQuantKVCache(
        head_dim=d, key_bits=3, value_bits=2, buffer_size=128, device=device
    )
    cache.prefill(k, v)

    scores = cache.attention_scores(q)
    assert scores.shape == (batch, heads, 8, seq)
    weights = F.softmax(scores.float(), dim=-1)
    out = cache.attend(weights)

    # fp32 baseline
    ref_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(d)
    ref_w = F.softmax(ref_scores, dim=-1)
    ref_out = torch.matmul(ref_w, v.float())

    cos = F.cosine_similarity(out.float().reshape(-1, d), ref_out.reshape(-1, d), dim=-1)
    # Random isotropic K/V is the worst case for TurboQuant: true inner
    # products are tiny (~0.07) vs per-estimate QJL noise, so softmax
    # weights deviate. Real LLM KV (concentrated logits) measures ~0.94
    # (see reference repo). Gate on the synthetic-data level.
    assert cos.mean().item() > 0.85, f"attention cos {cos.mean().item()}"


def test_gqa_attention_e2e(device):
    """GQA: 8 query heads over 4 kv heads through the compressed cache."""
    batch, kv_heads, q_heads, seq, d = 1, 4, 8, 512, 128
    k, v = _rand_kv(batch, kv_heads, seq, d, device, seed=12)
    g = torch.Generator(device="cpu").manual_seed(13)
    q = torch.randn(batch, q_heads, 4, d, generator=g).to(device)

    cache = TurboQuantKVCache(
        head_dim=d, key_bits=3, value_bits=2, buffer_size=128, device=device
    )
    cache.prefill(k, v)

    scores = cache.attention_scores(q)
    assert scores.shape == (batch, q_heads, 4, seq)

    weights = F.softmax(scores.float(), dim=-1)
    out = cache.attend(weights)
    assert out.shape == (batch, q_heads, 4, d)

    # fp32 reference with GQA expansion
    g_ratio = q_heads // kv_heads
    kk = k.float().unsqueeze(2).expand(batch, kv_heads, g_ratio, seq, d).reshape(batch, q_heads, seq, d)
    vv = v.float().unsqueeze(2).expand(batch, kv_heads, g_ratio, seq, d).reshape(batch, q_heads, seq, d)
    ref_scores = torch.matmul(q.float(), kk.transpose(-2, -1)) / math.sqrt(d)
    ref_out = torch.matmul(F.softmax(ref_scores, dim=-1), vv)

    cos = F.cosine_similarity(out.float().reshape(-1, d), ref_out.reshape(-1, d), dim=-1)
    assert cos.mean().item() > 0.83, f"gqa cos {cos.mean().item()}"


def test_decode_append(device):
    """Prefill then decode steps; buffer flush moves tokens to compressed store."""
    batch, heads, d = 1, 4, 64
    seq0 = 200
    k, v = _rand_kv(batch, heads, seq0, d, device, seed=2)

    cache = TurboQuantKVCache(
        head_dim=d, key_bits=3, value_bits=2, buffer_size=64, device=device
    )
    cache.prefill(k, v)
    assert cache.key_quantized is not None

    n_quant_before = cache.key_quantized.norms.shape[-1]

    # decode 100 tokens one at a time
    g2 = torch.Generator(device="cpu").manual_seed(22)
    for step in range(100):
        kk = torch.randn(batch, heads, 1, d, generator=g2).to(device)
        vv = torch.randn(batch, heads, 1, d, generator=g2).to(device)
        cache.append(kk, vv)
        # append the same to the reference too
        k = torch.cat([k, kk], dim=2)
        v = torch.cat([v, vv], dim=2)

    assert cache.seq_len == seq0 + 100
    assert cache.key_quantized.norms.shape[-1] > n_quant_before
    assert cache.key_buffer.shape[-2] <= cache.buffer_size

    q = torch.randn(batch, heads, 1, d, generator=g2).to(device)
    scores = cache.attention_scores(q)
    assert scores.shape[-1] == seq0 + 100

    weights = F.softmax(scores.float(), dim=-1)
    out = cache.attend(weights)

    ref_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(d)
    ref_w = F.softmax(ref_scores, dim=-1)
    ref_out = torch.matmul(ref_w, v.float())

    cos = F.cosine_similarity(out.float().squeeze(), ref_out.squeeze(), dim=-1)
    # worst-case synthetic data threshold (see test_prefill_and_attention_e2e)
    assert cos.mean().item() > 0.85, f"decode cos {cos.mean().item()}"


def test_prefill_attention_quality_4bit_values(device):
    """Quality-sensitive path: 4-bit values measurably better than 2-bit."""
    batch, heads, seq, d = 1, 4, 512, 128
    k, v = _rand_kv(batch, heads, seq, d, device, seed=11)
    q = torch.randn(batch, heads, 8, d, device=device)

    outs = {}
    for vb in (2, 4):
        cache = TurboQuantKVCache(
            head_dim=d, key_bits=3, value_bits=vb, buffer_size=128, device=device
        )
        cache.prefill(k, v)
        scores = cache.attention_scores(q)
        outs[vb] = cache.attend(F.softmax(scores.float(), dim=-1))

    ref_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(d)
    ref_out = torch.matmul(F.softmax(ref_scores, dim=-1), v.float())

    def mean_cos(a):
        return F.cosine_similarity(
            a.float().reshape(-1, d), ref_out.reshape(-1, d), dim=-1
        ).mean().item()

    c2, c4 = mean_cos(outs[2]), mean_cos(outs[4])
    assert c4 > 0.91, f"4-bit value cos {c4}"
    assert c4 > c2, f"4-bit ({c4}) should beat 2-bit ({c2})"


def test_memory_savings(device):
    """Compressed storage is smaller than fp16 baseline."""
    batch, heads, seq, d = 1, 8, 4096, 128
    k, v = _rand_kv(batch, heads, seq, d, device, seed=3)

    cache = TurboQuantKVCache(
        head_dim=d, key_bits=3, value_bits=2, buffer_size=128, device=device
    )
    cache.prefill(k, v)

    info = cache.memory_bytes()
    fp16_bytes = 2 * batch * heads * seq * d * 2  # K + V in fp16
    ratio = fp16_bytes / info["total"]
    # keys ~3.1x, values ~8x -> combined ~4x+ (minus fp buffer)
    assert ratio > 3.5, f"memory ratio {ratio:.2f}x"
