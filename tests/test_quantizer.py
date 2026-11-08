import math

import pytest
import torch

from turboquant_npu.bitpack import pack_indices, unpack_indices, pack_signs, unpack_signs
from turboquant_npu.quantizer import TurboQuantMSE, TurboQuantProd


def _rand_unit(n, d, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g).to(device)
    return x / x.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_pack_unpack_roundtrip(bits, device):
    d = 64
    g = torch.Generator(device="cpu").manual_seed(1)
    idx = torch.randint(0, 2**bits, (3, 5, d), generator=g).to(device)
    packed = pack_indices(idx, bits)
    unpacked = unpack_indices(packed, bits, d)
    assert torch.equal(idx, unpacked)


def test_sign_pack_roundtrip(device):
    d = 128
    signs = (torch.randn(4, d, device=device) > 0)
    packed = pack_signs(signs)
    unpacked = unpack_signs(packed, d)
    expected = torch.where(signs, 1.0, -1.0)
    assert torch.equal(unpacked, expected)


@pytest.mark.parametrize("bits", [2, 3])
def test_mse_reconstruction_quality(bits, device):
    d = 128
    q = TurboQuantMSE(dim=d, bits=bits, device=device)
    x = _rand_unit(512, d, device, seed=2)

    out = q(x)
    cos = torch.nn.functional.cosine_similarity(out, x, dim=-1)
    # 2-bit: cos ~ 0.93; 3-bit: ~ 0.98 (per-coordinate Beta quantization)
    thr = 0.90 if bits == 2 else 0.97
    assert cos.mean().item() > thr, f"bits={bits} cos={cos.mean().item()}"
    # reconstruction preserves norms well
    rel_norm_err = (out.norm(dim=-1) - x.norm(dim=-1)).abs().mean().item()
    assert rel_norm_err < 0.2


def test_prod_reconstruction_quality(device):
    d = 128
    q = TurboQuantProd(dim=d, bits=3, device=device)
    x = _rand_unit(256, d, device, seed=3)

    out = q(x)
    cos = torch.nn.functional.cosine_similarity(out, x, dim=-1)
    # MSE(2-bit) ~0.93 + QJL(1-bit) residual sketch; observed ~0.92 on
    # random isotropic vectors (real LLM keys behave better)
    assert cos.mean().item() > 0.90, f"cos={cos.mean().item()}"


def test_inner_product_unbiasedness(device):
    """E[estimated <q,k>] ~= true <q,k> (paper Thm 2).

    QJL estimates are unbiased per-sample but noisy; assert the empirical
    mean deviation is within 4 standard errors of zero.
    """
    d = 128
    q = TurboQuantProd(dim=d, bits=3, device=device)

    n = 4096
    x = _rand_unit(n, d, device, seed=4)
    y = _rand_unit(n, d, device, seed=5)

    xq = q.quantize(x)
    est = q.attention_score(y.unsqueeze(-2), xq).squeeze(-1)  # (n,)
    true_ip = (y * x).sum(-1)

    dev = (est - true_ip)
    se = dev.std().item() / math.sqrt(n)  # standard error of the mean
    mean_dev = dev.mean().item()
    assert abs(mean_dev) < 4.0 * se, (
        f"bias {mean_dev:.5f} exceeds 4*SE ({4*se:.5f})"
    )


def test_attention_score_matches_dequantize(device):
    """attention_score path == <query, dequantize(key)> exactly (same math)."""
    d = 64
    q = TurboQuantProd(dim=d, bits=3, device=device)
    x = _rand_unit(32, d, device, seed=6)
    y = _rand_unit(4, d, device, seed=7)

    xq = q.quantize(x)
    scores = q.attention_score(y.unsqueeze(-2), xq).squeeze(-2)
    ref = y @ q.dequantize(xq).T
    assert torch.allclose(scores, ref, atol=1e-3, rtol=1e-3)


def test_compression_ratio(device):
    """fp16 baseline vs compressed storage for keys."""
    d = 128
    n = 4096
    q = TurboQuantProd(dim=d, bits=3, device=device)
    x = torch.randn(n, d, device=device)

    xq = q.quantize(x)
    bytes_fp16 = n * d * 2
    bytes_comp = (
        xq.mse_indices.nelement()
        + xq.qjl_signs.nelement()
        + xq.residual_norms.nelement() * 2
        + xq.norms.nelement() * 2
    )
    ratio = bytes_fp16 / bytes_comp
    # 3-bit key: 2-bit MSE (packed as 4-bit -> 0.5B/coord) + 1-bit sign (0.125B)
    # + 2 fp16 norms per vector / 128 coords = ~0.64 B/coord vs 2 B/coord fp16
    assert ratio > 2.5, f"compression ratio {ratio:.2f}x too low"


def test_npu_cpu_consistency(device):
    """Same seeds -> npu and cpu produce identical packed data."""
    d = 64
    g = torch.Generator(device="cpu").manual_seed(8)
    x = torch.randn(64, d, generator=g)

    q_cpu = TurboQuantProd(dim=d, bits=3, device=torch.device("cpu"))
    q_dev = TurboQuantProd(dim=d, bits=3, device=device)

    xq_cpu = q_cpu.quantize(x.clone())
    xq_dev = q_dev.quantize(x.clone().to(device))

    assert torch.equal(xq_cpu.mse_indices.cpu(), xq_dev.mse_indices.cpu())
    assert torch.equal(xq_cpu.qjl_signs.cpu(), xq_dev.qjl_signs.cpu())
    assert torch.allclose(
        xq_cpu.norms, xq_dev.norms.cpu(), rtol=1e-4, atol=1e-4
    )
