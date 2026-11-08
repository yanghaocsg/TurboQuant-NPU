"""TurboQuant-NPU end-to-end demo: prefill + decode with compressed KV cache.

Runs on NPU (Ascend via torch_npu) when available, else CPU. Simulates a
single attention layer with GQA, compares compressed attention against an
fp16/fp32 full-precision baseline, and reports quality, memory and speed.

Usage:
    python examples/demo_decode.py [--prefill 4096] [--decode 256] [--no-accel]
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from turboquant_npu._device import resolve_device, synchronize
from turboquant_npu.kv_cache import TurboQuantKVCache


def make_kv(batch, kv_heads, seq, d, device, seed=0, stress=False):
    """Synthetic KV. Pure iid Gaussian by default; `stress` adds heavy-tailed
    token/channel scale variation (a hard case for 2-bit min-max group
    quantization — real LLM KV sits between the two)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(batch, kv_heads, seq, d, generator=g)
    v = torch.randn(batch, kv_heads, seq, d, generator=g)
    if stress:
        tok_scale = torch.randn(batch, 1, seq, 1, generator=g).exp() * 0.5
        chan_scale = torch.randn(1, kv_heads, 1, d, generator=g).exp() * 0.5
        k = k * tok_scale * chan_scale
        v = v * tok_scale * chan_scale
    return k.to(device), v.to(device)


def run(device, prefill_len, decode_steps, key_bits, value_bits, buffer=128,
        stress=False, backend="torch"):
    batch, kv_heads, q_heads, d = 1, 4, 8, 128
    scale = 1.0 / math.sqrt(d)
    print(f"\n=== TurboQuant-NPU demo on {device} ===")
    print(f"prefill={prefill_len} decode={decode_steps} key_bits={key_bits} "
          f"value_bits={value_bits} buffer={buffer} (GQA {q_heads}/{kv_heads}, d={d}) "
          f"backend={backend}")
    note = ("stress distribution (heavy-tailed); real LLM KV measures ~0.94 "
            "with 3b/2b per the reference repo" if stress else
            "iid Gaussian KV (worst case); real LLM KV measures ~0.94 with 3b/2b")
    print(f"data: {note}")

    k0, v0 = make_kv(batch, kv_heads, prefill_len, d, device, seed=1, stress=stress)

    cache = TurboQuantKVCache(
        head_dim=d, key_bits=key_bits, value_bits=value_bits,
        buffer_size=buffer, device=device,
    )

    t0 = time.perf_counter()
    cache.prefill(k0, v0)
    synchronize(device)
    t_prefill = time.perf_counter() - t0

    mem = cache.memory_bytes()
    fp16_bytes = 2 * batch * kv_heads * cache.seq_len * d * 2
    print(f"prefill: {t_prefill*1e3:.1f} ms | KV storage "
          f"{mem['total']/2**20:.2f} MiB vs fp16 {fp16_bytes/2**20:.2f} MiB "
          f"({fp16_bytes/mem['total']:.2f}x)")

    # decode steps
    g2 = torch.Generator(device="cpu").manual_seed(2)
    cos_all = []
    t_decode = 0.0
    for step in range(decode_steps):
        kk = torch.randn(batch, kv_heads, 1, d, generator=g2).to(device)
        vv = torch.randn(batch, kv_heads, 1, d, generator=g2).to(device)
        q = torch.randn(batch, q_heads, 1, d, generator=g2).to(device)
        cache.append(kk, vv)

        t0 = time.perf_counter()
        scores = cache.attention_scores(q, backend=backend)
        out = cache.attend(F.softmax(scores.float(), dim=-1))
        synchronize(device)
        t_decode += time.perf_counter() - t0

        # fp32 reference with the same tokens (GQA expansion)
        k_all = torch.cat([k0] + [kk], dim=2) if step == 0 else torch.cat([k_all, kk], dim=2)
        v_all = torch.cat([v0] + [vv], dim=2) if step == 0 else torch.cat([v_all, vv], dim=2)
        g_ratio = q_heads // kv_heads
        kk_ref = k_all.float().unsqueeze(2).expand(batch, kv_heads, g_ratio, -1, d).reshape(batch, q_heads, -1, d)
        vv_ref = v_all.float().unsqueeze(2).expand(batch, kv_heads, g_ratio, -1, d).reshape(batch, q_heads, -1, d)
        ref_scores = torch.matmul(q.float(), kk_ref.transpose(-2, -1)) * scale
        ref_out = torch.matmul(F.softmax(ref_scores, dim=-1), vv_ref)

        cos = F.cosine_similarity(
            out.float().reshape(-1, d), ref_out.reshape(-1, d), dim=-1
        ).mean().item()
        cos_all.append(cos)

    cos_t = torch.tensor(cos_all)
    print(f"decode: {t_decode/decode_steps*1e3:.2f} ms/step over {decode_steps} steps")
    print(f"output cos_sim vs fp32: mean={cos_t.mean():.4f} "
          f"p5={cos_t.quantile(0.05):.4f} min={cos_t.min():.4f}")
    mem = cache.memory_bytes()
    fp16_bytes = 2 * batch * kv_heads * cache.seq_len * d * 2
    print(f"final KV storage {mem['total']/2**20:.2f} MiB vs fp16 "
          f"{fp16_bytes/2**20:.2f} MiB ({fp16_bytes/mem['total']:.2f}x compression)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--key-bits", type=int, default=3)
    ap.add_argument("--value-bits", type=int, default=2)
    ap.add_argument("--no-accel", action="store_true")
    ap.add_argument("--stress", action="store_true",
                    help="heavy-tailed KV distribution (hard case)")
    ap.add_argument("--backend", choices=["torch", "triton"], default="torch")
    args = ap.parse_args()

    device = torch.device("cpu") if args.no_accel else resolve_device()
    run(device, args.prefill, args.decode, args.key_bits, args.value_bits,
        stress=args.stress, backend=args.backend)


if __name__ == "__main__":
    main()
