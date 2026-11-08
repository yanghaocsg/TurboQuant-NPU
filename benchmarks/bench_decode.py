"""Benchmark: torch vs triton decode paths on Ascend NPU.

Decode step = attention scores from compressed keys (+ softmax + value
aggregation). Measures the torch path, triton score kernels, and the fully
fused triton decode kernel across history lengths.

Usage:
    python benchmarks/bench_decode.py [--seqs 1024 4096 16384] [--reps 20]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch
import torch.nn.functional as F

from turboquant_npu._device import resolve_device, synchronize
from turboquant_npu.kv_cache import dequantize_values, quantize_values
from turboquant_npu.quantizer import TurboQuantProd

try:
    from turboquant_npu import triton_kernels
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


def bench(fn, reps, device):
    for _ in range(3):
        fn()
    synchronize(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run(seqs, reps):
    device = resolve_device()
    print(f"device: {device} | triton available: {HAVE_TRITON}")
    print(f"{'N':>7} | {'torch (ms)':>10} | {'triton score (ms)':>17} | "
          f"{'triton fused (ms)':>17} | fused speedup")
    print("-" * 78)

    kvh, hq, d = 8, 64, 128
    for N in seqs:
        g = torch.Generator(device="cpu").manual_seed(N)
        x = torch.randn(1, kvh, N, d, generator=g).to(device)
        v = torch.randn(1, kvh, N, d, generator=g).to(device)
        query = torch.randn(1, hq, 1, d, generator=g).to(device)

        q = TurboQuantProd(dim=d, bits=3, device=device)
        xq = q.quantize(x)
        vq = quantize_values(v, bits=2, group_size=32)
        sm_scale = 1.0 / math.sqrt(d)

        def torch_path():
            scores = q.attention_score(query, xq, backend="torch") * sm_scale
            weights = F.softmax(scores.float(), dim=-1)
            G = hq // kvh
            v_e = dequantize_values(vq, 32)
            if G > 1:
                B, H, Nn, D = v_e.shape
                v_e = v_e.unsqueeze(2).expand(B, H, G, Nn, D).reshape(B, hq, Nn, D)
            return torch.matmul(weights, v_e.float())

        t_torch = bench(torch_path, reps, device)

        t_score = float("nan")
        t_fused = float("nan")
        if HAVE_TRITON:
            def score_path():
                return triton_kernels.attention_score(q, query, xq)

            t_score = bench(score_path, reps, device)

            def fused_path():
                return triton_kernels.fused_decode(q, query, xq, vq, sm_scale)

            t_fused = bench(fused_path, reps, device)

        speedup = t_torch / t_fused if t_fused == t_fused else float("nan")
        print(f"{N:>7} | {t_torch:>10.2f} | {t_score:>17.2f} | "
              f"{t_fused:>17.2f} | {speedup:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    run(args.seqs, args.reps)


if __name__ == "__main__":
    main()
