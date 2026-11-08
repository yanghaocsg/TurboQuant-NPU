"""Benchmark: quantization throughput (tokens/s) on the accelerator.

Usage:
    python benchmarks/bench_quantize.py [--n 8192] [--kvh 8] [--d 128]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from turboquant_npu._device import resolve_device, synchronize
from turboquant_npu.kv_cache import quantize_values
from turboquant_npu.quantizer import TurboQuantMSE, TurboQuantProd


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


def run(n, kvh, d, reps):
    device = resolve_device()
    print(f"device: {device} | n={n} kv_heads={kvh} head_dim={d}")

    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, kvh, n, d, generator=g).to(device)

    mse = TurboQuantMSE(dim=d, bits=2, device=device)
    prod = TurboQuantProd(dim=d, bits=3, device=device)

    t_mse = bench(lambda: mse.quantize(x), reps, device)
    t_prod = bench(lambda: prod.quantize(x), reps, device)
    t_val = bench(lambda: quantize_values(x, bits=2, group_size=32), reps, device)
    t_deq = bench(lambda: prod.dequantize(prod.quantize(x)), reps, device)

    print(f"MSE quantize    (2-bit): {t_mse:8.2f} ms  {n/t_mse*1e3/1e3:8.0f}k tok/s")
    print(f"Prod quantize (3-bit TQ): {t_prod:8.2f} ms  {n/t_prod*1e3/1e3:8.0f}k tok/s")
    print(f"Value quantize (2-bit):   {t_val:8.2f} ms  {n/t_val*1e3/1e3:8.0f}k tok/s")
    print(f"Prod dequantize:          {t_deq:8.2f} ms  {n/t_deq*1e3/1e3:8.0f}k tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--kvh", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    run(args.n, args.kvh, args.d, args.reps)


if __name__ == "__main__":
    main()
