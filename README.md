# TurboQuant-NPU

TurboQuant KV Cache 量化在**华为昇腾 Ascend NPU** 上的实现。

- 论文: TurboQuant — Near-optimal KV cache quantization（ICLR 2026, arXiv:2504.19874）
- 参考实现: [0xSero/turboquant](https://github.com/0xSero/turboquant)（CUDA/Triton, GPLv3）
- 两个后端：**torch-npu**（纯 PyTorch 算子）与 **triton-npu**（triton-ascend 融合 kernel），`backend` 参数切换，triton 不可用时自动回退

## 状态

- ✅ **M0** 环境探测：torch_npu 关键算子、triton-ascend kernel 原语全部验证通过
- ✅ **Phase 1 (torch-npu)**: Lloyd-Max 码本（与参考逐位一致）/ 随机旋转 / QJL / bit-packing /
  KV Cache（GQA）/ CompressedKVStore / hybrid attention
- ✅ **Phase 2 (triton-npu)**: score kernel + flash-decoding split-KV fused decode kernel
- ✅ 59/59 测试在 CPU 与 Ascend 910 上通过

## 基准（Ascend 910, GQA 8/4 头, d=128, 3-bit key + 2-bit value）

decode 一步（压缩 KV 上 score + softmax + value 聚合）：

| 历史 N | torch 路径 | triton fused | 加速比 |
|-------:|-----------:|-------------:|-------:|
| 1,024 | 4.54 ms | 1.65 ms | **2.75x** |
| 4,096 | 11.85 ms | 5.23 ms | **2.26x** |
| 16,384 | 41.29 ms | 19.22 ms | **2.15x** |

triton score kernel 单独对比 torch score 路径 **~8x**；
KV 存储 **4.55x** 压缩（vs fp16）。详见 [docs/npu_notes.md](docs/npu_notes.md)。

## 算法概要

```
Key   (3-bit): x → ‖x‖归一化 → 随机正交旋转 Π → Lloyd-Max (2-bit) → QJL 残差符号 (1-bit)
Value (2-bit): 分组量化 (group=32, per-group scale/zero) → bit-packing
Score:        <q,k> ≈ <q, k̃_mse> + √(π/2)/d · ‖r‖ · <Sᵀ·sign(Sr), q>   （无偏估计）
```

解码时 query 旋转到压缩域直接算分，key 无需解压；近期 token 保留全精度 buffer。

## 环境

| 组件 | 版本 |
|------|------|
| NPU | Ascend 910 (64GB HBM) ×8, aarch64 |
| CANN | 26.1.1 |
| Python | 3.11.10 |
| torch / torch_npu | 2.10.0 / 2.10.0.post7 |
| triton / triton_ascend | 3.6.0 / 3.6.0 |

## 快速开始

```bash
pip install -e .
python examples/demo_decode.py                     # torch-npu 后端
python examples/demo_decode.py --backend triton    # triton-npu score kernel
python benchmarks/bench_decode.py                  # torch vs triton 基准
python -m pytest tests/ -v                         # 测试（自动检测 NPU，无 NPU 时跑 CPU）
```

## License

GPLv3（随上游 [0xSero/turboquant](https://github.com/0xSero/turboquant)）。
