# TurboQuant-NPU

TurboQuant KV Cache 量化在**华为昇腾 Ascend NPU** 上的实现。

- 论文: TurboQuant — Near-optimal KV cache quantization（ICLR 2026, arXiv:2504.19874）
- 参考实现: [0xSero/turboquant](https://github.com/0xSero/turboquant)（CUDA/Triton, GPLv3）
- 本项目: 先实现 **torch-npu** 版本（纯 PyTorch 算子），再实现 **triton-npu** 版本（triton-ascend 融合 kernel）

## 特性（规划中，见 [PLAN.md](PLAN.md)）

- ✅ **M0** 环境探测：torch_npu 关键算子、triton-ascend kernel 原语全部验证通过
- 🔜 **Phase 1 (torch-npu)**: Lloyd-Max 码本 / 随机旋转 / QJL / bit-packing / KV Cache / hybrid attention
- 🔜 **Phase 2 (triton-npu)**: score 计算 + fused decode 融合 kernel（在线 softmax，单 pass 读压缩 KV）

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

## 快速开始（Phase 1 完成后可用）

```bash
pip install -e .
python examples/demo_decode.py          # prefill 4k + decode，与 fp16 基线对比
python -m pytest tests/ -v              # 测试（自动检测 NPU，无 NPU 时跑 CPU）
```

## License

GPLv3（随上游 [0xSero/turboquant](https://github.com/0xSero/turboquant)）。
