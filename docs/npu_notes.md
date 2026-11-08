# Ascend NPU 移植笔记

环境：Ascend 910 (64GB) ×8 · CANN 26.1.1 · torch 2.10.0 + torch_npu 2.10.0.post7 ·
triton 3.6.0 + triton-ascend 3.6.0 · aarch64

## torch_npu 算子差异（Phase 1 踩坑）

| 问题 | 现象 | 对策 |
|------|------|------|
| `aclnnArange` 无 uint8 输出 | `arange(..., dtype=uint8)` 报 `AclNN_Parameter_Error: DT_UINT8 not in [FLOAT16, FLOAT, DOUBLE, INT64, INT32, BFLOAT16]` | `arange` 用 int32 生成再 `.to(uint8)`，或本仓库的标量循环方案 |
| 位移算子不支持张量广播 | `packed.unsqueeze(-1) >> shifts_tensor` 报 shape 错误（要求 out 与 RHS 同形） | 标量移位循环（每 byte ≤8 次元素级 kernel），见 `bitpack.py` |
| 标量移位/与/或 | 正常支持（`u >> 2`、`& 3`） | — |
| `searchsorted` | 支持良好 | 直接使用 |
| uint8 张量本身 | 支持（randint/存取/转换均正常），仅部分算子（arange）限制输出 dtype | — |

## triton-ascend kernel 经验（Phase 2）

1. **无硬件 gather**。`tl.load(ptr + tensor_idx)` 可用但编译成慢速访存；
   codebook 查表（≤16 项）改用 `tl.where` 选择链（寄存器标量 + 向量选择），
   score kernel 从 gather 版的 ~12.9ms (N=4096) 降到 **1.44ms**。
2. **串行 flash 循环浪费 AI core**。单 program 扫全序列的 fused kernel
   （grid=BH=64）只有 64 个核在忙，N=4096 时 33.5ms。
   改成 flash-decoding split-KV（partial 并行 + merge kernel）后 **5.23ms**。
3. **UB（统一缓冲）192KB 限制**。tile 过大或多种 tile 多缓冲时报
   `ub overflow, requires N bits while 1572864 bits available`：
   - bits=4 值路径：partial kernel `BLOCK_N` 从 128 降到 64
   - merge kernel：`BLOCK_NB` 上限 64（NB=256 × D=128 的 fp32 tile = 128KB 会超）
4. **2D tile + 行内 reduce** 正常且高效；`tl.static_range` 展开循环、
   `tl.exp/tl.max/tl.where/在线 softmax` 均可用。
5. `uint8` 指针 load 后 `.to(tl.int32)` 再移位，稳定。

## 基准（Ascend 910, GQA 8/4 头, d=128, 3-bit key + 2-bit value）

decode 一步（压缩 KV 上的 score + softmax + value 聚合，中位数 / 20 次）：

| 历史 N | torch 路径 | triton score | triton fused | fused 加速比 |
|-------:|-----------:|-------------:|-------------:|-------------:|
| 1,024 | 4.54 ms | 0.56 ms | 1.65 ms | **2.75x** |
| 4,096 | 11.85 ms | 1.44 ms | 5.23 ms | **2.26x** |
| 16,384 | 41.29 ms | 4.88 ms | 19.22 ms | **2.15x** |

量化吞吐（torch-npu 路径，8 kv 头 × d=128）：
- Prod 量化（3-bit TQ key）：~48k tok/s
- Value 量化（2-bit 分组）：~67k tok/s

## 质量参考

- 码本与参考实现逐位一致（最大偏差 ~1e-16）
- 随机各向同性 KV（最坏情况）：e2e attention 输出 cos ~0.85（2-bit value）、~0.93（4-bit value）
- 真实 LLM KV 上参考实现测得 ~0.94（3b key/2b value）、0.997（4-bit value）
