# TurboQuant-NPU 实施规划

> TurboQuant KV Cache 量化算法在华为昇腾 Ascend NPU 上的实现路线
>
> 目标仓库: https://github.com/yanghaocsg/TurboQuant-NPU
> 参考实现: https://github.com/0xSero/turboquant （CUDA/Triton，GPLv3）
> 论文: TurboQuant (ICLR 2026, arXiv:2504.19874)

---

## 1. 项目背景与目标

TurboQuant 是一种接近最优的 LLM KV Cache 压缩方法（典型配置 3-bit key / 2-bit value，
压缩比 ~4.4x），核心性质是**内积估计无偏**，因此可以直接在压缩域计算 attention score，
而不需要先解压再算。

本项目将 TurboQuant 移植到昇腾 NPU，分两个阶段：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | torch-npu 版本：纯 PyTorch 算子实现，跑在 `torch_npu` (CANN) 后端 | 🔜 进行中 |
| **Phase 2** | triton-npu 版本：用 `triton-ascend` 编写融合 kernel（score 计算 / fused decode） | 🔜 |

两阶段共享同一套数据结构与码本，Phase 2 作为可插拔后端叠加在 Phase 1 之上，
通过 `backend` 开关选择，triton 不可用时自动回退 torch 路径。

---

## 2. 目标环境（已在本机验证）

| 组件 | 版本 |
|------|------|
| NPU | Ascend 910 ×8（每卡 64GB HBM），aarch64 |
| CANN / npu-smi | 26.1.1 |
| Python | 3.11.10 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post7 |
| triton / triton_ascend | 3.6.0 / 3.6.0+dev |

### 可行性探测结果（已完成）

torch_npu 后端关键算子全部可用：
`matmul / einsum` ✅ `searchsorted` ✅ `uint8 & int32 位运算（<< >> & |）` ✅ `gather（张量下标索引）` ✅

triton-ascend 后端关键原语全部可用（已跑通最小 kernel）：
`int32 位运算` ✅ `gather（codebook 查表）` ✅ `2D tile 加载 + 行内 reduce` ✅

> 结论：Phase 1 / Phase 2 均无阻塞性障碍。

---

## 3. 算法要点（对齐参考实现）

**Key 压缩（TurboQuantProd，论文 Algorithm 2）**

1. `‖x‖` 归一化 → 随机正交旋转 `y = x·Πᵀ`（Π 由固定种子 Gaussian 矩阵 QR 分解得到）
2. 旋转后每维服从 Beta 分布 → **Lloyd-Max 最优标量码本**量化（b−1 bit，数值积分离线计算）
3. 残差 `r = x − x̃` 过 **QJL 投影**：`sign(S·r)` 每维 1 bit
4. 内积无偏估计：
   `<q, k> ≈ <q, k̃_mse> + √(π/2)/d · ‖r‖ · <Sᵀ·sign(Sr), q>`
5. 索引与符号全部 bit-packing（3-bit 按 4-bit 打包，2 个/字节）

**Value 压缩**：2/4-bit 分组量化（group=32，per-group scale/zero）+ bit-packing（2-bit 4 个/字节）

**解码路径关键优化**：不解压 key，而是把 query 旋转/投影到压缩域直接算分
（`q_rot = q·Πᵀ` 与码本 gather 点积；`q_sketch = q·Sᵀ` 与 sign bits 点积）。

**近期 token**（buffer_size，默认 128）保留全精度，与压缩历史段 log-sum-exp 合并。

---

## 4. 仓库结构规划

```
TurboQuant-NPU/
├── README.md
├── PLAN.md                  # 本文件
├── LICENSE                  # GPLv3（随上游）
├── pyproject.toml
├── turboquant_npu/
│   ├── __init__.py
│   ├── _device.py           # 设备抽象：npu 优先，回退 cuda/cpu
│   ├── codebook.py          # Lloyd-Max 码本（scipy 数值积分，CPU，JSON 缓存）
│   ├── codebooks/           # 预生成码本 d=64/128, bits=1..4
│   ├── rotation.py          # Π / S 矩阵（CPU 固定种子生成 → .npu()）
│   ├── bitpack.py           # 索引/符号 bit-packing 与解包（torch 算子）
│   ├── quantizer.py         # TurboQuantMSE / TurboQuantProd（torch-npu 后端）
│   ├── kv_cache.py          # Value 组量化 + TurboQuantKVCache
│   ├── store.py             # CompressedKVStore（chunk 追加 + 惰性 flatten）
│   ├── score.py             # hybrid attention（压缩历史 + 精确 buffer）
│   ├── backend.py           # 后端调度："torch" | "triton"（自动降级）
│   └── triton_kernels.py    # Phase 2：triton-ascend 融合 kernel
├── tests/
│   ├── test_codebook.py     # 码本 MSE 对齐论文 Table 1 / 参考实现
│   ├── test_quantizer.py    # 重建质量、无偏性、压缩比（CPU↔NPU 一致）
│   ├── test_kv_cache.py     # prefill/append/attend 端到端
│   ├── test_store_score.py  # chunk store + hybrid attention
│   └── test_triton.py       # Phase 2：kernel 与 torch 参考路径对齐
├── benchmarks/
│   ├── bench_quantize.py    # 量化吞吐（torch vs triton 后端）
│   └── bench_decode.py      # decode attention 时延 vs 序列长度
├── examples/
│   └── demo_decode.py       # 模拟 prefill+decode 的 KV 压缩演示
└── docs/
    └── npu_notes.md         # NPU 移植笔记（算子差异、坑、探测脚本结论）
```

---

## 5. Phase 1 — torch-npu 版本（M1–M3）

### M1 基础设施
- `_device.py`：`resolve_device()` — `npu` 优先（`torch.npu.is_available()`），回退 cuda/cpu
- `codebook.py`：移植 scipy 数值积分 Lloyd-Max（纯 CPU，与设备无关）；预生成
  d∈{64,128} × bits∈{1,2,3,4} 码本 JSON 进仓库；d=256 按需生成
- `rotation.py`：CPU 固定种子生成 Π（QR + 对角符号修正）与 S，`.to(device)`；
  提供 `rotate_forward/backward`
- `bitpack.py`：`_pack_indices/_unpack_indices`（bits=1: 8/字节，2: 4/字节，
  3: 升 4-bit 2/字节）+ QJL sign 打包

**验收**：码本 MSE 与参考实现 JSON 逐位一致；`‖ΠᵀΠ−I‖∞ < 1e-5`；
packing↔unpacking 往返无损。

### M2 核心量化器（NPU）
- `TurboQuantMSE.quantize/dequantize`：norm → rotate → `searchsorted` → pack；
  解码端 gather centroids → 反旋转 → 乘 norm
- `TurboQuantProd.quantize/dequantize/attention_score`：两级量化 + QJL；
  score 路径 = MSE 段 + QJL 段（`q @ Sᵀ` 与解包 signs 的 matmul）
- Value 组量化 `quantize_values/dequantize_values`（2/4-bit）

**验收**（NPU 与 CPU 双跑，结果一致）：
- 3-bit key 单位向量重建 cos_sim ≥ 0.999；2-bit value cos_sim ≈ 0.94（对齐参考实测）
- 内积无偏性：随机 4096 向量，`|E[est]−true| / |true| < 0.5%`
- 存储压缩比 ≈ 4.4x（head_dim=128, 3b key + 2b val，扣除 Π/S 与 buffer 后 ≥ 4x）

### M3 KV Cache 端到端（NPU）
- `TurboQuantKVCache`：`prefill / append / _flush_buffer / attention_scores / attend`
- `CompressedKVStore` + `compute_hybrid_attention`（GQA 广播避免 repeat_interleave）
- `examples/demo_decode.py`：随机（或真实分布）KV，prefill 4096 → decode 步进，
  与 fp16 全精度 attention 对比输出误差

**验收**：NPU 上 prefill 4k + decode 到 8k，
hybrid attention 输出与 fp16 基线 cos_sim ≥ 0.98；无 OOM（测试张量适配被占用的 HBM）。

---

## 6. Phase 2 — triton-npu 版本（M4–M6）

### M4 kernel 移植 + 向量化改造
参考实现的 CUDA Triton kernel 在每个坐标上做标量 `tl.load`（循环内逐维取 q 值），
这种写法在 Ascend 上效率差。改造为 **2D tile 向量化**：

- **kernel 1 `mse_score`**：`(BLOCK_N, D)` tile 一次性解包索引 →
  `tl.load(CENTROIDS + idx)` 向量 gather → 与 `q_rot` tile 相乘后沿 D 归约 → 乘 norms
- **kernel 2 `qjl_score`**：sign byte → `±1` 的位展开用算术式
  `sign = 2*((packed>>bit)&1) − 1` 向量化，与 `q_sketch` tile 点积 → 乘 `‖r‖·√(π/2)/d`
- **kernel 3 `fused_decode`**：单 pass 在线 softmax（flash 风格）+
  value 解量化（`v_quant·scale + zero`）加权累加，一次读压缩 KV 产出最终输出
- 探测已确认 gather/位运算/2D tile 可用；若个别原语在特定 CANN 版本受限，
  fallback：codebook gather → 16 路 `tl.where` 选择链（bits≤4）

### M5 正确性对齐
- 每个 kernel 与 torch 参考路径 `torch.allclose(atol=1e-3)`（fp32 累加）
- `backend.py`：`select_backend("triton")` 失败自动降级 `"torch"` 并告警

### M6 性能基准
- `benchmarks/`：torch-npu vs triton-npu，NPU 事件计时，扫 seq_len ∈ {1k, 4k, 16k, 64k}
- 目标：decode attention 路径 triton 版相对 torch 版 ≥ 1.5x（消除中间大张量物化）
- 记录 HBM 带宽利用率，写入 `docs/npu_notes.md`

---

## 7. 里程碑总览

| 里程碑 | 交付物 | 验收标准 |
|--------|--------|----------|
| M0 可行性探测（✅ 已完成） | 算子/kernel 探测结论 | 全部通过 |
| M0.5 规划入库（✅ 本文件） | PLAN.md / README / LICENSE | push 到 GitHub |
| M1 | 码本 + 旋转 + bitpack | 对齐参考实现 |
| M2 | 量化器（torch-npu） | cos_sim / 无偏性 / 压缩比 |
| M3 | KV cache + hybrid attention 端到端 | demo + 测试全绿 |
| M4 | triton-ascend 三个 kernel | 编译通过 |
| M5 | kernel 对齐测试 | allclose(atol=1e-3) |
| M6 | 基准测试 + 文档 | ≥1.5x 加速目标 |
| M7 | README 成果更新 / 发布 tag | v0.1.0 |

---

## 8. 风险与对策

| 风险 | 对策 |
|------|------|
| triton-ascend 对个别原语（动态 gather、复杂 mask）在特定版本不支持 | 已探测通过；仍保留 torch fallback，`backend.py` 自动降级 |
| NPU HBM 被其他任务大量占用（各卡 ~60/64GB） | 测试用小张量；`ASCEND_RT_VISIBLE_DEVICES` 选最空卡；CI 级测试控制在一卡 |
| uint8 位运算在旧 CANN 支持不全 | 已验证本机可用；代码中位运算统一走 int32 中间类型 |
| `searchsorted` NPU 数值边界行为差异 | 测试覆盖正/负边界；必要时以 `torch.bucketize` 替代 |
| vLLM(vllm-ascend) 集成范围过大 | 本期只做核心库 + demo；vllm-ascend adapter 列为 v0.2 目标 |
| HBM 占用导致 bench 波动 | 多次取样取中位数；报告注明环境 |

## 9. License

上游 0xSero/turboquant 为 GPLv3，本项目为衍生实现，采用 **GPLv3** 并在 README 标注来源。
