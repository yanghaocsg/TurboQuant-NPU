"""TurboQuant fused Triton kernels for Ascend NPU (triton-ascend).

Architectural notes (vs the reference CUDA kernels):
  - No per-coordinate scalar loads: everything is vectorized 2D tiles.
  - No codebook gather: Ascend has no hardware gather, so centroid lookup
    is a tl.where select-chain over the (<=16-entry) codebook held in
    registers — pure vector ALU.
  - The fused decode kernel uses flash-decoding split-KV: partial programs
    parallelize over KV blocks, a small merge kernel combines the online
    softmax states. The single-program serial loop would leave most AI
    cores idle at decode batch sizes.

Verified on Ascend 910 / CANN 26.1.1 / triton 3.6 + triton-ascend 3.6.

Derived from 0xSero/turboquant (GPLv3).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from turboquant_npu.bitpack import packing_params


@triton.jit
def _select_centroid(idx, CENTROIDS_ptr, NCLUST: tl.constexpr):
    """centroid[idx] without gather: select-chain over a tiny codebook."""
    c = tl.load(CENTROIDS_ptr + 0) + tl.zeros_like(idx).to(tl.float32)
    for i in tl.static_range(1, NCLUST):
        c = tl.where(idx == i, tl.load(CENTROIDS_ptr + i), c)
    return c


@triton.jit
def _kv_head_of(pid_qh, HQ, HKV, GQA: tl.constexpr):
    if GQA == 1:
        return pid_qh
    b = pid_qh // HQ
    h = pid_qh % HQ
    return b * HKV + h // GQA


# ─── Kernel 1: MSE scores ────────────────────────────────────────────

@triton.jit
def _mse_score_kernel(
    Q_ROT_ptr,       # (BHQ, D) rotated queries q @ Pi^T
    MSE_ptr,         # (BHKV, N, PACKED_D) uint8 packed indices
    NORMS_ptr,       # (BHKV, N)
    CENTROIDS_ptr,   # (n_clusters,)
    OUT_ptr,         # (BHQ, N)
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    N,
    HQ, HKV,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    BITS: tl.constexpr,
    VPD: tl.constexpr,
    NCLUST: tl.constexpr,
    GQA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_qh = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh_kv = _kv_head_of(pid_qh, HQ, HKV, GQA)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    byte_offs = tl.arange(0, PACKED_D)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    for sub in tl.static_range(VPD):
        packed = tl.load(
            MSE_ptr + bh_kv * stride_m_bh
            + n_offs[:, None] * stride_m_n + byte_offs[None, :] * stride_m_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        idx = (packed >> (sub * BITS)) & BIT_MASK
        cent = _select_centroid(idx, CENTROIDS_ptr, NCLUST)

        coord = byte_offs * VPD + sub
        q_val = tl.load(
            Q_ROT_ptr + pid_qh * stride_q_bh + coord * stride_q_d,
            mask=coord < D, other=0.0,
        ).to(tl.float32)
        scores += tl.sum(q_val[None, :] * cent, 1)

    norms = tl.load(
        NORMS_ptr + bh_kv * stride_n_bh + n_offs * stride_n_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    tl.store(
        OUT_ptr + pid_qh * stride_o_bh + n_offs * stride_o_n,
        scores * norms, mask=n_mask,
    )


# ─── Kernel 2: QJL scores (accumulated into OUT) ─────────────────────

@triton.jit
def _qjl_score_kernel(
    Q_SKETCH_ptr,    # (BHQ, D)
    SIGNS_ptr,       # (BHKV, N, PACKED_DS)
    RES_NORMS_ptr,   # (BHKV, N)
    OUT_ptr,         # (BHQ, N) — MSE scores added into
    stride_q_bh, stride_q_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_r_bh, stride_r_n,
    stride_o_bh, stride_o_n,
    N, HQ, HKV,
    D: tl.constexpr,
    PACKED_DS: tl.constexpr,
    QJL_SCALE,
    GQA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_qh = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh_kv = _kv_head_of(pid_qh, HQ, HKV, GQA)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    byte_offs = tl.arange(0, PACKED_DS)

    dot = tl.zeros([BLOCK_N], dtype=tl.float32)
    for bit in tl.static_range(8):
        packed = tl.load(
            SIGNS_ptr + bh_kv * stride_s_bh
            + n_offs[:, None] * stride_s_n + byte_offs[None, :] * stride_s_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        sign_val = (((packed >> bit) & 1).to(tl.float32) * 2.0) - 1.0

        coord = byte_offs * 8 + bit
        q_val = tl.load(
            Q_SKETCH_ptr + pid_qh * stride_q_bh + coord * stride_q_d,
            mask=coord < D, other=0.0,
        ).to(tl.float32)
        dot += tl.sum(q_val[None, :] * sign_val, 1)

    res_norms = tl.load(
        RES_NORMS_ptr + bh_kv * stride_r_bh + n_offs * stride_r_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)

    existing = tl.load(
        OUT_ptr + pid_qh * stride_o_bh + n_offs * stride_o_n,
        mask=n_mask, other=0.0,
    )
    tl.store(
        OUT_ptr + pid_qh * stride_o_bh + n_offs * stride_o_n,
        existing + dot * res_norms * QJL_SCALE,
        mask=n_mask,
    )


# ─── Kernel 3a: fused decode, partial (split-KV) ─────────────────────

@triton.jit
def _fused_decode_partial_kernel(
    Q_ROT_ptr, Q_SKETCH_ptr,
    MSE_ptr, SIGNS_ptr, NORMS_ptr, RES_NORMS_ptr, CENTROIDS_ptr,
    V_DATA_ptr, V_SCALES_ptr, V_ZEROS_ptr,
    OUT_M_ptr,     # (BHQ, NB) block max
    OUT_L_ptr,     # (BHQ, NB) block exp-sum
    OUT_ACC_ptr,   # (BHQ, NB, D) block weighted value sums (real dim order)
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_r_bh, stride_r_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_mo_bh, stride_mo_nb,
    stride_lo_bh, stride_lo_nb,
    stride_ao_bh, stride_ao_nb, stride_ao_d,
    N, HQ, HKV,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    PACKED_DS: tl.constexpr,
    PACKED_DV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BITS: tl.constexpr, VPD: tl.constexpr, NCLUST: tl.constexpr,
    V_BITS: tl.constexpr, V_VPD: tl.constexpr,
    QJL_SCALE, SM_SCALE,
    GQA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_qh = tl.program_id(0)
    pid_nb = tl.program_id(1)
    bh_kv = _kv_head_of(pid_qh, HQ, HKV, GQA)

    n_offs = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    byte_offs = tl.arange(0, PACKED_D)
    sign_byte_offs = tl.arange(0, PACKED_DS)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1
    V_MASK: tl.constexpr = (1 << V_BITS) - 1

    # ── scores ──
    mse = tl.zeros([BLOCK_N], dtype=tl.float32)
    for sub in tl.static_range(VPD):
        packed = tl.load(
            MSE_ptr + bh_kv * stride_m_bh
            + n_offs[:, None] * stride_m_n + byte_offs[None, :] * stride_m_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        idx = (packed >> (sub * BITS)) & BIT_MASK
        cent = _select_centroid(idx, CENTROIDS_ptr, NCLUST)
        coord = byte_offs * VPD + sub
        q_val = tl.load(
            Q_ROT_ptr + pid_qh * stride_q_bh + coord * stride_q_d,
            mask=coord < D, other=0.0,
        ).to(tl.float32)
        mse += tl.sum(q_val[None, :] * cent, 1)

    key_norms = tl.load(
        NORMS_ptr + bh_kv * stride_n_bh + n_offs * stride_n_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    mse = mse * key_norms

    qjl = tl.zeros([BLOCK_N], dtype=tl.float32)
    for bit in tl.static_range(8):
        packed = tl.load(
            SIGNS_ptr + bh_kv * stride_s_bh
            + n_offs[:, None] * stride_s_n + sign_byte_offs[None, :] * stride_s_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        sign_val = (((packed >> bit) & 1).to(tl.float32) * 2.0) - 1.0
        coord = sign_byte_offs * 8 + bit
        q_val = tl.load(
            Q_SKETCH_ptr + pid_qh * stride_q_bh + coord * stride_q_d,
            mask=coord < D, other=0.0,
        ).to(tl.float32)
        qjl += tl.sum(q_val[None, :] * sign_val, 1)

    res_norms = tl.load(
        RES_NORMS_ptr + bh_kv * stride_r_bh + n_offs * stride_r_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)

    scores = (mse + qjl * res_norms * QJL_SCALE) * SM_SCALE
    scores = tl.where(n_mask, scores, float("-inf"))

    # local softmax
    m_l = tl.max(scores, 0)
    p = tl.exp(scores - m_l)
    l_l = tl.sum(p, 0)

    # ── values: dequantize and accumulate per real dim ──
    # Group scales/zeros are loaded per group (contiguous rows) and expanded
    # to byte columns with a select-chain — avoids computed-index gathers,
    # which Ascend executes slowly.
    v_byte_offs = tl.arange(0, PACKED_DV)
    BYTES_PER_GROUP: tl.constexpr = GROUP_SIZE // V_VPD
    g_of_byte = v_byte_offs // BYTES_PER_GROUP

    s0 = tl.load(V_SCALES_ptr + bh_kv * stride_vs_bh
                 + n_offs * stride_vs_n + 0 * stride_vs_g,
                 mask=n_mask, other=0.0).to(tl.float32)
    z0 = tl.load(V_ZEROS_ptr + bh_kv * stride_vz_bh
                 + n_offs * stride_vz_n + 0 * stride_vz_g,
                 mask=n_mask, other=0.0).to(tl.float32)
    scale_row = s0[:, None] + tl.zeros([BLOCK_N, PACKED_DV], dtype=tl.float32)
    zero_row = z0[:, None] + tl.zeros([BLOCK_N, PACKED_DV], dtype=tl.float32)
    for g in tl.static_range(1, N_GROUPS):
        s_g = tl.load(V_SCALES_ptr + bh_kv * stride_vs_bh
                      + n_offs * stride_vs_n + g * stride_vs_g,
                      mask=n_mask, other=0.0).to(tl.float32)
        z_g = tl.load(V_ZEROS_ptr + bh_kv * stride_vz_bh
                      + n_offs * stride_vz_n + g * stride_vz_g,
                      mask=n_mask, other=0.0).to(tl.float32)
        sel = (g_of_byte == g)[None, :]
        scale_row = tl.where(sel, s_g[:, None], scale_row)
        zero_row = tl.where(sel, z_g[:, None], zero_row)

    acc0 = tl.zeros([PACKED_DV], dtype=tl.float32)
    acc1 = tl.zeros([PACKED_DV], dtype=tl.float32)
    acc2 = tl.zeros([PACKED_DV], dtype=tl.float32)
    acc3 = tl.zeros([PACKED_DV], dtype=tl.float32)

    for sub in tl.static_range(V_VPD):
        vp = tl.load(
            V_DATA_ptr + bh_kv * stride_v_bh
            + n_offs[:, None] * stride_v_n + v_byte_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        vq = ((vp >> (sub * V_BITS)) & V_MASK).to(tl.float32)

        contrib = tl.sum(p[:, None] * (vq * scale_row + zero_row), 0)
        if sub == 0:
            acc0 = contrib
        elif sub == 1:
            acc1 = contrib
        elif sub == 2:
            acc2 = contrib
        else:
            acc3 = contrib

    tl.store(OUT_M_ptr + pid_qh * stride_mo_bh + pid_nb * stride_mo_nb, m_l)
    tl.store(OUT_L_ptr + pid_qh * stride_lo_bh + pid_nb * stride_lo_nb, l_l)

    base = pid_qh * stride_ao_bh + pid_nb * stride_ao_nb
    tl.store(OUT_ACC_ptr + base + (v_byte_offs * V_VPD + 0) * stride_ao_d, acc0)
    if V_VPD > 1:
        tl.store(OUT_ACC_ptr + base + (v_byte_offs * V_VPD + 1) * stride_ao_d, acc1)
    if V_VPD > 2:
        tl.store(OUT_ACC_ptr + base + (v_byte_offs * V_VPD + 2) * stride_ao_d, acc2)
    if V_VPD > 3:
        tl.store(OUT_ACC_ptr + base + (v_byte_offs * V_VPD + 3) * stride_ao_d, acc3)


# ─── Kernel 3b: merge partials ───────────────────────────────────────

@triton.jit
def _fused_decode_merge_kernel(
    IN_M_ptr, IN_L_ptr, IN_ACC_ptr,
    OUT_ptr,          # (BHQ, D)
    stride_mi_bh, stride_mi_nb,
    stride_li_bh, stride_li_nb,
    stride_ai_bh, stride_ai_nb, stride_ai_d,
    stride_o_bh, stride_o_d,
    NB,
    D: tl.constexpr,
    BLOCK_NB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    nb_offs = tl.arange(0, BLOCK_NB)
    nb_mask = nb_offs < NB

    m_all = tl.load(IN_M_ptr + pid * stride_mi_bh + nb_offs * stride_mi_nb,
                    mask=nb_mask, other=float("-inf"))
    m_tot = tl.max(m_all, 0)

    l_all = tl.load(IN_L_ptr + pid * stride_li_bh + nb_offs * stride_li_nb,
                    mask=nb_mask, other=0.0)
    w = tl.exp(m_all - m_tot)  # per-block rescale factors
    l_tot = tl.sum(w * l_all, 0)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(0, NB, BLOCK_NB):
        cols = i + nb_offs
        cmask = cols < NB
        w_blk = tl.exp(
            tl.load(IN_M_ptr + pid * stride_mi_bh + cols * stride_mi_nb,
                    mask=cmask, other=float("-inf")) - m_tot
        )
        a = tl.load(
            IN_ACC_ptr + pid * stride_ai_bh
            + cols[:, None] * stride_ai_nb + d_offs[None, :] * stride_ai_d,
            mask=cmask[:, None] & d_mask[None, :], other=0.0,
        )
        acc += tl.sum(w_blk[:, None] * a, 0)

    tl.store(OUT_ptr + pid * stride_o_bh + d_offs * stride_o_d,
             acc / l_tot, mask=d_mask)


# ─── Python wrappers ─────────────────────────────────────────────────

def _prep(quantizer, query, quantized_key):
    B, Hq, n, d = query.shape
    assert n == 1, "triton kernels target the decode path (n_q == 1)"
    HKV = quantized_key.norms.shape[1]
    assert Hq % HKV == 0
    GQA = Hq // HKV

    q = query.reshape(B * Hq, d).float()
    q_rot = torch.matmul(q, quantizer.mse_quantizer.Pi.T)
    q_sketch = torch.matmul(q, quantizer.S.T)

    flat = lambda t: t.reshape(B * HKV, *t.shape[2:])
    return (
        B, Hq, HKV, d, GQA, q_rot, q_sketch,
        flat(quantized_key.mse_indices),
        flat(quantized_key.qjl_signs),
        flat(quantized_key.norms),
        flat(quantized_key.residual_norms),
    )


def attention_score(quantizer, query: torch.Tensor, quantized_key) -> torch.Tensor:
    """MSE + QJL attention scores via triton kernels (decode, n_q == 1).

    Matches TurboQuantProd._attention_score_torch. Returns (B, Hq, 1, N) fp32.
    """
    B, Hq, HKV, d, GQA, q_rot, q_sketch, mse_packed, qjl_signs, norms, res_norms = _prep(
        quantizer, query, quantized_key
    )

    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    packed_ds = qjl_signs.shape[2]
    bits_eff, vpd = packing_params(quantized_key.mse_bits)
    nclust = 1 << bits_eff

    out = torch.empty(B * Hq, N, device=query.device, dtype=torch.float32)
    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (B * Hq, triton.cdiv(N, BLOCK_N))

    _mse_score_kernel[grid](
        q_rot, mse_packed, norms, quantizer.mse_quantizer.centroids, out,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, HQ=Hq, HKV=HKV, D=d, PACKED_D=packed_d,
        BITS=bits_eff, VPD=vpd, NCLUST=nclust, GQA=GQA,
        BLOCK_N=BLOCK_N,
    )
    _qjl_score_kernel[grid](
        q_sketch, qjl_signs, res_norms, out,
        q_sketch.stride(0), q_sketch.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        res_norms.stride(0), res_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, HQ=Hq, HKV=HKV, D=d, PACKED_DS=packed_ds,
        QJL_SCALE=quantizer.qjl_scale, GQA=GQA,
        BLOCK_N=BLOCK_N,
    )
    return out.reshape(B, Hq, 1, N)


def fused_decode(quantizer, query: torch.Tensor, quantized_key, value_quantized,
                 sm_scale: float, group_size: int = 32) -> torch.Tensor:
    """Split-KV fused decode: scores + online softmax + value dequant.

    query: (B, Hq, 1, d). Returns (B, Hq, 1, d) fp32.
    """
    B, Hq, HKV, d, GQA, q_rot, q_sketch, mse_packed, qjl_signs, norms, res_norms = _prep(
        quantizer, query, quantized_key
    )

    v_data = value_quantized.data.reshape(B * HKV, *value_quantized.data.shape[2:])
    v_scales = value_quantized.scales.reshape(B * HKV, *value_quantized.scales.shape[2:])
    v_zeros = value_quantized.zeros.reshape(B * HKV, *value_quantized.zeros.shape[2:])

    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    packed_ds = qjl_signs.shape[2]
    packed_dv = v_data.shape[2]
    bits_eff, vpd = packing_params(quantized_key.mse_bits)
    v_bits_eff, v_vpd = packing_params(value_quantized.bits)
    nclust = 1 << bits_eff

    assert group_size % v_vpd == 0, "group_size must be a multiple of the packing width"
    BLOCK_N = 64  # bounded to keep the tile working set inside Ascend's UB
    NB = triton.cdiv(N, BLOCK_N)
    BH = B * Hq

    part_m = torch.empty(BH, NB, device=query.device, dtype=torch.float32)
    part_l = torch.empty(BH, NB, device=query.device, dtype=torch.float32)
    part_acc = torch.empty(BH, NB, d, device=query.device, dtype=torch.float32)
    out = torch.empty(BH, d, device=query.device, dtype=torch.float32)

    _fused_decode_partial_kernel[(BH, NB)](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms,
        quantizer.mse_quantizer.centroids,
        v_data, v_scales, v_zeros,
        part_m, part_l, part_acc,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        part_m.stride(0), part_m.stride(1),
        part_l.stride(0), part_l.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        N=N, HQ=Hq, HKV=HKV, D=d,
        PACKED_D=packed_d, PACKED_DS=packed_ds, PACKED_DV=packed_dv,
        GROUP_SIZE=group_size, N_GROUPS=d // group_size,
        BITS=bits_eff, VPD=vpd, NCLUST=nclust,
        V_BITS=v_bits_eff, V_VPD=v_vpd,
        QJL_SCALE=quantizer.qjl_scale, SM_SCALE=sm_scale,
        GQA=GQA, BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    BLOCK_NB = min(64, triton.next_power_of_2(NB))  # bounded: UB is 192KB
    _fused_decode_merge_kernel[(BH,)](
        part_m, part_l, part_acc, out,
        part_m.stride(0), part_m.stride(1),
        part_l.stride(0), part_l.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        out.stride(0), out.stride(1),
        NB=NB, D=d,
        BLOCK_NB=BLOCK_NB, BLOCK_D=triton.next_power_of_2(d),
        num_warps=4,
    )
    return out.reshape(B, Hq, 1, d)
