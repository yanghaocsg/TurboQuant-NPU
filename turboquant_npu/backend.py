"""Backend dispatch for TurboQuant-NPU.

"torch"  : pure PyTorch ops (works on npu via torch_npu, cuda, cpu)
"triton" : triton-ascend fused kernels (Phase 2); falls back to torch
           automatically if triton-ascend is unavailable or fails to compile.
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger("turboquant_npu.backend")

_AVAILABLE_BACKENDS = ("torch",)
_TRITON_PROBED = False


def _probe_triton() -> bool:
    """Probe triton-ascend availability once per process."""
    global _TRITON_PROBED, _AVAILABLE_BACKENDS
    if _TRITON_PROBED:
        return "triton" in _AVAILABLE_BACKENDS
    _TRITON_PROBED = True
    try:
        import triton  # noqa: F401
        import triton_ascend  # noqa: F401

        from turboquant_npu import triton_kernels  # noqa: F401

        _AVAILABLE_BACKENDS = ("torch", "triton")
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("triton backend unavailable (%s); using torch", e)
    return "triton" in _AVAILABLE_BACKENDS


def available_backends() -> tuple[str, ...]:
    _probe_triton()
    return _AVAILABLE_BACKENDS


def select_backend(requested: str) -> str:
    """Resolve a requested backend, degrading gracefully."""
    if requested == "torch":
        return "torch"
    if requested == "triton":
        return "triton" if _probe_triton() else "torch"
    raise ValueError(f"unknown backend {requested!r}")


# ── triton entry points (wired in Phase 2) ─────────────────────────────

def triton_attention_score(quantizer, query, quantized_key):
    """Score computation via triton-ascend kernels; falls back to torch."""
    try:
        from turboquant_npu.triton_kernels import attention_score as _tq_score

        return _tq_score(quantizer, query, quantized_key)
    except Exception as e:
        logger.warning("triton score failed (%s); falling back to torch", e)
        return quantizer._attention_score_torch(query, quantized_key)


def triton_fused_decode(quantizer, query, quantized_key, value_quantized,
                        sm_scale, group_size=32):
    """Fused decode attention via triton-ascend; falls back to torch path."""
    try:
        from turboquant_npu.triton_kernels import fused_decode as _tq_decode

        return _tq_decode(quantizer, query, quantized_key, value_quantized,
                          sm_scale, group_size)
    except Exception as e:
        logger.warning("triton fused decode failed (%s); torch fallback", e)
        from turboquant_npu.kv_cache import dequantize_values
        from turboquant_npu.score import _matmul_attend
        import math

        k = quantizer.dequantize(quantized_key)
        v = dequantize_values(value_quantized, group_size)
        return _matmul_attend(query.unsqueeze(0), k, v, 1, sm_scale).squeeze(0)
