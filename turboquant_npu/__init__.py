"""TurboQuant-NPU: TurboQuant KV cache quantization on Ascend NPU.

Backends: torch (torch_npu) and triton (triton-ascend).
Derived from 0xSero/turboquant (GPLv3).
"""

from turboquant_npu._device import resolve_device
from turboquant_npu.codebook import compute_lloyd_max_codebook, get_codebook
from turboquant_npu.quantizer import (
    MSEQuantized,
    ProdQuantized,
    TurboQuantMSE,
    TurboQuantProd,
)
from turboquant_npu.kv_cache import (
    TurboQuantKVCache,
    ValueQuantized,
    quantize_values,
    dequantize_values,
)
from turboquant_npu.store import CompressedKVStore
from turboquant_npu.score import compute_hybrid_attention

__version__ = "0.1.0"
