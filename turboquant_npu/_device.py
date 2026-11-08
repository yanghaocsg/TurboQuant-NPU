"""Device resolution for TurboQuant-NPU.

Priority: npu (Ascend via torch_npu) > cuda > cpu.

The heavy lifting (rotation QR, codebook integration) always runs on CPU for
reproducibility; results are moved to the target device.
"""

from __future__ import annotations

import torch

_NPU_CHECKED = False
_NPU_OK = False


def _try_import_torch_npu() -> bool:
    """Import torch_npu once and report whether NPU is usable."""
    global _NPU_CHECKED, _NPU_OK
    if not _NPU_CHECKED:
        _NPU_CHECKED = True
        try:
            import torch_npu  # noqa: F401

            _NPU_OK = torch.npu.is_available() if hasattr(torch, "npu") else False
        except Exception:
            _NPU_OK = False
    return _NPU_OK


def resolve_device(device=None) -> torch.device:
    """Return the best available device, or the requested one.

    Accepts torch.device / str / None. With None, prefers npu > cuda > cpu.
    """
    if device is not None:
        return torch.device(device)
    if _try_import_torch_npu():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize(device: torch.device):
    """Synchronize the given device (no-op on cpu)."""
    if device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
