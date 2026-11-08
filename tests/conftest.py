import pytest
import torch

from turboquant_npu._device import resolve_device


@pytest.fixture(params=["cpu", "accel"])
def device(request):
    """Run tests on cpu and (if available) npu/cuda."""
    if request.param == "accel":
        accel = resolve_device()
        if accel.type == "cpu":
            pytest.skip("no accelerator (npu/cuda) available")
        yield accel
    else:
        yield torch.device("cpu")
