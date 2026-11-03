import torch


def scale(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Returns alpha * x, computed by our own CUDA kernel.

    x must be a complex64 tensor on a CUDA device. Non-contiguous inputs are copied to a
    contiguous layout first. The result is a new contiguous tensor.
    """
    return torch.ops.spectral.scale.default(x, alpha)
