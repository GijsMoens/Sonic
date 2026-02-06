"""Optional VkFFT backend for GPU-accelerated FFT via CuPy/pyvkfft.

To use, install ``cupy`` and ``pyvkfft``, then::

    from sonic.vkfft import VkFFTBackend
    layer = Sonic(...)
    layer.fft_backend = VkFFTBackend()
"""

import torch
import cupy
from torch.utils.dlpack import to_dlpack
from pyvkfft.fft import rfftn as vk_rfftn, irfftn as vk_irfftn


def _axes_from_dim(tensor: torch.Tensor, dim):
    if dim is None:
        return None
    return tuple(d if d >= 0 else tensor.ndim + d for d in dim)


class VkRFFTn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.shape = tensor.shape
        ctx.dim = dim
        axes = _axes_from_dim(tensor, dim)
        cp_in = cupy.from_dlpack(to_dlpack(tensor.contiguous()))
        try:
            cp_out = vk_rfftn(cp_in, axes=axes)
        except Exception:
            return torch.fft.rfftn(tensor, dim=dim)
        return torch.from_dlpack(cp_out)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        dim = ctx.dim
        axes = _axes_from_dim(grad_output, dim)
        s = [shape[d] for d in dim]
        cp_in = cupy.from_dlpack(to_dlpack(grad_output.contiguous()))
        try:
            cp_out = vk_irfftn(cp_in, axes=axes, s=s)
        except Exception:
            return torch.fft.irfftn(grad_output, s=s, dim=dim), None
        return torch.from_dlpack(cp_out), None


class VkIRFFTn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, s, dim):
        ctx.shape = tensor.shape
        ctx.dim = dim
        axes = _axes_from_dim(tensor, dim)
        if s is not None:
            default_last = 2 * (tensor.shape[axes[-1]] - 1)
            default_s = [tensor.shape[a] for a in axes[:-1]] + [default_last]
            if tuple(s) != tuple(default_s):
                return torch.fft.irfftn(tensor, s=s, dim=dim)
        cp_in = cupy.from_dlpack(to_dlpack(tensor.contiguous()))
        try:
            cp_out = vk_irfftn(cp_in, axes=axes, s=s)
        except Exception:
            return torch.fft.irfftn(tensor, s=s, dim=dim)
        return torch.from_dlpack(cp_out)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        axes = _axes_from_dim(grad_output, dim)
        cp_in = cupy.from_dlpack(to_dlpack(grad_output.contiguous()))
        try:
            cp_out = vk_rfftn(cp_in, axes=axes)
        except Exception:
            return torch.fft.rfftn(grad_output, dim=dim), None, None
        return torch.from_dlpack(cp_out), None, None


class VkFFTBackend:
    """Drop-in FFT backend using VkFFT via CuPy for faster transforms."""

    @staticmethod
    def rfftn(tensor: torch.Tensor, dim=(-2, -1)):
        return VkRFFTn.apply(tensor, dim)

    @staticmethod
    def irfftn(tensor: torch.Tensor, s=None, dim=(-2, -1)):
        return VkIRFFTn.apply(tensor, s, dim)
