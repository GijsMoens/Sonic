import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeDropout(nn.Module):
    """Spectral mode dropout: randomly zeros entire frequency modes."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        num_modes = x.shape[1]
        mask = torch.bernoulli((1 - self.p) * torch.ones(num_modes, device=x.device))
        mask /= 1 - self.p
        view_shape = [1, num_modes] + [1] * (x.dim() - 2)
        return x * mask.view(view_shape)


def normalize_input(dim: int, x: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean unit-variance normalisation over channel + spatial dims.

    Normalising over all non-batch dimensions (C, H, W) instead of only spatial
    (H, W) prevents near-zero variance in individual channels (common in deeper
    layers with small, sparse post-ReLU feature maps) from amplifying noise and
    gradients, which otherwise leads to NaN during training.
    """
    dims_to_reduce = tuple(range(1, len(x.shape)))
    mean = x.to(torch.float32).mean(dim=dims_to_reduce, keepdim=True)
    var = x.to(torch.float32).var(dim=dims_to_reduce, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5)


def pad_input(dim: int, x: torch.Tensor, pad_linear: bool):
    """Optionally zero-pad the spatial dimensions to avoid FFT wrap-around."""
    if not pad_linear:
        if dim == 2:
            return x, None, x.shape[-2], x.shape[-1]
        return x, x.shape[-3], x.shape[-2], x.shape[-1]
    if dim == 2:
        B, C, H, W = x.shape
        return F.pad(x, (0, W, 0, H)), None, H, W
    if dim == 3:
        B, C, D, H, W = x.shape
        return F.pad(x, (0, W, 0, H, 0, D)), D, H, W


@torch.no_grad()
def get_freq_grids_2d(H: int, W: int, dx_eff: float, dy_eff: float, device, dtype):
    """Construct 2-D frequency grids for the real FFT."""
    kx = torch.fft.rfftfreq(W, d=dx_eff).to(device=device, dtype=dtype) * (2 * np.pi)
    ky = torch.fft.fftfreq(H, d=dy_eff).to(device=device, dtype=dtype) * (2 * np.pi)
    OY, OX = torch.meshgrid(ky, kx, indexing="ij")
    return OX, OY


@torch.no_grad()
def get_freq_grids_3d(D: int, H: int, W: int, dz_eff: float, dx_eff: float,
                      dy_eff: float, device, dtype):
    """Construct 3-D frequency grids for the real FFT."""
    kx = torch.fft.rfftfreq(W, d=dx_eff).to(device=device, dtype=dtype) * (2 * np.pi)
    ky = torch.fft.fftfreq(H, d=dy_eff).to(device=device, dtype=dtype) * (2 * np.pi)
    kz = torch.fft.fftfreq(D, d=dz_eff).to(device=device, dtype=dtype) * (2 * np.pi)
    OZ, OY, OX = torch.meshgrid(kz, ky, kx, indexing="ij")
    return OZ, OY, OX
