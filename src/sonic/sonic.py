"""SONIC: Spectral Oriented Neural Invariant Convolutions.

Reference implementation for the paper's supplementary material.
Checkpoint-compatible with the production code: state_dict keys
and parameter shapes match exactly, so .pth files load correctly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    ModeDropout, normalize_input, pad_input,
    get_freq_grids_2d, get_freq_grids_3d,
    unit_complex, init_direction_angles, angles_to_unit_vectors,
)


class Sonic(nn.Module):
    """Spectral Oriented Neural Invariant Convolution operator.

    Each mode m has a direction v_m, complex pole a_m, speed s_m,
    transverse decay tau_m, DC gain, and Butterworth bandwidth b_m.
    The transfer function is::

        T_m(w) = dc_m * conj(D_m) / |D_m|^2
        D_m    = (j * s_m * (v_m . w) - a_m + tau_m * |w_perp|^2)
               * (1 + (|w|/w_c,m)^{2n})

    The full operator in frequency space is::

        Y = C @ diag(T) @ B @ X

    where B (M x C) mixes input channels into M modes, T applies per-mode
    filtering, and C (K x M) mixes modes into K output channels.

    Args:
        dim: Spatial dimensionality (2 or 3).
        in_channels: Number of input channels.
        num_hidden: Number of output channels.
        M_modes: Number of spectral modes.
        normalize_input: Per-sample normalisation before FFT.
        dx, dy, dz: Grid spacing in physical units (1.0 if unknown).
        blockdiag_per_channel: Use block-diagonal mixer mask on B.
        dropout_p: Mode dropout probability during training.
        dtype: Computation dtype.
        fix_v: Fix directional vectors (register as buffers, not parameters).
        v_noise: Initial noise magnitude for learned directions.
        bandlimit: Enable learnable Butterworth bandlimiting per mode.
        bandlimit_order: Order of the Butterworth filter (higher = sharper).
        set_beta_zero: Initialise beta (imaginary part of poles) to zero.
    """

    def __init__(
        self,
        dim: int = 2,
        in_channels: int = 3,
        num_hidden: int = 64,
        M_modes: int = 12,
        normalize_input: bool = True,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        blockdiag_per_channel: bool = False,
        dropout_p: float = 0.0,
        dtype: torch.dtype = torch.float32,
        fix_v: bool = False,
        v_noise: float = 0.05,
        bandlimit: bool = True,
        bandlimit_order: int = 4,
        set_beta_zero: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.C = int(in_channels)
        self.K = int(num_hidden)
        self.M = int(M_modes)
        self.dim = int(dim)
        self.normalize_input = bool(normalize_input)
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.dtype = dtype
        self.fix_v = bool(fix_v)
        self.bandlimit = bool(bandlimit)
        self.bandlimit_order = int(bandlimit_order)
        self.mode_dropout = ModeDropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # --- Direction vectors (spherical parameterisation) ---
        theta0, phi0 = init_direction_angles(self.dim, self.M, fix_v, v_noise, dtype)
        if not fix_v:
            self.theta_v = nn.Parameter(theta0)
            if self.dim == 3:
                self.phi_v = nn.Parameter(phi0)
        else:
            self.register_buffer("theta_v", theta0)
            if self.dim == 3:
                self.register_buffer("phi_v", phi0)

        # --- Complex channel mixers B (M, C) and C (K, M) ---
        C_re, C_im = unit_complex((self.K, self.M), norm_dim=0, dtype=dtype)
        self.C_re = nn.Parameter(C_re)
        self.C_im = nn.Parameter(C_im)

        B_re, B_im = unit_complex((self.M, self.C), norm_dim=1, dtype=dtype)
        self.B_re = nn.Parameter(B_re)
        self.B_im = nn.Parameter(B_im)

        # --- Block-diagonal mixer mask ---
        if blockdiag_per_channel:
            groups = torch.tensor_split(torch.arange(self.M), self.C)
            mask = torch.zeros(self.M, self.C)
            for c, g in enumerate(groups):
                mask[g, c] = 1.0
            self.register_buffer("Bmask", mask)
        else:
            self.register_buffer("Bmask", None)

        # --- Spectral parameters (simple defaults; checkpoints overwrite) ---
        self.alpha_raw = nn.Parameter(torch.zeros(self.M, dtype=dtype))
        self.log_dc_gain = nn.Parameter(torch.zeros(self.M, dtype=dtype))
        self.log_tau = nn.Parameter(torch.zeros(self.M, dtype=dtype))
        self.log_scale = nn.Parameter(torch.zeros(self.M, dtype=dtype))
        self.beta = nn.Parameter(
            torch.zeros(self.M, dtype=dtype) if set_beta_zero
            else 0.1 * torch.randn(self.M, dtype=dtype)
        )

        # --- Learnable Butterworth bandwidth ---
        if self.bandlimit:
            self.log_bandwidth = nn.Parameter(torch.zeros(self.M, dtype=dtype))

    # ------------------------------------------------------------------ #
    #  Checkpoint migration
    # ------------------------------------------------------------------ #

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Migrate old checkpoints that used log_alpha to alpha_raw + log_dc_gain."""
        old_key = prefix + "log_alpha"
        if old_key in state_dict:
            log_alpha = state_dict.pop(old_key)
            alpha = torch.exp(log_alpha)
            state_dict[prefix + "alpha_raw"] = torch.log(torch.exp(alpha) - 1.0)
            state_dict[prefix + "log_dc_gain"] = log_alpha  # log(alpha) = old DC norm
        # Migrate old checkpoints missing log_bandwidth
        bw_key = prefix + "log_bandwidth"
        if bw_key not in state_dict and hasattr(self, 'log_bandwidth'):
            state_dict[bw_key] = torch.zeros(self.M, dtype=self.dtype)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------ #
    #  Parameter unpacking
    # ------------------------------------------------------------------ #

    def _get_params(self):
        """Unpack learnable parameters into tensors used by forward."""
        cdtype = torch.complex64 if self.dtype != torch.float64 else torch.complex128

        # Pole:  a = -softplus(alpha_raw) + j*beta   (real part always negative)
        a = torch.complex(
            -F.softplus(self.alpha_raw).float(), self.beta.float()
        ).to(cdtype)

        s = torch.exp(self.log_scale)
        tau = torch.exp(self.log_tau)
        dc_gain = torch.exp(self.log_dc_gain)

        # Direction unit vectors from spherical angles
        v = angles_to_unit_vectors(
            self.dim, self.theta_v, getattr(self, 'phi_v', None), self.dtype)

        # Complex mixers
        B = torch.complex(self.B_re.float(), self.B_im.float()).to(cdtype)
        C = torch.complex(self.C_re.float(), self.C_im.float()).to(cdtype)
        if self.Bmask is not None:
            B = B * self.Bmask.to(dtype=B.dtype, device=B.device)

        bandwidth = torch.sigmoid(self.log_bandwidth) if self.bandlimit else None

        return a, s, tau, v, B, C, dc_gain, bandwidth

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, pad_linear: bool = False, **kwargs):
        """Forward pass of the Sonic operator.

        Args:
            x: Input ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
            pad_linear: Zero-pad spatially to avoid FFT wrap-around.
            **kwargs: Override ``dx``, ``dy`` [, ``dz``] for resolution-aware
                inference.

        Returns:
            Output ``(B, K, H, W)`` or ``(B, K, D, H, W)``.
        """
        if self.normalize_input:
            x = normalize_input(self.dim, x)
        x, D, H, W = pad_input(self.dim, x, pad_linear)

        a, s, tau, v, B_mix, C_mix, dc_gain, bandwidth = self._get_params()
        cdtype = B_mix.dtype

        # 1. FFT
        spatial_dims = (-2, -1) if self.dim == 2 else (-3, -2, -1)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            Xf = torch.fft.rfftn(x.to(self.dtype), dim=spatial_dims)

        # 2. Frequency grids (use padded spatial dims, not Xf dims)
        dx = float(kwargs.get("dx", self.dx))
        dy = float(kwargs.get("dy", self.dy))
        if self.dim == 2:
            Hp, Wp = x.shape[-2], x.shape[-1]
            OX, OY = get_freq_grids_2d(Hp, Wp, dx, dy, x.device, self.dtype)
            omega = torch.stack([OX, OY], dim=0)                # (2, Hp, Wq)
        else:
            dz = float(kwargs.get("dz", self.dz))
            Dp, Hp, Wp = x.shape[-3], x.shape[-2], x.shape[-1]
            OZ, OY, OX = get_freq_grids_3d(
                Dp, Hp, Wp, dz, dx, dy, x.device, self.dtype)
            omega = torch.stack([OX, OY, OZ], dim=0)            # (3, Dp, Hp, Wq)

        # 3. Physical-space direction normalisation
        if self.dim == 2:
            d_spacing = torch.tensor([dx, dy], device=v.device, dtype=v.dtype)
        else:
            d_spacing = torch.tensor([dx, dy, dz], device=v.device, dtype=v.dtype)
        v_phys = v / (d_spacing.unsqueeze(1) + 1e-8)            # (ndim, M)
        v_phys = v_phys / v_phys.norm(dim=0, keepdim=True).clamp_min(1e-8)

        # 4. Transfer function T(omega) per mode
        dot = torch.einsum('dm, d... -> m...', v_phys, omega)   # (M, *spatial)
        wn2 = (omega * omega).sum(dim=0)                        # (*spatial)
        wperp = (wn2 - dot * dot).clamp_min(0.0)                # (M, *spatial)

        ones = (1,) * (omega.dim() - 1)  # broadcast shape for (M,) params
        s_   = s.reshape(-1, *ones)
        tau_ = tau.reshape(-1, *ones)
        a_   = a.reshape(-1, *ones)
        dc_  = dc_gain.reshape(-1, *ones).clamp_min(1e-8)

        denom = 1j * s_ * dot - a_ + tau_ * wperp

        # Per-mode Butterworth anti-aliasing (absorbed into denominator)
        if bandwidth is not None:
            max_d = max(dx, dy) if self.dim == 2 else max(dx, dy, dz)
            nyq_sq = (np.pi / max(max_d, 1e-8)) ** 2
            bw = bandwidth.reshape(-1, *ones)
            ratio_sq = (wn2 / nyq_sq) / bw.square().clamp_min(1e-12)
            denom = denom * (1.0 + ratio_sq.pow(self.bandlimit_order))

        mag_sq = (denom.real.square() + denom.imag.square()).clamp_min(1e-8)
        T = dc_ * denom.conj() / mag_sq                        # (M, *spatial)

        # Soft clamp resonance peaks (tanh saturation at magnitude 50)
        T_mag = T.abs().clamp_min(1e-8)
        T = T * (50.0 * torch.tanh(T_mag / 50.0) / T_mag)

        # 5. Spectral mixing
        U  = torch.einsum('mc, bc... -> bm...', B_mix, Xf.to(cdtype))
        V  = self.mode_dropout(U * T.unsqueeze(0))
        Yf = torch.einsum('km, bm... -> bk...', C_mix, V)

        # 6. Inverse FFT and crop
        pad_shape = x.shape[2:]
        Yf[..., 0].imag.zero_()
        if pad_shape[-1] % 2 == 0:
            Yf[..., -1].imag.zero_()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            y = torch.fft.irfftn(Yf, s=pad_shape, dim=spatial_dims)

        if self.dim == 2:
            return y[..., :H, :W].contiguous()
        return y[..., :D, :H, :W].contiguous()
