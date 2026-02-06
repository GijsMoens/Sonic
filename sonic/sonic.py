import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ModeDropout, normalize_input, pad_input, get_freq_grids_2d


class Sonic(nn.Module):
    """Spectral Oriented Neural Invariant Convolution operator.

    Continuous spectral parameterisation that models convolutional operators
    using a small set of shared, orientation-selective components.  Produces
    smooth responses across the full frequency domain, yielding global
    receptive fields and filters that adapt naturally across resolutions.

    Args:
        dim: Spatial dimensionality (2 or 3).
        in_channels: Number of input channels.
        num_hidden: Number of output channels.
        M_modes: Number of spectral modes.
        normalize_input: Per-sample normalisation before FFT.
        dx, dy, dz: Grid spacing in physical units (leave 1.0 if unknown).
        blockdiag_per_channel: Use block-diagonal mixer mask.
        dropout_p: Mode dropout probability.
        dtype: Computation dtype.
        fix_v: Fix directional vectors (do not learn them).
        v_noise: Initial noise magnitude for learned directions.
        rho: Scaling for imaginary part of alpha (damping oscillations).
        depth_idx, depth_total: Position in network for init scheduling.
        alpha_range: (min, max) for the alpha initialisation schedule.
        tau_range: (high, low) for the tau initialisation schedule.
        per_mode_jitter: Small random perturbation to promote mode diversity.
        set_beta_zero: Initialise beta to zero.
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
        rho: float = 1.0,
        depth_idx: int = 0,
        depth_total: int = 1,
        alpha_range: tuple = (0.2, 2.0),
        tau_range: tuple = (0.20, 0.01),
        per_mode_jitter: float = 0.03,
        set_beta_zero: bool = True,
    ):
        super().__init__()
        self.C = int(in_channels)
        self.dim = int(dim)
        self.M = int(M_modes)
        self.normalize_input = bool(normalize_input)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.dtype = dtype
        self.mode_dropout = ModeDropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fix_v = bool(fix_v)
        self.v_noise = float(v_noise)
        self.fft_backend = torch.fft
        self.K = int(num_hidden)
        self.rho_val = float(rho)

        # --- optional block-diagonal mixer mask ---
        if blockdiag_per_channel:
            groups = torch.tensor_split(torch.arange(self.M), self.C)
            mask = torch.zeros(self.M, self.C)
            for c, g in enumerate(groups):
                mask[g, c] = 1.0
            self.register_buffer("Bmask", mask)
        else:
            self.register_buffer("Bmask", None)

        # --- directions v ---
        if self.dim == 2:
            base = torch.linspace(0, np.pi, steps=self.M + 2, dtype=torch.float32)[1:-1]
            vx0 = torch.cos(base)
            vy0 = torch.sin(base)
            if not self.fix_v:
                vx0 = vx0 + self.v_noise * torch.randn(self.M, dtype=torch.float32)
                vy0 = vy0 + self.v_noise * torch.randn(self.M, dtype=torch.float32)
                v = torch.stack([vx0, vy0], dim=0)
                v = v / (v.norm(dim=0, keepdim=True) + 1e-8)
                self.vx = nn.Parameter(v[0].to(self.dtype))
                self.vy = nn.Parameter(v[1].to(self.dtype))
            else:
                self.register_buffer("vx", vx0.to(self.dtype))
                self.register_buffer("vy", vy0.to(self.dtype))

        # --- mixers (unit-complex rows/cols) ---
        def _unit_complex(shape, dim_to_normalize):
            real = torch.randn(*shape, dtype=torch.float32)
            imag = torch.randn(*shape, dtype=torch.float32)
            denom = (
                torch.sqrt((real**2 + imag**2).sum(dim=dim_to_normalize, keepdim=True))
                .clamp_min(1e-12)
            )
            real = (real / denom).to(self.dtype)
            imag = (imag / denom).to(self.dtype)
            return real, imag

        C_re, C_im = _unit_complex((self.K, self.M), dim_to_normalize=0)
        self.C_re = nn.Parameter(C_re)
        self.C_im = nn.Parameter(C_im)

        B_re, B_im = _unit_complex((self.M, self.C), dim_to_normalize=1)
        if self.Bmask is not None:
            mask = self.Bmask.to(B_re.dtype)
            B_re = B_re * mask
            B_im = B_im * mask
            denom = torch.sqrt((B_re**2 + B_im**2).sum(dim=1, keepdim=True)).clamp_min(1e-12)
            keep = denom > 1e-12
            B_re[keep] /= denom[keep]
            B_im[keep] /= denom[keep]
            dead = ~keep.squeeze(-1)
            if dead.any():
                rr, ii = _unit_complex((dead.sum().item(), self.C), dim_to_normalize=1)
                B_re[dead] = rr
                B_im[dead] = ii
        self.B_re = nn.Parameter(B_re)
        self.B_im = nn.Parameter(B_im)

        # --- init scheduling ---
        def inv_softplus(y):
            y = torch.clamp(torch.as_tensor(y, dtype=torch.float32), min=1e-12)
            return torch.log(torch.expm1(y))

        t = float(depth_idx) / max(1, depth_total - 1) if depth_total > 1 else 0.0

        a_min, a_max = map(float, alpha_range)
        a_min = max(a_min, 1e-6)
        a_fwd = a_min * ((a_max / a_min) ** t)

        tau_hi, tau_lo = float(tau_range[0]), float(tau_range[1])
        tau_lo = max(tau_lo, 1e-6)
        tau_fwd_sched = tau_hi * ((tau_lo / tau_hi) ** t)

        tau_phys = 1.0 / (self.dim * (np.pi**2))
        tau_fwd = 0.9 * tau_fwd_sched + 0.1 * tau_phys

        max_d = max(self.dx, self.dy) if self.dim == 2 else max(self.dx, self.dy, self.dz)
        s_target = 0.25 * (2.0 * np.pi / max(max_d, 1e-12))

        alpha_raw = inv_softplus(a_fwd).repeat(self.M).to(self.dtype)
        tau_raw = inv_softplus(tau_fwd).repeat(self.M).to(self.dtype)
        scale_raw = inv_softplus(s_target).repeat(self.M).to(self.dtype)

        if per_mode_jitter and per_mode_jitter > 0:
            j = float(per_mode_jitter)
            with torch.no_grad():
                alpha_raw += j * torch.randn_like(alpha_raw) * alpha_raw.abs().clamp_min(1e-3)
                tau_raw += j * torch.randn_like(tau_raw) * tau_raw.abs().clamp_min(1e-3)
                scale_raw += j * torch.randn_like(scale_raw) * scale_raw.abs().clamp_min(1e-3)

        self.alpha = nn.Parameter(alpha_raw)
        self.tau_raw = nn.Parameter(tau_raw)
        self.scale_raw = nn.Parameter(scale_raw)

        if set_beta_zero:
            self.beta_raw = nn.Parameter(torch.zeros(self.M, dtype=self.dtype))
        else:
            self.beta_raw = nn.Parameter(0.01 * torch.randn(self.M, dtype=self.dtype))
        self.rho = nn.Parameter(torch.tensor(self.rho_val, dtype=self.dtype))

    def _get_params(self, real_mixers=False):
        complex_dtype = (
            torch.complex64
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.complex128
        )

        a_re = -(F.softplus(self.alpha))
        tau = F.softplus(self.tau_raw)
        s = F.softplus(self.scale_raw)
        a_im = self.rho * torch.tanh(self.beta_raw.float())
        a = torch.complex(a_re, a_im).to(complex_dtype)

        if self.dim == 3:
            v = torch.stack([self.vx.float(), self.vy.float(), self.vz.float()], dim=0)
        else:
            v = torch.stack([self.vx.float(), self.vy.float()], dim=0)
        v = (v / (v.norm(dim=0, keepdim=True) + 1e-6)).to(self.dtype)

        B = torch.complex(
            self.B_re.float(),
            self.B_im.float() if not real_mixers else torch.zeros_like(self.B_im),
        ).to(complex_dtype)
        C_mixer = torch.complex(
            self.C_re.float(),
            self.C_im.float() if not real_mixers else torch.zeros_like(self.C_im),
        ).to(complex_dtype)
        if self.Bmask is not None:
            B = B * self.Bmask.to(dtype=B.dtype, device=B.device)

        return a, s, tau, v, B, C_mixer

    def forward(self, x: torch.Tensor, pad_linear: bool = False, block_h=None, **kwargs):
        """Forward pass of the Sonic operator.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            pad_linear: Pad input to avoid FFT wrap-around artifacts.
            block_h: Optional block height for memory-efficient processing.
            **kwargs: Override ``dx``, ``dy``, ``dz`` for resolution-aware
                inference (e.g. train at low resolution, infer at high).

        Returns:
            Output tensor of shape ``(B, K, H, W)``.
        """
        if self.normalize_input:
            x = normalize_input(self.dim, x)

        x, D, H, W = pad_input(self.dim, x, pad_linear)

        dx_eff = float(kwargs.get("dx", self.dx))
        dy_eff = float(kwargs.get("dy", self.dy))

        a, s, tau, v, B_mixer, C_mixer = self._get_params(real_mixers=False)

        # Adjust scale when resolution changes at inference time
        if ("dx" in kwargs) or ("dy" in kwargs) or ("dz" in kwargs):
            base = max(self.dx, self.dy) if self.dim == 2 else max(self.dx, self.dy, self.dz)
            eff = max(dx_eff, dy_eff) if self.dim == 2 else max(dx_eff, dy_eff, float(kwargs.get("dz", self.dz)))
            s = s * (base / eff)

        if self.dim == 2:
            return self._forward_2d(x, H, W, dx_eff, dy_eff, block_h, a, s, tau, v, B_mixer, C_mixer)
        else:
            raise NotImplementedError("3D forward pass not included in this release.")

    def _forward_2d(self, x, H, W, dx_eff, dy_eff, block_h, a, s, tau, v, B_mixer, C_mixer):
        Hp, Wp = x.shape[-2:]
        OX, OY = get_freq_grids_2d(Hp, Wp, dx_eff, dy_eff, x.device, self.dtype)

        complex_dtype = (
            torch.complex64
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.complex128
        )

        with torch.amp.autocast(device_type="cuda", enabled=False):
            Xf = self.fft_backend.rfftn(x.to(self.dtype), dim=(-2, -1))

        Bsz, C, Wq = Xf.shape[0], Xf.shape[1], Xf.shape[-1]
        K, M = int(self.K), int(v[0].numel())
        B_tc = B_mixer.transpose(0, 1).contiguous() if B_mixer.shape == (M, C) else B_mixer.contiguous()

        # Physical-space direction awareness
        scale = torch.tensor([dx_eff, dy_eff], device=v.device, dtype=v.dtype)[:, None]
        v_phys = v / (scale + 1e-8)
        v_phys = v_phys / (v_phys.norm(dim=0, keepdim=True) + 1e-8)
        v_vx, v_vy = v_phys[0][None, None, :], v_phys[1][None, None, :]

        s_s, t_t, a_a = s[None, None, :], tau[None, None, :], a[None, None, :]

        # Process in horizontal slabs for memory efficiency
        Yf_total = x.new_zeros((Bsz, K, Hp, Wq), dtype=complex_dtype)
        bh = Hp if (block_h is None or block_h >= Hp) else int(block_h)

        for y0 in range(0, Hp, bh):
            y1 = min(Hp, y0 + bh)

            OX_sl, OY_sl = OX[y0:y1, :, None], OY[y0:y1, :, None]

            dot = OX_sl * v_vx + OY_sl * v_vy
            wn2 = OX_sl**2 + OY_sl**2
            wperp = (wn2 - dot**2).clamp_min(0.0)

            denom = 1j * (s_s * dot) - a_a + t_t * wperp
            magn_dn = (denom.real.square() + denom.imag.square()).clamp_min(1e-8)
            T = denom.conj() / magn_dn

            T2 = T.real**2 + T.imag**2
            rms = torch.sqrt(T2.mean(dim=(0, 1), keepdim=True).clamp_min(1e-8))
            T = T / rms
            T = T.permute(2, 0, 1).contiguous()

            # Mixing
            Xf_ = Xf[:, :, y0:y1, :]
            U = torch.einsum("bchq,cm->bmhq", Xf_, B_tc)
            V = U * T.unsqueeze(0)
            V = self.mode_dropout(V)
            Yf = torch.einsum("km,bmhq->bkhq", C_mixer, V)
            Yf_total[:, :, y0:y1, :] = Yf

        # Ensure real output
        if Wq > 0:
            Yf_total[..., 0].imag.zero_()
        if Wp % 2 == 0:
            Yf_total[..., -1].imag.zero_()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            y_spatial = self.fft_backend.irfftn(Yf_total, s=(Hp, Wp), dim=(-2, -1))
        return y_spatial[..., :H, :W].contiguous()
