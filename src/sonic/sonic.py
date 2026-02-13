import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ModeDropout, normalize_input, pad_input, get_freq_grids_2d, get_freq_grids_3d


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

        elif self.dim == 3:
            # Fibonacci hemisphere lattice for 3-D direction init
            golden = (1 + np.sqrt(5)) / 2
            indices = torch.arange(self.M, dtype=torch.float32)
            theta = torch.acos(1 - 2 * (indices + 0.5) / (2 * self.M))
            phi = 2 * np.pi * indices / golden
            vx0 = torch.sin(theta) * torch.cos(phi)
            vy0 = torch.sin(theta) * torch.sin(phi)
            vz0 = torch.cos(theta)
            if not self.fix_v:
                vx0 = vx0 + self.v_noise * torch.randn(self.M, dtype=torch.float32)
                vy0 = vy0 + self.v_noise * torch.randn(self.M, dtype=torch.float32)
                vz0 = vz0 + self.v_noise * torch.randn(self.M, dtype=torch.float32)
                v = torch.stack([vx0, vy0, vz0], dim=0)
                v = v / (v.norm(dim=0, keepdim=True) + 1e-8)
                self.vx = nn.Parameter(v[0].to(self.dtype))
                self.vy = nn.Parameter(v[1].to(self.dtype))
                self.vz = nn.Parameter(v[2].to(self.dtype))
            else:
                self.register_buffer("vx", vx0.to(self.dtype))
                self.register_buffer("vy", vy0.to(self.dtype))
                self.register_buffer("vz", vz0.to(self.dtype))

        # --- mixers (unit-complex rows/cols) ---
        def _unit_complex(shape, dim_to_normalize):
            real = torch.randn(*shape, dtype=torch.float32)
            imag = torch.randn(*shape, dtype=torch.float32)
            denom = (
                torch.sqrt((real**2 + imag**2).sum(dim=dim_to_normalize, keepdim=True))
                .clamp_min(1e-12)
            )
            return (real / denom).to(self.dtype), (imag / denom).to(self.dtype)

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

        # Frequency grid cache (keyed by grid shape + spacing + device + dtype)
        self._freq_cache: dict = {}

    # ------------------------------------------------------------------
    #  Cached frequency grid helpers
    # ------------------------------------------------------------------

    def _get_freq_grids_2d_cached(self, H, W, dx, dy, device, dtype):
        key = (H, W, dx, dy, device, dtype)
        cached = self._freq_cache.get(key)
        if cached is not None:
            return cached
        grids = get_freq_grids_2d(H, W, dx, dy, device, dtype)
        self._freq_cache[key] = grids
        return grids

    def _get_freq_grids_3d_cached(self, D, H, W, dz, dx, dy, device, dtype):
        key = (D, H, W, dz, dx, dy, device, dtype)
        cached = self._freq_cache.get(key)
        if cached is not None:
            return cached
        grids = get_freq_grids_3d(D, H, W, dz, dx, dy, device, dtype)
        self._freq_cache[key] = grids
        return grids

    def _get_params(self, real_mixers=False):
        complex_dtype = (
            torch.complex64
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.complex128
        )

        a_re = -F.softplus(self.alpha)
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
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
            pad_linear: Pad input to avoid FFT wrap-around artifacts.
            block_h: Optional block height for memory-efficient processing.
            **kwargs: Override ``dx``, ``dy`` (and ``dz`` for 3-D) for
                resolution-aware inference.

        Returns:
            Output tensor of shape ``(B, K, H, W)`` or ``(B, K, D, H, W)``.
        """
        if self.normalize_input:
            x = normalize_input(self.dim, x)

        x, D, H, W = pad_input(self.dim, x, pad_linear)

        a, s, tau, v, B_mixer, C_mixer = self._get_params()

        if self.dim == 2:
            dx_eff = float(kwargs.get("dx", self.dx))
            dy_eff = float(kwargs.get("dy", self.dy))

            return self._forward_2d(x, H, W, dx_eff, dy_eff, block_h,
                                    a, s, tau, v, B_mixer, C_mixer)

        # dim == 3
        dx_eff = kwargs.get("dx", self.dx)
        dy_eff = kwargs.get("dy", self.dy)
        dz_eff = kwargs.get("dz", self.dz)

        # Support per-batch resolution tensors
        if not isinstance(dx_eff, torch.Tensor):
            dx_eff = torch.tensor([float(dx_eff)] * x.shape[0], device=x.device, dtype=self.dtype)
        if not isinstance(dy_eff, torch.Tensor):
            dy_eff = torch.tensor([float(dy_eff)] * x.shape[0], device=x.device, dtype=self.dtype)
        if not isinstance(dz_eff, torch.Tensor):
            dz_eff = torch.tensor([float(dz_eff)] * x.shape[0], device=x.device, dtype=self.dtype)

        return self._forward_3d(x, D, H, W, dz_eff, dx_eff, dy_eff, block_h,
                                a, s, tau, v, B_mixer, C_mixer)

    def _forward_2d(self, x, H, W, dx_eff, dy_eff, block_h, a, s, tau, v, B_mixer, C_mixer):
        Hp, Wp = x.shape[-2:]
        OX, OY = self._get_freq_grids_2d_cached(Hp, Wp, dx_eff, dy_eff, x.device, self.dtype)

        complex_dtype = (
            torch.complex64
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.complex128
        )

        with torch.amp.autocast(device_type="cuda", enabled=False):
            Xf = torch.fft.rfftn(x.to(self.dtype), dim=(-2, -1))

        Bsz, C, Wq = Xf.shape[0], Xf.shape[1], Xf.shape[-1]
        K, M = int(self.K), int(v[0].numel())
        B_tc = B_mixer.transpose(0, 1).contiguous() if B_mixer.shape == (M, C) else B_mixer.contiguous()
        # Precompute transpose for matmul: B_tc is (C, M), B_tc_t is (M, C)
        B_tc_t = B_tc.t()

        # Physical-space direction awareness
        scale = torch.tensor([dx_eff, dy_eff], device=v.device, dtype=v.dtype)[:, None]
        v_phys = v / (scale + 1e-8)
        v_phys = v_phys / (v_phys.norm(dim=0, keepdim=True) + 1e-8)

        # Reshape params as (M, 1, 1) so T is computed directly in (M, H, Wq)
        v_vx = v_phys[0].reshape(M, 1, 1)
        v_vy = v_phys[1].reshape(M, 1, 1)
        s_s = s.reshape(M, 1, 1)
        t_t = tau.reshape(M, 1, 1)
        a_a = a.reshape(M, 1, 1)

        # Precompute wn2 = OX^2 + OY^2 once (full grid, no copy needed for slabs)
        wn2_full = OX.square() + OY.square()

        # Process in horizontal slabs for memory efficiency
        Yf_total = x.new_zeros((Bsz, K, Hp, Wq), dtype=complex_dtype)
        bh = Hp if (block_h is None or block_h >= Hp) else int(block_h)

        for y0 in range(0, Hp, bh):
            y1 = min(Hp, y0 + bh)
            h_slab = y1 - y0

            # Frequency grid slices: (h_slab, Wq) — views, no copy
            OX_sl = OX[y0:y1, :]
            OY_sl = OY[y0:y1, :]

            # Compute T directly in (M, h_slab, Wq) layout — no permute needed
            dot = OX_sl * v_vx + OY_sl * v_vy          # (M, h_slab, Wq)
            wn2_sl = wn2_full[y0:y1, :]                 # (h_slab, Wq) view
            wperp = (wn2_sl - dot**2).clamp_min(0.0)    # (M, h_slab, Wq)

            denom = 1j * (s_s * dot) - a_a + t_t * wperp
            magn_dn = (denom.real.square() + denom.imag.square()).clamp_min(1e-8)
            T = denom.conj() / magn_dn

            # Normalise by DC gain |T(0)| = 1/|a| (resolution-invariant)
            T = T * a_a.abs().clamp_min(1e-8)
            # Clamp resonance peaks to prevent overflow
            T_mag = (T.real.square() + T.imag.square()).sqrt().clamp_min(1e-8)
            T = T * (T_mag.clamp(max=5.0) / T_mag)
            # T is already (M, h_slab, Wq) — no permute needed

            # --- einsum replacements with matmul ---
            # Xf_: (Bsz, C, h_slab, Wq) → flatten spatial → (Bsz, C, h_slab*Wq)
            Xf_ = Xf[:, :, y0:y1, :]
            hw = h_slab * Wq
            Xf_flat = Xf_.reshape(Bsz, C, hw)
            # U = B_tc_t @ Xf_flat → (Bsz, M, hw) → (Bsz, M, h_slab, Wq)
            U = torch.matmul(B_tc_t.unsqueeze(0), Xf_flat).reshape(Bsz, M, h_slab, Wq)
            V = self.mode_dropout(U * T.unsqueeze(0))
            # Y = C_mixer @ V_flat → (Bsz, K, hw) → (Bsz, K, h_slab, Wq)
            V_flat = V.reshape(Bsz, M, hw)
            Yf_total[:, :, y0:y1, :] = torch.matmul(C_mixer.unsqueeze(0), V_flat).reshape(Bsz, K, h_slab, Wq)

        # Ensure real output
        if Wq > 0:
            Yf_total[..., 0].imag.zero_()
        if Wp % 2 == 0:
            Yf_total[..., -1].imag.zero_()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            y_spatial = torch.fft.irfftn(Yf_total, s=(Hp, Wp), dim=(-2, -1))
        return y_spatial[..., :H, :W].contiguous()

    def _forward_3d(self, x_work, D, H, W, dz_eff, dx_eff, dy_eff, block_h,
                    a, s, tau, v, B_mixer, C_mixer, k_chunk: int = 8):
        Dp, Hp, Wp = x_work.shape[-3:]
        Bsz = x_work.shape[0]
        complex_dtype = (
            torch.complex64
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.complex128
        )

        with torch.amp.autocast(device_type="cuda", enabled=False):
            Xf = torch.fft.rfftn(x_work.to(self.dtype), dim=(-3, -2, -1))

        C_ch, Wq = Xf.shape[1], Xf.shape[-1]
        K, M = int(self.K), int(v[0].numel())
        B_tc = (B_mixer.transpose(0, 1).contiguous()
                if B_mixer.shape == (M, C_ch) else B_mixer.contiguous())
        # Precompute transpose for matmul: B_tc is (C, M), B_tc_t is (M, C)
        B_tc_t = B_tc.t()

        # Per-batch physical-space directions
        phys_scale = torch.stack([dx_eff, dy_eff, dz_eff], dim=1)
        v_phys = v.unsqueeze(0) / (phys_scale.unsqueeze(-1) + 1e-8)
        v_phys = v_phys / (v_phys.norm(dim=1, keepdim=True) + 1e-8)

        # Per-batch frequency grids (cached)
        OZ, OY, OX = self._get_freq_grids_3d_cached(
            Dp, Hp, Wp,
            dz_eff[0].item(), dx_eff[0].item(), dy_eff[0].item(),
            x_work.device, self.dtype,
        )
        OX = OX.unsqueeze(0).expand(Bsz, -1, -1, -1)
        OY = OY.unsqueeze(0).expand(Bsz, -1, -1, -1)
        OZ = OZ.unsqueeze(0).expand(Bsz, -1, -1, -1)

        if s.dim() == 1:
            s_s = s[None, None, None, None, :]
        else:
            s_s = s[:, None, None, None, :]
        t_t = tau[None, None, None, None, :]
        a_a = a[None, None, None, None, :]

        # Precompute wn2 = OX^2 + OY^2 + OZ^2 once before slab loop
        wn2_full = OX.square() + OY.square() + OZ.square()

        Yf_total = x_work.new_zeros((Bsz, K, Dp, Hp, Wq), dtype=complex_dtype)
        bh = Hp if (block_h is None or block_h >= Hp) else int(block_h)

        for y0 in range(0, Hp, bh):
            y1 = min(Hp, y0 + bh)
            h_slab = y1 - y0

            OX_sl = OX[:, :, y0:y1, :, None]
            OY_sl = OY[:, :, y0:y1, :, None]
            OZ_sl = OZ[:, :, y0:y1, :, None]

            v_vx = v_phys[:, 0, :].reshape(Bsz, 1, 1, 1, M)
            v_vy = v_phys[:, 1, :].reshape(Bsz, 1, 1, 1, M)
            v_vz = v_phys[:, 2, :].reshape(Bsz, 1, 1, 1, M)

            dot = OX_sl * v_vx + OY_sl * v_vy + OZ_sl * v_vz
            wn2_sl = wn2_full[:, :, y0:y1, :, None]  # view, no copy
            wperp = (wn2_sl - dot**2).clamp_min(0.0)

            denom = 1j * (s_s * dot) - a_a + t_t * wperp
            magn_dn = (denom.real.square() + denom.imag.square()).clamp_min(1e-8)
            T = denom.conj() / magn_dn

            # Normalise by DC gain |T(0)| = 1/|a| (resolution-invariant)
            T = T * a_a.abs().clamp_min(1e-8)
            # Clamp resonance peaks to prevent overflow
            T_mag = (T.real.square() + T.imag.square()).sqrt().clamp_min(1e-8)
            T = T * (T_mag.clamp(max=5.0) / T_mag)

            # --- einsum replacements with matmul ---
            # Xf_sl: (Bsz, C_ch, Dp, h_slab, Wq) → flatten spatial → (Bsz, C_ch, Dp*h_slab*Wq)
            Xf_sl = Xf[:, :, :, y0:y1, :]
            dhw = Dp * h_slab * Wq
            Xf_flat = Xf_sl.reshape(Bsz, C_ch, dhw)
            # U = B_tc_t @ Xf_flat → (Bsz, M, dhw) → (Bsz, M, Dp, h_slab, Wq)
            U = torch.matmul(B_tc_t.unsqueeze(0), Xf_flat).reshape(Bsz, M, Dp, h_slab, Wq)
            V = self.mode_dropout(U * T)

            # C_mixer matmul with k-chunking
            V_flat = V.reshape(Bsz, M, dhw)
            for k0 in range(0, K, k_chunk):
                k1 = min(K, k0 + k_chunk)
                Yf_total[:, k0:k1, :, y0:y1, :] = torch.matmul(
                    C_mixer[k0:k1].unsqueeze(0), V_flat
                ).reshape(Bsz, k1 - k0, Dp, h_slab, Wq)

        if Wq > 0:
            Yf_total[..., 0].imag.zero_()
        if Wp % 2 == 0:
            Yf_total[..., -1].imag.zero_()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            y_spatial = torch.fft.irfftn(Yf_total, s=(Dp, Hp, Wp), dim=(-3, -2, -1))
        return y_spatial[..., :D, :H, :W].contiguous()
