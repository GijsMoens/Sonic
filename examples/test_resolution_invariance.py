"""Resolution invariance test for SONIC.

Verifies that a Sonic layer produces consistent outputs across resolutions:
a signal defined on the same physical domain, sampled at different grid
spacings, should yield outputs that agree after resampling to a common grid.

Usage::

    python examples/test_resolution_invariance.py
    python examples/test_resolution_invariance.py --modes 24 --channels 8
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sonic import Sonic


# --------------------------------------------------------------------------- #
#  Analytic test signal (sum of oriented sinusoids)
# --------------------------------------------------------------------------- #

def make_signal(C: int, H: int, W: int, dx: float, dy: float):
    """Sample a band-limited analytic signal on a physical grid.

    The signal is a sum of sinusoids whose frequencies are well below the
    Nyquist limit of the *coarsest* grid tested, so aliasing does not
    confound the comparison.
    """
    ys = torch.arange(H, dtype=torch.float32) * dy
    xs = torch.arange(W, dtype=torch.float32) * dx
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    # Exact multiples of 1/L (L = N*dx = physical domain size) so the signal
    # is periodic on the grid → zero spectral leakage.
    # All |f| well below Nyquist = 1/(2*dx) = 0.5 cycles/unit for dx=1.
    L = H * dy  # physical domain extent (same for both dims when dx=dy)
    freqs = [(2/L, 1/L), (1/L, 3/L), (3/L, 2/L), (1/L, -1/L)]
    signal = torch.zeros(1, C, H, W)
    for c in range(C):
        acc = torch.zeros_like(X)
        for i, (fx, fy) in enumerate(freqs):
            phase = (c + i) * 0.4
            acc = acc + torch.sin(2 * np.pi * (fx * X + fy * Y) + phase)
        signal[0, c] = acc / len(freqs)
    return signal


# --------------------------------------------------------------------------- #
#  Main test
# --------------------------------------------------------------------------- #

def spectral_downsample(y_hr: torch.Tensor, target_h: int, target_w: int):
    """Downsample by truncating Fourier coefficients (exact for band-limited)."""
    Yf = torch.fft.rfftn(y_hr.float(), dim=(-2, -1))

    # Keep only the low-frequency bins that fit the target grid
    Wq_lo = target_w // 2 + 1
    Yf_lo = Yf.new_zeros(*Yf.shape[:-2], target_h, Wq_lo)

    # Copy the shared low-frequency quadrants
    h_half = target_h // 2
    Yf_lo[..., :h_half, :Wq_lo] = Yf[..., :h_half, :Wq_lo]
    Yf_lo[..., -h_half:, :Wq_lo] = Yf[..., -h_half:, :Wq_lo]

    # Scale to preserve energy under the new grid size
    scale = (target_h * target_w) / (y_hr.shape[-2] * y_hr.shape[-1])
    return torch.fft.irfftn(Yf_lo * scale, s=(target_h, target_w), dim=(-2, -1))


@torch.no_grad()
def test_resolution_invariance(
    modes: int = 12,
    channels: int = 4,
    num_hidden: int = 16,
    base_size: int = 32,
    scales: tuple = (1, 2, 4),
    device: str = "cpu",
):
    """Run resolution invariance test.

    Creates a Sonic layer, evaluates the same physical signal at multiple
    resolutions, and compares the outputs via spectral downsampling
    (Fourier truncation — exact for band-limited signals).
    """
    dx_base, dy_base = 1.0, 1.0

    layer = Sonic(
        dim=2,
        in_channels=channels,
        num_hidden=num_hidden,
        M_modes=modes,
        normalize_input=False,
        dx=dx_base,
        dy=dy_base,
    ).to(device).eval()

    # Reference output at base resolution
    x_base = make_signal(channels, base_size, base_size, dx_base, dy_base).to(device)
    y_base = layer(x_base)

    print(f"{'Scale':>6}  {'Grid':>10}  {'dx':>5}  {'Rel L2 err':>12}  {'Max abs err':>12}  {'Cosine sim':>11}")
    print("-" * 75)
    print(f"{'1x':>6}  {f'{base_size}x{base_size}':>10}  {dx_base:5.2f}  {'(reference)':>12}")

    results = {}
    for s in scales:
        if s == 1:
            continue
        H_s = base_size * s
        W_s = base_size * s
        dx_s = dx_base / s
        dy_s = dy_base / s

        x_hr = make_signal(channels, H_s, W_s, dx_s, dy_s).to(device)
        y_hr = layer(x_hr, dx=dx_s, dy=dy_s)

        # Spectral downsample (exact for band-limited signals)
        y_hr_ds = spectral_downsample(y_hr, base_size, base_size)

        # Metrics
        diff = y_hr_ds - y_base
        rel_l2 = diff.norm() / y_base.norm().clamp_min(1e-8)
        max_abs = diff.abs().max()
        cos_sim = F.cosine_similarity(y_base.flatten(), y_hr_ds.flatten(), dim=0)

        results[s] = {"rel_l2": rel_l2.item(), "max_abs": max_abs.item(), "cosine": cos_sim.item()}
        print(f"{f'{s}x':>6}  {f'{H_s}x{W_s}':>10}  {dx_s:5.2f}  {rel_l2.item():12.6f}  {max_abs.item():12.6f}  {cos_sim.item():11.6f}")

    # Summary
    print()
    worst_l2 = max(r["rel_l2"] for r in results.values())
    best_cos = min(r["cosine"] for r in results.values())

    if worst_l2 < 0.05 and best_cos > 0.99:
        print(f"PASS  — worst relative L2 = {worst_l2:.4f}, min cosine = {best_cos:.4f}")
    elif worst_l2 < 0.15:
        print(f"MARGINAL  — worst relative L2 = {worst_l2:.4f}, min cosine = {best_cos:.4f}")
    else:
        print(f"FAIL  — worst relative L2 = {worst_l2:.4f}, min cosine = {best_cos:.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Without resolution scaling (control: should be WORSE)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_without_scaling(
    modes: int = 12,
    channels: int = 4,
    num_hidden: int = 16,
    base_size: int = 32,
    scale: int = 4,
    device: str = "cpu",
):
    """Control test: run high-res input WITHOUT passing dx/dy.

    This should produce significantly worse agreement, confirming the
    resolution-aware mechanism matters.
    """
    dx_base, dy_base = 1.0, 1.0

    layer = Sonic(
        dim=2,
        in_channels=channels,
        num_hidden=num_hidden,
        M_modes=modes,
        normalize_input=False,
        dx=dx_base,
        dy=dy_base,
    ).to(device).eval()

    x_base = make_signal(channels, base_size, base_size, dx_base, dy_base).to(device)
    y_base = layer(x_base)

    H_s = base_size * scale
    dx_s = dx_base / scale
    dy_s = dy_base / scale
    x_hr = make_signal(channels, H_s, H_s, dx_s, dy_s).to(device)

    # With scaling (correct)
    y_scaled = layer(x_hr, dx=dx_s, dy=dy_s)
    y_scaled_ds = spectral_downsample(y_scaled, base_size, base_size)

    # Without scaling (control — treats high-res as if dx=1)
    y_naive = layer(x_hr)
    y_naive_ds = spectral_downsample(y_naive, base_size, base_size)

    l2_scaled = (y_scaled_ds - y_base).norm() / y_base.norm().clamp_min(1e-8)
    l2_naive = (y_naive_ds - y_base).norm() / y_base.norm().clamp_min(1e-8)
    cos_scaled = F.cosine_similarity(y_base.flatten(), y_scaled_ds.flatten(), dim=0)
    cos_naive = F.cosine_similarity(y_base.flatten(), y_naive_ds.flatten(), dim=0)

    print(f"{'Method':<20}  {'Rel L2 err':>12}  {'Cosine sim':>11}")
    print("-" * 50)
    print(f"{'With dx/dy scaling':<20}  {l2_scaled.item():12.6f}  {cos_scaled.item():11.6f}")
    print(f"{'Without (naive)':<20}  {l2_naive.item():12.6f}  {cos_naive.item():11.6f}")
    print()

    if l2_naive > l2_scaled:
        print(f"PASS  — naive error ({l2_naive:.4f}) > scaled error ({l2_scaled:.4f}), "
              "resolution scaling helps")
    else:
        print(f"NOTE  — naive error ({l2_naive:.4f}) <= scaled error ({l2_scaled:.4f})")


def main():
    parser = argparse.ArgumentParser(description="SONIC resolution invariance test")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--num-hidden", type=int, default=16)
    parser.add_argument("--base-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 75)
    print("  SONIC Resolution Invariance Test")
    print("=" * 75)
    print()

    print("Test 1: Multi-scale agreement (with resolution-aware dx/dy)")
    print("-" * 75)
    test_resolution_invariance(
        modes=args.modes, channels=args.channels, num_hidden=args.num_hidden,
        base_size=args.base_size, scales=(1, 2, 4), device=device,
    )

    print()
    print("Test 2: Control — with vs. without resolution scaling at 4x")
    print("-" * 75)
    test_without_scaling(
        modes=args.modes, channels=args.channels, num_hidden=args.num_hidden,
        base_size=args.base_size, scale=4, device=device,
    )


if __name__ == "__main__":
    main()
