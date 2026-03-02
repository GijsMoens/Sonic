"""Synthetic shape segmentation experiment with robustness evaluation.

Trains SONIC and baseline models (ConvNet, ViT, NIFF, GFNet) on a synthetic
multi-class shape segmentation task, then evaluates robustness against
geometric transformations, noise, and resolution shifts.

Usage::

    python examples/synthshape.py --epochs 2000 --modes 64
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sonic import Sonic

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


# ---------------------------------------------------------------------------
#  Tensor helpers
# ---------------------------------------------------------------------------

def to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 5:
        B, C, D, H, W = x.shape
        if D != 1:
            raise ValueError(f"Expected D==1 for 5D input, got D={D}.")
        return x.squeeze(2)
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}.")


# ---------------------------------------------------------------------------
#  Synthetic data generation
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Multi-class shape volume generator."""

    @staticmethod
    def generate_single_multi_class_volume(volume_size, num_objects, noise_sigma,
                                           segmentation_criterion="shape"):
        D, H, W = volume_size
        vol_final = np.zeros((D, H, W, 3), dtype=np.float32)
        semantic_mask = np.zeros((D, H, W), dtype=np.int32)
        collision_mask = np.zeros((D, H, W), dtype=bool)

        _, yy, xx = np.ogrid[:D, :H, :W]
        all_shapes = ["circle", "triangle", "square", "cross", "star"]
        shape_class_dict = {"circle": 1, "triangle": 2, "square": 3, "cross": 4, "star": 5}

        base_colors = [
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.5, 0.5, 0.5],
        ]

        shapes_to_place = (
            random.sample(all_shapes, k=min(num_objects, len(all_shapes)))
            if num_objects <= len(all_shapes)
            else random.choices(all_shapes, k=num_objects)
        )

        for shape_label in shapes_to_place:
            for _ in range(100):
                current_shape_mask = np.zeros((D, H, W), dtype=bool)
                is_valid = False

                if shape_label == "circle":
                    r_min = max(1, min(H, W) // 10)
                    r_max = min(H, W) // 2
                    if r_min >= r_max:
                        r_max = r_min + 1
                    r = np.random.randint(r_min, r_max)
                    if H > 2 * r and W > 2 * r:
                        cy, cx = np.random.randint(r, H - r), np.random.randint(r, W - r)
                        current_shape_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
                        is_valid = True

                elif shape_label == "square":
                    s_min, s_max = 3, min(H, W) // 6 + 1
                    if s_min >= s_max:
                        s_max = s_min + 1
                    s = np.random.randint(s_min, s_max)
                    if H > s and W > s:
                        y0, x0 = np.random.randint(0, H - s), np.random.randint(0, W - s)
                        current_shape_mask[:, y0:y0 + s, x0:x0 + s] = True
                        is_valid = True

                elif shape_label == "triangle":
                    base = np.random.randint(5, min(H, W) // 5 + 1)
                    h = int(base * np.sqrt(3) / 2)
                    if h > 0 and H > h and W > base:
                        cy = np.random.randint(0, H - h)
                        cx = np.random.randint(base // 2, W - base // 2)
                        frac = (yy - cy) / float(h)
                        half = np.maximum((base / 2.0) * (1 - frac), 0)
                        current_shape_mask = (np.abs(xx - cx) <= half) & (yy >= cy) & (yy < cy + h)
                        is_valid = True

                elif shape_label == "cross":
                    a = np.random.randint(5, min(H, W) // 5 + 1)
                    w = np.random.randint(max(1, a // 4), a // 2 + 1)
                    if H > 2 * a and W > 2 * a:
                        cy, cx = np.random.randint(a, H - a), np.random.randint(a, W - a)
                        horiz = ((yy >= cy - w // 2) & (yy < cy + w // 2 + w % 2)
                                 & (xx >= cx - a) & (xx < cx + a))
                        vert = ((xx >= cx - w // 2) & (xx < cx + w // 2 + w % 2)
                                & (yy >= cy - a) & (yy < cy + a))
                        current_shape_mask = horiz | vert
                        is_valid = True

                elif shape_label == "star":
                    r = np.random.randint(5, min(H, W) // 5 + 1)
                    if H > 2 * r and W > 2 * r:
                        cy, cx = np.random.randint(r, H - r), np.random.randint(r, W - r)
                        dist_yx = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                        angle = np.arctan2(yy - cy, xx - cx)
                        inner_r = 0.4 * r
                        max_r = inner_r + (r - inner_r) * (np.cos(5 * angle) + 1) / 2
                        current_shape_mask = dist_yx <= max_r
                        is_valid = True

                if is_valid and not np.any(current_shape_mask & collision_mask):
                    chosen_base_color = random.choice(base_colors)
                    final_color = (
                        np.array(chosen_base_color) * np.random.uniform(0.7, 1.2)
                        + np.random.uniform(-0.1, 0.1, 3)
                    )
                    vol_final[current_shape_mask] = final_color
                    semantic_mask[current_shape_mask] = shape_class_dict[shape_label]
                    collision_mask |= current_shape_mask
                    break

        if noise_sigma > 0:
            vol_final += noise_sigma * np.random.randn(D, H, W, 3).astype(np.float32)
        vol_final = np.clip(vol_final, 0, 1.0)
        return vol_final, semantic_mask

    @staticmethod
    def generate_data(batch_size, volume_size, num_objects=5, noise_sigma=0.1,
                      segmentation_criterion="shape", experiment="multiclass"):
        D, H, W = volume_size
        x = torch.zeros(batch_size, 3, D, H, W)
        y_mask = torch.zeros(batch_size, 1, D, H, W)
        for i in range(batch_size):
            vol_np, mask_np = SyntheticDataGenerator.generate_single_multi_class_volume(
                volume_size, num_objects, noise_sigma, segmentation_criterion
            )
            x[i] = torch.from_numpy(vol_np.transpose(3, 0, 1, 2).astype(np.float32))
            y_mask[i, 0] = torch.from_numpy(mask_np.astype(np.float32))
        return x, y_mask


# ---------------------------------------------------------------------------
#  Building blocks (baselines + SONIC)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Stack of 3x3 convolutions with GroupNorm + GELU."""

    def __init__(self, in_channels, out_channels, num_layers=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.block(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4.0))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBlock(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, no_upsample=False):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.H_patch = img_size // patch_size
        self.W_patch = img_size // patch_size
        num_patches = self.H_patch * self.W_patch
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.no_upsample = no_upsample

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.H_patch, self.W_patch)
        if self.no_upsample:
            return x
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


class GlobalFilter2d(nn.Module):
    """GFNet-style global filter."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.H = None
        self.W = None
        self.register_parameter("complex_weight", None)

    def _build_weight(self, H, W, device, dtype):
        fy = torch.linspace(-1.0, 1.0, H, device=device)
        fx = torch.linspace(0.0, 1.0, W // 2 + 1, device=device)
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        r2 = fy ** 2 + fx ** 2
        base_mag = torch.exp(-r2 / (2 * 0.4 ** 2))
        base = torch.stack([base_mag, torch.zeros_like(base_mag)], dim=-1)
        weight = base.unsqueeze(0).repeat(self.dim, 1, 1, 1)
        weight = weight + 0.01 * torch.randn_like(weight)
        self.complex_weight = nn.Parameter(weight.to(torch.float32))
        self.H, self.W = H, W

    def forward(self, x):
        x = x.to(torch.float32)
        B, C, H, W = x.shape
        if self.complex_weight is None:
            self._build_weight(H, W, x.device, x.dtype)
        elif H != self.H or W != self.W:
            raise ValueError(f"Trained at H={self.H}, W={self.W}, got H={H}, W={W}.")
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        weight_c = torch.view_as_complex(self.complex_weight).to(x_ft.dtype)
        x_ft = x_ft * weight_c.unsqueeze(0)
        return torch.fft.irfft2(x_ft, s=(H, W), dim=(-2, -1), norm="ortho")


class GlobalFilterBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, stride=1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_proj = nn.Conv2d(in_dim, out_dim, 1, stride=stride, bias=False) if (stride != 1 or in_dim != out_dim) else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.global_filter = GlobalFilter2d(out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, 4 * out_dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(4 * out_dim, out_dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.global_filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLP_small(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.layer1 = nn.Conv2d(2, 8, 1)
        self.layer2 = nn.Conv2d(8, 4, 1)
        self.layer3 = nn.Conv2d(4, planes, 1)

    def forward(self, x):
        x = F.silu(self.layer1(x.unsqueeze(0)))
        x = F.silu(self.layer2(x))
        return self.layer3(x).squeeze(0)


class FreqConv_DW_fftifft(nn.Module):
    """NIFF depthwise FFT-based convolution."""

    def __init__(self, planes, device=None):
        super().__init__()
        self.mlp_imag = MLP_small(planes)
        self.mlp_real = MLP_small(planes)
        self.mask = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.mask is None or self.mask.shape[-2:] != (H, W):
            ys = torch.arange(-(H // 2), H // 2, device=x.device).float()
            xs = torch.arange(-(W // 2), W // 2, device=x.device).float()
            mask_y = ys[None, :].repeat(W, 1).unsqueeze(0)
            mask_x = xs[:, None].repeat(1, H).unsqueeze(0)
            self.mask = torch.cat([mask_y, mask_x], dim=0).permute(0, 2, 1).contiguous().to(x.device)
        x_ft = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask)).to(x_ft.device)
        x_ft = x_ft * weights.unsqueeze(0)
        return torch.fft.ifft2(torch.fft.ifftshift(x_ft)).real


class NiffBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
        self.niff = FreqConv_DW_fftifft(out_dim)
        self.norm1 = nn.GroupNorm(num_groups, out_dim)
        self.norm2 = nn.GroupNorm(num_groups, out_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.proj(x)
        y = self.mlp(self.norm2(self.niff(self.norm1(z))))
        return self.relu(z + y)


class SonicBlock(nn.Module):
    """Single Sonic block with GroupNorm and residual."""

    def __init__(self, in_channels, out_channels, M_modes, is_stem=False,
                 sonic_kwargs=None):
        super().__init__()
        self.sonic = Sonic(dim=2, in_channels=in_channels, num_hidden=out_channels,
                           M_modes=M_modes, **(sonic_kwargs or {}))
        self.bn_main = nn.GroupNorm(8, out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, **sonic_forward_kwargs):
        identity = self.proj(x)
        out = self.sonic(x, **sonic_forward_kwargs)
        out = self.bn_main(out)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
#  Unified model
# ---------------------------------------------------------------------------

class SynthshapeModel(nn.Module):
    """Segmentation model with pluggable backbone: sonic, conv, vit, niff, gfnet."""

    def __init__(self, model_type, H, W, in_channels=3, num_classes=6, K1=64,
                 conv_layers=4, vit_depth=4, vit_heads=4, modes=32, patch_size=8, **kwargs):
        super().__init__()
        self.model_type = model_type.lower()
        self.H, self.W = H, W
        self.C_mid = K1
        self.depth = conv_layers

        self.in_proj = nn.Conv2d(in_channels, self.C_mid, 1, bias=False)

        if self.model_type == "conv":
            self.backbone = ConvBlock(self.C_mid, self.C_mid, num_layers=conv_layers)
            self.sonic_blocks = None
        elif self.model_type == "sonic":
            self.backbone = None
            sonic_kwargs = kwargs.get("sonic_kwargs_stage1", {})
            self.sonic_blocks = nn.ModuleList([
                SonicBlock(self.C_mid, self.C_mid, modes, is_stem=(i == 0),
                           sonic_kwargs=sonic_kwargs)
                for i in range(self.depth)
            ])
        elif self.model_type == "niff":
            self.sonic_blocks = None
            self.backbone = nn.Sequential(*[NiffBlock(self.C_mid, self.C_mid) for _ in range(self.depth)])
        elif self.model_type == "gfnet":
            self.sonic_blocks = None
            self.backbone = nn.Sequential(*[GlobalFilterBlock(self.C_mid) for _ in range(self.depth)])
        elif self.model_type == "vit":
            self.sonic_blocks = None
            self.backbone = None
            self.vit_block = ViTBlock(H, patch_size, self.C_mid, self.C_mid, vit_depth, vit_heads, no_upsample=True)
        else:
            raise ValueError(f"Unknown model_type={self.model_type!r}")

        self.head = nn.Sequential(
            nn.Conv2d(self.C_mid, self.C_mid, 1, bias=False),
            nn.GroupNorm(8, self.C_mid),
            nn.GELU(),
            nn.Conv2d(self.C_mid, num_classes, 1, bias=True),
        )

    def forward(self, x, **kwargs):
        x = to_bchw(x)
        f0 = self.in_proj(x)
        x_res = f0

        if self.model_type == "sonic":
            h = f0
            for blk in self.sonic_blocks:
                h = blk(h, **kwargs)
        elif self.model_type == "vit":
            h = self.vit_block(f0)
            h = F.interpolate(h, size=(self.H, self.W), mode="bilinear", align_corners=False)
        else:
            h = self.backbone(f0)

        return self.head(h + x_res)


# ---------------------------------------------------------------------------
#  Metrics & training
# ---------------------------------------------------------------------------

def multiclass_dice(preds, labels, num_classes, ignore_index=0, eps=1e-6):
    preds = preds.squeeze(1) if preds.dim() == 4 and preds.shape[1] == 1 else preds
    labels = to_bchw(labels).squeeze(1)
    preds_oh = F.one_hot(preds.long(), num_classes).permute(0, 3, 1, 2).float()
    labels_oh = F.one_hot(labels.long(), num_classes).permute(0, 3, 1, 2).float()
    if ignore_index is not None:
        preds_oh[:, ignore_index] = 0
        labels_oh[:, ignore_index] = 0
    inter = (preds_oh * labels_oh).sum(dim=[0, 2, 3])
    sums = preds_oh.sum(dim=[0, 2, 3]) + labels_oh.sum(dim=[0, 2, 3])
    valid = sums > 0
    if not valid.any():
        return torch.tensor(1.0, device=preds.device)
    return ((2.0 * inter[valid] + eps) / (sums[valid] + eps)).mean()


def train_model(args, model_type, experiment="multiclass"):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SynthshapeModel(**vars(args), model_type=model_type).to(device)
    print(f"Training: {model_type.upper()} on {device.upper()}")

    with torch.no_grad():
        _, yb = SyntheticDataGenerator.generate_data(512, (1, args.H, args.W), 5, args.train_noise, experiment=experiment)
        hist = torch.bincount(to_bchw(yb).long().flatten(), minlength=args.num_classes)
        freq = hist.float() / hist.sum().clamp_min(1)
        cls_weights = (1.0 / freq.clamp_min(1e-6)).to(device)
        cls_weights = (cls_weights / cls_weights.mean()).to(device)

    ce_loss_fn = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    dice_log = []
    for ep in range(args.epochs):
        x, y = SyntheticDataGenerator.generate_data(args.bs, (1, args.H, args.W), 5, args.train_noise, experiment=experiment)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            dice_score = multiclass_dice(preds, y, args.num_classes, 0)
            ce = ce_loss_fn(logits, to_bchw(y).squeeze(1).long())
            loss = 0.5 * ce + 0.5 * (1.0 - dice_score)

        dice_log.append(dice_score.item())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (ep + 1) % args.log_every == 0:
            print(f"  [{model_type.upper()}] Ep {ep+1:04d}/{args.epochs} | "
                  f"Loss: {loss.item():.4f} | Dice: {dice_score.item():.3f}")

    print(f"  Finished {model_type.upper()}.")
    return model, dice_log


# ---------------------------------------------------------------------------
#  Perturbation helpers
# ---------------------------------------------------------------------------

def apply_gaussian_noise(x, sigma):
    return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


def make_affine_grid(x, angle=0.0, translate=(0.0, 0.0), scale=1.0):
    B, C, H, W = x.shape
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad) / scale
    sin_a = math.sin(angle_rad) / scale
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    tx_pix, ty_pix = translate
    theta[:, 0, 2] = 2 * tx_pix / max(W - 1, 1)
    theta[:, 1, 2] = 2 * ty_pix / max(H - 1, 1)
    return F.affine_grid(theta, size=x.size(), align_corners=False)


def apply_affine_xy(x, y, angle=0.0, translate=(0.0, 0.0), scale=1.0):
    x_in = to_bchw(x)
    y_in = to_bchw(y).float()
    grid = make_affine_grid(x_in, angle=angle, translate=translate, scale=scale)
    x_out = F.grid_sample(x_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_out = F.grid_sample(y_in, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x_out, y_out.round().long().unsqueeze(1)


def apply_rescaling_xy(x, y, factor):
    x_in, y_in = to_bchw(x), to_bchw(y).float()
    B, C, H, W = x_in.shape
    new_H, new_W = max(4, int(round(H * factor))), max(4, int(round(W * factor)))
    x_rs = F.interpolate(x_in, size=(new_H, new_W), mode="bilinear", align_corners=False)
    y_rs = F.interpolate(y_in, size=(new_H, new_W), mode="nearest")
    if factor >= 1.0:
        y0, x0 = (new_H - H) // 2, (new_W - W) // 2
        x_out, y_out = x_rs[:, :, y0:y0+H, x0:x0+W], y_rs[:, :, y0:y0+H, x0:x0+W]
    else:
        pad_y, pad_x = H - new_H, W - new_W
        pad = (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
        x_out = F.pad(x_rs, pad, value=0.0)
        y_out = F.pad(y_rs, pad, value=0.0)
    return x_out, y_out.round().long().unsqueeze(1)


def apply_rotation_xy(x, y, degrees):
    return apply_affine_xy(x, y, angle=degrees)


def apply_translation_xy(x, y, percent):
    x_in, y_in = to_bchw(x), to_bchw(y)
    B, C, H, W = x_in.shape
    shift_y = (percent / 100.0) * H
    shift_x = (percent / 100.0) * W
    sign_y = torch.randint(0, 2, (B, 1), device=x_in.device) * 2 - 1
    sign_x = torch.randint(0, 2, (B, 1), device=x_in.device) * 2 - 1
    ty = float((shift_y * sign_y).mean())
    tx = float((shift_x * sign_x).mean())
    return apply_affine_xy(x_in, y_in, translate=(tx, ty))


def apply_distortion_xy(x, y, severity):
    x_in, y_in = to_bchw(x), to_bchw(y).float()
    B, C, H, W = x_in.shape
    device = x_in.device
    disp_y = torch.randn(B, 1, H, W, device=device) * (severity / 20.0)
    disp_x = torch.randn(B, 1, H, W, device=device) * (severity / 20.0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                             torch.linspace(-1, 1, W, device=device), indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    grid[..., 0] += disp_x.squeeze(1) / max(W, 1)
    grid[..., 1] += disp_y.squeeze(1) / max(H, 1)
    x_out = F.grid_sample(x_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_out = F.grid_sample(y_in, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x_out, y_out.round().long().unsqueeze(1)


def apply_combined_xy(x, y, percent):
    sev = percent / 10.0
    x_out, y_out = apply_distortion_xy(x, y, severity=sev * 2.0)
    x_out, y_out = apply_rescaling_xy(x_out, y_out, factor=1.0 + 0.1 * (sev - 1.0))
    x_out, y_out = apply_rotation_xy(x_out, y_out, degrees=5.0 * sev)
    x_out, y_out = apply_translation_xy(x_out, y_out, percent=percent / 2.0)
    x_out = apply_gaussian_noise(x_out, sigma=0.05 * sev)
    return x_out, y_out


# ---------------------------------------------------------------------------
#  Robustness evaluation
# ---------------------------------------------------------------------------

def evaluate_model_under_perturbations(model, device, args, perturb_type, values,
                                       num_batches=25, experiment="multiclass"):
    model.eval()
    results = {v: [] for v in values}
    for v in values:
        for _ in range(num_batches):
            x, y = SyntheticDataGenerator.generate_data(
                args.bs, (1, args.H, args.W), 5, args.train_noise, experiment=experiment)
            x, y = to_bchw(x).to(device), y.to(device)
            if perturb_type == "distortion":
                x_p, y_p = apply_distortion_xy(x, y, v)
            elif perturb_type == "noise":
                x_p, y_p = apply_gaussian_noise(x, v), y
            elif perturb_type == "rescaling":
                if getattr(model, "model_type", "") == "sonic" and v != 1.0:
                    # Run on the naturally-resized tensor (no crop/pad artifacts)
                    # and tell Sonic the effective grid spacing.
                    B_s, C_s, H_orig, W_orig = x.shape
                    new_H = max(4, int(round(H_orig * v)))
                    new_W = max(4, int(round(W_orig * v)))
                    x_p = F.interpolate(x, (new_H, new_W), mode="bilinear", align_corners=False)
                    y_p = F.interpolate(to_bchw(y).float(), (new_H, new_W), mode="nearest").round().long()
                else:
                    x_p, y_p = apply_rescaling_xy(x, y, v)
            elif perturb_type == "rotation":
                x_p, y_p = apply_rotation_xy(x, y, v)
            elif perturb_type == "translation":
                x_p, y_p = apply_translation_xy(x, y, v)
            elif perturb_type == "combined":
                x_p, y_p = apply_combined_xy(x, y, v)
            elif perturb_type == "baseline":
                x_p, y_p = x, y
            else:
                raise ValueError(f"Unknown perturbation: {perturb_type}")
            with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                fwd_kwargs = {}
                if perturb_type == "rescaling" and v != 1.0 and getattr(model, "model_type", "") == "sonic":
                    fwd_kwargs = {"dx": 1.0 / v, "dy": 1.0 / v}
                logits = model(x_p, **fwd_kwargs)
                preds = torch.argmax(logits, dim=1, keepdim=True)
                # Resize predictions to match y_p if spatial dims differ
                if preds.shape[-2:] != y_p.shape[-2:]:
                    preds = F.interpolate(preds.float(), y_p.shape[-2:], mode="nearest").long()
                results[v].append(multiclass_dice(preds, y_p, args.num_classes, 0).item())
    return {v: float(np.mean(vals)) for v, vals in results.items()}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def print_model_stats(model_stats):
    print("\n" + "=" * 60)
    print(" " * 18 + "MODEL COMPARISON STATS")
    print("=" * 60)
    for name, stats in model_stats.items():
        print(f"  {name}")
        for key, value in stats.items():
            print(f"    {key:<20}: {value:.3f}")
        print("-" * 60)


def print_robustness_table(all_results):
    name_map = {"sonic": "SonicNet", "conv": "ConvNet", "vit": "ViT", "niff": "NIFF", "gfnet": "GFNet"}
    models_order = [k for k in name_map if k in all_results]

    def get(m, section, value):
        d = all_results.get(m, {}).get(section, {})
        return f"{d[value]:.2f}" if value in d else "--"

    print("\n" + "=" * 80)
    print(" " * 20 + "ROBUSTNESS TABLE (mean Dice)")
    print("=" * 80)
    header = "{:<18} {:<8} ".format("Experiment", "Value") + " ".join(f"{name_map[m]:>10}" for m in models_order)
    print(header)
    print("-" * 80)
    sections = [
        ("Distortion", "distortion", [2.0, 4.0, 6.0]),
        ("Gaussian Noise", "noise", [0.1, 0.2, 0.3]),
        ("Rescaling", "rescaling", [0.75, 1.0, 1.5]),
        ("Rotation (deg)", "rotation", [15.0, 30.0, 45.0]),
        ("Translation (%)", "translation", [10.0, 20.0, 30.0]),
        ("Combined (%)", "combined", [10.0, 20.0, 30.0]),
        ("Baseline", "baseline", [1.0]),
    ]
    for label, key, vals in sections:
        print(label)
        for v in vals:
            row = " ".join(f"{get(m, key, v):>10}" for m in models_order)
            print(f"{'':<18} {v:<8} {row}")
        print("-" * 80)
    print("=" * 80)


def plot_single_example_predictions(trained_models, args, experiment="multiclass"):
    name_map = {"conv": "ConvNet", "vit": "ViT", "niff": "NIFF", "gfnet": "GFNet", "sonic": "SonicNet"}
    methods = [m for m in ["conv", "vit", "niff", "gfnet", "sonic"] if m in trained_models]
    device = next(iter(trained_models.values())).parameters().__next__().device

    x_base, y_base = SyntheticDataGenerator.generate_data(1, (1, args.H, args.W), 5, 0.0, experiment=experiment)
    x_base, y_base = x_base.to(device), y_base.to(device)

    perturbations = [
        ("Rescaling", lambda x, y: apply_rescaling_xy(x, y, factor=0.75)),
        ("Rotation", lambda x, y: apply_rotation_xy(x, y, degrees=30.0)),
        ("Translation", lambda x, y: apply_translation_xy(x, y, percent=20.0)),
        ("Distortion", lambda x, y: apply_distortion_xy(x, y, severity=4.0)),
        ("Noise", lambda x, y: (apply_gaussian_noise(x, sigma=0.2), y)),
        ("Combined", lambda x, y: apply_combined_xy(x, y, percent=20.0)),
    ]

    n_rows, n_cols = len(perturbations), 2 + len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    cmap_mask = plt.get_cmap("tab10")
    vmin, vmax = 0, args.num_classes - 1
    col_titles = ["Input", "GT"] + [name_map.get(m, m) for m in methods]

    for ri, (row_name, pert_fn) in enumerate(perturbations):
        x_p, y_p = pert_fn(x_base, y_base)
        x_p, y_p = x_p.to(device), y_p.to(device)
        x_bchw, y_bchw = to_bchw(x_p), to_bchw(y_p)
        inp_img = x_bchw[0].permute(1, 2, 0).cpu().numpy()
        gt_mask = y_bchw[0, 0].cpu().numpy()

        axes[ri, 0].imshow(inp_img)
        axes[ri, 1].imshow(gt_mask, cmap=cmap_mask, vmin=vmin, vmax=vmax)

        for mi, mkey in enumerate(methods):
            model = trained_models[mkey]
            model.eval()
            with torch.no_grad():
                pred = torch.argmax(model(x_p), dim=1)[0].cpu().numpy()
            axes[ri, 2 + mi].imshow(pred, cmap=cmap_mask, vmin=vmin, vmax=vmax)
        axes[ri, 0].set_ylabel(row_name, fontsize=11, rotation=90, labelpad=18)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                         transform=ax.transAxes, lw=1, ec="#ccc", fc="#f6f6f6", zorder=-1))
    for c in range(n_cols):
        axes[0, c].set_title(col_titles[c] if c < len(col_titles) else "", fontsize=12, pad=6)

    plt.tight_layout()
    plt.savefig("synthshape_perturbations_grid.png", dpi=200, bbox_inches="tight")
    print("Saved: synthshape_perturbations_grid.png")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SONIC synthetic shape robustness experiment")
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)
    parser.add_argument("--experiment", type=str, default="multiclass")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--K1", type=int, default=64)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=4)
    parser.add_argument("--conv_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--train_noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", type=str, default="./results")
    parser.add_argument("--log_every", type=int, default=250)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # models_to_run = ["sonic", "conv", "vit", "niff", "gfnet"]
    models_to_run = ["sonic", "conv"]
    # models_to_run = ["sonic"]
    trained_models: Dict[str, nn.Module] = {}
    model_stats: Dict[str, dict] = {}
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models_to_run:
        model, dice_log = train_model(args, model_name, experiment=args.experiment)
        trained_models[model_name] = model
        model.eval()
        with torch.no_grad():
            final_dice = dice_log[-1] if dice_log else 0.0
            params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            stats = {"Final Dice": final_dice, "Params (M)": params_m}
            if thop_profile is not None:
                dummy = torch.randn(1, 3, args.H, args.W).to(device)
                macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
                stats["GMACs"] = macs / 1e9
        model_stats[model_name.upper()] = stats

    print_model_stats(model_stats)

    # Robustness sweeps
    all_results: Dict[str, Dict[str, Dict[float, float]]] = {}
    perturbation_configs = [
        ("baseline", [1.0], 30),
        ("distortion", [2.0, 4.0, 6.0], 20),
        ("noise", [0.1, 0.2, 0.3], 20),
        ("rescaling", [0.75, 1.0, 1.5], 20),
        ("rotation", [15.0, 30.0, 45.0], 20),
        ("translation", [10.0, 20.0, 30.0], 20),
        ("combined", [10.0, 20.0, 30.0], 20),
    ]
    for model_name, model in trained_models.items():
        print(f"\nRobustness sweeps: {model_name.upper()} ...")
        all_results[model_name] = {}
        for ptype, vals, nb in perturbation_configs:
            all_results[model_name][ptype] = evaluate_model_under_perturbations(
                model, device, args, ptype, vals, num_batches=nb, experiment=args.experiment)

    print_robustness_table(all_results)
    plot_single_example_predictions(trained_models, args, experiment=args.experiment)
