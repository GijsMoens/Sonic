"""Sonic classification architecture.

All spatial mixing is done by bandlimited Sonic operators.
Channel mixing uses 1x1 convolutions with GELU.

Architecture
------------
- 4-stage hierarchical backbone (dims=[64,128,256,512], depths=[3,3,9,3])
- SonicBlock: Sonic for spatial mixing, 1x1 conv MLP for channel mixing
- Classification head: GroupNorm -> GAP -> Linear
"""

from typing import List

import torch
import torch.nn as nn

from sonic import Sonic

_SIZE_CONFIGS = {
    "tiny": dict(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        mlp_ratio=[4, 4, 2, 1],
    ),
    "normal": dict(
        dims=[64, 128, 256, 512],
        depths=[3, 3, 9, 3],
        mlp_ratio=4.0,
    ),
    "large": dict(
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        mlp_ratio=4.0,
    ),
}


class DropPath(nn.Module):
    """Stochastic depth: drop the entire residual branch with probability *p*."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype).add_(keep).floor_()
        return x * mask / keep


class LayerScale(nn.Module):
    """Per-channel learnable scaling initialised to a small value."""

    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, *([1] * (x.ndim - 2)))


class SonicBlock(nn.Module):
    """Sonic block: Sonic for spatial mixing, 1x1 conv for channel mixing.

    Structure (two residual connections):
        Spatial: GN -> Sonic -> LayerScale -> DropPath + x
        Channel: GN -> Conv1x1 -> GELU -> Conv1x1 -> LayerScale -> DropPath + x

    Set *mlp_ratio=0* to skip the channel-mixing branch.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, sonic_kwargs: dict = None):
        super().__init__()
        sk = dict(sonic_kwargs or {})
        sk.setdefault("normalize_input", False)

        self.norm1 = nn.GroupNorm(1, dim)
        self.sonic = Sonic(dim=2, in_channels=dim, num_hidden=dim, **sk)
        self.ls1 = LayerScale(dim)
        self.drop1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.has_mlp = mlp_ratio > 0
        if self.has_mlp:
            mlp_hidden = int(dim * mlp_ratio)
            self.norm2 = nn.GroupNorm(1, dim)
            self.pwconv1 = nn.Conv2d(dim, mlp_hidden, kernel_size=1)
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(mlp_hidden, dim, kernel_size=1)
            self.ls2 = LayerScale(dim)
            self.drop2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.drop1(self.ls1(self.sonic(self.norm1(x), **kwargs)))
        if self.has_mlp:
            h = self.pwconv2(self.act(self.pwconv1(self.norm2(x))))
            x = x + self.drop2(self.ls2(h))
        return x


class SonicBackbone(nn.Module):
    """4-stage hierarchical backbone using SonicBlocks.

    Args:
        dims: Channel dimensions per stage, e.g. [64, 128, 256, 512].
        depths: Number of blocks per stage, e.g. [3, 3, 9, 3].
        drop_path_rate: Maximum stochastic depth rate (linearly increasing).
        mlp_ratio: MLP expansion ratio.  A single float applied to all stages,
            or a list of per-stage floats (must match ``len(dims)``).
        sonic_kwargs: Dict forwarded to every Sonic instance.
    """

    def __init__(self, dims: List[int] = [96, 192, 384, 768],
                 depths: List[int] = [3, 3, 9, 3],
                 drop_path_rate: float = 0.2,
                 mlp_ratio: "float | list[float]" = 4.0,
                 sonic_kwargs: dict = None):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.num_stages = len(dims)

        if isinstance(mlp_ratio, (int, float)):
            mlp_ratios = [float(mlp_ratio)] * self.num_stages
        else:
            mlp_ratios = [float(r) for r in mlp_ratio]
            if len(mlp_ratios) != self.num_stages:
                raise ValueError(
                    f"mlp_ratio list length {len(mlp_ratios)} != num_stages {self.num_stages}"
                )

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )

        self.downsamples = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.downsamples.append(nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_idx = 0

        self.stages = nn.ModuleList()
        for stage_i in range(self.num_stages):
            stage_blocks = nn.ModuleList()
            for _ in range(depths[stage_i]):
                stage_blocks.append(SonicBlock(
                    dim=dims[stage_i],
                    mlp_ratio=mlp_ratios[stage_i],
                    drop_path=dp_rates[block_idx],
                    sonic_kwargs=dict(sonic_kwargs or {}),
                ))
                block_idx += 1
            self.stages.append(stage_blocks)

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage_i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x, **kwargs)
            features.append(x)
            if stage_i < len(self.downsamples):
                x = self.downsamples[stage_i](x)
        return features


class SonicClassifier(nn.Module):
    """Sonic backbone + classification head.

    Head: GroupNorm -> AdaptiveAvgPool2d(1) -> Linear
    """

    def __init__(self, n_classes: int = 1000, **backbone_kwargs):
        super().__init__()
        self.backbone = SonicBackbone(**backbone_kwargs)
        last_dim = self.backbone.dims[-1]
        self.norm = nn.GroupNorm(1, last_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_dim, n_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.backbone(x, **kwargs)
        x = self.pool(self.norm(features[-1]))
        return self.fc(torch.flatten(x, 1))


def sonic_net(n_classes: int = 1000, drop_path_rate: float = 0.2,
              sonic_kwargs: dict = None, M_modes: int = 128,
              size: str = "normal"):
    """Build a SonicClassifier.

    Args:
        size: ``'tiny'`` (~5.7M), ``'normal'`` (~15.0M), or ``'large'`` (~31.7M).
    """
    if size not in _SIZE_CONFIGS:
        raise ValueError(f"size must be one of {list(_SIZE_CONFIGS)}, got '{size}'")

    cfg = _SIZE_CONFIGS[size].copy()
    sk = dict(sonic_kwargs or {})
    sk.setdefault("normalize_input", False)
    sk.setdefault("M_modes", M_modes)
    sk.setdefault("bandlimit", True)

    return SonicClassifier(
        n_classes=n_classes,
        drop_path_rate=drop_path_rate,
        sonic_kwargs=sk,
        **cfg,
    )
