"""ResNet-50 backbone with SONIC blocks replacing standard convolutions."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sonic import Sonic


class SequentialWithKwargs(nn.Sequential):
    """Sequential container that forwards ``**kwargs`` to every module."""

    def forward(self, x, **kwargs):
        for module in self._modules.values():
            x = module(x, **kwargs)
        return x


class Sonic2d(nn.Module):
    """Thin wrapper: optional pooling + Sonic operator."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 sonic_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.op = Sonic(dim=2, in_channels=in_ch, num_hidden=out_ch, **(sonic_kwargs or {}))

    def forward(self, x: torch.Tensor, **resolution_kwargs):
        x = self.pool(x)
        return self.op(x, **resolution_kwargs)


def conv1x1(in_planes, out_planes, stride=1, **_kwargs):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _gn(num_channels, num_groups=32):
    """GroupNorm with clamped group count (resolution-invariant alternative to BN)."""
    return nn.GroupNorm(min(num_groups, num_channels), num_channels)


class SonicBottleneck(nn.Module):
    """Bottleneck block: Conv1x1 reduce → Sonic (spectral 3x3) → Conv1x1 expand."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 use_sonic: bool = False, sonic_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        mid = int(planes * (base_width / 64.0)) * groups
        out_ch = planes * self.expansion

        # 1x1 reduce
        self.conv1 = conv1x1(inplanes, mid)
        self.bn1 = _gn(mid)

        # Sonic spectral operator (replaces 3x3 conv)
        self.sonic2 = Sonic2d(mid, mid, stride=stride, sonic_kwargs=sonic_kwargs)
        self.bn2 = _gn(mid)

        # 1x1 expand
        self.conv3 = conv1x1(mid, out_ch)
        self.bn3 = _gn(out_ch)

    def forward(self, x, **resolution_kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.sonic2(out, **resolution_kwargs)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, *, use_sonic: bool = False,
                 sonic_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = _gn if use_sonic else nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_sonic = use_sonic
        self.sonic_kwargs = sonic_kwargs or {}
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        # Always use a standard conv stem – a Sonic layer at 112×112 is a
        # huge FFT bottleneck, and the stem is just edge detection anyway.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Track global block index for depth-aware Sonic init scheduling.
        self._block_idx = 0
        self._block_total = sum(layers)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            if self.use_sonic:
                raise NotImplementedError("replace_stride_with_dilation not supported in Sonic mode.")
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        block_sonic_kwargs = {**self.sonic_kwargs,
                              "depth_idx": self._block_idx,
                              "depth_total": self._block_total}
        self._block_idx += 1
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer,
            use_sonic=self.use_sonic, sonic_kwargs=block_sonic_kwargs,
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            block_sonic_kwargs = {**self.sonic_kwargs,
                                  "depth_idx": self._block_idx,
                                  "depth_total": self._block_total}
            self._block_idx += 1
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer, use_sonic=self.use_sonic,
                sonic_kwargs=block_sonic_kwargs,
            ))
        return SequentialWithKwargs(*layers)

    def forward(self, x, **resolution_kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, **resolution_kwargs)
        x = self.layer2(x, **resolution_kwargs)
        x = self.layer3(x, **resolution_kwargs)
        x = self.layer4(x, **resolution_kwargs)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet_sonic(layers=(1, 1, 1, 1), M_modes=128, **kwargs):
    """Construct a shallow SONIC-ResNet with fewer blocks and richer spectral capacity.

    Sonic already provides a global receptive field per layer (FFT-based),
    so deep stacking is unnecessary.  Defaults: 4 blocks with 128 modes
    (~7.4M params) vs. ResNet-50-Sonic's 16 blocks with 32 modes (~14.7M).
    """
    kwargs.setdefault("sonic_kwargs", {})
    kwargs["sonic_kwargs"].setdefault("M_modes", M_modes)
    kwargs["sonic_kwargs"].setdefault("normalize_input", False)
    return ResNet(SonicBottleneck, list(layers), use_sonic=True, **kwargs)


def resnet50_sonic(**kwargs):
    """Construct a ResNet-50 model with SONIC blocks."""
    kwargs.setdefault("sonic_kwargs", {})
    kwargs["sonic_kwargs"].setdefault("M_modes", 32)
    kwargs["sonic_kwargs"].setdefault("normalize_input", False)
    return ResNet(SonicBottleneck, [3, 4, 6, 3], use_sonic=True, **kwargs)
