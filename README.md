# SONIC: Spectral Oriented Neural Invariant Convolutions

Official PyTorch implementation of the SONIC operator.

> **SONIC: Spectral Oriented Neural Invariant Convolutions**
> Gijs Moens, Regina Beets-Tan, Eduardo Pooch
> ICLR 2026 &mdash; [Paper](https://arxiv.org/abs/2601.19884) &bull; [OpenReview](https://openreview.net/forum?id=qDGiMrUVmc)

SONIC is a continuous spectral parameterisation that models convolutional
operators using a small set of shared, orientation-selective components.
Each mode is defined by a learnable direction, complex pole, speed, transverse
decay, and Butterworth bandwidth — all evaluated analytically on the frequency
grid. This gives SONIC several desirable properties:

- **Global receptive fields** with learnable directional selectivity
- **Resolution invariance** — train at one resolution, infer at another by passing the physical grid spacing (`dx`, `dy`)
- **High parameter efficiency** — a single SONIC layer with 12 modes uses ~2 orders of magnitude fewer parameters than an equivalent convolution
- **Native 2D and 3D** support

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.9, PyTorch >= 2.0.

## Quick Start

### Basic usage (2D)

```python
import torch
from sonic import Sonic

layer = Sonic(
    dim=2,
    in_channels=64,
    num_hidden=64,
    M_modes=12,
)

x = torch.randn(1, 64, 128, 128)
y = layer(x)  # (1, 64, 128, 128)
```

### 3D volumetric data

```python
layer_3d = Sonic(
    dim=3,
    in_channels=32,
    num_hidden=32,
    M_modes=16,
)

x = torch.randn(1, 32, 48, 64, 64)
y = layer_3d(x)  # (1, 32, 48, 64, 64)
```

### Resolution-aware inference

Train at one resolution and infer at a different one by passing the
physical grid spacing. SONIC's continuous spectral parameterisation adapts
automatically:

```python
# Train at 1.0 mm spacing
layer = Sonic(dim=2, in_channels=64, num_hidden=64, M_modes=12, dx=1.0, dy=1.0)

# Infer at 0.5 mm spacing (2x higher resolution)
y_hires = layer(x_hires, dx=0.5, dy=0.5)
```

## Architecture

SONIC provides a drop-in spectral mixing layer. For classification, we include
a full hierarchical backbone (`SonicClassifier`) with three size variants:

| Variant | Dims | Depths | Params |
|---------|------|--------|--------|
| `tiny` | [64, 128, 256, 512] | [2, 2, 6, 2] | ~5.7M |
| `normal` | [64, 128, 256, 512] | [3, 3, 9, 3] | ~15.0M |
| `large` | [96, 192, 384, 768] | [3, 3, 9, 3] | ~31.7M |

Each stage uses `SonicBlock` (GroupNorm → Sonic → LayerScale → DropPath +
1x1 conv MLP channel mixing) with a patchify stem and strided-conv
downsampling between stages.

## Examples

| Script | Description |
|---|---|
| [`examples/resnet50_sonic.py`](examples/resnet50_sonic.py) | SONIC backbone + classifier (tiny / normal / large) |
| [`examples/imagenet_train.py`](examples/imagenet_train.py) | ImageNet training loop (DDP-ready, AMP, cosine schedule) |
| [`examples/synthshape.py`](examples/synthshape.py) | Synthetic shape segmentation + robustness evaluation |
| [`examples/test_resolution_invariance.py`](examples/test_resolution_invariance.py) | Verify resolution invariance numerically |

### ImageNet training

```bash
pip install -e ".[examples]"

# Single GPU
python examples/imagenet_train.py /path/to/imagenet --epochs 300 --size normal

# Multi-GPU (torchrun)
torchrun --nproc_per_node=4 examples/imagenet_train.py /path/to/imagenet \
    --size normal --modes 128 --batch-size 1024 --amp
```

### Synthetic shape segmentation

```bash
python examples/synthshape.py --epochs 2000 --modes 64
```

Trains SONIC and baseline models (ConvNet, ViT, NIFF, GFNet) on a synthetic
multi-class shape segmentation task, then evaluates robustness against
geometric transformations, noise, and resolution shifts.

### Resolution invariance test

```bash
python examples/test_resolution_invariance.py --modes 12 --channels 4
```

Evaluates the same physical signal at 1x, 2x, and 4x resolution and confirms
that outputs agree after spectral downsampling.

## Project Structure

```
├── src/sonic/
│   ├── __init__.py          # Public API: Sonic
│   ├── sonic.py             # Core SONIC operator
│   └── utils.py             # Frequency grids, direction init, mode dropout
├── examples/
│   ├── resnet50_sonic.py    # Classification backbone
│   ├── imagenet_train.py    # ImageNet training script
│   ├── synthshape.py        # Synthetic segmentation experiment
│   └── test_resolution_invariance.py
├── checkpoints/
│   ├── SonicNet-S.pth.tar   # Checkpoint Sonic for ImageNet
├── pyproject.toml
└── LICENSE                  # CC BY 4.0
```

## Citation

```bibtex
@inproceedings{moens2026sonic,
    title={{SONIC}: Spectral Oriented Neural Invariant Convolutions},
    author={Gijs Joppe Moens and Regina Beets-Tan and Eduardo H. P. Pooch},
    booktitle={International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=qDGiMrUVmc},
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
