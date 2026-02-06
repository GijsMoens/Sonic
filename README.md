# SONIC: Spectral Oriented Neural Invariant Convolutions

Official implementation of the SONIC operator.

> **SONIC: Spectral Oriented Neural Invariant Convolutions**
> ICLR 2026 &mdash; [OpenReview](https://openreview.net/forum?id=qDGiMrUVmc)

SONIC is a continuous spectral parameterisation that models convolutional
operators using a small set of shared, orientation-selective components. It
provides global receptive fields with directional selectivity, resolution
invariance, and high parameter efficiency.

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.9, PyTorch >= 2.0.

## Quick start

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

Resolution-aware inference (train low-res, infer high-res):

```python
y_hires = layer(x_hires, dx=0.5, dy=0.5)
```

## Examples

| Script | Description |
|---|---|
| `examples/resnet50_sonic.py` | ResNet-50 with SONIC blocks for ImageNet |
| `examples/imagenet_train.py` | ImageNet training loop (DDP-ready) |
| `examples/synthshape.py` | Synthetic shape segmentation + robustness eval |

Run the synthetic experiment:

```bash
pip install -e ".[examples]"
python examples/synthshape.py --epochs 2000 --modes 64
```

## Citation

```bibtex
@inproceedings{sonic2026,
    title     = {{SONIC}: Spectral Oriented Neural Invariant Convolutions},
    booktitle = {International Conference on Learning Representations},
    year      = {2026},
    url       = {https://openreview.net/forum?id=qDGiMrUVmc}
}
```

## License

CC BY 4.0
