"""Named image-transform factories.

Train-time augmentation follows the BLIP recipe verbatim
(https://github.com/salesforce/BLIP/blob/main/data/__init__.py): RandomResizedCrop
with scale (0.5, 1.0), RandomHorizontalFlip, and RandAugment(N=2, M=5) restricted
to BLIP's 10-op subset that excludes color-altering operations
(Color/Contrast/Posterize/Solarize/Invert) — these are unsafe for VizWiz because
the dataset is captured by blind users and colour information is already
unreliable. Eval/test path is the standard CLIP/BLIP deterministic preprocessing
(Resize-shortest + CenterCrop, BICUBIC).
"""
from __future__ import annotations

from typing import Mapping

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


_BLIP_RANDAUG_OPS = {
    # Geometric (semantics-preserving)
    "Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
    # Photometric (intensity only — no hue/saturation distortion)
    "AutoContrast", "Brightness", "Sharpness", "Equalize",
}


class _BLIPRandAugment(T.RandAugment):
    """torchvision RandAugment restricted to BLIP's 10-op subset.

    BLIP's `data/randaugment.py` lists exactly these ops and explicitly excludes
    Color, Posterize, Solarize, Invert (all colour-destructive). For VizWiz the
    same exclusion applies: photographs are captured under uncontrolled
    lighting / focus by blind users, and adding colour noise on top harms more
    than it helps (Li et al. 2022, BLIP §3.2).
    """

    def _augmentation_space(self, num_bins, image_size):
        space = super()._augmentation_space(num_bins, image_size)
        return {k: v for k, v in space.items() if k in _BLIP_RANDAUG_OPS}


def _identity(x):
    return x


def _normalise_or_identity(mean, std):
    if mean is None or std is None:
        return T.Lambda(_identity)
    return T.Normalize(mean=mean, std=std)


def build_transform(preset_cfg: Mapping, train: bool = False) -> T.Compose:
    """Build a torchvision Compose from a preset dict.

    Expected preset keys: `resize_shortest` (int), `crop` (int | None),
    `mean` (list | None), `std` (list | None). The preset only fixes the target
    geometry and normalisation stats; augmentation behaviour is determined
    entirely by the `train` flag and follows BLIP for `train=True`.
    """
    resize_shortest = preset_cfg["resize_shortest"]
    crop = preset_cfg.get("crop")
    mean, std = preset_cfg.get("mean"), preset_cfg.get("std")

    if train and crop is not None:
        # BLIP recipe (Salesforce 2022) for caption fine-tuning of vision encoders.
        return T.Compose([
            T.RandomResizedCrop(crop, scale=(0.5, 1.0),
                                interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            _BLIPRandAugment(num_ops=2, magnitude=5,
                             interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            _normalise_or_identity(mean, std),
        ])

    # Eval / test (and the rare crop=None preset): canonical CLIP/BLIP determinism.
    steps: list = [T.Resize(resize_shortest, interpolation=InterpolationMode.BICUBIC)]
    if crop is not None:
        steps.append(T.CenterCrop(crop))
    steps.append(T.ToTensor())
    steps.append(_normalise_or_identity(mean, std))
    return T.Compose(steps)


def denormalise(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    """Reverse a Normalise step for visualisation. Returns a new clipped tensor in [0, 1]."""
    if mean is None or std is None:
        return tensor.clamp(0.0, 1.0)
    mean_t = torch.tensor(mean).view(-1, 1, 1).to(tensor)
    std_t = torch.tensor(std).view(-1, 1, 1).to(tensor)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
