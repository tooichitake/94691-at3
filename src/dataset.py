"""VizWiz caption datasets.

Three dataset classes for three distinct use cases:

1. `VizWizCaptionDataset` — training / caption-level loss (one row per caption).
2. `VizWizEvalDataset`    — metric scoring (one row per image + its list of reference captions).
3. `VizWizInferenceDataset` — beam search / qualitative demos (one row per image, no captions).

A split manifest (written by the Phase 1 notebook) is a JSON list of per-image records:
    {
      "image_id":     int,
      "file_name":    str,
      "captions":     [str,  ...],          # cleaned
      "tokens":       [[str, ...], ...],    # after <start>/<end> wrap
      "token_ids":    [[int, ...], ...],    # padded to max_len + 2
      "caption_lens": [int,  ...]           # true length incl. <start>/<end>, pre-padding
    }
"""
from __future__ import annotations

import json
import random
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transform


def _load_manifest(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Image-level deterministic split (used by Phase 1 to write the manifests)
# -----------------------------------------------------------------------

def split_image_ids(image_ids: Sequence[int], ratios=(0.8, 0.1, 0.1), seed: int = 2026):
    """Split a list of unique image ids into (train, val, test) with the given ratios.

    Uses ``random.Random(seed).shuffle`` so results do not depend on global RNG state.
    Ratios must sum to 1.0. Any rounding slack is absorbed by the train split.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(round(n * ratios[1]))
    n_test = int(round(n * ratios[2]))
    n_train = n - n_val - n_test
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return train, val, test


class VizWizCaptionDataset(Dataset):
    """Flattens per-image records into per-caption rows.

    __getitem__ returns `(image_tensor, caption_ids, caption_len, image_id)`.
    """

    def __init__(
        self,
        split_manifest_path: str | Path,
        images_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        records = _load_manifest(split_manifest_path)

        rows: List[Tuple[int, str, List[int], int]] = []
        for rec in records:
            img_id = rec["image_id"]
            fname = rec["file_name"]
            for ids, length in zip(rec["token_ids"], rec["caption_lens"]):
                rows.append((img_id, fname, ids, length))
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img_id, fname, ids, length = self.rows[i]
        with Image.open(self.images_dir / fname) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long), img_id


class VizWizEvalDataset(Dataset):
    """One row per IMAGE. Returns `(image_tensor, image_id, references)`.

    `references` is the list of cleaned caption strings (usually 5) from the split manifest —
    exactly what BLEU / CIDEr / ROUGE need as ground-truth.
    """

    def __init__(
        self,
        split_manifest_path: str | Path,
        images_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.records = _load_manifest(split_manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        with Image.open(self.images_dir / rec["file_name"]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(rec["image_id"]), list(rec["captions"])

    @cached_property
    def references_by_image_id(self) -> Dict[int, List[str]]:
        """``{image_id: [ref, ...]}`` dict consumed by ``compute_captioning_metrics``.

        Cached because callers query it once per validation epoch and the
        underlying records list never mutates after ``__init__``.
        """
        return {int(rec["image_id"]): list(rec["captions"]) for rec in self.records}


class VizWizInferenceDataset(Dataset):
    """One row per IMAGE, no captions. Returns `(image_tensor, image_id)`.

    Used for beam search on test set and for qualitative demo grids where ground-truth
    captions are irrelevant.
    """

    def __init__(
        self,
        split_manifest_path: str | Path,
        images_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.records = _load_manifest(split_manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        with Image.open(self.images_dir / rec["file_name"]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(rec["image_id"])


def collate_fn(batch):
    """Stack a batch of (image, caption_ids, length, image_id) rows."""
    imgs, caps, lens, ids = zip(*batch)
    if isinstance(imgs[0], torch.Tensor):
        imgs = torch.stack(imgs, dim=0)
    caps = torch.stack(caps, dim=0)
    lens = torch.stack(lens, dim=0)
    ids = torch.tensor(ids, dtype=torch.long)
    return imgs, caps, lens, ids


def eval_collate_fn(batch):
    """Collate for `VizWizEvalDataset`: stacks images, keeps ids and variable-length refs as lists."""
    imgs, ids, refs = zip(*batch)
    if isinstance(imgs[0], torch.Tensor):
        imgs = torch.stack(imgs, dim=0)
    ids = torch.tensor(ids, dtype=torch.long)
    return imgs, ids, list(refs)


def inference_collate_fn(batch):
    """Collate for `VizWizInferenceDataset`."""
    imgs, ids = zip(*batch)
    if isinstance(imgs[0], torch.Tensor):
        imgs = torch.stack(imgs, dim=0)
    ids = torch.tensor(ids, dtype=torch.long)
    return imgs, ids


# -----------------------------------------------------------------------
# DataLoaders container (fastai `DataLoaders` / Lightning `LightningDataModule` style)
# -----------------------------------------------------------------------

class DataLoaders:
    """Container for the 4 DataLoaders needed by a captioning experiment.

    - `train`        — `VizWizCaptionDataset`, shuffled, per-caption rows (cross-entropy training)
    - `val_caption`  — same dataset type over val split (val loss / teacher-forcing val)
    - `val_eval`     — `VizWizEvalDataset` over val split (beam search + BLEU/CIDEr/ROUGE in-loop)
    - `test_eval`    — `VizWizEvalDataset` over test split (final evaluation)

    Mirrors fastai `DataLoaders(train, valid)` / PyTorch Lightning `LightningDataModule` patterns.
    """

    def __init__(self,
                 train: DataLoader,
                 val_caption: DataLoader,
                 val_eval: DataLoader,
                 test_eval: DataLoader):
        self.train = train
        self.val_caption = val_caption
        self.val_eval = val_eval
        self.test_eval = test_eval

    @classmethod
    def from_config(cls, cfg_model: dict, data_cfg, device) -> "DataLoaders":
        """Build all 4 loaders from a model config + the shared data config.

        Paths inside `data_cfg.paths` are relative to the project root; this method
        assumes the process cwd IS the project root (see the notebook preamble that
        does `os.chdir` into it).

        Image-transform preset is read from `cfg_model["data"]["transform_preset"]`;
        training DS gets augmentation on, eval DSs get the deterministic transform.
        """
        preset = data_cfg.image.presets[cfg_model["data"]["transform_preset"]]
        tf_train = build_transform(preset, train=True)
        tf_eval = build_transform(preset, train=False)

        images_dir = Path(data_cfg.paths.images_dir)
        train_split = Path("data/processed/train_split.json")
        val_split = Path("data/processed/val_split.json")
        test_split = Path("data/processed/test_split.json")

        train_ds = VizWizCaptionDataset(train_split, images_dir, transform=tf_train)
        val_cap_ds = VizWizCaptionDataset(val_split, images_dir, transform=tf_eval)
        val_eval_ds = VizWizEvalDataset(val_split, images_dir, transform=tf_eval)
        test_eval_ds = VizWizEvalDataset(test_split, images_dir, transform=tf_eval)

        bs = cfg_model["training"]["batch_size"]
        pin = (device.type == "cuda")
        nw = cfg_model["training"].get("num_workers", 0)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                                   collate_fn=collate_fn, pin_memory=pin)
        val_cap_loader = DataLoader(val_cap_ds, batch_size=bs, shuffle=False, num_workers=nw,
                                     collate_fn=collate_fn, pin_memory=pin)
        val_eval_loader = DataLoader(val_eval_ds, batch_size=bs, shuffle=False, num_workers=nw,
                                      collate_fn=eval_collate_fn, pin_memory=pin)
        test_eval_loader = DataLoader(test_eval_ds, batch_size=bs, shuffle=False, num_workers=nw,
                                       collate_fn=eval_collate_fn, pin_memory=pin)
        return cls(train_loader, val_cap_loader, val_eval_loader, test_eval_loader)

    def __repr__(self) -> str:
        return (f"DataLoaders(train={len(self.train.dataset)} caps, "
                f"val_caption={len(self.val_caption.dataset)} caps, "
                f"val_eval={len(self.val_eval.dataset)} imgs, "
                f"test_eval={len(self.test_eval.dataset)} imgs)")
