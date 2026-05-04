"""General utilities: reproducibility, model introspection, I/O, dict formatting."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


# =======================================================================
# Reproducibility
# =======================================================================

def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU + CUDA) RNGs and ``PYTHONHASHSEED``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =======================================================================
# Model introspection
# =======================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    ps = (p for p in model.parameters() if (not trainable_only) or p.requires_grad)
    return sum(p.numel() for p in ps)


# =======================================================================
# Metric / dict formatting
# =======================================================================

def pretty_print_metrics(metrics: Dict[str, Any], title: str = "") -> str:
    """Format a metrics dict for notebook print cells."""
    order = ["bleu1", "bleu2", "bleu3", "bleu4", "cider", "rouge_l"]
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * max(len(title), 40))
    for k in order:
        if k in metrics:
            v = metrics[k]
            lines.append(f"  {k:>8s}: {v:.4f}" if isinstance(v, float) else f"  {k:>8s}: {v}")
    for k, v in metrics.items():
        if k in order:
            continue
        lines.append(f"  {k:>8s}: {v}")
    return "\n".join(lines)


# =======================================================================
# I/O
# =======================================================================

def append_csv(row: Dict[str, Any], path: Path) -> None:
    """Append one row to a CSV file, writing the header if the file is new.

    `f.flush()` + `os.fsync()` ensure the row reaches disk before returning — so
    training interrupted mid-epoch still leaves a complete, durable log of every
    epoch that did finish (crucial when a notebook kernel is killed).
    """
    path = Path(path)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file; return ``default`` if missing."""
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
