"""YAML config loader with attribute access (cfg.text.vocab_size)."""
from pathlib import Path
import yaml


class AttrDict(dict):
    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path) -> AttrDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AttrDict(raw)
