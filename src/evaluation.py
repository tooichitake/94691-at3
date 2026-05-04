"""Captioning metrics via pycocoevalcap (BLEU / CIDEr / ROUGE-L)."""
from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Mapping, Sequence

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from .vocab import clean_caption, tokenize


@contextlib.contextmanager
def _silence_stdout():
    """Swallow pycocoevalcap's diagnostic prints (testlen/reflen/ratio)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _tok(s: str) -> List[str]:
    return tokenize(clean_caption(s))


def _tok_joined(s: str) -> str:
    return " ".join(_tok(s))


# Sentinel for EOS-fix n-gram scoring; survives ``clean_caption`` and won't
# appear in a real caption.
_EOS_SENTINEL = "eossentinelvizwiz"


def compute_captioning_metrics(predictions: Sequence[dict],
                               references_by_id: Dict[int, Sequence[str]]) -> Dict[str, float]:
    """Corpus-level BLEU-1..4 + CIDEr + ROUGE-L."""
    hyp: Dict[int, List[str]] = {}
    ref: Dict[int, List[str]] = {}
    for p in predictions:
        iid = int(p["image_id"])
        if iid not in references_by_id:
            raise KeyError(f"prediction references unknown image_id={iid}")
        hyp[iid] = [_tok_joined(p["caption"])]
        ref[iid] = [_tok_joined(r) for r in references_by_id[iid]]

    with _silence_stdout():
        bleu_scores, _ = Bleu(4).compute_score(ref, hyp)
        cider, _ = Cider().compute_score(ref, hyp)
        rouge, _ = Rouge().compute_score(ref, hyp)
    return {
        "bleu1": float(bleu_scores[0]),
        "bleu2": float(bleu_scores[1]),
        "bleu3": float(bleu_scores[2]),
        "bleu4": float(bleu_scores[3]),
        "cider": float(cider),
        "rouge_l": float(rouge),
    }


def compute_cider_per_image(predictions: Sequence[dict],
                            references_by_id: Mapping[int, Sequence[str]],
                            *,
                            enforce_eos: bool = True) -> np.ndarray:
    """Per-image CIDEr-D for GRPO reward. With ``enforce_eos`` (Stefanini 2023
    sacreeos STANDARD), only ``terminated_with_eos=True`` predictions get the
    sentinel appended, so non-terminating fragments cannot game n-gram TF-IDF."""
    # Key by enumerate index: GRPO sends G samples per image_id under the same
    # iid; image_id keying would collapse them.
    hyp: Dict[int, List[str]] = {}
    ref: Dict[int, List[str]] = {}
    for k, p in enumerate(predictions):
        iid = int(p["image_id"])
        if iid not in references_by_id:
            raise KeyError(f"prediction references unknown image_id={iid}")
        h = _tok_joined(p["caption"])
        if enforce_eos and bool(p.get("terminated_with_eos", True)):
            h = f"{h} {_EOS_SENTINEL}".strip()
        hyp[k] = [h]
        refs_joined = [_tok_joined(r) for r in references_by_id[iid]]
        if enforce_eos:
            refs_joined = [f"{r} {_EOS_SENTINEL}".strip() for r in refs_joined]
        ref[k] = refs_joined

    with _silence_stdout():
        _, per_image = Cider().compute_score(ref, hyp)
    return np.asarray(per_image, dtype=np.float32)
