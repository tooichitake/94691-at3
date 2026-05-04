"""Batched caption decoding for an entire DataLoader.

For a single image / tensor, call ``ImageCaptioner.generate(method=...)`` directly.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from tqdm import tqdm  # plain text tqdm — see note in src/training.py


def _empty_pred(img_id: int) -> Dict[str, Any]:
    return {"image_id": int(img_id), "caption": "", "terminated_with_eos": False}


@torch.no_grad()
def decode_loader(
    model,
    loader,
    vocab,
    device: torch.device,
    *,
    method: Union[str, Sequence[str], None] = "greedy",
    methods: Optional[Iterable[str]] = None,
    beam_size: int = 5,
    max_len: int = 52,
    length_penalty: float = 1.0,
    log_prefix: Optional[str] = "decode",
) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Run ``model.generate(...)`` across every batch of ``loader``.

    Two call shapes:

    - **Single method** (``method="greedy"`` etc., default): returns a flat list
      of prediction dicts.
    - **Multiple methods** (pass ``methods=("greedy", "beam")``): runs the
      encoder once per batch and decodes with every requested method, returning
      ``{method_name: [pred, ...]}``. Lets ``Trainer.evaluate_on_test`` get
      greedy + beam in one pass without re-encoding the whole test set.

    Each prediction dict has:
      - ``image_id``: int
      - ``caption``: detokenised text (pre-metric tokenisation)
      - ``terminated_with_eos``: bool — True iff the decoder emitted ``<end>``
        before hitting ``max_len``. Consumed by the EOS-fix CIDEr reward in
        ``src.evaluation.compute_cider_per_image``.

    ``length_penalty`` is forwarded to beam search only (no-op for greedy /
    sampling). ``log_prefix=None`` disables the progress bar.

    Accepts batches of either ``(imgs, ids)`` (inference dataset) or
    ``(imgs, ids, refs)`` (eval dataset). Refs are ignored.
    """
    method_list = list(methods) if methods is not None else [method]
    if not method_list:
        raise ValueError("decode_loader: at least one method must be specified")
    multi = methods is not None

    model.eval()
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in method_list}
    iterator = tqdm(loader, desc=log_prefix, leave=False) if log_prefix else loader
    for batch in iterator:
        if len(batch) == 2:
            imgs, ids = batch
        elif len(batch) == 3:
            imgs, ids, _ = batch
        else:
            raise ValueError(f"unexpected batch tuple size {len(batch)}")
        imgs = imgs.to(device, non_blocking=True)

        # One encoder forward per batch — shared across all requested methods.
        encoder_out = model.encoder(imgs)

        for m in method_list:
            token_lists = model.decoder.generate(
                encoder_out, max_len=max_len, method=m,
                beam_size=beam_size, length_penalty=length_penalty,
            )
            for img_id, tok_ids in zip(ids.tolist(), token_lists):
                # The decoder truncates at <end>; if the truncated length
                # equals max_len the run never emitted <end>.
                terminated = len(tok_ids) < max_len
                results[m].append({
                    "image_id": int(img_id),
                    "caption": vocab.detokenise(tok_ids),
                    "terminated_with_eos": bool(terminated),
                })

    if multi:
        return results
    return results[method_list[0]]
