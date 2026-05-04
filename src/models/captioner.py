"""ImageCaptioner wrapper + build_captioner factory."""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .decoders import build_decoder
from .encoders import build_encoder


class ImageCaptioner(nn.Module):
    """Thin wrapper that holds encoder + decoder and exposes one ``forward`` + one ``generate``.

    Special-token ids (``start_id`` / ``end_id`` / ``pad_id``) live on the
    decoder; this wrapper only knows ``encoder`` + ``decoder`` modules.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def start_id(self) -> int:
        return self.decoder.start_id

    @property
    def end_id(self) -> int:
        return self.decoder.end_id

    @property
    def pad_id(self) -> int:
        return self.decoder.pad_id

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        """Teacher-forcing training pass.

        images:   [B, 3, H, W]
        captions: [B, T] full padded sequence starting with <start> and ending with <end>/<pad>
        lengths:  [B] true length of captions incl. <start>/<end>

        Returns logits over captions[:, :-1] → [B, T-1, V].
        Callers compute loss against captions[:, 1:].
        """
        encoder_out = self.encoder(images)
        captions_in = captions[:, :-1]
        lengths_in = (lengths - 1).clamp(min=1)
        return self.decoder(encoder_out, captions_in, lengths_in)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 52,
        method: str = "greedy",
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ) -> Union[List[List[int]], Tuple[torch.Tensor, torch.Tensor]]:
        """Decode captions from images.

        - ``method="greedy"`` / ``"beam"``: returns ``List[List[int]]`` token ids
          (without the leading ``<start>`` and stopping at ``<end>``).
        - ``method="sample"``: returns ``(sampled_ids [B, T], log_probs [B, T])``
          for GRPO / REINFORCE training. Finished positions are masked to pad id
          and ``log_prob = 0``.
        - ``length_penalty`` only applies to ``method="beam"`` and follows Wu
          et al. 2016 GNMT's ``score / (((5 + seq_len) / 6) ** alpha)``
          convention. ``1.0`` applies full GNMT-style normalisation.
        """
        encoder_out = self.encoder(images)
        return self.decoder.generate(
            encoder_out,
            max_len=max_len,
            method=method,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )


def build_captioner(cfg: dict, vocab) -> ImageCaptioner:
    """Factory.

    cfg: the parsed model config YAML (has ``model.encoder`` and ``model.decoder``).
    vocab: ``Vocabulary`` instance (provides start_id / end_id / pad_id and size).
    """
    encoder = build_encoder(dict(cfg["model"]["encoder"]))
    decoder = build_decoder(
        dict(cfg["model"]["decoder"]),
        vocab_size=len(vocab),
        encoder_dim=encoder.d_enc,
        start_id=vocab.start_idx,
        end_id=vocab.end_idx,
        pad_id=vocab.pad_idx,
    )
    return ImageCaptioner(encoder, decoder)
