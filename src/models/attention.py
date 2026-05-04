"""Attention modules for caption decoders.

- ``BahdanauAttention`` / ``LuongAttention``: RNN decoder cross-attention.
- ``ScaledDotProductAttention`` / ``MultiHeadAttention``: Vaswani 2017
  Transformer attention, used by ``TransformerDecoderLayer``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (MLP) attention — Bahdanau et al. 2015.

    score(h, m) = v^T tanh(W_h h + W_m m)
    where h is the decoder hidden state [B, H] and m is a spatial memory feature [B, N, M].

    Attention-weight dropout follows the modern d2l.ai ``AdditiveAttention``
    convention (absent from the 2015 paper) and mirrors how ``nn.MultiheadAttention``
    regularises its softmax weights.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, attn_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_m = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor):
        """
        hidden: [B, H]
        memory: [B, N, M]
        Returns (context [B, M], weights [B, N]).
        """
        scores = self.v(torch.tanh(self.W_h(hidden).unsqueeze(1) + self.W_m(memory))).squeeze(-1)  # [B, N]
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(self.dropout(weights).unsqueeze(1), memory).squeeze(1)  # [B, M]
        return context, weights


class LuongAttention(nn.Module):
    """Multiplicative (dot-product) attention — Luong et al. 2015, general form.

    score(h, m) = h^T W m

    Attention-weight dropout follows modern practice (mirrors ``nn.MultiheadAttention``).
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W = nn.Linear(hidden_dim, encoder_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor):
        scores = torch.bmm(memory, self.W(hidden).unsqueeze(-1)).squeeze(-1)  # [B, N]
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(self.dropout(weights).unsqueeze(1), memory).squeeze(1)  # [B, M]
        return context, weights


# =======================================================================
# Transformer attention (Vaswani 2017)
# =======================================================================

class ScaledDotProductAttention(nn.Module):
    """Vaswani 2017 §3.2.1: ``Attention(Q,K,V) = softmax(QK^T/√d_k) V``.

    Backed by ``F.scaled_dot_product_attention`` (PyTorch 2.0+), which
    dispatches to FlashAttention-2 / memory-efficient kernels on Ampere+
    GPUs (Dao et al. 2022, *FlashAttention*; Dao 2023, *FlashAttention-2*).
    The math fallback is bit-equivalent to the explicit ``softmax(QK^T)V``
    formula above.

    Q/K/V shape ``[B, h, T, d_k]``; mask is bool (``True`` blocks).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = float(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = None
        if mask is not None:
            # SDPA expects an additive float mask (-inf at blocked positions)
            # or a bool mask where True = participate. Our convention is
            # True = block, so convert to additive form.
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )


class MultiHeadAttention(nn.Module):
    """Vaswani 2017 §3.2.2 multi-head attention. Hand-written (no
    ``nn.MultiheadAttention``); HuggingFace ``q_proj``/``k_proj``/``v_proj``/
    ``out_proj`` field naming.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # No bias on Q/K/V projections — GPT-2 (Radford 2019), PaLM (Chowdhery
        # 2022), LLaMA (Touvron 2023) all drop these biases; ablations show
        # zero quality impact and a small parameter saving. ``out_proj``
        # keeps its bias (still standard in GPT-2 / LLaMA).
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] → [B, n_heads, T, d_k]."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_heads, T, d_k] → [B, T, d_model]."""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        mask = None
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(0).unsqueeze(0)              # [1, 1, T_q, T_k]
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)        # [B, 1, 1, T_k]
            mask = kpm if mask is None else (mask | kpm)

        out = self.attention(q, k, v, mask=mask)
        return self.out_proj(self._combine_heads(out))
