"""Caption decoders. Uniform contract:

  forward(encoder_out, captions_in, lengths) -> logits [B, T, V]
  generate(encoder_out, max_len=, method=, beam_size=, length_penalty=)
    method "greedy" / "beam" -> List[List[int]];
    method "sample" -> (ids [B, T], log_probs [B, T]) for GRPO.

Special-token ids stored on the decoder at construction (not threaded through
generate). Tied input/output embedding (Press & Wolf 2017). Transformer side
is hand-written from ``MultiHeadAttention`` + ``FeedForward`` +
``TransformerDecoderLayer``, mirroring Vaswani 2017 §3 with the BERT/GPT-2
norm-first variant; beam search uses a Hugging Face-style length penalty
inspired by Wu et al. 2016 GNMT §7.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    BahdanauAttention,
    LuongAttention,
    MultiHeadAttention,
)


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------

def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Bool ``[T, T]`` mask: ``True`` where ``j > i`` (future positions)."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def _sinusoidal_pe(T: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(d, device=device).unsqueeze(0).float()
    div = torch.exp((-(i // 2) * 2 * math.log(10000.0) / d))
    pe = pos * div
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe  # [T, d]


def _init_bert_like(module: nn.Module) -> None:
    """BERT/GPT-2 standard init: trunc-normal(std=0.02) on Linear/Embedding,
    ones+zeros on LayerNorm. Brings initial CE loss near ``log(|V|)``.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _init_rnn(rnn: nn.RNNBase) -> None:
    """Modern RNN init (fastai / AllenNLP / OpenNMT default).

    - ``weight_ih_*``: Xavier uniform (input projection — feedforward-like).
    - ``weight_hh_*``: orthogonal *per gate* (Saxe 2014, Jozefowicz 2015).
      PyTorch packs the gates into a single ``[k*H, H]`` tensor (k=4 for LSTM,
      k=3 for GRU); calling ``orthogonal_`` on the whole thing only gives
      column-orthonormality. We split per gate so each ``H x H`` recurrent
      block is row-orthonormal, preserving the recurrent activation norm.
    - ``bias_*``: zeros, then for LSTM set forget-gate bias to 1
      (Jozefowicz 2015 §3 — accelerates convergence on long sequences).
    """
    H = rnn.hidden_size
    if isinstance(rnn, nn.LSTM):
        n_gates = 4
    elif isinstance(rnn, nn.GRU):
        n_gates = 3
    else:
        n_gates = 1
    for name, p in rnn.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "weight_hh" in name:
            for k in range(n_gates):
                nn.init.orthogonal_(p.data[k * H : (k + 1) * H])
        elif "bias" in name:
            nn.init.zeros_(p)
            if isinstance(rnn, nn.LSTM):
                # PyTorch LSTM bias layout: [b_ii | b_if | b_ig | b_io] of length 4*H
                p.data[H : 2 * H].fill_(1.0)


def _truncate_at_end(ids: List[int], end_id: int, pad_id: int) -> List[int]:
    """Truncate a generated id list at the first ``end_id`` or ``pad_id`` token."""
    out: List[int] = []
    for tid in ids:
        if tid == end_id or tid == pad_id:
            break
        out.append(tid)
    return out


# --------------------------------------------------------------------------------
# RNN decoder (LSTM / GRU, optionally with Bahdanau / Luong cross-attention)
# --------------------------------------------------------------------------------

class RNNDecoder(nn.Module):
    """LSTM or GRU decoder, optionally with cross-attention over encoder spatial
    features. ``rnn_type`` ∈ {``"lstm"``, ``"gru"``}; ``attention`` ∈
    {``None``, ``"bahdanau"``, ``"luong"``}.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        *,
        start_id: int,
        end_id: int,
        pad_id: int,
        rnn_type: str = "lstm",
        embed_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 1,
        dropout: float = 0.3,
        attention: Optional[str] = None,
        attn_dim: int = 512,
    ):
        super().__init__()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"unknown rnn_type {rnn_type!r}")
        self.rnn_type = rnn_type
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_attn = attention is not None
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_id)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim) if rnn_type == "lstm" else None
        self.dropout = nn.Dropout(dropout)

        if self.use_attn:
            if attention == "bahdanau":
                self.attn = BahdanauAttention(hidden_dim, encoder_dim, attn_dim, dropout=dropout)
            elif attention == "luong":
                self.attn = LuongAttention(hidden_dim, encoder_dim, dropout=dropout)
            else:
                raise ValueError(f"unknown attention {attention!r}")
            rnn_in = embed_dim + encoder_dim
        else:
            self.attn = None
            rnn_in = embed_dim

        self.rnn = rnn_cls(rnn_in, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        # Tied output head when hidden_dim == embed_dim (Press & Wolf 2017).
        if hidden_dim == embed_dim:
            self.out = nn.Linear(hidden_dim, vocab_size, bias=True)
            self.out.weight = self.embed.weight
        else:
            self.out = nn.Linear(hidden_dim, vocab_size)

        self.apply(_init_bert_like)
        # ``apply`` re-randomises the pad row via tied weight; restore zero.
        with torch.no_grad():
            self.embed.weight[self.embed.padding_idx].zero_()
        # ``apply`` skips RNN flat params (weight_ih/hh, bias_ih/hh); init explicitly.
        _init_rnn(self.rnn)

    def _init_state(self, global_feat: torch.Tensor):
        h = torch.tanh(self.init_h(global_feat)).unsqueeze(0).expand(self.n_layers, -1, -1).contiguous()
        if self.rnn_type == "lstm":
            c = torch.tanh(self.init_c(global_feat)).unsqueeze(0).expand(self.n_layers, -1, -1).contiguous()
            return (h, c)
        return h

    def _hidden_from_state(self, state):
        if self.rnn_type == "lstm":
            return state[0][-1]  # top-layer hidden
        return state[-1]

    # ---- forward (teacher-forcing) ----
    def forward(self, encoder_out: Dict[str, torch.Tensor], captions_in: torch.Tensor, lengths: torch.Tensor):
        """captions_in: [B, T], lengths: [B] (true non-pad length of captions_in)."""
        B, T = captions_in.shape
        g = encoder_out["global"]
        spatial = encoder_out["spatial"]
        emb = self.dropout(self.embed(captions_in))    # [B, T, embed_dim]

        state = self._init_state(g)

        if not self.use_attn:
            # Pack-and-feed; faster for variable lengths
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, state = self.rnn(packed, state)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            return self.out(self.dropout(out))

        # With attention: step-by-step so each step sees the current hidden
        logits = emb.new_zeros(B, T, self.vocab_size)
        for t in range(T):
            h_t = self._hidden_from_state(state)                 # [B, H]
            ctx, _ = self.attn(h_t, spatial)                      # [B, D_enc]
            x_t = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)  # [B, 1, embed_dim + D_enc]
            out_t, state = self.rnn(x_t, state)                   # [B, 1, H]
            logits[:, t] = self.out(self.dropout(out_t.squeeze(1)))
        return logits

    # ---- inference ----
    @torch.no_grad()
    def generate(self, encoder_out, *, max_len: int = 52, method: str = "greedy",
                 beam_size: int = 5, length_penalty: float = 1.0):
        if method == "greedy":
            return self._greedy(encoder_out, max_len)
        if method == "beam":
            return self._beam(encoder_out, max_len, beam_size, length_penalty)
        if method == "sample":
            return self._sample(encoder_out, max_len)
        raise ValueError(method)

    def _step(self, tok: torch.Tensor, state, spatial: torch.Tensor):
        """One decoder step. tok: [B, 1] (current-step token id). Returns (logits [B, V], new_state)."""
        emb_t = self.embed(tok)                                 # [B, 1, E]
        if self.use_attn:
            h_t = self._hidden_from_state(state)
            ctx, _ = self.attn(h_t, spatial)
            x_t = torch.cat([emb_t.squeeze(1), ctx], dim=-1).unsqueeze(1)
        else:
            x_t = emb_t
        out_t, state = self.rnn(x_t, state)
        logits = self.out(out_t.squeeze(1))                     # [B, V]
        return logits, state

    def _greedy(self, encoder_out, max_len: int) -> List[List[int]]:
        g = encoder_out["global"]
        spatial = encoder_out["spatial"]
        B = g.shape[0]
        device = g.device
        state = self._init_state(g)
        tok = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_tokens: List[torch.Tensor] = []
        for _ in range(max_len):
            logits, state = self._step(tok, state, spatial)
            nxt = logits.argmax(dim=-1)                                      # [B]
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            finished = finished | (nxt == self.end_id)
            all_tokens.append(nxt)
            tok = nxt.unsqueeze(1)
            if finished.all():
                break
        ids = torch.stack(all_tokens, dim=1).cpu().tolist()                  # [B, T]
        return [_truncate_at_end(seq, self.end_id, self.pad_id) for seq in ids]

    def _sample(self, encoder_out, max_len: int):
        """Multinomial sampling — used by GRPO. Returns (ids [B, T], logp [B, T]).

        Positions after the first ``<end>`` emission are masked to ``pad_id``
        and ``log_prob = 0`` so downstream masking can use ``ids != pad`` as
        the valid-token mask. The ``<end>`` token itself IS included in the
        returned sequence (matches Liang 2025 ``_sample`` convention).
        """
        g = encoder_out["global"]
        spatial = encoder_out["spatial"]
        B = g.shape[0]
        device = g.device
        state = self._init_state(g)
        tok = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        pad_tensor = torch.full((B,), self.pad_id, dtype=torch.long, device=device)
        zero_tensor = torch.zeros(B, dtype=torch.float32, device=device)
        all_ids: List[torch.Tensor] = []
        all_logp: List[torch.Tensor] = []
        for _ in range(max_len):
            logits, state = self._step(tok, state, spatial)
            dist = torch.distributions.Categorical(logits=logits)
            nxt = dist.sample()                                              # [B]
            logp = dist.log_prob(nxt).to(zero_tensor.dtype)                  # [B]
            nxt_masked = torch.where(finished, pad_tensor, nxt)
            logp_masked = torch.where(finished, zero_tensor, logp)
            all_ids.append(nxt_masked)
            all_logp.append(logp_masked)
            finished = finished | (nxt == self.end_id)
            tok = nxt_masked.unsqueeze(1)
            if finished.all():
                break
        return torch.stack(all_ids, dim=1), torch.stack(all_logp, dim=1)

    def _beam(self, encoder_out, max_len: int, beam_size: int, length_penalty: float = 1.0):
        """Per-sample beam search with Wu et al. 2016 GNMT length normalisation.

        Uses the original ``score / (((5 + seq_len) / 6) ** alpha)`` form from
        GNMT §7. ``length_penalty=1.0`` applies full GNMT-style normalisation.
        """
        g = encoder_out["global"]
        spatial = encoder_out["spatial"]
        B = g.shape[0]
        device = g.device
        all_outputs: List[List[int]] = []

        def _norm(score: float, seq_len: int) -> float:
            return score / (((5.0 + max(1, seq_len)) / 6.0) ** length_penalty)

        for b in range(B):
            state = self._init_state(g[b:b+1])
            state = self._replicate_state(state, beam_size)
            spat_b = spatial[b:b+1].expand(beam_size, -1, -1).contiguous()
            tok = torch.full((beam_size, 1), self.start_id, dtype=torch.long, device=device)

            log_probs = torch.zeros(beam_size, device=device)
            log_probs[1:] = -1e9
            sequences: List[List[int]] = [[] for _ in range(beam_size)]
            finished: List[bool] = [False] * beam_size
            fin_log_probs: List[float] = [0.0] * beam_size
            fin_seqs: List[List[int]] = [[] for _ in range(beam_size)]

            for step in range(max_len):
                logits, state = self._step(tok, state, spat_b)
                logp = F.log_softmax(logits, dim=-1)              # [beam, V]
                combined = log_probs.unsqueeze(1) + logp           # [beam, V]
                flat = combined.view(-1)
                top_vals, top_idx = flat.topk(beam_size)
                beam_idx = top_idx // self.vocab_size
                tok_idx = top_idx % self.vocab_size

                new_sequences = []
                new_finished = []
                new_fin_seqs = list(fin_seqs)
                new_fin_log_probs = list(fin_log_probs)
                kept_beams = []
                for rank in range(beam_size):
                    parent = int(beam_idx[rank].item())
                    tid = int(tok_idx[rank].item())
                    seq = sequences[parent] + ([] if tid == self.end_id else [tid])
                    if tid == self.end_id:
                        norm = _norm(top_vals[rank].item(), len(seq))
                        prev_norm = (_norm(new_fin_log_probs[parent], len(new_fin_seqs[parent]))
                                     if new_fin_seqs[parent] else -1e9)
                        if norm > prev_norm:
                            new_fin_seqs[parent] = seq
                            new_fin_log_probs[parent] = top_vals[rank].item()
                        new_sequences.append(seq)
                        new_finished.append(True)
                    else:
                        new_sequences.append(seq)
                        new_finished.append(False)
                    kept_beams.append(parent)

                state = self._reorder_state(state, torch.tensor(kept_beams, device=device))
                sequences = new_sequences
                finished = new_finished
                fin_seqs = new_fin_seqs
                fin_log_probs = new_fin_log_probs
                log_probs = top_vals.clone()
                for rank in range(beam_size):
                    if finished[rank]:
                        log_probs[rank] = -1e9
                tok = tok_idx.unsqueeze(1)

                if all(finished):
                    break

            candidates = [(_norm(fin_log_probs[i], len(fin_seqs[i])), fin_seqs[i])
                          for i in range(beam_size) if fin_seqs[i]]
            if not candidates:
                candidates = [(_norm(log_probs[i].item(), len(sequences[i])),
                               _truncate_at_end(sequences[i], self.end_id, self.pad_id))
                              for i in range(beam_size)]
            best = max(candidates, key=lambda x: x[0])[1]
            all_outputs.append(best)

        return all_outputs

    def _replicate_state(self, state, k: int):
        if self.rnn_type == "lstm":
            h, c = state
            return (h.repeat(1, k, 1), c.repeat(1, k, 1))
        return state.repeat(1, k, 1)

    def _reorder_state(self, state, index: torch.Tensor):
        if self.rnn_type == "lstm":
            h, c = state
            return (h.index_select(1, index), c.index_select(1, index))
        return state.index_select(1, index)


# --------------------------------------------------------------------------------
# Transformer decoder (hand-written, mirrors Vaswani 2017 §3 + BERT/GPT-2 norm-first)
# --------------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise FFN (Vaswani 2017 §3.3) with GELU (Hendrycks & Gimpel 2016).
    ``linear1`` / ``linear2`` field names mirror PyTorch ``nn.TransformerDecoderLayer``.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):
    """Pre-norm decoder layer (Xiong et al. ICML 2020 norm-first variant of
    Vaswani 2017 §3): self-attn → cross-attn → FFN with three residual adds.
    ``norm1/2/3`` mirror PyTorch ``nn.TransformerDecoderLayer``;
    ``self_attn`` / ``cross_attn`` follow HuggingFace Bart.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        ))
        x = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(x, memory, memory))
        x = self.norm3(tgt)
        tgt = tgt + self.dropout(self.ffn(x))
        return tgt


class TransformerDecoder(nn.Module):
    """N-layer Transformer decoder, hand-written. Tied embedding (Press & Wolf
    2017); BERT trunc-normal init (Devlin 2019); GPT-2 / Megatron 1/√(2N)
    scaling on residual-write projections (Radford 2019, Shoeybi 2019).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        *,
        start_id: int,
        end_id: int,
        pad_id: int,
        n_layers: int = 4,
        n_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 52,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        # Memory projection LayerNorm (Meshed-Memory CVPR 2020) stabilises
        # cross-attn across encoder feature scales.
        self.memory_proj = nn.Sequential(
            nn.Linear(encoder_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.register_buffer("pe", _sinusoidal_pe(max_len + 8, d_model, torch.device("cpu")))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        # Tied output head (Press & Wolf 2017): shares weights with ``embed``.
        self.out = nn.Linear(d_model, vocab_size, bias=True)
        self.out.weight = self.embed.weight

        self.apply(_init_bert_like)
        # 1/√(2N) residual-write scaling on (self_attn.out_proj, cross_attn.out_proj,
        # ffn.linear2) keeps residual-stream variance bounded as depth grows.
        scale = (2.0 * n_layers) ** -0.5
        for layer in self.layers:
            for proj in (layer.self_attn.out_proj,
                         layer.cross_attn.out_proj,
                         layer.ffn.linear2):
                nn.init.trunc_normal_(proj.weight,
                                      std=0.02 * scale,
                                      a=-2 * 0.02 * scale,
                                      b=2 * 0.02 * scale)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        # ``apply`` re-randomises the pad row via tied weight; restore zero.
        with torch.no_grad():
            self.embed.weight[self.embed.padding_idx].zero_()

    def _positional_encoding(self, T: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal PE of length ``T``; lazily extends ``self.pe`` if needed."""
        if T > self.pe.shape[0]:
            self.pe = _sinusoidal_pe(T + 8, self.d_model, self.pe.device)
        return self.pe[:T].to(device)

    def _prep_memory(self, encoder_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.memory_proj(encoder_out["spatial"])            # [B, N, d_model]

    def _decode(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor],
                tgt_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.norm_out(x)

    def forward(self, encoder_out, captions_in: torch.Tensor, lengths: torch.Tensor):
        B, T = captions_in.shape
        emb = self.embed(captions_in) * math.sqrt(self.d_model)
        pe = self._positional_encoding(T, emb.device)
        x = self.dropout(emb + pe.unsqueeze(0))                   # [B, T, d_model]
        memory = self._prep_memory(encoder_out)
        tgt_mask = _causal_mask(T, device=x.device)
        tgt_key_padding_mask = (captions_in == self.pad_id)
        h = self._decode(x, memory, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out(h)                                         # [B, T, V]

    @torch.no_grad()
    def generate(self, encoder_out, *, max_len: int = 52, method: str = "greedy",
                 beam_size: int = 5, length_penalty: float = 1.0):
        if method == "greedy":
            return self._greedy(encoder_out, max_len)
        if method == "beam":
            return self._beam(encoder_out, max_len, beam_size, length_penalty)
        if method == "sample":
            return self._sample(encoder_out, max_len)
        raise ValueError(method)

    def _greedy(self, encoder_out, max_len: int) -> List[List[int]]:
        memory = self._prep_memory(encoder_out)
        B = memory.shape[0]
        device = memory.device
        tokens = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_tokens: List[torch.Tensor] = []
        for _ in range(max_len):
            T = tokens.shape[1]
            emb = self.embed(tokens) * math.sqrt(self.d_model)
            x = emb + self._positional_encoding(T, device).unsqueeze(0)
            tgt_mask = _causal_mask(T, device=device)
            h = self._decode(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
            logits = self.out(h[:, -1])                            # [B, V]
            nxt = logits.argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            finished = finished | (nxt == self.end_id)
            all_tokens.append(nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            if finished.all():
                break
        ids = torch.stack(all_tokens, dim=1).cpu().tolist()
        return [_truncate_at_end(seq, self.end_id, self.pad_id) for seq in ids]

    def _sample(self, encoder_out, max_len: int):
        """Multinomial sampling — used by GRPO. Returns (ids [B, T], logp [B, T])."""
        memory = self._prep_memory(encoder_out)
        B = memory.shape[0]
        device = memory.device
        tokens = torch.full((B, 1), self.start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        pad_tensor = torch.full((B,), self.pad_id, dtype=torch.long, device=device)
        zero_tensor = torch.zeros(B, dtype=torch.float32, device=device)
        all_ids: List[torch.Tensor] = []
        all_logp: List[torch.Tensor] = []
        for _ in range(max_len):
            T = tokens.shape[1]
            emb = self.embed(tokens) * math.sqrt(self.d_model)
            x = emb + self._positional_encoding(T, device).unsqueeze(0)
            tgt_mask = _causal_mask(T, device=device)
            h = self._decode(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
            logits = self.out(h[:, -1])
            dist = torch.distributions.Categorical(logits=logits)
            nxt = dist.sample()
            logp = dist.log_prob(nxt).to(zero_tensor.dtype)
            nxt_masked = torch.where(finished, pad_tensor, nxt)
            logp_masked = torch.where(finished, zero_tensor, logp)
            all_ids.append(nxt_masked)
            all_logp.append(logp_masked)
            finished = finished | (nxt == self.end_id)
            tokens = torch.cat([tokens, nxt_masked.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return torch.stack(all_ids, dim=1), torch.stack(all_logp, dim=1)

    def _beam(self, encoder_out, max_len: int, beam_size: int, length_penalty: float = 1.0):
        memory_all = self._prep_memory(encoder_out)                # [B, N, D]
        B = memory_all.shape[0]
        device = memory_all.device
        all_outputs: List[List[int]] = []

        def _norm(score: float, seq_len: int) -> float:
            return score / (((5.0 + max(1, seq_len)) / 6.0) ** length_penalty)

        for b in range(B):
            memory = memory_all[b:b+1].expand(beam_size, -1, -1).contiguous()
            tokens = torch.full((beam_size, 1), self.start_id, dtype=torch.long, device=device)
            log_probs = torch.zeros(beam_size, device=device)
            log_probs[1:] = -1e9
            finished = [False] * beam_size
            fin_seqs: List[List[int]] = []
            fin_scores: List[float] = []

            for step in range(max_len):
                T = tokens.shape[1]
                emb = self.embed(tokens) * math.sqrt(self.d_model)
                x = emb + self._positional_encoding(T, device).unsqueeze(0)
                tgt_mask = _causal_mask(T, device=device)
                h = self._decode(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
                logp = F.log_softmax(self.out(h[:, -1]), dim=-1)    # [beam, V]
                combined = log_probs.unsqueeze(1) + logp
                flat = combined.view(-1)
                top_vals, top_idx = flat.topk(beam_size)
                beam_idx = top_idx // self.vocab_size
                tok_idx = top_idx % self.vocab_size

                new_tokens = torch.cat([tokens[beam_idx], tok_idx.unsqueeze(1)], dim=1)
                new_log_probs = top_vals.clone()
                new_finished = []
                for rank in range(beam_size):
                    tid = int(tok_idx[rank].item())
                    if tid == self.end_id:
                        seq = [t for t in new_tokens[rank, 1:-1].tolist()]
                        fin_seqs.append(seq)
                        fin_scores.append(_norm(new_log_probs[rank].item(), len(seq)))
                        new_log_probs[rank] = -1e9
                        new_finished.append(True)
                    else:
                        new_finished.append(False)

                tokens = new_tokens
                log_probs = new_log_probs
                finished = new_finished
                if all(finished):
                    break

            if fin_seqs:
                best = fin_seqs[int(torch.tensor(fin_scores).argmax().item())]
            else:
                # No beam finished — take best length-normalised in-progress.
                norm = [_norm(log_probs[i].item(), tokens.shape[1] - 1) for i in range(beam_size)]
                best_idx = int(torch.tensor(norm).argmax().item())
                best = _truncate_at_end(tokens[best_idx, 1:].tolist(), self.end_id, self.pad_id)
            all_outputs.append(best)
        return all_outputs


# --------------------------------------------------------------------------------
# Registry + factory
# --------------------------------------------------------------------------------

DECODER_REGISTRY = {
    "lstm":        partial(RNNDecoder, rnn_type="lstm", attention=None),
    "lstm_attn":   partial(RNNDecoder, rnn_type="lstm"),
    "gru":         partial(RNNDecoder, rnn_type="gru",  attention=None),
    "gru_attn":    partial(RNNDecoder, rnn_type="gru"),
    "transformer": TransformerDecoder,
}


def build_decoder(cfg: dict, vocab_size: int, encoder_dim: int, *,
                  start_id: int, end_id: int, pad_id: int) -> nn.Module:
    name = cfg["name"]
    if name not in DECODER_REGISTRY:
        raise KeyError(f"Unknown decoder {name!r}; known: {list(DECODER_REGISTRY)}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return DECODER_REGISTRY[name](
        vocab_size=vocab_size, encoder_dim=encoder_dim,
        start_id=start_id, end_id=end_id, pad_id=pad_id,
        **kwargs,
    )
