"""SentencePiece BPE vocabulary for image captioning.

Wraps `sentencepiece.SentencePieceProcessor` (Kudo & Richardson EMNLP 2018,
arXiv:1808.06226) with the stable attribute / method surface the Phase 2/3 code
already relies on:

    len(vocab)                                   ->  int
    vocab.pad_idx / start_idx / end_idx / unk_idx -> int
    vocab.encode(text_or_tokens, pad_to=None)    -> List[int]
    vocab.decode(ids, strip_specials=True)       -> List[str] of subword pieces
    vocab.detokenise(ids)                        -> str (SentencePiece native join)
    vocab.save(path) / Vocabulary.load(path)     -> persistence
    vocab.meta                                   -> dict (tokenizer, vocab_size, ...)

Special-token ids are fixed by convention (matches Phase 1's prior scheme):
    <pad>=0, <start>=1, <end>=2, <unk>=3

Phase 1 trains the tokenizer on the train-split cleaned captions via
`Vocabulary.train_from_texts(...)`, which writes a binary `spm_bpe_{V}.model`
and a small pointer `vocab.json`. Phase 2/3 notebooks load via
`Vocabulary.load("data/processed/vocab.json")` (or directly pointing at the
`.model` file) — the `.model` is what everyone on the team actually shares.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm


# Fixed special-token ids (constant across all Phase 2/3 checkpoints).
PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = ("<pad>", "<start>", "<end>", "<unk>")


# =======================================================================
# Word-level caption cleaning + whitespace tokenisation (metric-side).
# =======================================================================
# Independent of the SentencePiece BPE used by ``Vocabulary``: captioning
# convention (Rennie 2017, Anderson 2018, Liang 2025) scores BLEU / CIDEr /
# ROUGE-L on space-delimited word tokens even when the underlying model
# uses subwords. ``src.evaluation`` uses these to align preds + refs to the
# n-gram form pycocoevalcap expects.

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_caption(s: str) -> str:
    """Lowercase, strip non-``[a-z0-9\\s]`` punctuation, collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    """Whitespace tokenisation."""
    return s.split() if s else []


class Vocabulary:
    """SentencePiece BPE vocabulary with a stable Phase 2/3-facing API."""

    def __init__(self, sp_processor, meta: dict | None = None):
        self.sp = sp_processor
        self.meta: dict = dict(meta or {})

    # -------------------------------- shape / specials --------------------------------

    def __len__(self) -> int:
        return int(self.sp.vocab_size())

    @property
    def pad_idx(self) -> int:
        return PAD_ID

    @property
    def start_idx(self) -> int:
        return START_ID

    @property
    def end_idx(self) -> int:
        return END_ID

    @property
    def unk_idx(self) -> int:
        return UNK_ID

    @property
    def specials(self) -> List[str]:
        return list(SPECIAL_TOKENS)

    # --------------------------------- encode / decode --------------------------------

    def encode(self, text, pad_to: int | None = None) -> List[int]:
        """Encode a raw string (or whitespace-joinable token list) to subword ids.

        Accepts either a ``str`` or a ``Sequence[str]`` (the latter is space-joined
        before SentencePiece encoding, for backward compatibility with callers that
        still pass pre-tokenised lists). Does **not** prepend/append ``<start>`` or
        ``<end>`` — callers that need those should wrap explicitly.
        """
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        ids = self.sp.encode(text, out_type=int)
        if pad_to is not None:
            if len(ids) < pad_to:
                ids = ids + [self.pad_idx] * (pad_to - len(ids))
            else:
                ids = ids[:pad_to]
        return ids

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> List[str]:
        """Decode ids to a list of subword pieces, stopping at ``<end>``.

        Returns the underlying SentencePiece pieces (leading ``▁`` marks a word
        boundary). For a cleanly detokenised sentence use ``detokenise(ids)``.
        """
        out: List[str] = []
        for i in ids:
            i = int(i)
            if i == self.end_idx:
                break
            if strip_specials and i in (self.pad_idx, self.start_idx):
                continue
            out.append(self.sp.id_to_piece(i))
        return out

    def detokenise(self, ids: Sequence[int]) -> str:
        """Return a detokenised string — SentencePiece's native ``decode`` over
        the id prefix up to ``<end>``, with specials stripped.
        """
        clean: List[int] = []
        for i in ids:
            i = int(i)
            if i == self.end_idx:
                break
            if i in (self.pad_idx, self.start_idx, self.unk_idx):
                # include <unk> as a literal subword so decode produces the " ⁇ " placeholder
                if i == self.unk_idx:
                    clean.append(i)
                continue
            clean.append(i)
        return self.sp.decode(clean)

    # ----------------------------------- training -------------------------------------

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        output_dir,
        *,
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        pointer_filename: str = "vocab.json",
    ) -> "Vocabulary":
        """Train a SentencePiece BPE model on the supplied texts.

        Determinism: SentencePiece BPE training is deterministic given the same
        input file content — identical training corpus and hyperparameters
        produce a bit-exact ``.model`` file across machines. This makes the
        binary model file safe to commit and share across the 4-student team.

        Writes three artifacts to ``output_dir`` that together form the shared
        Phase 1 handoff:

        - ``spm_bpe_{V}.model``          binary SentencePiece model
        - ``spm_bpe_{V}.vocab``          human-readable token list
        - ``{pointer_filename}``         small JSON pointing at the ``.model`` —
                                         this is what Phase 2/3 notebooks open,
                                         so the code path ``Vocabulary.load(
                                         "data/processed/vocab.json")`` stays
                                         identical to the old word-level scheme
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # SentencePiece's Python trainer prefers file input; materialise to a temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as tmp:
            for t in texts:
                tmp.write(str(t).strip())
                tmp.write("\n")
            tmp_path = tmp.name

        model_prefix = str(output_dir / f"spm_bpe_{vocab_size}")
        try:
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=PAD_ID, bos_id=START_ID, eos_id=END_ID, unk_id=UNK_ID,
                pad_piece="<pad>", bos_piece="<start>",
                eos_piece="<end>", unk_piece="<unk>",
                normalization_rule_name="nmt_nfkc",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + ".model")

        meta = {
            "tokenizer": "spm_bpe",
            "vocab_size": int(sp.vocab_size()),
            "model_type": model_type,
            "character_coverage": character_coverage,
            "specials": list(SPECIAL_TOKENS),
            "pad_id": PAD_ID, "start_id": START_ID,
            "end_id": END_ID, "unk_id": UNK_ID,
            "model_file": f"spm_bpe_{vocab_size}.model",
        }

        # Write the pointer file that Phase 2/3 code opens.
        pointer_path = output_dir / pointer_filename
        pointer_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cls(sp, meta)

    # -------------------------------- persistence -------------------------------------

    def save(self, path) -> None:
        """(Re)write the pointer JSON. The binary ``.model`` is already on disk
        from ``train_from_texts``; ``save`` only refreshes the ``meta`` pointer."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocabulary":
        """Load a trained vocabulary. ``path`` may be:

        - ``data/processed/vocab.json``          pointer file (recommended)
        - ``data/processed/spm_bpe_8000.model``  binary model directly
        - ``data/processed/``                    directory containing an ``spm_bpe_*.model``
        """
        p = Path(path)
        meta: dict | None = None

        if p.is_dir():
            candidates = sorted(p.glob("spm_bpe_*.model"))
            if not candidates:
                raise FileNotFoundError(f"No spm_bpe_*.model in {p}")
            model_file = candidates[0]
        elif p.suffix == ".json":
            meta = json.loads(p.read_text(encoding="utf-8"))
            if "model_file" not in meta:
                raise ValueError(f"{p} is not a SentencePiece pointer JSON "
                                 "(missing 'model_file' field)")
            model_file = p.parent / meta["model_file"]
        else:
            model_file = p

        if not Path(model_file).exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")

        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))

        if meta is None:
            meta = {
                "tokenizer": "spm_bpe",
                "vocab_size": int(sp.vocab_size()),
                "specials": list(SPECIAL_TOKENS),
                "pad_id": PAD_ID, "start_id": START_ID,
                "end_id": END_ID, "unk_id": UNK_ID,
                "model_file": Path(model_file).name,
            }
        return cls(sp, meta)
