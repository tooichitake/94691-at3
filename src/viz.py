"""Visualisation helpers for notebooks and the final report.

Styling: seaborn ``whitegrid`` theme + colorblind palette, minimal spines, subtle
grids. Public API: see ``__all__`` in ``src/__init__.py`` — split into plot
functions (``plot_*``) and table builders that return ``pandas.DataFrame`` / ``Styler``.

Seaborn palettes used: ``colorblind`` (categorical), ``crest`` / ``flare``
(sequential), ``rocket_r`` (heatmap, dark = high).
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from IPython.display import display
from PIL import Image as _PIL

from .transforms import build_transform, denormalise


# =======================================================================
# Shared style
# =======================================================================

_PALETTE = "colorblind"
_CONTEXT = "notebook"
_STYLE = "whitegrid"

_PHASE_M2_MARKER = "_m2_"
_CE_SUFFIXES = (" (CE)", " (ce)")
_GRPO_SUFFIXES = (" (GRPO)", " (grpo)")
_RUN_SUFFIXES = (*_CE_SUFFIXES, *_GRPO_SUFFIXES)

# (pred_mean / ref_mean) length verdict — ordered by ascending threshold; first
# bucket whose threshold > ratio wins.
_LENGTH_VERDICTS = [
    (0.5, "COLLAPSED", "crimson"),
    (0.8, "SHORT",     "#c48400"),
    (1.3, "OK",        "seagreen"),
    (float("inf"), "LONG", "#c48400"),
]


def _base_run_name(name: str) -> str:
    """Strip any '' (CE)'' / '' (GRPO)'' suffix a paired CE/GRPO entry carries."""
    for suffix in _RUN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _student_of(name: str) -> str:
    """``s4_m1_resnet_transformer`` -> ``S4``."""
    return _base_run_name(name).split("_")[0].upper()


def _student_idx(name: str) -> int:
    """``s4_m1_resnet_transformer`` -> 4."""
    return int(_base_run_name(name).split("_")[0][1:])


def _is_m2(name: str) -> bool:
    return _PHASE_M2_MARKER in name


def _phase_of(name: str) -> str:
    return "M2" if _is_m2(name) else "M1"


def _is_grpo(name: str) -> bool:
    return any(name.endswith(s) for s in _GRPO_SUFFIXES)


_ARCH_DISPLAY = {
    # Encoder pieces
    "resnet50":  "ResNet-50",
    "resnet":    "ResNet",
    "smallcnn":  "SmallCNN",
    "vit":       "ViT",
    "clip":      "CLIP",
    # Decoder pieces
    "lstm":         "LSTM",
    "gru":          "GRU",
    "transformer":  "Transformer",
    # Attention variants
    "bahdanau":  "Bahdanau",
    "luong":     "Luong",
}


def _pretty_run_name(run_name: str) -> str:
    """Render a config-style run name as a paper-friendly model label.

    Examples
    --------
    ``s4_m1_resnet_transformer``  →  ``S4-M1 ResNet + Transformer``
    ``s1_m2_resnet_lstm_bahdanau`` →  ``S1-M2 ResNet + LSTM (Bahdanau)``
    ``s2_m1_smallcnn_gru``          →  ``S2-M1 SmallCNN + GRU``
    """
    base = _base_run_name(run_name)
    parts = base.split("_")
    if len(parts) < 3:
        return base
    student, phase, *arch_parts = parts
    pieces = [_ARCH_DISPLAY.get(p, p.capitalize()) for p in arch_parts]
    # Last piece is treated as "(Variant)" for attention names; otherwise join with " + ".
    if len(pieces) >= 3 and arch_parts[-1] in ("bahdanau", "luong"):
        head = " + ".join(pieces[:-1])
        tail = f" ({pieces[-1]})"
        arch = head + tail
    else:
        arch = " + ".join(pieces)
    return f"{student.upper()}-{phase.upper()} {arch}"


def _strip_spm_marker(text: str) -> str:
    """Remove SentencePiece word-boundary markers (▁ U+2581) from caption text.

    Predictions saved by ``inference.py`` before the detokenise switch contain
    space-separated SentencePiece pieces like ``"▁a ▁person ▁holding"``.
    Stripping the ▁ in place yields clean English ``"a person holding"`` —
    the leading marker collapses to nothing, mid-string ones merge with the
    surrounding spaces.
    """
    return text.replace("▁", "").strip()


def set_plot_style() -> None:
    """Apply the project's unified matplotlib + seaborn style.

    Called automatically by every plot function in this module (idempotent).
    Call it **once** at the top of a notebook (after imports) to unify styling for
    any manual matplotlib plots as well (Phase 1 notebook does this).

    Style specs: seaborn whitegrid + colorblind palette, DejaVu Sans, DPI 300
    (report-quality), subdued spines and grids, semibold left-aligned titles.
    """
    sns.set_theme(style=_STYLE, context=_CONTEXT, palette=_PALETTE)
    mpl.rcParams.update({
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.titleweight":      "semibold",
        "axes.titlesize":        11.5,
        "axes.titlepad":         9,
        "axes.titlelocation":    "left",
        "axes.labelsize":        10,
        "axes.edgecolor":        "#444",
        "axes.linewidth":        0.9,
        "legend.frameon":        False,
        "legend.fontsize":       9,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "xtick.color":           "#444",
        "ytick.color":           "#444",
        "figure.dpi":            300,
        "grid.alpha":            0.25,
        "grid.linewidth":        0.7,
        "font.family":           ["DejaVu Sans"],
        # Report-quality export defaults: embed TrueType (so PDF text stays
        # selectable / searchable), use STIX for maths so Δ β ε ρ render
        # properly next to body text, and trim whitespace on savefig.
        "mathtext.fontset":      "stix",
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
        "svg.fonttype":          "none",
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
    })


# =======================================================================
# Per-experiment / per-student helpers
# =======================================================================

def plot_qualitative_samples(preds_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
                             test_eval_dataset,
                             preset: Mapping[str, Any],
                             *,
                             n: int = 6,
                             seed: int = 0,
                             title: str = "") -> None:
    """Image grid + N predicted captions + up-to-3 human references, all read from disk.

    `preds_by_label` is `{label: beam_preds}` where each `beam_preds` is the list
    saved in `predictions_test.json` (e.g. `payload["ce_beam"]`, `payload["grpo_beam"]`).
    Pass one entry for a single-system gallery; pass two (CE / GRPO) and they are
    rendered side-by-side under each image — the Liang 2025 / Rennie 2017 SCST
    qualitative layout for showing what RL changed word-by-word.

    Each prediction line is colour-coded by its label (palette index in label order);
    references are muted grey. No model forward — captions come from the JSON files
    `Trainer.evaluate_on_test()` already wrote.
    """

    set_plot_style()
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(test_eval_dataset), size=n, replace=False))

    labels = list(preds_by_label)
    caps_by_label = [
        {p["image_id"]: p["caption"] for p in preds_by_label[lbl]}
        for lbl in labels
    ]
    palette = sns.color_palette(_PALETTE, n_colors=max(3, len(labels)))

    cols = min(n, 3)
    rows = max(1, (n + cols - 1) // cols)
    # Vertical room scales with how many prediction lines stack under each image.
    hspace = 1.45 + 0.35 * max(0, len(labels) - 1)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 6.0 * rows),
                             gridspec_kw={"hspace": hspace, "wspace": 0.18})
    axes = np.atleast_1d(axes).ravel()

    wrap_width = 50

    for ax, i in zip(axes, picks):
        img_tensor, img_id, refs = test_eval_dataset[i]
        vis = denormalise(img_tensor, preset.get("mean"), preset.get("std")) \
                  .permute(1, 2, 0).cpu().numpy()
        ax.imshow(vis)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"id {img_id}", fontsize=9.5, color="#555", pad=4)

        # Predictions — one wrapped block per label, stacked top-to-bottom.
        y_cursor = -0.04
        for lbl, caps_map, color in zip(labels, caps_by_label, palette):
            pred = caps_map.get(img_id, "[missing]")
            tag = f"{lbl}: "
            wrapped = textwrap.fill(f"{tag}{pred}", width=wrap_width,
                                    subsequent_indent=" " * len(tag),
                                    break_long_words=False, break_on_hyphens=False)
            ax.text(0.0, y_cursor, wrapped,
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8, family="monospace", color=color,
                    fontweight="semibold", linespacing=1.5)
            n_lines = wrapped.count("\n") + 1
            y_cursor -= 0.06 * n_lines + 0.01

        # References — wrap each ref individually with hanging indent.
        ref_indent = " " * len("ref:  ")
        ref_block = "\n".join(
            textwrap.fill(f"ref:  {r}", width=wrap_width,
                          subsequent_indent=ref_indent,
                          break_long_words=False, break_on_hyphens=False)
            for r in refs[:3]
        )
        ax.text(0.0, y_cursor, ref_block,
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, family="monospace", color="#6a6a6a",
                linespacing=1.7)

    for ax in axes[len(picks):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, y=1.0, fontsize=13, fontweight="semibold", x=0.02, ha="left")
    plt.show()


def plot_training_curves(run_dirs: Sequence[Path],
                         labels: Sequence[str] | None = None,
                         *,
                         cols: Sequence[str] = ("train_loss", "val_cider", "val_bleu4"),
                         titles: Sequence[str] | None = None,
                         smooth: bool = False,
                         mark_best: bool = True,
                         group_by: Callable[[str], Any] | None = None,
                         style_by: Callable[[str], Any] | None = None,
                         style_order: Sequence[Any] | None = None) -> None:
    """Overlay `train_log.csv` curves from multiple runs.

    For val-* columns (higher-is-better) the peak epoch is marked with a ★.
    For loss columns (lower-is-better) the min epoch is marked.

    Parameters
    ----------
    group_by
        Optional callable ``name -> group_key`` that maps each label to a
        grouping identifier. Runs in the same group share a colour. When
        ``None`` (default) every run gets its own colour.
    style_by
        Optional callable ``name -> style_key`` that maps each label to a
        line-style bucket (first style key → solid, second → dashed, ...).
        Lets a 2D axis × phase design be encoded as colour × dash pattern.
    style_order
        Optional explicit ordering of style keys, controlling which bucket
        gets which style. The first key gets solid ``"-"``, second dashed
        ``"--"``, then ``"-."`` and ``":"``. Pass e.g. ``("m2", "m1")`` to
        make M2 curves solid and M1 curves dashed (the headline/"after"
        experiment as the primary visual). If ``None`` (default), keys are
        ordered by first-encounter in the ``labels`` list.
    """

    set_plot_style()
    if labels is None:
        labels = [Path(p).name for p in run_dirs]
    if titles is None:
        titles = [c.replace("_", " ").title() for c in cols]

    dfs = [pd.read_csv(Path(p) / "train_log.csv") for p in run_dirs]

    # Colour assignment: one colour per group (default = one group per run)
    group_keys = [group_by(lbl) for lbl in labels] if group_by is not None else list(labels)
    unique_groups = list(dict.fromkeys(group_keys))
    group_palette = sns.color_palette(_PALETTE, n_colors=max(4, len(unique_groups)))
    group_colour = {g: group_palette[i] for i, g in enumerate(unique_groups)}
    colours = [group_colour[g] for g in group_keys]

    # Line style assignment: first style key → solid, then dashed, dashdot, dotted
    style_cycle = ["-", "--", "-.", ":"]
    if style_by is not None:
        style_keys = [style_by(lbl) for lbl in labels]
        if style_order is not None:
            unique_styles = list(style_order)
        else:
            unique_styles = list(dict.fromkeys(style_keys))
        style_map = {s: style_cycle[i % len(style_cycle)] for i, s in enumerate(unique_styles)}
        linestyles = [style_map[s] for s in style_keys]
    else:
        linestyles = ["-"] * len(labels)

    crowded = len(dfs) > 4
    panel_w = 5.4 if not crowded else 5.0
    # Crowded: extra height reserved at the bottom for the multi-row figure
    # legend so it never overlaps the x-axis tick labels above it.
    panel_h = 4.2 if not crowded else 5.6
    fig, axes = plt.subplots(1, len(cols), figsize=(panel_w * len(cols), panel_h),
                             sharex=True, constrained_layout=False)
    axes = np.atleast_1d(axes)

    for ax, col, title in zip(axes, cols, titles):
        lower_is_better = "loss" in col
        for df, label, color, ls in zip(dfs, labels, colours, linestyles):
            y = df[col].rolling(3, min_periods=1).mean() if smooth else df[col]
            ax.plot(df["epoch"], y, marker="o", markersize=3.5, linewidth=1.7,
                    color=color, linestyle=ls, label=label, alpha=0.88)
            if mark_best:
                idx = y.idxmin() if lower_is_better else y.idxmax()
                ax.scatter([df["epoch"][idx]], [y[idx]], s=230, marker="*",
                           color=color, edgecolor="white", linewidth=1.4, zorder=5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        if lower_is_better:
            ax.set_yscale("log")

    if crowded:
        # Reserve bottom space for the legend (turned off constrained_layout
        # above so this is honoured exactly).
        plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.30,
                            wspace=0.22)
        handles, lbls = axes[0].get_legend_handles_labels()
        # Cap ncol at 2 — run names like s5_m2_vit_gru_bahdanau are long, and
        # 3+ columns force horizontal overlap once the cohort hits 10 runs.
        ncol = min(len(lbls), 2)
        fig.legend(handles, lbls, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=ncol, fontsize=8.5, frameon=False,
                   columnspacing=3.0, handletextpad=0.6,
                   labelspacing=0.6)
    else:
        plt.tight_layout()
        for ax in axes:
            ax.legend(fontsize=8.5, loc="best")
    plt.show()


def plot_grpo_curves(run_dirs: Sequence[Path],
                     labels: Sequence[str] | None = None,
                     *,
                     ce_baselines: Mapping[str, float] | None = None) -> None:
    """Visualise Stage-2 GRPO training from `grpo_log.csv`.

    Four panels in a 2×2 grid (Liang 2025 / Shao 2024 layout):
      - reward_mean (with ±std band) — the RL signal driving updates
      - val_cider — the actual eval metric we care about
      - KL(π_θ ‖ π_ref) — drift from the Stage-1 reference policy
      - entropy of π_θ — exploration / mode-collapse diagnostic

    Reference threshold lines:
      - KL = 1.0 (Schulman 2017 PPO heuristic — above this is "unstable")
      - entropy = 1.0 (≈ exp(1) ≈ 2.7 effective candidates per token, the
        usual mode-collapse warning level for ~10k-vocab captioning)

    Parameters
    ----------
    run_dirs
        One or more output directories each containing a `grpo_log.csv`.
    labels
        Optional run labels (default: directory basenames).
    ce_baselines
        Optional `{label: ce_val_cider}` to plot a horizontal dashed line
        on the val-cider panel for the pre-RL Stage-1 score, so the reader
        can see whether GRPO actually improved over CE.
    """

    set_plot_style()
    if labels is None:
        labels = [Path(p).name for p in run_dirs]
    dfs = [pd.read_csv(Path(p) / "grpo_log.csv") for p in run_dirs]
    n = len(dfs)
    palette = sns.color_palette(_PALETTE, n_colors=max(4, n))

    # When there are many runs, widen the figure and hoist the legend out to
    # the right so curves stay readable. ≤4 runs: compact single legend per
    # axes. >4 runs: one shared legend outside the figure.
    crowded = n > 4
    fig_w = 12.5 if crowded else 11.5
    fig_h = 8.4 if crowded else 7.5
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
    (ax_r, ax_v), (ax_kl, ax_h) = axes

    band_alpha = 0.03 if crowded else 0.13
    for df, lbl, c in zip(dfs, labels, palette):
        ax_r.plot(df["epoch"], df["grpo_reward_mean"], marker="o", markersize=4,
                  linewidth=1.8, color=c, label=lbl)
        ax_r.fill_between(df["epoch"],
                          df["grpo_reward_mean"] - df["grpo_reward_std"],
                          df["grpo_reward_mean"] + df["grpo_reward_std"],
                          color=c, alpha=band_alpha)
    ax_r.set_xlabel("GRPO epoch"); ax_r.set_ylabel("reward (CIDEr-D, ±std)")
    ax_r.set_title("Reward trajectory")

    # 2. val_cider (★ = peak; CE baselines as dotted lines, only for ≤4 runs to
    # avoid clutter — otherwise they blur into the data).
    for df, lbl, c in zip(dfs, labels, palette):
        ax_v.plot(df["epoch"], df["val_cider"], marker="o", markersize=4,
                  linewidth=1.8, color=c, label=lbl)
        idx = df["val_cider"].idxmax()
        ax_v.scatter([df["epoch"][idx]], [df["val_cider"][idx]], s=220, marker="*",
                     color=c, edgecolor="white", linewidth=1.4, zorder=5)
        if not crowded and ce_baselines is not None and lbl in ce_baselines:
            ax_v.axhline(ce_baselines[lbl], color=c, linestyle=":", linewidth=1.0,
                         alpha=0.7)
    ax_v.set_xlabel("GRPO epoch"); ax_v.set_ylabel("val CIDEr")
    title_v = "Val CIDEr (★ = peak)"
    if not crowded and ce_baselines:
        title_v += " — dotted = CE baseline"
    ax_v.set_title(title_v)

    # 3. KL with instability threshold
    for df, lbl, c in zip(dfs, labels, palette):
        ax_kl.plot(df["epoch"], df["grpo_kl"], marker="o", markersize=4,
                   linewidth=1.8, color=c, label=lbl)
    ax_kl.axhline(1.0, color="#888", linestyle="--", linewidth=1.0,
                  label="KL = 1.0 (PPO instability)")
    ax_kl.set_xlabel("GRPO epoch"); ax_kl.set_ylabel("KL(π_θ ‖ π_ref)")
    ax_kl.set_title("Policy drift from reference")

    # 4. entropy with mode-collapse threshold
    for df, lbl, c in zip(dfs, labels, palette):
        ax_h.plot(df["epoch"], df["grpo_entropy"], marker="o", markersize=4,
                  linewidth=1.8, color=c, label=lbl)
    ax_h.axhline(1.0, color="#888", linestyle="--", linewidth=1.0,
                 label="H = 1.0 (≈ 2.7 eff. tokens, collapse warning)")
    ax_h.set_xlabel("GRPO epoch"); ax_h.set_ylabel("entropy of π_θ")
    ax_h.set_title("Policy entropy (exploration vs collapse)")

    if crowded:
        handles, lbls = ax_r.get_legend_handles_labels()
        # Cap ncol at 3 — run names like s5_m2_vit_gru_bahdanau are long and
        # collide horizontally at 4 columns once the cohort hits 10 runs.
        ncol = min(len(lbls), 3)
        fig.legend(handles, lbls, loc="lower center",
                   bbox_to_anchor=(0.5, -0.08),
                   ncol=ncol, fontsize=8, frameon=False,
                   columnspacing=2.2, handletextpad=0.6)
    else:
        for ax in (ax_r, ax_v, ax_kl, ax_h):
            ax.legend(fontsize=8.5, loc="best")

    fig.suptitle("GRPO Stage-2 training diagnostics",
                 y=1.02, fontsize=13, fontweight="semibold", x=0.02, ha="left")
    plt.show()


def plot_caption_lengths_compared(preds_by_model: Mapping[str, Iterable[Mapping[str, Any]]],
                                  references_by_id: Mapping[int, Sequence[str]],
                                  *,
                                  title: str = "",
                                  y_clip_quantile: float = 0.995) -> None:
    """Single figure: N models + reference caption-length distributions as violins.

    The violin shows full distribution shape + quartiles (`inner="quart"`) and a
    dashed horizontal line at the reference mean. Numeric stats (mean, std, median,
    Q1/Q3, mean_ratio vs ref) live in a separate `caption_length_stats_table(...)`
    call so the figure stays uncluttered and the report can quote exact values.
    """

    set_plot_style()
    ref_lens = np.asarray([len(r.split()) for refs in references_by_id.values() for r in refs])
    ref_mean = float(ref_lens.mean())

    rows = [{"source": "references", "length": int(l)} for l in ref_lens]
    for name, preds in preds_by_model.items():
        for p in preds:
            rows.append({"source": name, "length": int(len(p["caption"].split()))})
    df = pd.DataFrame(rows)

    n_models = len(preds_by_model)
    order = ["references"] + list(preds_by_model)
    palette = [sns.color_palette(_PALETTE)[7]] + list(sns.color_palette(_PALETTE, n_colors=max(4, n_models)))[:n_models]

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * (n_models + 1) + 2), 5.2))
    sns.violinplot(data=df, x="source", y="length", hue="source", order=order,
                   palette=palette, ax=ax, cut=0, bw_adjust=0.6, inner="quart",
                   linewidth=1.1, saturation=0.85, legend=False)

    ax.axhline(ref_mean, color=sns.color_palette(_PALETTE)[7], linestyle="--",
               linewidth=1.0, alpha=0.55, zorder=0)

    y_cap = float(df["length"].quantile(y_clip_quantile))
    ax.set_ylim(0, y_cap * 1.08)

    ax.set_xlabel("")
    ax.set_ylabel("caption length (words)")
    ax.set_title(title or "Caption length — every model vs. references")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def caption_length_stats_table(preds_by_model: Mapping[str, Iterable[Mapping[str, Any]]],
                               references_by_id: Mapping[int, Sequence[str]]):
    """Return a styled table of caption-length stats for N models + references.

    One row per source: n, mean, std, median, Q1, Q3, min, max, mean_ratio
    (model_mean / ref_mean). Use alongside `plot_caption_lengths_compared` when
    you want richer pre/post-RL coverage in the table than the violin shows
    (e.g. CE vs CE→GRPO for both models).
    """

    ref_lens = np.asarray([len(r.split()) for refs in references_by_id.values() for r in refs])
    ref_mean = float(ref_lens.mean())

    rows = [{
        "source": "references",
        "n": int(ref_lens.size),
        "mean": ref_mean, "std": float(ref_lens.std()),
        "median": float(np.median(ref_lens)),
        "Q1": float(np.quantile(ref_lens, 0.25)),
        "Q3": float(np.quantile(ref_lens, 0.75)),
        "min": float(ref_lens.min()), "max": float(ref_lens.max()),
        "mean_ratio": 1.0,
    }]
    for name, preds in preds_by_model.items():
        lens = np.asarray([len(p["caption"].split()) for p in preds])
        rows.append({
            "source": name,
            "n": int(lens.size),
            "mean": float(lens.mean()), "std": float(lens.std()),
            "median": float(np.median(lens)),
            "Q1": float(np.quantile(lens, 0.25)),
            "Q3": float(np.quantile(lens, 0.75)),
            "min": float(lens.min()), "max": float(lens.max()),
            "mean_ratio": float(lens.mean()) / ref_mean,
        })
    stats_df = pd.DataFrame(rows).set_index("source")
    return (stats_df.style
            .background_gradient(subset=["mean"], cmap="Blues")
            .background_gradient(subset=["std"], cmap="Oranges")
            .background_gradient(subset=["mean_ratio"], cmap="RdYlGn",
                                 vmin=0.5, vmax=1.5)
            .format({"n": "{:.0f}", "mean": "{:.2f}", "std": "{:.2f}",
                     "median": "{:.0f}", "Q1": "{:.0f}", "Q3": "{:.0f}",
                     "min": "{:.0f}", "max": "{:.0f}",
                     "mean_ratio": "{:.2f}×"}))


def plot_caption_length_distribution(predictions: Iterable[Mapping[str, Any]],
                                     references_by_id: Mapping[int, Sequence[str]],
                                     *,
                                     title: str = "",
                                     label: str = "predictions") -> None:
    """Overlay predicted vs reference caption length (word counts) with hist + KDE.

    Prints a verdict — useful VizWiz-specific signal:
      • ratio < 0.5     → caption collapse (model emits ~5-7 words vs refs' ~10-15)
      • 0.5 ≤ r < 0.8   → short
      • 0.8 ≤ r < 1.3   → ok
      • r ≥ 1.3          → long
    """

    set_plot_style()
    palette = sns.color_palette(_PALETTE)
    pred_lens = np.asarray([len(p["caption"].split()) for p in predictions])
    ref_lens = np.asarray([len(r.split()) for refs in references_by_id.values() for r in refs])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    sns.histplot(ref_lens, bins=30, stat="density", ax=ax,
                 color=palette[7], alpha=0.45, label="references", edgecolor="none")
    sns.kdeplot(ref_lens, ax=ax, color=palette[7], linewidth=1.8, alpha=0.9)
    sns.histplot(pred_lens, bins=30, stat="density", ax=ax,
                 color=palette[0], alpha=0.65, label=label, edgecolor="none")
    sns.kdeplot(pred_lens, ax=ax, color=palette[0], linewidth=2.0)

    ref_mean, pred_mean = float(ref_lens.mean()), float(pred_lens.mean())
    ref_std, pred_std = float(ref_lens.std()), float(pred_lens.std())
    ax.axvline(ref_mean, color=palette[7], linestyle="--", linewidth=1.1)
    ax.axvline(pred_mean, color=palette[0], linestyle="--", linewidth=1.3)

    ratio = pred_mean / ref_mean
    verdict, badge_color = next(
        (label, color) for threshold, label, color in _LENGTH_VERDICTS if ratio < threshold
    )
    ax.text(0.98, 0.95,
            f"pred = {pred_mean:5.1f} ± {pred_std:.1f} words\n"
            f" ref = {ref_mean:5.1f} ± {ref_std:.1f} words\n"
            f"pred/ref = {ratio:.2f}  [{verdict}]",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, family="monospace", color=badge_color,
            bbox=dict(facecolor="white", edgecolor=badge_color, boxstyle="round,pad=0.5", alpha=0.9))

    ax.set_xlabel("caption length (words)")
    ax.set_ylabel("density")
    ax.set_title(title or "Caption length — predictions vs. references")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def metrics_comparison_table(m1: Mapping[str, Any], m2: Mapping[str, Any],
                             names: Sequence[str] = ("Model 1", "Model 2"),
                             *,
                             styled: bool = False):
    """Compare two metric dicts; optional pandas Styler with inline bars + delta colour map."""

    keys = ["bleu1", "bleu2", "bleu3", "bleu4", "cider", "rouge_l"]
    rows = []
    for k in keys:
        v1, v2 = m1.get(k), m2.get(k)
        delta = round(v2 - v1, 4) if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else None
        rows.append({
            "metric": k,
            names[0]: round(v1, 4) if isinstance(v1, (int, float)) else None,
            names[1]: round(v2, 4) if isinstance(v2, (int, float)) else None,
            "delta": delta,
        })
    df = pd.DataFrame(rows).set_index("metric")
    if not styled:
        return df
    return (df.style
              .background_gradient(subset=list(names), cmap="Greens", axis=1)
              .background_gradient(subset=["delta"], cmap="RdYlGn", vmin=-0.2, vmax=0.2)
              .bar(subset=list(names), color=["#c3e7d4", "#8ed1b1"], align="left", height=60)
              .format("{:.4f}", na_rep="-"))


# =======================================================================
# Group-level cross-model comparisons
# =======================================================================

def plot_metric_bars(metric_dicts: Mapping[str, Mapping[str, float]],
                     *,
                     metrics: Sequence[str] = ("bleu1", "bleu4", "cider", "rouge_l"),
                     title: str = "",
                     highlight: str | None = None,
                     show_values: bool = True,
                     orient: str | None = None) -> None:
    """Grouped bar chart comparing several models across several metrics.

    `orient='h'` for horizontal bars (auto-selected when N > 5 so labels fit).
    `highlight`: name of one model to draw in a stronger colour.
    """

    set_plot_style()
    models = list(metric_dicts)
    n = len(models)
    if orient is None:
        orient = "h" if n > 5 else "v"

    df = pd.DataFrame(
        [{"model": m, "metric": k, "value": metric_dicts[m].get(k, np.nan)}
         for m in models for k in metrics]
    )
    palette = sns.color_palette(_PALETTE, n_colors=n)
    if highlight and highlight in models:
        base = sns.color_palette("pastel", n_colors=n)
        palette = [sns.color_palette("colorblind")[3] if m == highlight else base[i]
                   for i, m in enumerate(models)]

    if orient == "v":
        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * n * len(metrics) / 2 + 2), 4.4))
        sns.barplot(data=df, x="metric", y="value", hue="model",
                    palette=palette, ax=ax, edgecolor="white", linewidth=0.8)
        ax.set_xlabel(""); ax.set_ylabel("score")
    else:
        fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.45 * n * len(metrics) + 1.5)))
        sns.barplot(data=df, y="metric", x="value", hue="model",
                    palette=palette, ax=ax, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("score"); ax.set_ylabel("")

    ax.set_title(title)
    ax.legend(title="", fontsize=9, ncols=min(n, 4),
              loc="upper right" if orient == "v" else "lower right")
    ax.grid(True, axis="x" if orient == "h" else "y", alpha=0.25)

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2,
                         color="#333")
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(metric_dicts: Mapping[str, Mapping[str, float]],
                        *,
                        metrics: Sequence[str] = ("bleu1", "bleu2", "bleu3", "bleu4", "cider", "rouge_l"),
                        title: str = "",
                        normalise_per_column: bool = True,
                        highlight_max: bool = True,
                        group_separator: str | None = "_m2_",
                        cmap: str = "rocket_r",
                        block_size: int | None = None) -> None:
    """Heatmap with rows = models, cols = metrics.

    - `normalise_per_column=True` scales each metric to [0, 1] for colouring
      (raw values still annotated on cells).
    - `highlight_max=True` draws a thick border around the column-maximum cell.
    - `group_separator` inserts a horizontal separator between rows whose names
      contain that substring — e.g. `_m2_` draws a line between Phase 2 and Phase 3.
    """

    set_plot_style()
    models = list(metric_dicts)
    raw = np.array([[metric_dicts[m].get(k, np.nan) for k in metrics] for m in models], dtype=float)
    norm = raw.copy()
    if normalise_per_column:
        cmin = np.nanmin(raw, axis=0, keepdims=True)
        cmax = np.nanmax(raw, axis=0, keepdims=True)
        denom = np.where(cmax - cmin < 1e-9, 1.0, cmax - cmin)
        norm = (raw - cmin) / denom

    df = pd.DataFrame(norm, index=models, columns=list(metrics))
    annot = pd.DataFrame(raw, index=models, columns=list(metrics))

    fig, ax = plt.subplots(figsize=(1.4 * len(metrics) + 2.5, 0.58 * len(models) + 2.2))
    sns.heatmap(df, annot=annot.round(3).astype(str), fmt="s",
                cmap=cmap, cbar_kws={"label": "normalised score" if normalise_per_column else "score",
                                            "shrink": 0.75, "pad": 0.02},
                linewidths=0.6, linecolor="white", ax=ax,
                annot_kws={"fontsize": 9.5})
    ax.set_title(title)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    if highlight_max and len(models) > 1:
        for col_idx in range(len(metrics)):
            col_vals = raw[:, col_idx]
            if np.isnan(col_vals).all():
                continue
            best_row = int(np.nanargmax(col_vals))
            ax.add_patch(mp.Rectangle((col_idx, best_row), 1, 1, fill=False,
                                       edgecolor="#222", linewidth=2.2, zorder=5))

    # Row separators: solid line where the student changes; dotted line at the
    # first row whose name matches ``group_separator`` (legacy 2-run use case).
    if len(models) > 1:
        prev_key = _student_of(models[0])
        for row_idx in range(1, len(models)):
            key = _student_of(models[row_idx])
            if key != prev_key:
                ax.hlines(row_idx, *ax.get_xlim(), colors="#222", linewidth=1.3)
            prev_key = key
    if group_separator:
        for row_idx, name in enumerate(models):
            if group_separator in name and row_idx > 0:
                ax.hlines(row_idx, *ax.get_xlim(), colors="#222", linewidth=1.3,
                          linestyles=":")
                break

    if block_size and block_size > 1 and len(models) > block_size:
        for row_idx in range(block_size, len(models), block_size):
            ax.hlines(row_idx, *ax.get_xlim(), colors="#bbb", linewidth=0.8,
                      linestyles="-")

    plt.tight_layout()
    plt.show()


def plot_metric_scatter(metric_dicts: Mapping[str, Mapping[str, float]],
                        *,
                        x: str = "bleu4",
                        y: str = "cider",
                        title: str = "",
                        annotate: bool = True) -> None:
    """Pareto-style scatter across runs: x vs y metric.

    Encoding:
      - Colour = phase / stage (M1 CE, M2 CE, GRPO) — see legend
      - Shape  = student (one shape per student index)

    The colour + shape encoding plus the legend in the lower right is enough
    to identify each marker; no inline text labels are drawn.

    If ``metric_dicts`` contains paired CE / GRPO entries (same run with a
    ``" (CE)"`` / ``" (GRPO)"`` suffix), draws an arrow from CE → GRPO so the
    RL-induced movement direction is visually obvious.
    """

    set_plot_style()
    palette = sns.color_palette(_PALETTE)
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    # Per-student arrow palette (colorblind-safe). Lets the reader trace a
    # CE→GRPO transition inside a dense cluster: the arrow's colour matches
    # the student key in the lower-right legend, while marker colour still
    # encodes phase (M1 / M2 / GRPO).
    student_indices: list[int] = []
    for name in metric_dicts:
        s = _student_idx(name)
        if s not in student_indices:
            student_indices.append(s)
    student_indices.sort()
    arrow_palette = sns.color_palette("colorblind", n_colors=max(5, len(student_indices)))
    student_arrow_colour = {s: arrow_palette[i] for i, s in enumerate(student_indices)}

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    by_base: Dict[str, Dict[str, tuple]] = {}
    base_to_name: Dict[str, str] = {}
    for name, m in metric_dicts.items():
        xi, yi = m.get(x), m.get(y)
        if xi is None or yi is None:
            continue
        bn = _base_run_name(name)
        stage = "grpo" if _is_grpo(name) else "ce"
        by_base.setdefault(bn, {})[stage] = (xi, yi)
        base_to_name[bn] = name

    for bn, stages in by_base.items():
        if "ce" in stages and "grpo" in stages:
            (x0, y0), (x1, y1) = stages["ce"], stages["grpo"]
            s = _student_idx(base_to_name[bn])
            arrow_col = student_arrow_colour[s]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=arrow_col,
                                         linewidth=1.2, alpha=0.75,
                                         shrinkA=3, shrinkB=3,
                                         connectionstyle="arc3,rad=0.14"),
                        zorder=2)

    for name, m in metric_dicts.items():
        xi, yi = m.get(x), m.get(y)
        if xi is None or yi is None:
            continue
        s = _student_idx(name)
        marker = markers[(s - 1) % len(markers)]
        if _is_grpo(name):
            color, edge = palette[3], "#444"
            phase_label, size = "GRPO (Stage 2)", 9
        elif _is_m2(name):
            color, edge = palette[2], "#222"
            phase_label, size = "Phase 3 (M2 · CE)", 14
        else:
            color, edge = palette[0], "#444"
            phase_label, size = "Phase 2 (M1 · CE)", 14
        ax.scatter(xi, yi, s=size, marker=marker, color=color, edgecolor=edge,
                   linewidth=0.4, alpha=0.92, label=phase_label, zorder=3)

    # Two legends, both lower-right: phase legend on top, student-arrow key
    # underneath. Arrow colour identifies the student each CE→GRPO trajectory
    # belongs to, useful inside the dense central cluster.
    from matplotlib.lines import Line2D
    phase_handles_seen, phase_lbls_seen = ax.get_legend_handles_labels()
    unique_phase = dict(zip(phase_lbls_seen, phase_handles_seen))
    leg_phase = ax.legend(unique_phase.values(), unique_phase.keys(),
                          loc="lower right", title="Phase",
                          fontsize=8, title_fontsize=8.5, framealpha=0.92)
    ax.add_artist(leg_phase)

    student_handles = [
        Line2D([0], [0], color=student_arrow_colour[s], lw=2.2,
               marker="", label=f"S{s}")
        for s in student_indices
    ]
    ax.legend(handles=student_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.27),
              title="Student (arrow)", ncol=2,
              fontsize=8, title_fontsize=8.5, framealpha=0.92,
              handlelength=1.6, columnspacing=1.0, handletextpad=0.5)

    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())
    ax.set_title(title or f"{x.upper()} vs. {y.upper()} — Pareto view")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cross_model_gallery(per_run_preds: Mapping[str, Mapping[int, str]],
                             test_eval_ds,
                             preset: Mapping[str, Any],
                             *,
                             n: int = 6,
                             seed: int = 0,
                             cols: int = 3,
                             title: str = "") -> None:
    """For `n` test images, show the image once + every model's prediction stacked below.

    `per_run_preds`: `{run_name: {image_id: caption}}`
    `test_eval_ds`: a `VizWizEvalDataset` (needed for image tensor + references)
    `preset`: image transform preset dict (supplies `mean` / `std` for denormalisation)
    `cols`: number of columns (default 2 — fits a report page width better than
      the original 1-column long strip).

    The single most compact evidence figure for the Report § 6 — shows the reader
    *where* each model succeeds or fails on the *same* inputs.
    """

    set_plot_style()
    palette = sns.color_palette(_PALETTE, n_colors=max(4, len(per_run_preds)))
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(test_eval_ds), size=n, replace=False))

    tf = build_transform(preset, train=False)

    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    row_h = 5.0 + 0.42 * len(per_run_preds)
    fig, axes = plt.subplots(rows, cols, figsize=(8.5 * cols, row_h * rows),
                             gridspec_kw={"hspace": 1.45, "wspace": 0.18})
    axes = np.atleast_1d(axes).ravel()

    if cols >= 3:
        wrap_width = 50
    elif cols == 2:
        wrap_width = 72
    else:
        wrap_width = 95

    PRED_FS = 9.5
    REF_FS  = 9.0

    for ax, i in zip(axes, picks):
        rec = test_eval_ds.records[i]
        pil = _PIL.open(test_eval_ds.images_dir / rec["file_name"]).convert("RGB")
        img_t = tf(pil)
        vis = denormalise(img_t, preset.get("mean"), preset.get("std")).permute(1, 2, 0).cpu().numpy()
        ax.imshow(vis)
        ax.set_anchor("NW")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"image id {rec['image_id']}", fontsize=11, color="#555", pad=4, loc="left")

        line_h_pred = 0.048
        line_h_ref  = 0.044
        cap_indent  = "  "
        y = -0.04
        for k, (run_name, preds) in enumerate(per_run_preds.items()):
            cap = _strip_spm_marker(preds.get(rec["image_id"], "—"))
            label = _pretty_run_name(run_name)
            ax.text(0.0, y, label, transform=ax.transAxes, va="top", ha="left",
                    fontsize=PRED_FS, family="monospace",
                    color=palette[k % len(palette)], fontweight="semibold")
            y -= line_h_pred
            wrapped_cap = textwrap.fill(cap, width=wrap_width,
                                        initial_indent=cap_indent,
                                        subsequent_indent=cap_indent,
                                        break_long_words=False,
                                        break_on_hyphens=False)
            n_cap_lines = wrapped_cap.count("\n") + 1
            ax.text(0.0, y, wrapped_cap, transform=ax.transAxes, va="top", ha="left",
                    fontsize=PRED_FS, family="monospace",
                    color=palette[k % len(palette)])
            y -= line_h_pred * n_cap_lines + 0.006

        y -= 0.014
        ax.text(0.0, y, "refs:", transform=ax.transAxes, va="top", ha="left",
                fontsize=PRED_FS, family="monospace", color="#444", fontweight="semibold")
        y -= line_h_ref
        ref_indent = "  "
        for r in rec["captions"][:2]:
            wrapped = textwrap.fill(f"• {r}", width=wrap_width,
                                    subsequent_indent=ref_indent,
                                    break_long_words=False,
                                    break_on_hyphens=False)
            n_lines = wrapped.count("\n") + 1
            ax.text(0.02, y, wrapped, transform=ax.transAxes, va="top", ha="left",
                    fontsize=REF_FS, family="monospace", color="#6a6a6a")
            y -= line_h_ref * n_lines

    for ax in axes[len(picks):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, y=1.005, fontsize=13.5, fontweight="semibold", x=0.02, ha="left")
    plt.show()


# =======================================================================
# Table helpers (pandas DataFrame)
# =======================================================================

def _classify_param(name: str, requires_grad: bool) -> str:
    """Bucket a named_parameter into one of the 5 display groups.

    Schema (uniform across LSTM / LSTMAttn / GRU / GRUAttn / Transformer):

      encoder.frozen       — encoder params with requires_grad=False
      encoder.unfrozen     — encoder params unfrozen by partial fine-tuning
      decoder.embedding    — `decoder.embed` (token embeddings)
      decoder.core         — RNN cells / Transformer blocks / attention modules
      decoder.adapters     — memory_proj + init_h + positional norms + output head
    """
    if name.startswith("encoder."):
        return "encoder.frozen" if not requires_grad else "encoder.unfrozen"
    lname = name.lower()
    if ".embed" in lname:
        return "decoder.embedding"
    if any(k in lname for k in ("memory_proj", "init_h", "decoder.out",
                                "pos_enc", "positional")):
        return "decoder.adapters"
    return "decoder.core"


def parameter_breakdown_table(model: nn.Module,
                              *,
                              m_unit: float = 1e6) -> "pd.DataFrame":
    """Return a 6-row parameter-breakdown table (with TOTAL row).

    Columns: ``params_M``, ``trainable_M``, ``trainable_%``, ``note``.

    Notes column auto-detects:
      - Transformer tied embedding (``decoder.embed.weight is decoder.out.weight``)
      - Bahdanau / Luong attention presence
      - Per-family encoder unfreeze pattern (ResNet layer4 / ViT last-N / CLIP last-N)

    Designed for the report's "Justification of architectural choices"
    section — every architecture intervention lands on a dedicated row with
    its parameter count as evidence.
    """

    buckets: Dict[str, Dict[str, float]] = {}
    for name, p in model.named_parameters():
        key = _classify_param(name, p.requires_grad)
        b = buckets.setdefault(key, {"params": 0, "trainable": 0})
        b["params"] += p.numel()
        if p.requires_grad:
            b["trainable"] += p.numel()

    # Autodetect note per row
    notes: Dict[str, str] = {}

    # Tied embedding note (Transformer: decoder.embed.weight is decoder.out.weight)
    dec = getattr(model, "decoder", None)
    embed_mod = getattr(dec, "embed", None) if dec is not None else None
    out_mod = getattr(dec, "out", None) if dec is not None else None
    if (embed_mod is not None and out_mod is not None
            and getattr(embed_mod, "weight", None) is getattr(out_mod, "weight", None)):
        notes["decoder.embedding"] = "tied with output projection (Press & Wolf 2017)"

    # Attention presence
    attn_mod = getattr(dec, "attn", None) if dec is not None else None
    if attn_mod is not None:
        notes["decoder.core"] = f"{type(attn_mod).__name__} + RNN cells"

    # Encoder family fingerprint
    enc = getattr(model, "encoder", None)
    if enc is not None:
        enc_name = type(enc).__name__
        if "ResNet" in enc_name:
            notes["encoder.frozen"] = "ResNet stem + layer1-3 (frozen)"
            if "encoder.unfrozen" in buckets:
                notes["encoder.unfrozen"] = "ResNet layer4 (Kumar 2022)"
        elif "CLIP" in enc_name:
            notes["encoder.frozen"] = "CLIP ViT shallow blocks (frozen)"
            if "encoder.unfrozen" in buckets:
                notes["encoder.unfrozen"] = "CLIP last-N blocks + post_layernorm (Wei 2023)"
        elif "ViT" in enc_name:
            notes["encoder.frozen"] = "ViT shallow blocks (frozen)"
            if "encoder.unfrozen" in buckets:
                notes["encoder.unfrozen"] = "ViT last-N blocks + LN (Kumar 2022)"
        elif "SmallCNN" in enc_name:
            notes["encoder.unfrozen"] = "scratch-trained (no ImageNet)"

    # Build rows in canonical order (missing buckets omitted)
    order = ["encoder.frozen", "encoder.unfrozen",
             "decoder.embedding", "decoder.core", "decoder.adapters"]
    rows = []
    for key in order:
        if key not in buckets:
            continue
        b = buckets[key]
        pct = 100.0 * b["trainable"] / b["params"] if b["params"] > 0 else 0.0
        rows.append({
            "group":       key,
            "params_M":    round(b["params"] / m_unit, 3),
            "trainable_M": round(b["trainable"] / m_unit, 3),
            "trainable_%": round(pct, 1),
            "note":        notes.get(key, ""),
        })

    total_params = sum(b["params"] for b in buckets.values())
    total_trainable = sum(b["trainable"] for b in buckets.values())
    rows.append({
        "group":       "TOTAL",
        "params_M":    round(total_params / m_unit, 3),
        "trainable_M": round(total_trainable / m_unit, 3),
        "trainable_%": round(100.0 * total_trainable / max(1, total_params), 1),
        "note":        "",
    })

    return pd.DataFrame(rows).set_index("group")


def hyperparameter_summary_table(cfg: Mapping[str, Any]):
    """Side-by-side hyperparameter summary extracted from a run config.

    Left block = Stage-1 (CE) hyperparameters from ``cfg['training']`` /
    ``cfg['eval']``. Right block = Stage-2 (GRPO) hyperparameters from
    ``cfg['grpo']``, shown only when ``cfg['grpo']['enabled']``.

    Returns a pandas ``Styler`` ready for ``display(...)`` — the row index
    is hidden so the table reads as four flat columns under two grouped
    headers (``CE (Stage-1)`` / ``GRPO (Stage-2)``).
    """

    tr = dict(cfg.get("training", {}))
    ev = dict(cfg.get("eval", {}))
    grpo = dict(cfg.get("grpo", {}))

    def _fmt_lr(v):
        return f"{float(v):.1e}"

    ce_rows = [
        ("optimizer",           tr.get("optimizer", "adamw")),
        ("base lr",             _fmt_lr(tr.get("lr"))),
        ("encoder lr scale",    tr.get("encoder_lr_scale", 1.0)),
        ("encoder lr decay",    tr.get("encoder_lr_decay", 1.0)),
        ("betas",               str(tr.get("betas", "(0.9, 0.999)"))),
        ("weight decay",        tr.get("weight_decay", 0.0)),
        ("batch size",          tr.get("batch_size")),
        ("epochs",              tr.get("epochs")),
        ("scheduler",           tr.get("scheduler", "cosine")),
        ("warmup steps",        tr.get("warmup_steps", 0)),
        ("label smoothing",     tr.get("label_smoothing", 0.0)),
        ("grad clip",           tr.get("clip_norm", "—")),
        ("early stop patience", tr.get("early_stop_patience") or "disabled"),
        ("AMP",                 tr.get("amp", False)),
        ("beam size",           ev.get("beam_size", "—")),
        ("length penalty α",    ev.get("length_penalty", 1.0)),
        ("max gen len",         ev.get("max_gen_len", 52)),
    ]

    grpo_rows: list[tuple[str, Any]] = []
    if grpo.get("enabled"):
        grpo_rows = [
            ("epochs",     grpo.get("epochs")),
            ("lr",         _fmt_lr(grpo.get("lr"))),
            ("group size", grpo.get("group_size")),
            ("clip ε",     grpo.get("clip_eps")),
            ("KL β",       grpo.get("kl_beta")),
            ("reward",     grpo.get("reward")),
        ]

    n = max(len(ce_rows), len(grpo_rows))
    pad = ("", "")
    ce_rows   = ce_rows   + [pad] * (n - len(ce_rows))
    grpo_rows = grpo_rows + [pad] * (n - len(grpo_rows))

    df = pd.DataFrame({
        ("CE (Stage-1)",   "hyperparameter"): [r[0] for r in ce_rows],
        ("CE (Stage-1)",   "value"):          [r[1] for r in ce_rows],
        ("GRPO (Stage-2)", "hyperparameter"): [r[0] for r in grpo_rows],
        ("GRPO (Stage-2)", "value"):          [r[1] for r in grpo_rows],
    })
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.style.hide(axis="index")


def architecture_comparison_table(model_m1: nn.Module, model_m2: nn.Module,
                                  *,
                                  names: Sequence[str] = ("Model 1", "Model 2")
                                  ) -> "pd.DataFrame":
    """Side-by-side parameter breakdown for two models.

    Rows = five canonical groups + TOTAL; columns = ``(name, 'params_M')``
    and ``(name, 'trainable_M')`` under a multi-index, plus a shared ``note``
    column inferred from M1 (attention, tied embed, encoder family).

    Used in the per-student "Model 1 vs Model 2" comparison section to
    highlight how the architectural axis change redistributes parameters.
    """

    t1 = parameter_breakdown_table(model_m1)
    t2 = parameter_breakdown_table(model_m2)

    all_groups = [g for g in ["encoder.frozen", "encoder.unfrozen",
                              "decoder.embedding", "decoder.core",
                              "decoder.adapters", "TOTAL"]
                  if g in set(t1.index) | set(t2.index)]

    def _col(tbl, field):
        return [tbl.loc[g, field] if g in tbl.index else 0.0 for g in all_groups]

    frame = pd.DataFrame(
        {
            (names[0], "params_M"):    _col(t1, "params_M"),
            (names[0], "trainable_M"): _col(t1, "trainable_M"),
            (names[1], "params_M"):    _col(t2, "params_M"),
            (names[1], "trainable_M"): _col(t2, "trainable_M"),
        },
        index=all_groups,
    )
    frame.columns = pd.MultiIndex.from_tuples(list(frame.columns))
    frame.index.name = "group"
    return frame


def metrics_table_6block(metrics: Mapping[str, Any],
                         *,
                         metric_keys: Sequence[str] = ("bleu1", "bleu2", "bleu3",
                                                        "bleu4", "cider", "rouge_l")
                         ) -> "pd.DataFrame":
    """Render a 6-block ``metrics.json`` as a 4-row × K-col DataFrame.

    The 6-block schema holds: ``{greedy, beam, best_val, grpo_greedy, grpo_beam,
    grpo_best_val}`` — this helper keeps the four report-relevant rows
    (CE/GRPO × greedy/beam) and drops ``best_val`` which is diagnostic only.

    Missing blocks (e.g. GRPO skipped) are filled with NaN.
    """

    row_map = [
        ("CE · greedy",   metrics.get("greedy")),
        ("CE · beam",     metrics.get("beam")),
        ("GRPO · greedy", metrics.get("grpo_greedy")),
        ("GRPO · beam",   metrics.get("grpo_beam")),
    ]
    data = []
    for label, block in row_map:
        if block is None:
            data.append({k: None for k in metric_keys})
        else:
            data.append({k: block.get(k) for k in metric_keys})
    df = pd.DataFrame(data, index=[lbl for lbl, _ in row_map])
    df.index.name = "decoding"
    return df.round(4)


def prediction_diff_table(preds_a: Sequence[Mapping[str, Any]],
                          preds_b: Sequence[Mapping[str, Any]],
                          references_by_id: Mapping[int, Sequence[str]],
                          *,
                          names: Sequence[str] = ("A", "B"),
                          n: int = 5,
                          seed: int = 0) -> "pd.DataFrame":
    """Pick ``n`` images where predictions A and B differ; render 4-col table.

    Columns: ``image_id``, ``<name_A>``, ``<name_B>``, ``references``.

    Used for CE vs CE→GRPO qualitative diff (Liang 2025 Table 4 style): the
    clearest way to show whether RL fine-tuning changed caption content,
    length, or diversity on the same test images.
    """

    # Strip SentencePiece ▁ markers so the displayed caption matches paper
    # convention; the raw subword form is recoverable from token_ids if needed.
    map_a = {int(p["image_id"]): _strip_spm_marker(str(p["caption"])) for p in preds_a}
    map_b = {int(p["image_id"]): _strip_spm_marker(str(p["caption"])) for p in preds_b}
    common = sorted(set(map_a) & set(map_b))
    diff_ids = [i for i in common if map_a[i].strip() != map_b[i].strip()]
    if not diff_ids:
        return pd.DataFrame(
            [{"image_id": "—", names[0]: "(no diffs)", names[1]: "(no diffs)", "references": ""}]
        ).set_index("image_id")

    rng = np.random.default_rng(seed)
    picks = rng.choice(diff_ids, size=min(n, len(diff_ids)), replace=False).tolist()
    rows = []
    for i in sorted(picks):
        refs = references_by_id.get(i, references_by_id.get(str(i), []))
        ref_join = "  |  ".join(str(r) for r in list(refs)[:2])
        rows.append({
            "image_id":   int(i),
            names[0]:     map_a[i],
            names[1]:     map_b[i],
            "references": ref_join,
        })
    return pd.DataFrame(rows).set_index("image_id")


# =======================================================================
# Group-level table helpers
# =======================================================================

def runs_summary_table(runs: Mapping[str, Mapping[str, Any]]) -> "pd.DataFrame":
    """Build one row per run from the scanned ``outputs/`` directory.

    ``runs``: ``{run_name: {"dir": Path, "config": dict, "metrics": dict,
                             "log": DataFrame|None}}``
    (the schema produced by ``03_group_comparison.ipynb`` ``scan_runs()``).

    Columns: student, phase, encoder, decoder, attention, pretraining,
    total_M, trainable_M, CE_val_CIDEr, GRPO_val_CIDEr.

    Returns NaN for any missing field — never raises.
    """

    rows = []
    for name, r in runs.items():
        cfg = r.get("config") or {}
        enc = (cfg.get("model", {}) or {}).get("encoder", {}) or {}
        dec = (cfg.get("model", {}) or {}).get("decoder", {}) or {}
        metrics = r.get("metrics") or {}
        best_ce = (metrics.get("best_val") or {}).get("cider")
        best_grpo = (metrics.get("grpo_best_val") or {}).get("cider")
        row = {
            "student":      _student_of(name),
            "phase":        _phase_of(name),
            "encoder":      enc.get("name", "—"),
            "decoder":      dec.get("name", "—"),
            "attention":    dec.get("attention", "—"),
            "pretrain":     "ImageNet" if "resnet" in enc.get("name", "") or "vit" in enc.get("name", "") else
                            "CLIP (WIT-400M)" if "clip" in enc.get("name", "") else
                            "scratch",
            "val_CIDEr_CE":   round(best_ce, 4) if best_ce is not None else None,
            "val_CIDEr_GRPO": round(best_grpo, 4) if best_grpo is not None else None,
            "run_name":     name,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["student", "phase"]).set_index("run_name")


def architecture_axis_matrix(runs: Mapping[str, Mapping[str, Any]]):
    """Colour-coded 2D matrix highlighting the **design axis** per student.

    Rows = the 8 runs (4 students × 2 phases); columns = categorical axes
    (encoder / decoder / attention / pretraining / d_model).
    Cells with the same value share a colour — making the axis each student
    chose to vary visually obvious.

    Returns a **pandas Styler** — display directly in a notebook cell.
    """

    rows = []
    for name, r in runs.items():
        cfg = r.get("config") or {}
        enc = (cfg.get("model", {}) or {}).get("encoder", {}) or {}
        dec = (cfg.get("model", {}) or {}).get("decoder", {}) or {}
        rows.append({
            "student":   _student_of(name),
            "phase":     _phase_of(name),
            "encoder":   enc.get("name", "—"),
            "decoder":   dec.get("name", "—"),
            "attention": dec.get("attention", "—"),
            "pretrain":  ("CLIP" if "clip" in enc.get("name", "") else
                          "ImageNet" if enc.get("pretrained", False) else
                          "scratch"),
            "d_model":   dec.get("d_model", dec.get("hidden_dim", "—")),
        })
    df = pd.DataFrame(rows).sort_values(["student", "phase"]).reset_index(drop=True)
    df = df.set_index(["student", "phase"])

    # Build a categorical -> colour map per column (distinct colour per unique value)
    palette = sns.color_palette("pastel", n_colors=10).as_hex()

    def _color_fn(col):
        uniq = {v: palette[i % len(palette)] for i, v in enumerate(sorted(map(str, col.unique())))}
        return [f"background-color: {uniq[str(v)]}" for v in col]

    return df.style.apply(_color_fn, axis=0).set_properties(**{"text-align": "center"})


def ce_grpo_delta_table(runs: Mapping[str, Mapping[str, Any]]):
    """Per-run CE→GRPO improvement on test (beam search).

    Columns: ``CE_CIDEr``, ``GRPO_CIDEr``, ``ΔCIDEr``, ``CE_BLEU4``,
    ``GRPO_BLEU4``, ``ΔBLEU4``, ``CE_ROUGE_L``, ``GRPO_ROUGE_L``, ``ΔROUGE_L``.

    Sorted by ΔCIDEr descending. Returns a styled DataFrame with a red-green
    diverging colour map on the delta columns (symmetrically scaled).

    Only rows where both CE and GRPO blocks exist are kept.
    """

    def _get(metrics, key, block):
        b = metrics.get(block, {}) or {}
        return b.get(key)

    rows = []
    for name, r in runs.items():
        m = r.get("metrics") or {}
        ce, grpo = m.get("beam"), m.get("grpo_beam")
        if not ce or not grpo:
            continue
        d = {"run": name}
        for metric_key, short in [("cider", "CIDEr"), ("bleu4", "BLEU4"),
                                   ("rouge_l", "ROUGE_L")]:
            v_ce, v_g = ce.get(metric_key), grpo.get(metric_key)
            d[f"CE_{short}"] = round(v_ce, 4) if v_ce is not None else None
            d[f"GRPO_{short}"] = round(v_g, 4) if v_g is not None else None
            if v_ce is not None and v_g is not None:
                d[f"Δ{short}"] = round(v_g - v_ce, 4)
            else:
                d[f"Δ{short}"] = None
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("run")
    if "ΔCIDEr" in df.columns:
        df = df.sort_values("ΔCIDEr", ascending=False)
    delta_cols = [c for c in df.columns if c.startswith("Δ")]
    if delta_cols:
        vmax = float(np.nanmax(np.abs(df[delta_cols].to_numpy())))
        return df.style.background_gradient(
            subset=delta_cols, cmap="RdYlGn", vmin=-vmax, vmax=+vmax
        ).format("{:.4f}", na_rep="—")
    return df


def metrics_full_table(metric_dicts: Mapping[str, Mapping[str, float]],
                       *,
                       metrics: Sequence[str] = ("bleu1", "bleu2", "bleu3",
                                                  "bleu4", "cider", "rouge_l")
                       ) -> "pd.DataFrame":
    """Exact numeric companion to ``plot_metric_heatmap`` for group comparison.

    Rows = runs; columns = metrics; values = rounded to 4 decimals.
    """
    df = pd.DataFrame(
        [{k: metric_dicts[m].get(k) for k in metrics} for m in metric_dicts],
        index=list(metric_dicts),
    ).round(4)
    df.index.name = "run"
    return df

