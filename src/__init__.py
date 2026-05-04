"""DL AT3 VizWiz Image Captioning — top-level package facade.

Stateful workhorses follow PyTorch Lightning / HuggingFace / fastai conventions:
  - ``Trainer(cfg, model, loaders, vocab, device, run_dir).fit().fit_grpo().evaluate_on_test(checkpoint=...)``
  - ``DataLoaders.from_config(cfg_model, data_cfg, device)``
  - ``Vocabulary`` (SentencePiece BPE), ``ImageCaptioner``,
    ``BahdanauAttention``, ``LuongAttention``
  - ``EarlyStopping``, ``ModelCheckpoint`` (callbacks)

Stateless helpers stay as module functions:
  - ``train_one_epoch`` / ``validate`` / ``evaluate_on_val``     Stage 1 CE
  - ``train_grpo_epoch``                                          Stage 2 GRPO RL
  - ``decode_loader(method=...)``                                 batched seq2seq inference
  - ``compute_captioning_metrics``                                corpus BLEU/CIDEr/ROUGE
  - ``compute_cider_per_image``                                   per-image CIDEr (GRPO reward)
  - ``build_captioner``, ``build_encoder``, ``build_decoder``, ``build_transform``
  - ``clean_caption``, ``tokenize``, ``set_seed``, ``split_image_ids``
  - plotting + I/O in ``utils`` / ``viz``
"""
from .config import AttrDict, load_config
from .dataset import (
    DataLoaders,
    VizWizCaptionDataset,
    VizWizEvalDataset,
    VizWizInferenceDataset,
    collate_fn,
    eval_collate_fn,
    inference_collate_fn,
    split_image_ids,
)
from .evaluation import compute_captioning_metrics, compute_cider_per_image
from .inference import decode_loader
from .models import (
    BahdanauAttention,
    CLIPVisionEncoder,
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    FeedForward,
    ImageCaptioner,
    LuongAttention,
    MultiHeadAttention,
    RNNDecoder,
    ResNetEncoder,
    ScaledDotProductAttention,
    SmallCNNEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    ViTEncoder,
    build_captioner,
    build_decoder,
    build_encoder,
)
from .training import (
    EarlyStopping,
    GRPO_REWARDS,
    ModelCheckpoint,
    Trainer,
    evaluate_on_val,
    train_grpo_epoch,
    train_one_epoch,
    validate,
)
from .transforms import build_transform, denormalise
from .utils import (
    append_csv,
    count_parameters,
    pretty_print_metrics,
    save_json,
    set_seed,
)
from .viz import (
    architecture_axis_matrix,
    architecture_comparison_table,
    caption_length_stats_table,
    ce_grpo_delta_table,
    hyperparameter_summary_table,
    metrics_comparison_table,
    metrics_full_table,
    metrics_table_6block,
    parameter_breakdown_table,
    plot_caption_length_distribution,
    plot_caption_lengths_compared,
    plot_cross_model_gallery,
    plot_metric_bars,
    plot_metric_heatmap,
    plot_metric_scatter,
    plot_grpo_curves,
    plot_qualitative_samples,
    plot_training_curves,
    prediction_diff_table,
    runs_summary_table,
    set_plot_style,
)
from .vocab import Vocabulary, clean_caption, tokenize

__all__ = [
    # config / reproducibility
    "AttrDict", "load_config", "set_seed",
    # text
    "Vocabulary", "clean_caption", "tokenize",
    # splits / transforms
    "split_image_ids", "build_transform", "denormalise",
    # datasets + loaders
    "VizWizCaptionDataset", "VizWizEvalDataset", "VizWizInferenceDataset",
    "collate_fn", "eval_collate_fn", "inference_collate_fn",
    "DataLoaders",
    # models
    "ImageCaptioner", "build_captioner",
    "build_encoder", "build_decoder",
    "ENCODER_REGISTRY", "DECODER_REGISTRY",
    "ResNetEncoder", "ViTEncoder", "SmallCNNEncoder", "CLIPVisionEncoder",
    "RNNDecoder", "TransformerDecoder",
    "FeedForward", "TransformerDecoderLayer",
    "BahdanauAttention", "LuongAttention",
    "ScaledDotProductAttention", "MultiHeadAttention",
    # training (high-level Trainer + low-level workers + callbacks)
    "Trainer",
    "train_one_epoch", "validate", "evaluate_on_val", "train_grpo_epoch",
    "EarlyStopping", "ModelCheckpoint", "GRPO_REWARDS",
    # inference + evaluation
    "decode_loader", "compute_captioning_metrics", "compute_cider_per_image",
    # utils: model / formatting / I/O / paths
    "count_parameters", "pretty_print_metrics", "append_csv", "save_json",
    # viz — plots
    "set_plot_style",
    "plot_qualitative_samples", "plot_training_curves", "plot_grpo_curves",
    "plot_metric_bars", "plot_metric_heatmap",
    "plot_caption_length_distribution", "plot_caption_lengths_compared",
    "plot_metric_scatter", "plot_cross_model_gallery",
    # viz — tables
    "metrics_comparison_table",
    "parameter_breakdown_table", "hyperparameter_summary_table",
    "architecture_comparison_table", "metrics_table_6block",
    "prediction_diff_table",
    "runs_summary_table", "architecture_axis_matrix",
    "ce_grpo_delta_table", "metrics_full_table",
    "caption_length_stats_table",
]
