"""Visual encoders with a uniform contract.

Every encoder's ``forward(images)`` returns a dict:
    {"global":  [B, D_global]   — a single pooled feature (for RNN init or Transformer cross-attn memory token),
     "spatial": [B, N, D_enc]   — a sequence of feature vectors (for attention mechanisms)}

Per-family partial-unfreezing support:
    - ``ResNetEncoder(unfreeze_last_bottlenecks=N)`` unfreezes the last ``N``
      bottlenecks of ``layer4`` (typical N=3 → entire layer4; Keras / fast.ai / He 2016).
    - ``ViTEncoder(unfreeze_last_transformer_layers=N)`` unfreezes the last
      ``N`` transformer encoder blocks + the final LayerNorm (typical N=3;
      Kumar et al. NeurIPS 2022).
    - ``CLIPVisionEncoder(unfreeze_last_transformer_layers=N)`` unfreezes the
      last ``N`` transformer blocks + ``post_layernorm`` (typical N=2, more
      conservative due to contrastive pretraining sensitivity; Wei et al.
      ICCV 2023).
    - ``SmallCNNEncoder`` is trained from scratch so unfreezing is a no-op.

Each encoder exposes ``get_trainable_groups_by_depth() -> List[List[Parameter]]``
where index 0 is the **shallowest-from-output** (most recent) group and index
len-1 is the deepest-into-input group. ``src.training.Trainer._build_optimizer``
uses this to assign layer-wise-decayed LRs (Kumar 2022 / ELECTRA style).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as tvm
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from transformers import CLIPVisionModel
from transformers.utils import logging as hf_logging


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def _freeze_with_tail_unfreeze(
    encoder: nn.Module,
    blocks,
    n_unfreeze: int,
    *,
    tail_norm: nn.Module | None = None,
) -> int:
    """Freeze ``encoder`` then unfreeze the last ``n_unfreeze`` of ``blocks``
    (plus ``tail_norm`` if given). Returns the number actually unfrozen.

    Encapsulates the partial-fine-tune recipe shared by ResNet (``layer4``
    bottlenecks), ViT (last transformer blocks + ``encoder.ln``), and CLIP
    (last transformer blocks + ``post_layernorm``). Caller stores the
    returned int as ``self._n_unfrozen`` for layer-wise LR scheduling.
    """
    _freeze(encoder)
    if n_unfreeze <= 0:
        return 0
    blocks_list = list(blocks)
    n = min(n_unfreeze, len(blocks_list))
    for blk in blocks_list[-n:]:
        for p in blk.parameters():
            p.requires_grad_(True)
        blk.train()
    if tail_norm is not None:
        for p in tail_norm.parameters():
            p.requires_grad_(True)
        tail_norm.train()
    return n


def _train_keep_frozen_eval(module: nn.Module, mode: bool) -> nn.Module:
    """Replacement recursion for ``nn.Module.train(mode)`` that keeps any
    sub-tree whose parameters are *all* frozen (``requires_grad=False``) in
    ``eval`` mode regardless of ``mode``.

    Why this is needed
    ------------------
    PyTorch's default ``train(mode)`` flips every descendant indiscriminately.
    For an encoder that was partially frozen at __init__ (``_freeze(self)``
    plus a per-block re-enable of the last N layers), a later
    ``model.train()`` call from the trainer un-does the freeze in two silent
    ways:

      - **BatchNorm** (ResNet / SmallCNN) re-enables running-stats updates.
        ``running_mean`` / ``running_var`` then drift away from the
        ImageNet statistics on every forward pass — independent of
        ``requires_grad``, since buffer updates are not gated by autograd.
      - **Dropout** (ViT / CLIP transformer blocks) re-activates inside
        the ostensibly frozen blocks and injects per-step noise into a
        component that is supposed to behave as a deterministic feature
        extractor.

    Both effects break the "frozen pretrained encoder" premise that
    Rennie 2017 / Anderson 2018 / Liang 2025 captioning baselines assume.

    Sub-trees with at least one trainable parameter (ResNet ``layer4`` after
    partial unfreeze, ViT / CLIP last transformer blocks + final LayerNorm)
    follow ``mode`` normally so their gradient-receiving BatchNorms /
    Dropouts behave correctly during fine-tuning.
    """
    has_trainable = any(p.requires_grad for p in module.parameters(recurse=True))
    module.training = mode if has_trainable else False
    for child in module.children():
        _train_keep_frozen_eval(child, mode)
    return module


class ResNetEncoder(nn.Module):
    """ImageNet-pretrained ResNet-50, fc removed.

    Spatial output = 7×7 = 49 tokens of dim 2048. Global output = spatial-mean.

    Partial unfreezing: ``unfreeze_last_bottlenecks=N`` trains the last ``N``
    bottleneck blocks of ``layer4`` (N=3 unfreezes all of layer4). Earlier
    stages (layer1/2/3) stay frozen.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_last_bottlenecks: int = 0,
        **_kwargs,
    ):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        # Drop avgpool + fc; keep everything up to the last conv block (layer4 -> [B, 2048, 7, 7])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.d_enc = 2048
        self._n_unfrozen = (
            _freeze_with_tail_unfreeze(self, self.layer4, unfreeze_last_bottlenecks)
            if freeze else 0
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        b, c, h, w = x.shape
        spatial = x.flatten(2).transpose(1, 2).contiguous()  # [B, 49, 2048]
        global_feat = spatial.mean(dim=1)                    # [B, 2048]
        return {"global": global_feat, "spatial": spatial}

    def get_trainable_groups_by_depth(self) -> List[List[nn.Parameter]]:
        """Return trainable parameter groups ordered shallow→deep.

        Index 0 = most recent bottleneck (closest to output), index N-1 = deepest.
        Empty list if no unfreezing.
        """
        if self._n_unfrozen == 0:
            return []
        bks = list(self.layer4)[-self._n_unfrozen:]
        # Shallowest-from-output first: reverse so last bottleneck (closest to output) is index 0
        return [list(bk.parameters()) for bk in reversed(bks)]

    def train(self, mode: bool = True):
        return _train_keep_frozen_eval(self, mode)


class ViTEncoder(nn.Module):
    """ImageNet-pretrained ViT-B/16. Outputs 14×14 = 196 patch tokens (dim 768) + the CLS token as global.

    Partial unfreezing: ``unfreeze_last_transformer_layers=N`` trains the last
    ``N`` transformer encoder blocks + the final LayerNorm (``encoder.ln``).
    Patch embedding + earlier blocks stay frozen.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_last_transformer_layers: int = 0,
        **_kwargs,
    ):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = tvm.vit_b_16(weights=weights)
        self.d_enc = 768
        self._n_unfrozen = (
            _freeze_with_tail_unfreeze(
                self, self.vit.encoder.layers, unfreeze_last_transformer_layers,
                tail_norm=self.vit.encoder.ln,
            ) if freeze else 0
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Borrow torchvision's internal path up to the encoder output, then split CLS from patches
        x = self.vit._process_input(images)                     # [B, 196, 768]
        b = x.shape[0]
        cls = self.vit.class_token.expand(b, -1, -1)             # [B, 1, 768]
        x = torch.cat([cls, x], dim=1)                           # [B, 197, 768]
        x = self.vit.encoder(x)                                   # [B, 197, 768]
        global_feat = x[:, 0]                                    # [B, 768]
        spatial = x[:, 1:]                                       # [B, 196, 768]
        return {"global": global_feat, "spatial": spatial}

    def get_trainable_groups_by_depth(self) -> List[List[nn.Parameter]]:
        if self._n_unfrozen == 0:
            return []
        layers = list(self.vit.encoder.layers)[-self._n_unfrozen:]
        # Shallowest-from-output first (reverse so last transformer block comes first)
        groups: List[List[nn.Parameter]] = [list(blk.parameters()) for blk in reversed(layers)]
        # Fold the final LayerNorm into the shallowest (closest-to-output) group
        groups[0] = groups[0] + list(self.vit.encoder.ln.parameters())
        return groups

    def train(self, mode: bool = True):
        return _train_keep_frozen_eval(self, mode)


class SmallCNNEncoder(nn.Module):
    """3 conv blocks, trained from scratch.

    Designed to be the group's 'from-scratch baseline' control for S2 Phase 2.
    Input 224×224 -> output 28×28×256 -> flattened to 784 tokens OR avg-pooled to 7×7.
    """

    def __init__(self, out_dim: int = 256, **_kwargs):
        super().__init__()
        self.d_enc = out_dim
        self.features = nn.Sequential(
            # block 1: 224 -> 112
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 2: 112 -> 56
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 3: 56 -> 28 -> (adaptive pool) 7
            nn.Conv2d(128, out_dim, 3, padding=1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        # Modern from-scratch CNN init: Kaiming fan_out + ReLU (He 2015).
        # PyTorch's default Conv2d init is kaiming_uniform_(a=sqrt(5)) — a
        # legacy default that is *not* what He 2015 recommends for ReLU nets.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(images)                       # [B, out_dim, 7, 7]
        spatial = x.flatten(2).transpose(1, 2).contiguous()  # [B, 49, out_dim]
        global_feat = spatial.mean(dim=1)
        return {"global": global_feat, "spatial": spatial}

    def get_trainable_groups_by_depth(self) -> List[List[nn.Parameter]]:
        # Trained from scratch: no discriminative LR scheme needed; caller should
        # treat the whole encoder as part of the main optimizer at decoder LR.
        return []

    def train(self, mode: bool = True):
        # Trained from scratch — every BN is fully trainable, so this collapses
        # to the standard recursion. Kept for interface uniformity.
        return _train_keep_frozen_eval(self, mode)


class CLIPVisionEncoder(nn.Module):
    """CLIP ViT encoder backed by any HF CLIP vision checkpoint.

    Outputs ``N+1`` tokens (dim 768 for ViT-B checkpoints) from CLIP's
    transformer, where token 0 is CLS and ``N`` depends on patch size
    (e.g. 49 for patch32 at 224px, 196 for patch16 at 224px).

    Partial unfreezing: ``unfreeze_last_transformer_layers=N`` trains the last
    ``N`` transformer blocks of ``vision_model.encoder.layers`` plus
    ``vision_model.post_layernorm``. Wei et al. ICCV 2023 recommend a more
    conservative N for CLIP (typical N=2) because contrastive pretraining is
    sensitive to perturbation — full unfreeze easily causes catastrophic
    forgetting of image-text alignment.
    """

    def __init__(
        self,
        pretrained: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
        unfreeze_last_transformer_layers: int = 0,
        **_kwargs,
    ):
        super().__init__()
        _prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        disable_progress_bars()
        hf_logging.disable_progress_bar()
        try:
            self.vit = CLIPVisionModel.from_pretrained(pretrained)
        finally:
            hf_logging.set_verbosity(_prev_verbosity)
            hf_logging.enable_progress_bar()
            enable_progress_bars()
        self.d_enc = int(self.vit.config.hidden_size)
        self._n_unfrozen = (
            _freeze_with_tail_unfreeze(
                self, self.vit.vision_model.encoder.layers, unfreeze_last_transformer_layers,
                tail_norm=self.vit.vision_model.post_layernorm,
            ) if freeze else 0
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.vit(pixel_values=images)
        # out.last_hidden_state: [B, N+1, D], where the 0-th token is CLS
        seq = out.last_hidden_state
        global_feat = seq[:, 0]
        spatial = seq[:, 1:]
        return {"global": global_feat, "spatial": spatial}

    def get_trainable_groups_by_depth(self) -> List[List[nn.Parameter]]:
        if self._n_unfrozen == 0:
            return []
        layers = list(self.vit.vision_model.encoder.layers)[-self._n_unfrozen:]
        groups: List[List[nn.Parameter]] = [list(blk.parameters()) for blk in reversed(layers)]
        groups[0] = groups[0] + list(self.vit.vision_model.post_layernorm.parameters())
        return groups

    def train(self, mode: bool = True):
        return _train_keep_frozen_eval(self, mode)


# Registry used by `src.models.captioner.build_captioner`
ENCODER_REGISTRY = {
    "resnet50": ResNetEncoder,
    "vit_b16": ViTEncoder,
    "small_cnn": SmallCNNEncoder,
    "clip_vision": CLIPVisionEncoder,
}


def build_encoder(cfg: dict) -> nn.Module:
    name = cfg["name"]
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder {name!r}; known: {list(ENCODER_REGISTRY)}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ENCODER_REGISTRY[name](**kwargs)
