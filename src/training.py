"""CE Stage-1 + GRPO Stage-2 training (Lightning / HF Trainer style)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DataLoaders
from .evaluation import compute_captioning_metrics, compute_cider_per_image
from .inference import decode_loader
from .models import build_captioner
from .utils import append_csv, load_json, save_json


# yaml ``grpo.reward`` -> per-image fn(preds, refs, enforce_eos=True) -> np.ndarray
GRPO_REWARDS = {
    "cider": compute_cider_per_image,
}


def _optimizer_step(loss: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    model: nn.Module,
                    clip_norm: float,
                    scaler: Optional[torch.amp.GradScaler],
                    amp_enabled: bool) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
        optimizer.step()


def _merge_json(path: Path, updates: Dict[str, Any]) -> None:
    existing = load_json(path, default={})
    if not isinstance(existing, dict):
        existing = {}
    existing.update(updates)
    save_json(existing, path)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    *,
                    clip_norm: float = 5.0,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    amp_dtype: torch.dtype = torch.bfloat16,
                    amp_enabled: bool = True,
                    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                    log_prefix: str = "train") -> Dict[str, float]:
    """One CE epoch with teacher forcing. ``scheduler`` (cosine/warmup) is
    stepped per optimizer.step; ReduceLROnPlateau should be stepped externally."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for imgs, caps, lens, _ids in tqdm(loader, desc=log_prefix, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(imgs, caps, lens)                  # [B, T-1, V]
            targets = caps[:, 1:]                             # [B, T-1]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        _optimizer_step(loss, optimizer, model, clip_norm, scaler, amp_enabled)
        if scheduler is not None:
            scheduler.step()

        ntoks = int((lens - 1).clamp(min=0).sum().item())
        total_loss += float(loss.item()) * max(ntoks, 1)
        total_tokens += max(ntoks, 1)

    return {"loss": total_loss / max(total_tokens, 1)}


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             *,
             amp_dtype: torch.dtype = torch.bfloat16,
             amp_enabled: bool = True,
             log_prefix: str = "val") -> Dict[str, float]:
    """Teacher-forcing CE on the val caption loader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for imgs, caps, lens, _ids in tqdm(loader, desc=log_prefix, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(imgs, caps, lens)
            targets = caps[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        ntoks = int((lens - 1).clamp(min=0).sum().item())
        total_loss += float(loss.item()) * max(ntoks, 1)
        total_tokens += max(ntoks, 1)
    return {"loss": total_loss / max(total_tokens, 1)}


@torch.no_grad()
def evaluate_on_val(model: nn.Module,
                    val_eval_loader: DataLoader,
                    vocab,
                    device: torch.device,
                    *,
                    beam_size: int = 5,
                    max_len: int = 52,
                    length_penalty: float = 1.0,
                    log_prefix: str = "val beam") -> Dict[str, Any]:
    """Beam search on val + corpus BLEU/CIDEr/ROUGE."""
    preds = decode_loader(model, val_eval_loader, vocab, device,
                          method="beam", beam_size=beam_size, max_len=max_len,
                          length_penalty=length_penalty, log_prefix=log_prefix)
    refs = val_eval_loader.dataset.references_by_image_id
    return compute_captioning_metrics(preds, refs)


def _gather_logp(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1).gather(2, ids.unsqueeze(-1)).squeeze(-1)


def train_grpo_epoch(
    model: nn.Module,
    model_ref: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab,
    device: torch.device,
    refs_by_image_id: Mapping[int, Sequence[str]],
    *,
    reward_fn=compute_cider_per_image,
    group_size: int = 5,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    max_len: int = 52,
    clip_norm: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    amp_enabled: bool = True,
    log_prefix: str = "grpo",
) -> Dict[str, float]:
    """One GRPO epoch (Liang 2025, Schulman k3 KL). Returns
    ``{"loss", "kl", "entropy", "reward_mean", "reward_std"}``."""
    model.train()
    model.encoder.eval()      # RL fine-tunes only the language head
    model_ref.eval()

    start_id = int(getattr(vocab, "start_idx", 1))
    end_id = int(getattr(vocab, "end_idx", 2))
    pad_id = int(getattr(vocab, "pad_idx", 0))

    agg = {"loss": 0.0, "kl": 0.0, "entropy": 0.0,
           "reward_mean": 0.0, "reward_std": 0.0}
    n_batches = 0

    for imgs, _caps, _lens, ids in tqdm(loader, desc=log_prefix, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.shape[0]

        # Broadcast images to [B*G, ...]
        imgs_g = imgs.repeat_interleave(group_size, dim=0)
        ids_flat = [int(i) for i in ids.tolist() for _ in range(group_size)]

        # Encoder is shared between policy and reference (frozen) → one pass.
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                enc_out = model.encoder(imgs_g)
                sampled_ids, sampled_logp_old = model.decoder.generate(
                    enc_out, max_len=max_len, method="sample",
                )

        # Per-image reward (CIDEr-D + EOS fix or whatever ``reward_fn`` is)
        preds: List[dict] = []
        for k, seq in enumerate(sampled_ids.tolist()):
            terminated = end_id in seq
            clean = [t for t in seq if t != pad_id]
            words = vocab.decode(clean, strip_specials=True)
            preds.append({
                "image_id": ids_flat[k],
                "caption": " ".join(words),
                "terminated_with_eos": terminated,
            })
        rewards_np = reward_fn(preds, refs_by_image_id, enforce_eos=True)
        rewards = torch.from_numpy(rewards_np).to(device).view(B, group_size)

        # Group-normalised advantage
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True).clamp_min(1e-5)
        advantage = ((rewards - mean_r) / std_r).view(B * group_size, 1)

        # Re-score under policy and reference; clamp lengths >=1 so RNN
        # pack_padded_sequence does not error on all-pad samples.
        start_col = torch.full((B * group_size, 1), start_id, dtype=torch.long, device=device)
        sample_with_start = torch.cat([start_col, sampled_ids], dim=1)
        mask_tokens = (sampled_ids != pad_id).float()
        T_decoder = sampled_ids.shape[1]  # decoder input is sample_with_start[:, :-1] → T = sampled_ids.shape[1]
        lengths_decoder = (1 + mask_tokens.sum(dim=1)).clamp(min=1, max=T_decoder).long()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model.decoder(enc_out, sample_with_start[:, :-1], lengths_decoder)
            logp_new = _gather_logp(logits, sampled_ids)

            with torch.no_grad():
                logits_ref = model_ref.decoder(enc_out, sample_with_start[:, :-1], lengths_decoder)
                logp_ref = _gather_logp(logits_ref, sampled_ids)

            # PPO-clipped surrogate + Schulman k3 KL estimator
            ratio = (logp_new - sampled_logp_old.detach()).exp()
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.min(surrogate_1, surrogate_2)

            diff = logp_ref - logp_new
            kl = diff.exp() - diff - 1.0

            per_token = policy_loss + kl_beta * kl
            mask = mask_tokens
            mask_sum = mask.sum(dim=1).clamp_min(1.0)
            loss = ((per_token * mask).sum(dim=1) / mask_sum).mean()

        optimizer.zero_grad(set_to_none=True)
        _optimizer_step(loss, optimizer, model, clip_norm, scaler, amp_enabled)

        with torch.no_grad():
            mean_kl = float(((kl * mask).sum() / mask.sum().clamp_min(1.0)).item())
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(-1)            # [B*G, T]
            mean_entropy = float(((entropy * mask).sum() / mask.sum().clamp_min(1.0)).item())
        agg["loss"] += float(loss.item())
        agg["kl"] += mean_kl
        agg["entropy"] += mean_entropy
        agg["reward_mean"] += float(rewards.mean().item())
        agg["reward_std"] += float(rewards.std().item())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in agg.items()}


class EarlyStopping:
    """Patience-based early stop on an INCREASING scalar (e.g. CIDEr)."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.n_bad = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.n_bad = 0
            return False
        self.n_bad += 1
        return self.n_bad > self.patience


class ModelCheckpoint:
    """Save best-only checkpoint when ``val_metrics[monitor]`` improves."""

    def __init__(self, run_dir: Path, monitor: str = "cider", filename: str = "best.pt"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.best = -math.inf
        self.monitor = monitor

    def update(self, model: nn.Module, epoch: int,
               val_metrics: Dict[str, Any],
               config_snapshot: Dict[str, Any]) -> bool:
        metric = float(val_metrics.get(self.monitor, float("-inf")))
        if metric > self.best:
            self.best = metric
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": int(epoch),
                f"val_{self.monitor}": metric,
                "val_metrics": dict(val_metrics),
                "config": config_snapshot,
            }, self.path)
            return True
        return False

    def load_into(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return ckpt


class Trainer:
    """CE pretrain (``fit``) → GRPO fine-tune (``fit_grpo``) → ``evaluate_on_test``."""

    def __init__(self,
                 cfg: dict,
                 model: nn.Module,
                 loaders: DataLoaders,
                 vocab,
                 device: torch.device,
                 run_dir: Path):
        self.cfg = cfg
        self.model = model
        self.loaders = loaders
        self.vocab = vocab
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        tr_cfg = cfg["training"]
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=tr_cfg.get("label_smoothing", 0.0),
        )
        self.optimizer = self._build_optimizer(tr_cfg)
        self.scheduler = self._build_scheduler(tr_cfg)
        # bf16 on SM>=80, fp16+GradScaler otherwise.
        self.amp_enabled = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        elif self.amp_enabled:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler(device.type)
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
        self.monitor = str(tr_cfg["monitor"])
        self.checkpoint = ModelCheckpoint(self.run_dir, monitor=self.monitor, filename="best.pt")
        self.grpo_checkpoint = ModelCheckpoint(self.run_dir, monitor=self.monitor, filename="best_grpo.pt")
        patience = tr_cfg.get("early_stop_patience")
        self.early_stop = EarlyStopping(patience=int(patience)) if patience else None

    def _build_optimizer(self, tr_cfg: dict) -> torch.optim.Optimizer:
        """AdamW with layer-wise LR decay over unfrozen encoder blocks
        (Kumar 2022): decoder @ lr; depth-d block @ lr * scale * decay**d."""
        base_lr = float(tr_cfg["lr"])
        weight_decay = float(tr_cfg.get("weight_decay", 0.0))
        betas = tuple(tr_cfg.get("betas", [0.9, 0.999]))

        decoder_params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        param_groups: List[dict] = [{"params": decoder_params, "lr": base_lr}]

        get_groups = getattr(self.model.encoder, "get_trainable_groups_by_depth", None)
        if callable(get_groups):
            enc_groups = get_groups()
            enc_lr_scale = float(tr_cfg.get("encoder_lr_scale", 0.1))
            enc_lr_decay = float(tr_cfg.get("encoder_lr_decay", 0.65))
            for depth, params in enumerate(enc_groups):
                lr_scale = enc_lr_scale * (enc_lr_decay ** depth)
                param_groups.append({"params": params, "lr": base_lr * lr_scale})
        # Encoder params not covered above (e.g. scratch SmallCNN) get decoder LR.
        covered = {id(p) for g in param_groups for p in g["params"]}
        leftover_enc = [p for p in self.model.encoder.parameters()
                        if p.requires_grad and id(p) not in covered]
        if leftover_enc:
            param_groups.append({"params": leftover_enc, "lr": base_lr})

        name = str(tr_cfg.get("optimizer", "adamw")).lower()
        if name not in ("adam", "adamw"):
            raise ValueError(f"unknown optimizer {name!r}")
        return torch.optim.AdamW(param_groups, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, tr_cfg: dict):
        kind = tr_cfg.get("scheduler")
        if kind == "cosine":
            warmup = int(tr_cfg.get("warmup_steps", 0))
            total_steps = len(self.loaders.train) * tr_cfg["epochs"]

            def lr_lambda(step):
                if step < warmup:
                    return max(1e-6, step / max(1, warmup))
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        if kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=2,
            )
        return None

    # ------------------------ public methods ----------------------------

    def fit(self) -> "Trainer":
        """Stage 1 CE training loop. Returns self for chaining."""
        tr_cfg = self.cfg["training"]
        log_path = self.run_dir / "train_log.csv"
        if log_path.exists():
            log_path.unlink()

        n_epochs = tr_cfg["epochs"]
        for epoch in range(1, n_epochs + 1):
            print(f"\n=== epoch {epoch}/{n_epochs} ===", file=sys.stderr, flush=True)
            # Per-step schedulers stepped inside train_one_epoch; plateau is per-epoch.
            step_sched = (None if isinstance(self.scheduler,
                                             torch.optim.lr_scheduler.ReduceLROnPlateau)
                          else self.scheduler)
            tr = train_one_epoch(self.model, self.loaders.train, self.optimizer,
                                 self.criterion, self.device,
                                 clip_norm=tr_cfg.get("clip_norm", 5.0),
                                 scaler=self.scaler,
                                 amp_dtype=self.amp_dtype,
                                 amp_enabled=self.amp_enabled,
                                 scheduler=step_sched,
                                 log_prefix=f"epoch {epoch}")
            va = validate(self.model, self.loaders.val_caption, self.criterion, self.device,
                          amp_dtype=self.amp_dtype,
                          amp_enabled=self.amp_enabled)
            metrics = self.evaluate_on_val()

            monitor_val = float(metrics[self.monitor])
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitor_val)

            lr_now = self.optimizer.param_groups[0]["lr"]
            append_csv({
                "epoch": epoch,
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "val_bleu1": metrics["bleu1"], "val_bleu2": metrics["bleu2"],
                "val_bleu3": metrics["bleu3"], "val_bleu4": metrics["bleu4"],
                "val_cider": metrics["cider"], "val_rouge_l": metrics["rouge_l"],
                "lr": lr_now,
            }, log_path)

            improved = self.checkpoint.update(self.model, epoch, metrics, config_snapshot=self.cfg)
            stop = self.early_stop.step(monitor_val) if self.early_stop is not None else False

            print(f"epoch {epoch:2d}: train_loss={tr['loss']:.3f}  val_loss={va['loss']:.3f}  "
                  f"bleu4={metrics['bleu4']:.4f}  cider={metrics['cider']:.4f}  "
                  f"best({self.monitor})={self.checkpoint.best:.4f}{'  [best]' if improved else ''}  "
                  f"lr={lr_now:.2e}", file=sys.stderr, flush=True)

            if stop:
                print(f"  early stop at epoch {epoch} (patience exceeded)", file=sys.stderr, flush=True)
                break

        print(f"\nBest CE val {self.monitor}: {self.checkpoint.best:.4f}")
        return self

    def fit_grpo(self) -> "Trainer":
        """Load ``best.pt``, re-freeze encoder, GRPO fine-tune the decoder."""
        grpo_cfg = self.cfg.get("grpo", {})
        if not grpo_cfg.get("enabled", False):
            raise RuntimeError("grpo.enabled=false in config")
        if not (self.run_dir / "best.pt").exists():
            raise FileNotFoundError(f"{self.run_dir / 'best.pt'} not found; run Trainer.fit() first")
        reward_name = str(grpo_cfg["reward"])
        reward_fn = GRPO_REWARDS[reward_name]

        self.checkpoint.load_into(self.model, self.device)
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        self.model.encoder.eval()

        model_ref = build_captioner(self.cfg, self.vocab).to(self.device)
        self.checkpoint.load_into(model_ref, self.device)
        for p in model_ref.parameters():
            p.requires_grad_(False)
        model_ref.eval()

        grpo_params = [p for p in self.model.parameters() if p.requires_grad]
        grpo_optim = torch.optim.AdamW(
            grpo_params,
            lr=float(grpo_cfg["lr"]),
            weight_decay=float(grpo_cfg.get("weight_decay", 0.0)),
        )

        refs_map = _references_from_train_manifest(self.cfg)

        log_path = self.run_dir / "grpo_log.csv"
        if log_path.exists():
            log_path.unlink()

        n_epochs = int(grpo_cfg.get("epochs", 5))
        for epoch in range(1, n_epochs + 1):
            print(f"\n=== grpo epoch {epoch}/{n_epochs} ===", file=sys.stderr, flush=True)
            tr = train_grpo_epoch(
                self.model, model_ref,
                self.loaders.train, grpo_optim, self.vocab, self.device,
                refs_map,
                reward_fn=reward_fn,
                group_size=int(grpo_cfg.get("group_size", 5)),
                clip_eps=float(grpo_cfg.get("clip_eps", 0.2)),
                kl_beta=float(grpo_cfg.get("kl_beta", 0.04)),
                max_len=int(self.cfg["eval"]["max_gen_len"]),
                clip_norm=float(grpo_cfg.get("clip_norm", 1.0)),
                scaler=self.scaler,
                amp_dtype=self.amp_dtype,
                amp_enabled=self.amp_enabled,
                log_prefix=f"grpo ep{epoch}",
            )
            metrics = self.evaluate_on_val()

            lr_now = grpo_optim.param_groups[0]["lr"]
            append_csv({
                "epoch": epoch,
                "grpo_loss": tr["loss"], "grpo_kl": tr["kl"], "grpo_entropy": tr["entropy"],
                "grpo_reward_mean": tr["reward_mean"], "grpo_reward_std": tr["reward_std"],
                "val_bleu4": metrics["bleu4"], "val_cider": metrics["cider"],
                "val_rouge_l": metrics["rouge_l"], "lr": lr_now,
            }, log_path)

            improved = self.grpo_checkpoint.update(self.model, epoch, metrics, config_snapshot=self.cfg)
            print(f"grpo epoch {epoch:2d}: loss={tr['loss']:.4f}  kl={tr['kl']:.4f}  "
                  f"entropy={tr['entropy']:.3f}  reward({reward_name})={tr['reward_mean']:.3f}±{tr['reward_std']:.3f}  "
                  f"val_cider={metrics['cider']:.4f}  best({self.monitor})={self.grpo_checkpoint.best:.4f}"
                  f"{'  [best]' if improved else ''}", file=sys.stderr, flush=True)

        print(f"\nBest GRPO val {self.monitor}: {self.grpo_checkpoint.best:.4f}")
        return self

    def evaluate_on_val(self) -> Dict[str, Any]:
        """Beam search on val + corpus metrics."""
        length_penalty = float(self.cfg.get("eval", {}).get("length_penalty", 1.0))
        return evaluate_on_val(self.model, self.loaders.val_eval, self.vocab, self.device,
                               beam_size=self.cfg["eval"]["beam_size"],
                               max_len=self.cfg["eval"]["max_gen_len"],
                               length_penalty=length_penalty)

    def evaluate_on_test(self, *, checkpoint: str = "ce") -> Dict[str, Any]:
        """Reload ``best.pt`` (``"ce"``) or ``best_grpo.pt`` (``"grpo"``), run
        greedy + beam on test, merge into ``metrics.json`` / ``predictions_test.json``."""
        ckpt_choice = checkpoint.lower()
        ckpt_dispatch = {
            "ce":   (self.checkpoint,      "",      "ce"),
            "grpo": (self.grpo_checkpoint, "grpo_", "grpo"),
        }
        if ckpt_choice not in ckpt_dispatch:
            raise ValueError(f"checkpoint must be 'ce' or 'grpo', got {checkpoint!r}")
        cb, prefix, preds_key = ckpt_dispatch[ckpt_choice]

        if not cb.path.exists():
            raise FileNotFoundError(f"{cb.path} does not exist — "
                                    f"run {'fit()' if ckpt_choice == 'ce' else 'fit_grpo()'} first")

        model = build_captioner(self.cfg, self.vocab).to(self.device)
        ckpt = cb.load_into(model, self.device)

        refs = self.loaders.test_eval.dataset.references_by_image_id
        length_penalty = float(self.cfg.get("eval", {}).get("length_penalty", 1.0))
        decoded = decode_loader(model, self.loaders.test_eval, self.vocab, self.device,
                                methods=("greedy", "beam"),
                                beam_size=self.cfg["eval"]["beam_size"],
                                max_len=self.cfg["eval"]["max_gen_len"],
                                length_penalty=length_penalty,
                                log_prefix="test greedy+beam")
        greedy_preds = decoded["greedy"]
        beam_preds = decoded["beam"]

        m_greedy = compute_captioning_metrics(greedy_preds, refs)
        m_beam = compute_captioning_metrics(beam_preds, refs)

        # Merge with pre-existing metrics so CE and GRPO blocks coexist.
        _merge_json(self.run_dir / "metrics.json", {
            f"{prefix}greedy": m_greedy,
            f"{prefix}beam": m_beam,
            f"{prefix}best_val": ckpt["val_metrics"],
        })
        _merge_json(self.run_dir / "predictions_test.json", {
            f"{preds_key}_greedy": greedy_preds,
            f"{preds_key}_beam": beam_preds,
        })

        return {
            "model": model,
            "greedy_preds": greedy_preds,
            "beam_preds": beam_preds,
            "metrics_greedy": m_greedy,
            "metrics_beam": m_beam,
            "checkpoint_info": {
                "epoch": ckpt["epoch"],
                "monitor": self.monitor,
                f"val_{self.monitor}": ckpt.get(f"val_{self.monitor}", float("nan")),
                "val_metrics": ckpt["val_metrics"],
                "which": preds_key,
            },
        }


def _references_from_train_manifest(cfg: dict) -> Dict[int, List[str]]:
    """``{image_id: [ref, ...]}`` over the train split for GRPO reward."""
    train_split = cfg.get("data", {}).get("train_split") or "data/processed/train_split.json"
    records = json.loads(Path(train_split).read_text(encoding="utf-8"))
    return {int(rec["image_id"]): [str(c) for c in rec["captions"]] for rec in records}
