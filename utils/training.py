"""
utils/training.py - Reusable training loops for Stage-2 models.

Provides:
  train_one_epoch()           - single training epoch (with AMP support)
  validate_one_epoch()        - single validation epoch
  identity_cross_validate()   - identity-stratified k-fold CV (no subject leakage)
  cross_validate()            - legacy CV (kept for backward-compat)
  build_optimizer()           - construct optimizer from config
  build_scheduler()           - construct LR-scheduler from config
  build_criterion()           - construct loss with optional class weights
"""

from __future__ import annotations

import os
import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Within each batch, pulls embeddings of the same class together and pushes
    different-class embeddings apart. Highly effective for fine-grained tasks.

    Usage: apply to L2-normalised feature embeddings (before the classifier).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,  # (B, D) — L2-normalised embeddings
        labels: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        device = features.device
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix, scaled by temperature
        sim = torch.mm(features, features.T) / self.temperature  # (B, B)

        # Numerical stability: subtract row max before exp (like log-softmax)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exclude self from the denominator
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        exp_sim = torch.exp(sim) * (~mask_self).float()        # zero out diagonal

        # Positive mask: same class, not self
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        pos_mask  = (labels_eq & ~mask_self).float()

        # SupCon log-probability for each anchor-positive pair
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

        # Mean over positives for each anchor (skip anchors with no positives)
        pos_count = pos_mask.sum(dim=1)
        has_pos   = pos_count > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        return loss_per_anchor[has_pos].mean()


# ── CutMix augmentation ───────────────────────────────────────────────────────

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup: linearly interpolate pairs of images and labels."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix to a batch.

    Returns (mixed_images, labels_a, labels_b, lam) where
    loss = lam * CE(pred, labels_a) + (1-lam) * CE(pred, labels_b).
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    perm = torch.randperm(B, device=images.device)

    cx = int(np.random.uniform(0, W))
    cy = int(np.random.uniform(0, H))
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    # Adjust lam to actual patch ratio
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[perm], lam


# ── Single-epoch helpers ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    clip_grad: float | None = 1.0,
    cutmix_alpha: float = 0.0,
    supcon_weight: float = 0.0,
) -> tuple[float, float]:
    """
    One training epoch with optional AMP, gradient clipping, and CutMix.

    Args:
        cutmix_alpha: Beta distribution parameter for CutMix.
                      0.0 disables CutMix; 1.0 is the standard setting.
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"
    use_cutmix = cutmix_alpha > 0.0
    use_supcon = supcon_weight > 0.0 and hasattr(model, "encode")
    _supcon = SupConLoss(temperature=0.07).to(device) if use_supcon else None

    for inputs, roi_masks, labels in loader:
        inputs    = inputs.to(device, non_blocking=True)
        roi_masks = roi_masks.to(device, non_blocking=True) if roi_masks is not None else None
        labels    = labels.to(device, non_blocking=True)

        # Randomly alternate CutMix and Mixup (only when not using SupCon)
        if use_cutmix and not use_supcon and model.training:
            if np.random.random() < 0.5:
                inputs, labels_a, labels_b, lam = cutmix_batch(
                    inputs, labels, alpha=cutmix_alpha)
            else:
                inputs, labels_a, labels_b, lam = mixup_batch(
                    inputs, labels, alpha=cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(inputs, roi_masks)
            if use_cutmix and not use_supcon and model.training:
                loss = lam * criterion(outputs, labels_a) + \
                       (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            # SupCon auxiliary loss (only when backbone is not frozen)
            if use_supcon and model.training:
                emb  = model.encode(inputs)
                loss = loss + supcon_weight * _supcon(emb, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        # Accuracy against the primary label
        ref_labels = labels_a if (use_cutmix and not use_supcon and model.training) else labels
        correct += (predicted == ref_labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    tta: bool = False,
) -> tuple[float, float, float]:
    """
    One validation epoch with optional Test-Time Augmentation (TTA).

    TTA averages softmax over 8 views: original + 3 rotations + 4 flips of those.
    Returns (avg_loss, accuracy, balanced_accuracy).
    """
    import torchvision.transforms.functional as TF

    use_amp = scaler is not None and device.type == "cuda"
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_gt, all_pred = [], []

    def _tta_views(x: torch.Tensor) -> list[torch.Tensor]:
        """Return 8 augmented views of a batch tensor."""
        return [
            x,
            TF.hflip(x),
            TF.vflip(x),
            TF.hflip(TF.vflip(x)),
            torch.rot90(x, 1, [2, 3]),
            torch.rot90(x, 2, [2, 3]),
            torch.rot90(x, 3, [2, 3]),
            TF.hflip(torch.rot90(x, 1, [2, 3])),
        ]

    with torch.no_grad():
        for inputs, roi_masks, labels in loader:
            inputs    = inputs.to(device, non_blocking=True)
            roi_masks = roi_masks.to(device, non_blocking=True) if roi_masks is not None else None
            labels    = labels.to(device, non_blocking=True)

            if tta:
                views = _tta_views(inputs)
                probs = torch.zeros(inputs.size(0), 2, device=device)
                for v in views:
                    with autocast(enabled=use_amp):
                        probs += torch.softmax(model(v, roi_masks), dim=1)
                outputs = probs / len(views)
                total_loss += criterion(torch.log(outputs.clamp(min=1e-9)), labels).item()
            else:
                with autocast(enabled=use_amp):
                    outputs = model(inputs, roi_masks)
                    total_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
            all_gt.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    bal_acc = balanced_accuracy_score(all_gt, all_pred)
    return total_loss / len(loader), correct / total, bal_acc


# ── Factories ─────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: Any) -> optim.Optimizer:
    """
    Adam | AdamW | SGD from config.

    If the model exposes ``get_parameter_groups()`` (e.g. EfficientNetStage2)
    and the config sets ``backbone_lr_scale``, two parameter groups are created:
      - backbone: lr × backbone_lr_scale
      - head:     lr  (full learning rate)
    Otherwise all parameters share the same lr.
    """
    def _get(key, default=None):
        try:   v = cfg[key]
        except (KeyError, TypeError): v = getattr(cfg, key, default)
        if hasattr(v, "__class__") and v.__class__.__name__ == "DotMap" and not dict(v):
            return default
        return v if v is not None else default

    name = str(_get("optimizer", "AdamW"))
    lr   = float(cfg.lr_scheduler.lr)
    wd   = float(_get("weight_decay", 1e-4))

    # Differential LR: backbone gets a smaller LR than the head
    raw_scale = _get("backbone_lr_scale", None)
    if raw_scale is not None and hasattr(model, "get_parameter_groups"):
        backbone_lr = lr * float(raw_scale)
        param_groups = model.get_parameter_groups(
            backbone_lr=backbone_lr, head_lr=lr, weight_decay=wd
        )
        logger.info(f"Differential LR — backbone: {backbone_lr:.2e}  head: {lr:.2e}")
    else:
        param_groups = model.parameters()

    if name == "AdamW":
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif name == "Adam":
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    elif name == "SGD":
        return optim.SGD(param_groups, lr=lr,
                         weight_decay=wd, momentum=float(_get("momentum", 0.9)))
    raise ValueError(f"Unknown optimizer '{name}'.")


def build_scheduler(optimizer: optim.Optimizer, cfg: Any,
                    n_epochs: int | None = None) -> optim.lr_scheduler.LRScheduler:
    """StepLR (default) or CosineAnnealingLR from config."""
    try:   sched_type = cfg.lr_scheduler.get("type", "step")
    except Exception: sched_type = "step"
    if not sched_type or sched_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.lr_scheduler.step_size),
            gamma=float(cfg.lr_scheduler.gamma),
        )
    elif sched_type == "cosine":
        T_max = n_epochs or int(cfg.epochs)
        eta_min = float(getattr(cfg.lr_scheduler, "eta_min", 1e-6))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler type '{sched_type}'.")


def build_criterion(cfg: Any,
                    class_counts: list[int] | None = None,
                    device: torch.device | None = None) -> nn.Module:
    """
    CrossEntropyLoss with optional class weights (inverse-frequency weighting)
    and label smoothing.
    """
    try:   name = cfg["criterion"]
    except (KeyError, TypeError): name = getattr(cfg, "criterion", "CrossEntropyLoss")
    if hasattr(name, "__class__") and name.__class__.__name__ == "DotMap":
        name = "CrossEntropyLoss"
    name = str(name) if name else "CrossEntropyLoss"

    try:   smoothing = float(cfg.get("label_smoothing", 0.0))
    except Exception: smoothing = 0.0

    weight = None
    if class_counts is not None:
        counts  = torch.tensor(class_counts, dtype=torch.float)
        weight  = counts.sum() / (len(counts) * counts)
        weight  = weight / weight.sum() * len(counts)   # normalise
        if device:
            weight = weight.to(device)
        logger.info(f"Class weights: {weight.cpu().tolist()}")

    if name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)
    raise ValueError(f"Unknown criterion '{name}'.")


# ── Identity-stratified cross-validation (main entry point) ──────────────────

def _get_identity(path: str) -> str:
    """
    Extract subject ID from filename.
    IIITD: '100_20120910_L_Normal.png' → '100'
    UND:   '06184d209_R__Normal.tiff'  → 'und_06184'
    """
    fname = os.path.basename(path)
    first = fname.split("_")[0]
    # UND files contain 'd' separating subject and session: '06184d209'
    if "d" in first and first[0].isdigit():
        return "und_" + first.split("d")[0]
    return first


def identity_cross_validate(
    cfg: Any,
    model_builder: Callable[[], nn.Module],
    dataset: Dataset,
    save_dir: str,
    evaluate_fn: Callable,
    save_results_fn: Callable,
    device: torch.device,
    n_splits: int = 5,
) -> list[float]:
    """
    Identity-stratified k-fold cross-validation.

    Groups all images of the same subject into the same fold so no subject
    appears in both train and validation within a fold.  Uses
    StratifiedGroupKFold to also preserve class balance across folds.

    Args:
        dataset:  must expose a `.data` list of (img_path, mask_path, label).
        Others:   same as cross_validate().

    Returns:
        List of best balanced-accuracy scores, one per fold.
    """
    labels  = [lbl for *_, lbl in dataset.data]
    groups  = [_get_identity(p) for p, *_ in dataset.data]
    n_ids   = len(set(groups))
    logger.info(f"Identity-stratified {n_splits}-fold CV  "
                f"({len(dataset)} images, {n_ids} identities)")

    # Count per-class samples for weighted loss
    class_counts_map: dict[int, int] = {}
    for lbl in labels:
        class_counts_map[lbl] = class_counts_map.get(lbl, 0) + 1
    n_classes    = len(class_counts_map)
    class_counts = [class_counts_map.get(i, 1) for i in range(n_classes)]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    fold_accuracies: list[float] = []
    use_amp = device.type == "cuda"

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(dataset.data, labels, groups)):
        fold_no = fold + 1
        n_train_ids = len({groups[i] for i in train_idx})
        n_val_ids   = len({groups[i] for i in val_idx})
        logger.info(f"── Fold {fold_no}/{n_splits}  "
                    f"train={len(train_idx)} imgs ({n_train_ids} ids)  "
                    f"val={len(val_idx)} imgs ({n_val_ids} ids) ──")

        fold_train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=use_amp,
        )
        fold_val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=use_amp,
        )

        model     = model_builder().to(device)
        criterion = build_criterion(cfg, class_counts=class_counts, device=device)
        scaler    = None   # AMP disabled: gradient overflow with some architectures

        # CutMix, SupCon, and TTA settings from config
        try:    cutmix_alpha  = float(cfg["cutmix_alpha"])
        except Exception: cutmix_alpha = 0.0
        try:    supcon_weight = float(cfg["supcon_weight"])
        except Exception: supcon_weight = 0.0
        try:    use_tta = bool(cfg["tta"])
        except Exception: use_tta = False
        if fold == 0:
            logger.info(f"  CutMix alpha={cutmix_alpha}  SupCon weight={supcon_weight}  TTA={use_tta}")

        # Two-stage fine-tuning: freeze backbone for the first N epochs, then unfreeze.
        try:    freeze_epochs = int(cfg["freeze_epochs"])
        except Exception: freeze_epochs = 0
        if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
            model.freeze_backbone()
            logger.info(f"  Backbone frozen for first {freeze_epochs} epochs.")

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, n_epochs=cfg.epochs)

        best_bal_acc   = 0.0
        best_ckpt_path = os.path.join(save_dir, f"best_model_fold_{fold_no}.pth")

        train_writer = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/train"))
        val_writer   = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/val"))

        for epoch in range(cfg.epochs):
            # Unfreeze backbone after warm-up and rebuild optimizer with diff LR
            if freeze_epochs > 0 and epoch == freeze_epochs and hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()
                optimizer = build_optimizer(model, cfg)
                scheduler = build_scheduler(optimizer, cfg,
                                            n_epochs=cfg.epochs - freeze_epochs)
                logger.info(f"  Backbone unfrozen at epoch {epoch+1} — full fine-tuning.")

            tr_loss, tr_acc = train_one_epoch(
                model, fold_train_loader, criterion, optimizer, device,
                scaler=scaler, clip_grad=5.0, cutmix_alpha=cutmix_alpha,
                supcon_weight=supcon_weight)
            vl_loss, vl_acc, vl_bal = validate_one_epoch(
                model, fold_val_loader, criterion, device, scaler=scaler,
                tta=use_tta)
            scheduler.step()

            train_writer.add_scalar("Loss",             tr_loss, epoch)
            train_writer.add_scalar("Accuracy",         tr_acc,  epoch)
            val_writer.add_scalar("Loss",               vl_loss, epoch)
            val_writer.add_scalar("Accuracy",           vl_acc,  epoch)
            val_writer.add_scalar("Balanced_Accuracy",  vl_bal,  epoch)

            logger.info(
                f"  Fold {fold_no}  Ep {epoch+1:>3}/{cfg.epochs} │ "
                f"tr {tr_acc:.3f}  │  val acc {vl_acc:.3f}  bal {vl_bal:.3f}"
            )

            # Save on balanced accuracy (not plain accuracy) to avoid class bias
            if vl_bal > best_bal_acc:
                best_bal_acc = vl_bal
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"   ✓ Saved  (bal_acc={best_bal_acc:.4f})")

        train_writer.close()
        val_writer.close()

        # Final evaluation on fold val set
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        true_labels, pred_labels = evaluate_fn(model, fold_val_loader, device)

        eval_dir = os.path.join(save_dir, f"eval_results/fold_{fold_no}")
        os.makedirs(eval_dir, exist_ok=True)
        img_names = [os.path.basename(dataset.data[i][0]) for i in val_idx]
        save_results_fn(img_names, true_labels, pred_labels, idx2class, eval_dir,
                        title=f"Stage-2  Fold {fold_no}")

        fold_accuracies.append(best_bal_acc)
        logger.info(f"  Best balanced acc fold {fold_no}: {best_bal_acc:.4f}")

    mean_acc = float(np.mean(fold_accuracies))
    logger.info(f"CV complete │ mean bal_acc={mean_acc:.4f}  "
                f"per fold={[f'{a:.4f}' for a in fold_accuracies]}")
    return fold_accuracies


# ── Correct cross_validate: subject-stratified, test set never touched ────────

def proper_cross_validate(
    cfg: Any,
    model_builder: Callable[[], nn.Module],
    train_dataset: Dataset,
    save_dir: str,
    evaluate_fn: Callable,
    save_results_fn: Callable,
    device: torch.device,
    n_splits: int = 5,
    start_fold: int = 1,
    end_fold: int | None = None,
) -> list[float]:
    """
    Correct 5-fold CV for a paper:

    - StratifiedGroupKFold on training data only (val_dir is NEVER touched).
    - Subjects are kept intact: the same person never appears in both the
      fold's train and val splits, eliminating identity leakage.
    - Checkpoint selection is based on balanced accuracy on the fold's
      held-out training subjects — not on the held-out test set.
    - val_dir (the 20 % subject-disjoint split from setup_*.py) remains
      completely sealed and is used only by eval_two_stage.py.
    """
    labels  = [lbl for *_, lbl in train_dataset.data]
    groups  = [_get_identity(p) for p, *_ in train_dataset.data]
    n_ids   = len(set(groups))
    logger.info(
        f"Proper identity-stratified {n_splits}-fold CV  "
        f"({len(train_dataset)} images, {n_ids} training identities)"
    )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    fold_accuracies: list[float] = []
    use_amp = device.type == "cuda"

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(train_dataset.data, labels, groups)):
        fold_no = fold + 1
        if fold_no < start_fold:
            logger.info(f"── Fold {fold_no}/{n_splits} — skipped (resuming from fold {start_fold})")
            continue
        if end_fold is not None and fold_no > end_fold:
            logger.info(f"── Fold {fold_no}/{n_splits} — skipped (end_fold={end_fold})")
            continue

        n_train_ids = len({groups[i] for i in train_idx})
        n_val_ids   = len({groups[i] for i in val_idx})
        logger.info(
            f"── Fold {fold_no}/{n_splits}  "
            f"train={len(train_idx)} imgs ({n_train_ids} ids)  "
            f"val={len(val_idx)} imgs ({n_val_ids} ids) ──"
        )

        fold_train_loader = DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=use_amp,
        )
        fold_val_loader = DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=use_amp,
        )

        # Fix random seed per fold for reproducible, well-conditioned initialization.
        # Without this, each parallel fold process gets a different system-entropy
        # seed. For some seeds the MGSA gate_conv initializes to a bad local minimum
        # (loss stuck at ln(2) = 0.693) and never converges.
        torch.manual_seed(42 + fold_no)
        model     = model_builder().to(device)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, n_epochs=cfg.epochs)
        criterion = build_criterion(cfg)
        scaler    = None   # AMP disabled: ConvNeXt+attention has fp16 overflow

        best_bal_acc   = 0.0
        best_ckpt_path = os.path.join(save_dir, f"best_model_fold_{fold_no}.pth")

        train_writer = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/train"))
        val_writer   = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/val"))

        for epoch in range(cfg.epochs):
            tr_loss, tr_acc = train_one_epoch(
                model, fold_train_loader, criterion, optimizer, device,
                scaler=scaler, clip_grad=5.0)
            vl_loss, vl_acc, vl_bal = validate_one_epoch(
                model, fold_val_loader, criterion, device, scaler=scaler)
            scheduler.step()

            train_writer.add_scalar("Loss",             tr_loss, epoch)
            train_writer.add_scalar("Accuracy",         tr_acc,  epoch)
            val_writer.add_scalar("Loss",               vl_loss, epoch)
            val_writer.add_scalar("Accuracy",           vl_acc,  epoch)
            val_writer.add_scalar("Balanced_Accuracy",  vl_bal,  epoch)

            logger.info(
                f"  Fold {fold_no}  Ep {epoch+1:>3}/{cfg.epochs} │ "
                f"tr {tr_acc:.3f}  │  val {vl_acc:.3f}  bal {vl_bal:.3f}"
            )

            if vl_bal > best_bal_acc:
                best_bal_acc = vl_bal
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"   ✓ Saved  (bal_acc={best_bal_acc:.4f})")

        train_writer.close()
        val_writer.close()

        # Report per-fold performance on the fold's held-out training subjects
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        true_labels, pred_labels = evaluate_fn(model, fold_val_loader, device)

        eval_dir = os.path.join(save_dir, f"eval_results/fold_{fold_no}")
        os.makedirs(eval_dir, exist_ok=True)
        img_names = [os.path.basename(train_dataset.data[i][0]) for i in val_idx]
        save_results_fn(img_names, true_labels, pred_labels, idx2class, eval_dir,
                        title=f"Stage-2  Fold {fold_no}")

        fold_accuracies.append(best_bal_acc)
        logger.info(f"  Best balanced acc fold {fold_no}: {best_bal_acc:.4f}")

    mean_acc = float(np.mean(fold_accuracies))
    logger.info(
        f"CV complete │ mean bal_acc={mean_acc:.4f}  "
        f"per fold={[f'{a:.4f}' for a in fold_accuracies]}"
    )
    return fold_accuracies


# ── Legacy cross_validate (kept for backward-compat) ─────────────────────────

def cross_validate(
    cfg: Any,
    model_builder: Callable[[], nn.Module],
    train_dataset: Dataset,
    val_loader: DataLoader,
    save_dir: str,
    evaluate_fn: Callable,
    save_results_fn: Callable,
    device: torch.device,
    n_splits: int = 5,
    start_fold: int = 1,
) -> list[float]:
    """Original CV using a fixed val_loader (kept for backward-compat)."""
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = [lbl for *_, lbl in train_dataset.data]
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    fold_accuracies: list[float] = []
    use_amp = device.type == "cuda"

    for fold, (train_idx, _) in enumerate(skf.split(train_dataset, labels)):
        fold_no = fold + 1
        if fold_no < start_fold:
            logger.info(f"── Fold {fold_no}/{n_splits} — skipped (resuming from fold {start_fold})")
            continue
        logger.info(f"── Fold {fold_no}/{n_splits} ──────────────────────────")

        fold_train_loader = DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=use_amp,
        )

        model     = model_builder().to(device)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, n_epochs=cfg.epochs)
        criterion = build_criterion(cfg)
        scaler    = None   # AMP disabled: ConvNeXt+attention has fp16 overflow

        best_val_acc   = 0.0
        best_ckpt_path = os.path.join(save_dir, f"best_model_fold_{fold_no}.pth")

        train_writer = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/train"))
        val_writer   = SummaryWriter(os.path.join(save_dir, f"logs/fold_{fold_no}/val"))

        for epoch in range(cfg.epochs):
            tr_loss, tr_acc = train_one_epoch(
                model, fold_train_loader, criterion, optimizer, device,
                scaler=scaler, clip_grad=5.0)
            vl_loss, vl_acc, vl_bal = validate_one_epoch(
                model, val_loader, criterion, device, scaler=scaler)
            scheduler.step()

            train_writer.add_scalar("Loss",     tr_loss, epoch)
            train_writer.add_scalar("Accuracy", tr_acc,  epoch)
            val_writer.add_scalar("Loss",       vl_loss, epoch)
            val_writer.add_scalar("Accuracy",   vl_acc,  epoch)

            logger.info(
                f"  Fold {fold_no}  Ep {epoch+1:>3}/{cfg.epochs} │ "
                f"tr {tr_acc:.3f}  │  val {vl_acc:.3f}  bal {vl_bal:.3f}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"   ✓ Saved  (val_acc={best_val_acc:.4f})")

        train_writer.close()
        val_writer.close()

        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        true_labels, pred_labels = evaluate_fn(model, val_loader, device)

        eval_dir = os.path.join(save_dir, f"eval_results/fold_{fold_no}")
        os.makedirs(eval_dir, exist_ok=True)
        img_names = [os.path.basename(p) for p, *_ in val_loader.dataset.data]
        save_results_fn(img_names, true_labels, pred_labels, idx2class, eval_dir)

        fold_accuracies.append(best_val_acc)
        logger.info(f"  Best val acc fold {fold_no}: {best_val_acc:.4f}")

    logger.info(f"CV complete │ mean={np.mean(fold_accuracies):.4f}")
    return fold_accuracies
