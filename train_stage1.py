"""
train_stage1.py - Stage-1 DenseNet-121 trainer.

Trains a DenseNet-121 binary (live vs. artefact) classifier on cropped iris
images.  The full training loop (train + validation) runs for a fixed number
of epochs and the best checkpoint is saved based on validation accuracy.

Usage:
  python train_stage1.py \\
      --config    configs/stage1.yaml \\
      --save_dir  experiments/stage1 \\
      [--model_ckpt  pretrained/DesNet121_stage1.pth]  # optional fine-tune
      [--device       cuda:0]
      [--log_level    INFO]
"""

import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm

from lens_dataset import LensDatasetCropped
from utils.evaluation import save_eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-1 DenseNet-121 iris classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=str, default="configs/stage1.yaml",
                   help="Stage-1 YAML config.")
    p.add_argument("--save_dir",   type=str, default="experiments/stage1",
                   help="Directory for checkpoints and logs.")
    p.add_argument("--model_ckpt", type=str, default=None,
                   help="Optional Stage-1 checkpoint to fine-tune from.")
    p.add_argument("--device",     type=str, default=None,
                   help="Torch device string, e.g. 'cuda:0'. Auto-detected when omitted.")
    p.add_argument("--log_level",  type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> DotMap:
    with open(path) as fh:
        return DotMap(yaml.safe_load(fh))


def _resolve_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(n_classes: int) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str,
) -> tuple[float, float, list, list, list]:
    """Run one epoch.

    Returns (loss, accuracy, scores, labels, image_names).
    ``optimizer`` is required only when ``phase == 'train'``.
    """
    training = phase == "train"
    model.train(training)

    total_loss, n_correct, n_total = 0.0, 0, 0
    all_scores, all_labels, all_names = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, _mask, labels in tqdm(loader, desc=phase, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds      = torch.argmax(logits, dim=1)
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            all_scores.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_names.extend([""] * labels.size(0))

    avg_loss = total_loss / max(n_total, 1)
    accuracy = n_correct / max(n_total, 1)
    return avg_loss, accuracy, all_scores, all_labels, all_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = arg_parser()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    cfg    = _load_config(args.config)
    device = _resolve_device(args.device)

    run_dir = os.path.join(args.save_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log.info("Save directory: %s", run_dir)

    # Config values - all accessed via cfg.data.* and top-level keys
    train_dir   = cfg.data.train_dir
    val_dir     = cfg.data.val_dir
    inp_dim     = cfg.inp_dim           # [h, w] list from yaml
    batch_size  = int(cfg.batch_size)
    num_workers = int(cfg.num_workers)
    n_epochs    = int(cfg.epochs)
    lr          = float(cfg.lr_scheduler.lr)
    step_size   = int(cfg.lr_scheduler.step_size)
    gamma       = float(cfg.lr_scheduler.gamma)
    class_ids   = dict(cfg.class_ids)
    n_classes   = len(set(class_ids.values()))

    # Datasets - train and val loaded independently from their own directories
    train_ds = LensDatasetCropped(train_dir, class_ids, inp_dim, train=True)
    val_ds   = LensDatasetCropped(val_dir,   class_ids, inp_dim, train=False)
    log.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Model
    model = build_model(n_classes).to(device)
    if args.model_ckpt:
        log.info("Loading checkpoint for fine-tuning: %s", args.model_ckpt)
        weights = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(weights.get("state_dict", weights))

    # Optimiser, loss, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    best_val_acc  = 0.0
    ckpt_path     = os.path.join(run_dir, "DesNet121_best.pth")
    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, *_ = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        val_loss, val_acc, val_preds, val_labels, val_names = run_epoch(
            model, val_loader, criterion, None, device, "val"
        )
        scheduler.step()

        log.info(
            "Epoch %3d/%d | Train loss: %.4f  acc: %.4f | Val loss: %.4f  acc: %.4f",
            epoch, n_epochs, train_loss, train_acc, val_loss, val_acc,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "val_acc": val_acc, "state_dict": model.state_dict()},
                       ckpt_path)
            log.info("   New best val acc %.4f - checkpoint saved: %s", val_acc, ckpt_path)

    # Save training log
    log_path = os.path.join(run_dir, "training_log.json")
    with open(log_path, "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("Training log: %s", log_path)

    # Loss / accuracy curves
    epochs_list = [h["epoch"]      for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses   = [h["val_loss"]   for h in history]
    fig, ax = plt.subplots()
    ax.plot(epochs_list, train_losses, label="Train")
    ax.plot(epochs_list, val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage-1 Training Loss")
    ax.legend()
    loss_plot = os.path.join(run_dir, "DenseNet121_Loss.jpg")
    fig.savefig(loss_plot)
    plt.close(fig)
    log.info("Loss curve: %s", loss_plot)

    # Final evaluation on validation set using the best checkpoint
    log.info("Evaluating best model on validation set...")
    best_weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_weights["state_dict"])
    _, _, val_preds, val_labels, val_names = run_epoch(model, val_loader, criterion, None, device, "val")
    idx2class = {v: k for k, v in class_ids.items()}
    save_eval_results(val_names, val_labels, val_preds, idx2class, run_dir)
    log.info("Evaluation artefacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()