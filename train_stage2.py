"""
train_stage2.py - Stage-2 training: Normal vs. Clear lens classification.

Supports ConvNeXt and ResNet backbones with mask-guided attention.

--------------------------------------
--combine_data  Merges train + val into one pool and runs identity-stratified
                5-fold CV.  Every subject's images stay in one fold so there
                is zero identity leakage between train and validation.
                The model is saved on *balanced* accuracy (not plain accuracy)
                to prevent the clear-class bias observed before.

Usage
-----
  # Original behaviour (train-only, fixed val set):
  python train_stage2.py --config configs/stage2_convnext.yaml

  # Recommended: combine train+val, identity-stratified CV
  python train_stage2.py --config configs/stage2_convnext.yaml --combine_data
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil

import torch
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader

from lens_dataset import LensDatasetWithMask, LensDatasetCropped
from models import build_stage2_model
from utils.evaluation import evaluate_model, save_eval_results
from utils.training import cross_validate, identity_cross_validate, proper_cross_validate


# ── CLI ───────────────────────────────────────────────────────────────────────

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-2 lens classifier (ResNet or ConvNeXt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="configs/stage2_convnext.yaml")
    p.add_argument("--save_dir",     default="experiments/stage2_convnext")
    p.add_argument("--n_folds",      type=int, default=5)
    p.add_argument("--device",       default=None)
    p.add_argument("--log_level",    default="INFO")
    p.add_argument("--combine_data", action="store_true",
                   help="Merge train+val and run identity-stratified CV "
                        "(recommended — eliminates subject leakage).")
    p.add_argument("--start_fold", type=int, default=1,
                   help="Resume CV from this fold number (1-indexed). "
                        "Skips earlier folds without overwriting their checkpoints.")
    p.add_argument("--end_fold", type=int, default=None,
                   help="Stop after this fold (1-indexed, inclusive). "
                        "Use with --start_fold to run a single fold on one GPU. "
                        "E.g. --start_fold 3 --end_fold 3 runs only fold 3.")
    p.add_argument("--run_dir", default=None,
                   help="Use this exact directory instead of auto-creating a timestamped one. "
                        "Useful for resuming an interrupted run.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str) -> DotMap:
    with open(path) as f:
        return DotMap(yaml.safe_load(f))

def _resolve_device(cli: str | None) -> torch.device:
    if cli:
        return torch.device(cli)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = arg_parser()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg    = _load_config(args.config)
    device = _resolve_device(args.device)
    backbone = str(cfg.backbone).lower()
    use_cropped = backbone.startswith("efficientnet") or bool(cfg.get("use_cropped", False))
    logger.info(f"Device  : {device}")
    logger.info(f"Backbone: {cfg.backbone}")
    logger.info(f"Combine : {args.combine_data}")
    logger.info(f"Dataset : {'cropped iris' if use_cropped else 'full image + mask'}")

    for path, label in [(cfg.data.train_dir, "train"),
                         (cfg.data.val_dir,   "val")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} dir not found: {path}")

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = os.path.join(
            args.save_dir,
            f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            + ("_combined" if args.combine_data else ""),
        )
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, run_dir)
    logger.info(f"Run dir : {run_dir}")

    # ROI mask ids (only used for mask-guided backbones, not for efficientnet)
    seg_classes = list(cfg.seg_classes) if cfg.get("seg_classes") else []
    roi_classes = list(cfg.roi_classes) if cfg.get("roi_classes") else []
    roi_ids = ([seg_classes.index(r) + 1 for r in roi_classes]
               if roi_classes and seg_classes else None)

    def model_builder():
        return build_stage2_model(cfg)

    # Optional extra flat image directories (e.g. UND dataset)
    extra_dirs = list(cfg.get("extra_train_dirs") or []) or None

    def _make_dataset(root, train_flag):
        if use_cropped:
            return LensDatasetCropped(
                root, cfg.class_ids, inp_dim=cfg.inp_dim, train=train_flag,
                extra_dirs=extra_dirs if train_flag else None,
            )
        else:
            return LensDatasetWithMask(
                root, cfg.class_ids, roi_ids,
                inp_dim=cfg.inp_dim, train=train_flag,
            )

    if args.combine_data:
        # ── Identity-stratified CV on merged train+val ────────────────────────
        train_ds = _make_dataset(cfg.data.train_dir, train_flag=True)
        val_ds   = _make_dataset(cfg.data.val_dir,   train_flag=True)

        n_train_orig = len(train_ds.data)
        train_ds.data = train_ds.data + val_ds.data
        logger.info(f"Combined dataset: {len(train_ds)} images "
                    f"(train={n_train_orig}, val={len(val_ds.data)})")

        fold_accs = identity_cross_validate(
            cfg=cfg,
            model_builder=model_builder,
            dataset=train_ds,
            save_dir=run_dir,
            evaluate_fn=evaluate_model,
            save_results_fn=save_eval_results,
            device=device,
            n_splits=args.n_folds,
        )
    else:
        # ── Correct CV: subject-stratified on training data, test set sealed ──
        # Checkpoints are selected on held-out *training* subjects.
        # val_dir is never seen during training; it is used only by eval_two_stage.py.
        train_ds = _make_dataset(cfg.data.train_dir, train_flag=True)
        logger.info(f"Train: {len(train_ds)}  (val_dir sealed for final eval only)")

        fold_accs = proper_cross_validate(
            cfg=cfg,
            model_builder=model_builder,
            train_dataset=train_ds,
            save_dir=run_dir,
            evaluate_fn=evaluate_model,
            save_results_fn=save_eval_results,
            device=device,
            n_splits=args.n_folds,
            start_fold=args.start_fold,
            end_fold=args.end_fold,
        )

    logger.info(f"Fold scores : {[f'{a:.4f}' for a in fold_accs]}")
    logger.info(f"Mean        : {sum(fold_accs)/len(fold_accs):.4f}")
    logger.info(f"Artefacts   → {run_dir}")


if __name__ == "__main__":
    main()
