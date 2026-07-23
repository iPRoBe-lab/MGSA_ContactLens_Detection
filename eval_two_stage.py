"""eval_two_stage.py

Improved two-stage lens classification evaluation:
  - Stage 1: DenseNet-121  →  patterned vs. non-patterned  (cropped images)
  - Stage 2: ConvNeXt ensemble (all fold checkpoints) →  normal vs. clear
             (full image + iris ROI mask)

Improvements over previous version
------------------------------------
* Stage-2 **ensemble**: pass all fold checkpoints; predictions are averaged
  over all models (soft-vote) before the argmax.
* **Batched inference** – configurable batch size, not hard-coded to 1.
* **Per-stage metrics**: accuracy at Stage 1, Stage 2, and the full pipeline.
* **3-class confusion matrix** with class names (patterned / normal / clear).
* Saves a rich HTML-style summary to the output directory.

Usage
-----
    python eval_two_stage.py \\
        --config  configs/eval_two_stage.yaml \\
        --ckpt1   experiments/stage1/run_20260324_010640/DesNet121_best.pth \\
        --ckpt2   experiments/stage2_convnext/run_20260324_012051/best_model_fold_1.pth \\
                  experiments/stage2_convnext/run_20260324_012051/best_model_fold_2.pth \\
                  experiments/stage2_convnext/run_20260324_012051/best_model_fold_3.pth \\
                  experiments/stage2_convnext/run_20260324_012051/best_model_fold_4.pth \\
                  experiments/stage2_convnext/run_20260324_012051/best_model_fold_5.pth \\
        --save_dir experiments/eval_two_stage
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yaml
from dotmap import DotMap
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from lens_dataset import LensDatasetCombined
from models import build_stage2_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-stage lens-type evaluation with Stage-2 ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default="configs/eval_two_stage.yaml")
    p.add_argument("--ckpt1",     required=True,
                   help="Stage-1 DenseNet-121 checkpoint.")
    p.add_argument("--ckpt2",     required=True, nargs="+",
                   help="One or more Stage-2 checkpoints (all are ensembled).")
    p.add_argument("--save_dir",  default="experiments/eval_two_stage")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device",    default=None)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> DotMap:
    with open(path) as f:
        return DotMap(yaml.safe_load(f))


def resolve_device(cli: str | None) -> torch.device:
    if cli:
        return torch.device(cli)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stage1(ckpt: str, n_classes: int, device: torch.device) -> nn.Module:
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    w = torch.load(ckpt, map_location=device)
    m.load_state_dict(w.get("state_dict", w))
    return m.to(device).eval()


def load_stage2_ensemble(ckpts: list[str], cfg2: DotMap,
                          device: torch.device) -> list[nn.Module]:
    models_list = []
    for ckpt in ckpts:
        m = build_stage2_model(cfg2)
        w = torch.load(ckpt, map_location=device)
        m.load_state_dict(w)
        models_list.append(m.to(device).eval())
    return models_list


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                           title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)*2),
                                    max(4, len(class_names)*1.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = arg_parser()
    cfg    = load_cfg(args.config)
    cfg1   = load_cfg(cfg.stage1_config)
    cfg2   = load_cfg(cfg.stage2_config)
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Validate paths ────────────────────────────────────────────────────────
    for path, label in [(args.ckpt1, "Stage-1 ckpt"),
                         (cfg.data.test_dir, "test_dir")]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {label} not found: {path}")
    for ckpt in args.ckpt2:
        if not os.path.exists(ckpt):
            sys.exit(f"ERROR: Stage-2 ckpt not found: {ckpt}")

    print(f"Device      : {device}")
    print(f"Test dir    : {cfg.data.test_dir}")
    print(f"Stage-2 folds: {len(args.ckpt2)}")

    # ── Backbone-specific setup ───────────────────────────────────────────────
    s2_backbone = str(cfg2.get("backbone", "convnext")).lower()
    use_cropped_for_s2 = s2_backbone.startswith("efficientnet")

    seg_classes = list(cfg2.seg_classes) if cfg2.get("seg_classes") else []
    roi_classes = list(cfg2.roi_classes) if cfg2.get("roi_classes") else []
    roi_ids = ([seg_classes.index(r) + 1 for r in roi_classes]
               if roi_classes and seg_classes else None)

    # LensDatasetCombined provides full image, mask, AND cropped iris.
    # When Stage-2 uses EfficientNet (cropped), resize cropped images to cfg2.inp_dim
    # directly — avoid a double-resize which changes pixel values vs training.
    cropped_dim = cfg2.inp_dim if use_cropped_for_s2 else cfg1.inp_dim
    dataset = LensDatasetCombined(
        cfg.data.test_dir, cfg.class_ids, roi_ids,
        cfg2.inp_dim, cropped_dim,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(dataset)}")
    print(f"Stage-2 backbone: {s2_backbone} "
          f"({'cropped iris' if use_cropped_for_s2 else 'full image + mask'})")

    # ── Load models ───────────────────────────────────────────────────────────
    n_s1_classes = len(set(dict(cfg1.class_ids).values()))
    print(f"\nLoading Stage-1 (DenseNet-121, {n_s1_classes} classes)…")
    model1 = load_stage1(args.ckpt1, n_s1_classes, device)

    print(f"Loading Stage-2 ensemble ({len(args.ckpt2)} fold(s), "
          f"backbone={cfg2.get('backbone','convnext')})…")
    models2 = load_stage2_ensemble(args.ckpt2, cfg2, device)

    # ── Class mappings ────────────────────────────────────────────────────────
    # cfg.class_ids  e.g. {patterned:0, normal:1, clear:2}
    class_ids   = dict(cfg.class_ids)           # name → id
    idx2class   = {v: k for k, v in class_ids.items()}
    n_classes   = len(class_ids)
    class_names = [idx2class[i] for i in range(n_classes)]

    # Stage-2 maps to a subset of global class ids
    s2_idx2class = {v: k for k, v in dict(cfg2.class_ids).items()}
    pat_id       = class_ids["patterned"]

    # ── Inference ─────────────────────────────────────────────────────────────
    all_gt, all_pred, all_names = [], [], []
    # also track intermediate stage decisions per image
    stage1_preds_per_image = []   # 0=patterned, 1=non-patterned

    with torch.no_grad():
        for images, masks, cropped, labels, img_names in tqdm(loader, desc="Eval"):
            images  = images.to(device)
            masks   = masks.to(device)
            cropped = cropped.to(device)
            B = images.size(0)

            # Stage 1 – patterned vs. non-patterned
            logits1 = model1(cropped)                      # (B, 2)
            pred1   = torch.argmax(logits1, dim=1)         # (B,)

            # Stage 2 ensemble – average softmax over all fold models
            # EfficientNet uses cropped iris images; ConvNeXt/ResNet use full image + mask.
            logits2_sum = torch.zeros(B, 2, device=device)
            if use_cropped_for_s2:
                dummy_mask = torch.zeros(B, 1, 1, 1, device=device)
                for m2 in models2:
                    logits2_sum += F.softmax(m2(cropped, dummy_mask), dim=1)
            else:
                for m2 in models2:
                    logits2_sum += F.softmax(m2(images, masks), dim=1)
            pred2 = torch.argmax(logits2_sum, dim=1)      # (B,)

            # Combine: patterned → class_ids["patterned"]
            #          non-patterned → map Stage-2 output to global class id
            for i in range(B):
                if pred1[i].item() == pat_id:
                    final = pat_id
                else:
                    s2_name = s2_idx2class[pred2[i].item()]
                    final   = class_ids[s2_name]

                all_pred.append(final)
                all_gt.append(labels[i].item())
                all_names.append(img_names[i])
                stage1_preds_per_image.append(pred1[i].item())

    # ── Overall metrics ───────────────────────────────────────────────────────
    acc = accuracy_score(all_gt, all_pred)
    cm  = confusion_matrix(all_gt, all_pred,
                            labels=list(range(n_classes)))
    report = classification_report(all_gt, all_pred,
                                    target_names=class_names, digits=4)

    print(f"\n{'='*60}")
    print(f"  TWO-STAGE PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"\n{report}")

    # ── Stage-1 standalone accuracy ───────────────────────────────────────────
    # Stage 1 is binary: patterned(0) vs non-patterned(1)
    # Ground truth: patterned if global label == pat_id, else non-patterned
    gt_s1   = [1 if g != pat_id else 0 for g in all_gt]
    pred_s1 = [1 if p != pat_id else 0 for p in stage1_preds_per_image]
    acc_s1  = accuracy_score(gt_s1, pred_s1)

    # ── Stage-2 standalone accuracy (only non-patterned images) ───────────────
    # Among images that Stage 1 correctly identified as non-patterned,
    # how well does Stage 2 distinguish normal vs. clear?
    s2_gt, s2_pred = [], []
    for g, p, s1p in zip(all_gt, all_pred, stage1_preds_per_image):
        if g != pat_id:            # truly non-patterned
            s2_gt.append(g)
            s2_pred.append(p)
    acc_s2 = accuracy_score(s2_gt, s2_pred) if s2_gt else float("nan")

    print(f"  Stage-1 Accuracy (patterned/non-patterned): {acc_s1*100:.2f}%")
    print(f"  Stage-2 Accuracy (normal/clear, non-patterned only): "
          f"{acc_s2*100:.2f}%")
    print(f"  Full Pipeline Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print("\n  Per-class accuracy:")
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)
    for g, p in zip(all_gt, all_pred):
        per_class_total[g]   += 1
        per_class_correct[g] += int(g == p)
    for idx, name in sorted(idx2class.items()):
        tot = per_class_total[idx]
        cor = per_class_correct[idx]
        print(f"    {name:<12}: {cor}/{tot}  ({100*cor/tot:.2f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # CSV
    pd.DataFrame({
        "image":      all_names,
        "true_label": [idx2class[g] for g in all_gt],
        "predicted":  [idx2class[p] for p in all_pred],
        "correct":    [g == p for g, p in zip(all_gt, all_pred)],
    }).to_csv(os.path.join(args.save_dir, "results.csv"), index=False)

    # Classification report
    with open(os.path.join(args.save_dir, "classification_report.txt"), "w") as f:
        f.write(f"Stage-1 Accuracy : {acc_s1*100:.2f}%\n")
        f.write(f"Stage-2 Accuracy : {acc_s2*100:.2f}%\n")
        f.write(f"Pipeline Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        title=f"Two-Stage Pipeline — Confusion Matrix\n(Accuracy: {acc*100:.2f}%)",
        save_path=os.path.join(args.save_dir, "confusion_matrix.png"),
    )

    # Stage-1 confusion matrix
    cm1 = confusion_matrix(gt_s1, pred_s1)
    plot_confusion_matrix(
        cm1, ["patterned", "non-patterned"],
        title=f"Stage-1 — Confusion Matrix\n(Accuracy: {acc_s1*100:.2f}%)",
        save_path=os.path.join(args.save_dir, "confusion_matrix_stage1.png"),
    )

    # Stage-2 confusion matrix (non-patterned images only)
    if s2_gt:
        cm2 = confusion_matrix(s2_gt, s2_pred, labels=[class_ids["normal"],
                                                         class_ids["clear"]])
        plot_confusion_matrix(
            cm2, ["normal", "clear"],
            title=f"Stage-2 — Confusion Matrix (non-patterned only)\n"
                  f"(Accuracy: {acc_s2*100:.2f}%)",
            save_path=os.path.join(args.save_dir, "confusion_matrix_stage2.png"),
        )

    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
