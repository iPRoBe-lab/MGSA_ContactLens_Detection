"""
utils/evaluation.py - Reusable evaluation helpers for Stage-2 models.

Provides:
  evaluate_model()    - collect predictions from a DataLoader
  make_eval_reports() - build classification report + confusion matrix
  save_eval_results() - save CSV, txt report, and confusion-matrix PNG
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    Run inference on *dataloader* and return (true_labels, predicted_labels).

    The dataloader must yield (inputs, roi_masks, labels).
    roi_masks are forwarded to the model; pass None tensors if unused.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []

    with torch.no_grad():
        for inputs, roi_masks, labels in dataloader:
            inputs    = inputs.to(device)
            roi_masks = roi_masks.to(device) if roi_masks is not None else None
            labels    = labels.to(device)

            outputs = model(inputs, roi_masks)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return all_labels, all_preds


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def make_eval_reports(
    true_labels: list[int],
    predicted_labels: list[int],
) -> tuple[str, np.ndarray]:
    """Return (classification_report_str, confusion_matrix_array)."""
    clf_report  = classification_report(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return clf_report, conf_matrix


def save_eval_results(
    image_names: list[str],
    gt_labels: list[int],
    predictions: list[int],
    idx2class: dict[int, str],
    save_dir: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save evaluation artefacts to *save_dir*:

    * ``results.csv``              - per-image GT vs. prediction
    * ``classification_report.txt``
    * ``confusion_matrix.csv``
    * ``confusion_matrix.png``

    Args:
        image_names: list of image file-names (no path needed).
        gt_labels:   ground-truth class indices.
        predictions: predicted class indices.
        idx2class:   mapping  index -> class name.
        save_dir:    output directory (created if missing).
        title:       title for the confusion-matrix plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    clf_report, conf_matrix = make_eval_reports(gt_labels, predictions)

    # --- per-image CSV --------------------------------------------------
    gt_classes   = [idx2class[l] for l in gt_labels]
    pred_classes = [idx2class[p] for p in predictions]
    pd.DataFrame({
        "Image Name":    image_names,
        "Ground Truth":  gt_classes,
        "Prediction":    pred_classes,
    }).to_csv(os.path.join(save_dir, "results.csv"), index=False)

    # --- classification report -----------------------------------------
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        if isinstance(clf_report, str):
            fh.write(clf_report)
        else:
            json.dump(clf_report, fh, indent=2)

    # --- confusion matrix CSV ------------------------------------------
    np.savetxt(
        os.path.join(save_dir, "confusion_matrix.csv"),
        conf_matrix,
        delimiter=",",
        fmt="%d",
    )

    # --- confusion matrix PNG ------------------------------------------
    class_names = [idx2class[i] for i in range(len(idx2class))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print(f"[eval] Results saved to: {save_dir}")
