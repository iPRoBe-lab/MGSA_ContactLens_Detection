"""eval_stage1.py

Evaluate a trained Stage-1 DenseNet-121 classifier on a labelled test set of
cropped iris images.

Usage::

    python eval_stage1.py \\
        --config     configs/stage1.yaml \\
        --model_ckpt experiments/stage1/DesNet121_best.pth \\
        --test_data  data/Iris/Test_Cropped \\
        --save_dir   experiments/stage1/eval_output \\
        [--device    cuda:0]
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader

from lens_dataset import LensDatasetCropped
from utils.evaluation import evaluate_model, make_eval_reports, save_eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a Stage-1 DenseNet-121 model on a labelled test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=str, required=True,
                   help="Stage-1 YAML config (same file used for training).")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the Stage-1 model checkpoint (.pth).")
    p.add_argument("--test_data",  type=str, required=True,
                   help="Directory of labelled cropped test images.")
    p.add_argument("--save_dir",   type=str, default="experiments/stage1/eval_output",
                   help="Where to save the evaluation artefacts.")
    p.add_argument("--device",     type=str, default=None,
                   help="Torch device string, e.g. 'cuda:0'. Auto-detected when omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> DotMap:
    with open(path) as fh:
        return DotMap(yaml.safe_load(fh))


def _resolve_device(cli_device) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_stage1_model(n_classes: int) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = arg_parser()
    cfg    = _load_config(args.config)
    device = _resolve_device(args.device)

    for path, label in [(args.model_ckpt, "model ckpt"), (args.test_data, "test data")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Dataset
    test_dataset = LensDatasetCropped(
        args.test_data,
        cfg.class_ids,
        cfg.inp_dim,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Model
    n_classes = len(set(dict(cfg.class_ids).values()))
    print(f"Building Stage-1 model (DenseNet-121, {n_classes} classes)...")
    model = build_stage1_model(n_classes)
    weights = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(weights.get("state_dict", weights))
    model = model.to(device).eval()

    # Evaluate
    print("Running evaluation...")
    all_names, all_labels, all_preds = evaluate_model(model, test_loader, device)

    # Reports
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    print(make_eval_reports(all_labels, all_preds))

    os.makedirs(args.save_dir, exist_ok=True)
    save_eval_results(all_names, all_labels, all_preds, idx2class, args.save_dir)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
