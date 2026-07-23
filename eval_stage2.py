"""
eval_stage2.py - Stage-2 model evaluation.

Evaluates a trained Stage-2 model (ResNet or ConvNeXt backbone) on a labelled
test set. Outputs a CSV of per-image predictions, a classification report, and
a confusion-matrix image.

Usage:
  python eval_stage2.py \\
      --config     configs/stage2.yaml \\
      --model_ckpt experiments/stage2/fold1/best_model_fold_1.pth \\
      --test_data  data/Iris/Test_prepared \\
      --save_dir   experiments/stage2/eval_output \\
      [--device    cuda:0]
"""

import argparse
import os

import torch
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader

from lens_dataset import LensDatasetWithMask
from models import build_stage2_model
from utils.evaluation import evaluate_model, make_eval_reports, save_eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a Stage-2 model on a labelled test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=str, required=True,
                   help="Stage-2 YAML config (same file used for training).")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the Stage-2 model checkpoint (.pth).")
    p.add_argument("--test_data",  type=str, required=True,
                   help="Root directory of the prepared test set (images/ and masks/ sub-dirs).")
    p.add_argument("--save_dir",   type=str, default="experiments/stage2/eval_output",
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


def _resolve_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    seg_classes = list(cfg.seg_classes)
    roi_classes = list(cfg.roi_classes)
    roi_ids     = [seg_classes.index(r) + 1 for r in roi_classes] if roi_classes else None

    test_dataset = LensDatasetWithMask(
        args.test_data,
        cfg.class_ids,
        roi_ids,
        cfg.inp_dim,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Model
    backbone = cfg.get("backbone", "resnet")
    print(f"Building Stage-2 model (backbone: {backbone})…")
    model = build_stage2_model(cfg)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model = model.to(device).eval()

    # Evaluate
    print("Running evaluation…")
    all_names, all_labels, all_preds = evaluate_model(model, test_loader, device)

    # Reports
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    print(make_eval_reports(all_labels, all_preds))

    os.makedirs(args.save_dir, exist_ok=True)
    save_eval_results(all_names, all_labels, all_preds, idx2class, args.save_dir)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
