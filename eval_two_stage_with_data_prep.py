"""eval_two_stage_with_data_prep.py

Two-stage evaluation with automatic data preparation.

Given only a directory of raw images, this script:
  1. Crops the iris region using a YOLO model.
  2. Generates iris segmentation masks using the classical pipeline.
  3. Runs the two-stage classifier (Stage-1: DenseNet-121, Stage-2: ResNet/ConvNeXt).
  4. Saves per-image predictions to a CSV.

Prepared data is written next to the input directory::

    <images_dir>_prepared/
        images/    <- copy of originals
        cropped/   <- YOLO-cropped patches
        masks/     <- segmentation masks

Usage::

    python eval_two_stage_with_data_prep.py \\
        --config      configs/eval_two_stage.yaml \\
        --ckpt1       experiments/stage1/DesNet121_best.pth \\
        --ckpt2       experiments/stage2/best_model_fold_1.pth \\
        --images_dir  data/Iris/Test \\
        --yolo_path   pretrained_models/yolov5s.pt \\
        --save_dir    experiments/eval_two_stage \\
        --conf_thres  0.4 --iou_thres 0.5
"""

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader
from tqdm import tqdm

from lens_dataset import LensDatasetCombined
from models import build_stage2_model
from utils.evaluation import save_eval_results
from utils.prepare_dataset import prepare_masks
from utils.yolo_crop import YoloPredictor, crop_iris_using_yolo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-stage evaluation with automatic data preparation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",         type=str, default="configs/eval_two_stage.yaml")
    p.add_argument("--ckpt1",          type=str, required=True,
                   help="Stage-1 (DenseNet-121) checkpoint.")
    p.add_argument("--ckpt2",          type=str, required=True,
                   help="Stage-2 checkpoint.")
    p.add_argument("--images_dir",     type=str, required=True,
                   help="Directory of raw input images.")
    p.add_argument("--yolo_path",      type=str, required=True,
                   help="Path to YOLO weights for iris cropping.")
    p.add_argument("--save_dir",       type=str, default="experiments/eval_two_stage")
    p.add_argument("--conf_thres",     type=float, default=0.4)
    p.add_argument("--iou_thres",      type=float, default=0.5)
    p.add_argument("--max_detections", type=int,   default=10)
    p.add_argument("--device",         type=str,   default=None,
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


def load_stage1_model(ckpt_path: str, n_classes: int, device: torch.device) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(weights.get("state_dict", weights))
    return model.to(device).eval()


def load_stage2_model(ckpt_path: str, cfg2: DotMap, device: torch.device) -> nn.Module:
    model = build_stage2_model(cfg2)
    weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(weights)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

_YOLO_CLASSES = {0: "iris", 1: "pupil", 2: "sclera", 3: "upper_lashes", 4: "lower_lashes"}


def prepare_data(
    images_dir: str,
    prepared_dir: str,
    yolo_path: str,
    data_prep_cfg: DotMap,
    device: torch.device,
    conf_thres: float,
    iou_thres: float,
    max_dets: int,
) -> None:
    crop_dir   = os.path.join(prepared_dir, "cropped")
    mask_dir   = os.path.join(prepared_dir, "masks")
    images_out = os.path.join(prepared_dir, "images")
    for d in (crop_dir, mask_dir, images_out):
        os.makedirs(d, exist_ok=True)

    print("Cropping iris images with YOLO...")
    yolo = YoloPredictor(
        yolo_path,
        classes_dict=_YOLO_CLASSES,
        device=device,
        conf_threshold=conf_thres,
        iou_threshold=iou_thres,
        max_detections=max_dets,
    )
    crop_iris_using_yolo(images_dir, yolo, crop_dir)
    yolo.remove_models()
    del yolo

    print("Generating segmentation masks...")
    prepare_masks(images_dir, data_prep_cfg, mask_dir)

    print("Copying original images...")
    shutil.copytree(images_dir, images_out, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = arg_parser()
    config = _load_config(args.config)
    cfg1   = _load_config(config.stage1_config)
    cfg2   = _load_config(config.stage2_config)
    data_prep_cfg = _load_config(config.data_prep_config)
    device = _resolve_device(args.device)

    for path, label in [
        (args.ckpt1,      "Stage-1 ckpt"),
        (args.ckpt2,      "Stage-2 ckpt"),
        (args.yolo_path,  "YOLO weights"),
        (args.images_dir, "images_dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Derive prepared directory; fail early if it already exists
    parent, name = os.path.split(args.images_dir.rstrip("/"))
    prepared_dir = os.path.join(parent, f"{name}_prepared")
    if os.path.exists(prepared_dir):
        raise FileExistsError(
            f"Prepared directory already exists: {prepared_dir}\n"
            "Remove it manually before re-running."
        )

    prepare_data(
        args.images_dir, prepared_dir,
        args.yolo_path, data_prep_cfg, device,
        args.conf_thres, args.iou_thres, args.max_detections,
    )

    # ROI ids
    seg_classes = list(cfg2.seg_classes)
    roi_classes = list(cfg2.roi_classes)
    roi_ids = [seg_classes.index(r) + 1 for r in roi_classes] if roi_classes else None

    # Dataset
    test_dataset = LensDatasetCombined(
        prepared_dir, config.class_ids, roi_ids, cfg2.inp_dim, cfg1.inp_dim,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Models
    print("Loading Stage-1 model (DenseNet-121)...")
    model_1 = load_stage1_model(args.ckpt1, len(set(dict(cfg1.class_ids).values())), device)

    print(f"Loading Stage-2 model (backbone: {cfg2.get('backbone', 'resnet')})...")
    model_2 = load_stage2_model(args.ckpt2, cfg2, device)

    # Inference
    stage2_idx2class = {v: k for k, v in dict(cfg2.class_ids).items()}
    all_labels, all_predictions, all_image_names = [], [], []

    with torch.no_grad():
        for images, masks, cropped, labels, img_names in tqdm(test_loader, desc="Evaluating"):
            images  = images.to(device)
            masks   = masks.to(device)
            cropped = cropped.to(device)
            labels  = labels.to(device)

            # Stage 1
            pred1 = torch.argmax(model_1(cropped), dim=1).item()

            if pred1 == config.class_ids["patterned"]:
                final_pred = pred1
            else:
                # Stage 2
                pred2      = torch.argmax(model_2(images, masks), dim=1).item()
                class_name = stage2_idx2class[pred2]
                final_pred = config.class_ids[class_name]

            all_predictions.append(final_pred)
            all_labels.append(labels[0].item())
            all_image_names.append(img_names[0])

    os.makedirs(args.save_dir, exist_ok=True)
    idx2class = {v: k for k, v in dict(config.class_ids).items()}
    save_eval_results(all_image_names, all_labels, all_predictions, idx2class, args.save_dir)
    print(f"Evaluation complete. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
