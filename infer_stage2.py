"""
infer_stage2.py - Backbone-agnostic Stage-2 inference script.

Runs a trained Stage-2 model on a single image or a folder of images.
The iris ROI mask is extracted on-the-fly using the data-prep pipeline.
Results are saved to a CSV file.

Usage:
  python infer_stage2.py \\
      --config       configs/stage2_convnext.yaml \\
      --model_ckpt   experiments/stage2/run_.../best_model_fold_1.pth \\
      --image_path   data/images/ \\
      --data-config  configs/data_prep.yaml \\
      --save_dir     outputs/results
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from dotmap import DotMap
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models import build_stage2_model
from utils.prepare_dataset import get_roi_mask


def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 inference (ResNet or ConvNeXt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      type=str, required=True,
                   help="Path to model config YAML.")
    p.add_argument("--model_ckpt",  type=str, required=True,
                   help="Path to model .pth checkpoint.")
    p.add_argument("--image_path",  type=str, required=True,
                   help="Path to a single image or a directory of images.")
    p.add_argument("--data-config", type=str, default="configs/data_prep.yaml",
                   help="Path to data-prep config YAML.")
    p.add_argument("--save_dir",    type=str, default=None,
                   help="Directory to save results CSV.")
    p.add_argument("--device",      type=str, default=None,
                   help="Torch device string, e.g. 'cuda:0'.")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return DotMap(yaml.safe_load(f))


def prepare_input_image(img, data_cfg, img_transform, mask_transform, roi_ids):
    img_gray = np.array(img.convert('L'))
    mask, _, success = get_roi_mask(img_gray, configs=data_cfg, debug=False, return_success=True)
    mask_pil = Image.fromarray(mask).convert('L')
    if roi_ids is not None:
        mask = mask_pil.point(lambda p: p in roi_ids, '1')
    else:
        mask = mask_pil.point(lambda p: p > 0, '1')
    img = img_transform(img)
    mask = mask_transform(mask)
    return img, mask, success


def main():
    args = arg_parser()
    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_classes = cfg.seg_classes
    roi_classes = cfg.roi_classes
    roi_ids = [seg_classes.index(r) + 1 for r in roi_classes] if roi_classes else None

    model = build_stage2_model(cfg)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    images = []
    if os.path.isdir(args.image_path):
        images = [os.path.join(args.image_path, img) for img in os.listdir(args.image_path)]
    else:
        images.append(args.image_path)

    img_transform = transforms.Compose([
        transforms.Resize(cfg.inp_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(cfg.inp_dim, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    predictions = []
    is_correct_masks = []
    for img_path in tqdm(images, desc='Infering images'):
        img = Image.open(img_path).convert('RGB')
        img, mask, is_mask = prepare_input_image(img, data_cfg, img_transform, mask_transform, roi_ids)
        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        outputs = model(img, mask)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        predictions.append(prediction)
        is_correct_masks.append(is_mask)

    image_names = [os.path.basename(p) for p in images[:len(predictions)]]
    class_names = list(cfg.class_ids.keys())
    save_dir = args.save_dir or os.path.join(os.path.dirname(args.model_ckpt), "results")
    os.makedirs(save_dir, exist_ok=True)

    pred_classes = [class_names[p] for p in predictions]
    pd.DataFrame({
        "Image Name":  image_names,
        "Prediction":  pred_classes,
        "Mask Success": is_correct_masks,
    }).to_csv(os.path.join(save_dir, "results.csv"), index=False)

    print(f"Inference complete. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
