"""infer_stage1.py

Run Stage-1 (DenseNet-121) inference on a directory of unlabelled cropped iris
images and save the predictions to a CSV.

Usage::

    python infer_stage1.py \\
        --config     configs/stage1.yaml \\
        --model_ckpt experiments/stage1/DesNet121_best.pth \\
        --images_dir data/Iris/Test_Cropped \\
        --save_dir   experiments/stage1/infer_output \\
        [--device    cuda:0]
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import yaml
from dotmap import DotMap
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 DenseNet-121 inference on unlabelled images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=str, required=True,
                   help="Stage-1 YAML config.")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the Stage-1 model checkpoint (.pth).")
    p.add_argument("--images_dir", type=str, required=True,
                   help="Directory of cropped iris images to classify.")
    p.add_argument("--save_dir",   type=str, default="experiments/stage1/infer_output",
                   help="Directory to save results.csv.")
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


_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_images(directory: str) -> list:
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in _IMG_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    return paths


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

    for path, label in [(args.model_ckpt, "model ckpt"), (args.images_dir, "images_dir")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Model
    n_classes = len(set(dict(cfg.class_ids).values()))
    idx2class = {v: k for k, v in dict(cfg.class_ids).items()}
    print(f"Loading Stage-1 model (DenseNet-121, {n_classes} classes)...")
    model = build_stage1_model(n_classes)
    weights = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(weights.get("state_dict", weights))
    model = model.to(device).eval()

    # Transform
    inp_dim = cfg.inp_dim
    transform = T.Compose([
        T.Resize((inp_dim, inp_dim)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Inference
    image_paths = _collect_images(args.images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    names, preds = [], []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inferring"):
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            pred   = torch.argmax(model(tensor), dim=1).item()
            names.append(os.path.basename(img_path))
            preds.append(idx2class.get(pred, str(pred)))

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "results.csv")
    pd.DataFrame({"Image Name": names, "Prediction": preds}).to_csv(out_path, index=False)
    print(f"Inference complete. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
