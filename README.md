# Two-Stage Contact Lens Detection for Iris Recognition

Official implementation of the IJCB paper:

> **"Two-Stage Contact Lens Detection with Mask-Guided Spatial Attention for Iris Recognition"**

A two-stage deep-learning cascade that classifies each ocular image as:
- **Patterned** — cosmetic (coloured) contact lens → detected at Stage 1
- **Normal** — bare iris, no lens → detected at Stage 2
- **Clear** — transparent soft lens → detected at Stage 2

---

## Method Overview

```
Input image
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 · DenseNet-121                 │
│  256×256 iris crop                      │
│  Patterned vs. Non-patterned            │
└──────────┬──────────────────────────────┘
           │ non-patterned only
           ▼
┌─────────────────────────────────────────┐
│  Stage 2 · ConvNeXt-Base + MGSA         │
│  480×640 full periocular image + mask   │
│  Normal vs. Clear                       │
└─────────────────────────────────────────┘
```

**Stage 1** uses a fine-tuned DenseNet-121 (D-NetPAD backbone) to screen for patterned lens artefacts. Perfect accuracy (100%) on all four benchmarks.

**Stage 2** introduces **Mask-Guided Spatial Attention (MGSA)**: a Hough-derived anatomical ROI mask is injected directly into the ConvNeXt-Base feature hierarchy as a hard spatial prior, combined with a learned soft gate and SE channel recalibration to localise the subtle limbal evidence of transparent lens wear.

---

## Results

| Dataset | Pipeline Acc. | Macro-F1 | Test N |
|---|---|---|---|
| UND-AD100 | **90.00%** | 0.916 | 50 |
| UND-LG4000 | **95.20%** | 0.946 | 250 |
| IITD-Cogent | **97.50%** | 0.974 | 761 |
| IITD-Vista | **98.80%** | 0.988 | 584 |

VeriEye EER improvement after lens-type-guided score calibration:

| Dataset | EER Before | EER After | Rel. Improvement |
|---|---|---|---|
| UND-AD100 | 0.49% | 0.47% | −4.1% |
| UND-LG4000 | 0.75% | 0.64% | −14.7% |
| IITD-Cogent | 0.76% | 0.70% | −7.9% |
| IITD-Vista | 2.26% | 1.62% | −28.3% |

---

## Installation

```bash
# Python 3.10 recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:
```bash
uv venv -p 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Repository Structure

```
configs/
    stage1.yaml                 Stage-1 training config (DenseNet-121)
    stage2_convnext.yaml        Stage-2 ConvNeXt-Base config (paper results)
    stage2_resnet.yaml          Stage-2 ResNet-101 config
    eval_two_stage.yaml         End-to-end evaluation config
    data_prep.yaml              Data preparation / iris segmentation config

models/
    attention.py                MaskGuidedSpatialAttention, MultiHeadROIAttention, SEBlock
    convnext_net.py             AttentionConvNeXt (ConvNeXt-Base + MGSA)
    attention_resnet.py         AttentionResNet (ResNet + MGSA)
    efficientnet_net.py         EfficientNet variant
    model_factory.py            build_stage2_model(cfg) factory

utils/
    iris_seg.py                 Hough-based iris/pupil/outer-arc segmentation
    prepare_dataset.py          Full data preparation pipeline
    training.py                 Train/validate loops, subject-stratified 5-fold CV
    evaluation.py               evaluate_model, save_eval_results
    do_cropping.py              Mask-based iris crop extraction
    yolo_crop.py                YOLO-based iris detection and cropping

lens_dataset.py                 PyTorch Dataset: LensDatasetWithMask, LensDatasetCropped

train_stage1.py                 Train Stage-1
train_stage2.py                 Train Stage-2 (any backbone)
eval_stage1.py                  Evaluate Stage-1
eval_stage2.py                  Evaluate Stage-2
eval_two_stage.py               End-to-end pipeline evaluation with 5-fold ensemble
eval_two_stage_with_data_prep.py Two-stage eval with automatic preprocessing
infer_stage1.py                 Stage-1 inference on new images
infer_stage2.py                 Stage-2 inference on new images

weights/
    AD100/
        stage1/DesNet121_best.pth
        stage2/best_model_fold_{1..5}.pth
    LG4000/  ...
    IITDCogent/  ...
    IITDVista/  ...
```

---

## Pre-trained Weights

Model weights are stored in `weights/` and tracked with **Git LFS**.

After cloning, pull weights with:
```bash
git lfs pull
```

Each dataset has its own Stage-1 and Stage-2 (5-fold ensemble) weights:

| Path | Description |
|---|---|
| `weights/{dataset}/stage1/DesNet121_best.pth` | Stage-1 DenseNet-121 (~28 MB) |
| `weights/{dataset}/stage2/best_model_fold_N.pth` | Stage-2 ConvNeXt-Base + MGSA fold N (~335 MB) |

Available datasets: `AD100`, `LG4000`, `IITDCogent`, `IITDVista`.

---

## Data Preparation

Your raw images should be organised by class in a flat directory:

```
raw_images/
    normal/        *.jpg / *.png
    clear/         *.jpg / *.png
    patterned/     *.jpg / *.png  (Stage-1 only; not used in Stage-2)
```

**Step 1 — Generate iris masks and split into train/val**

```bash
python utils/prepare_dataset.py \
    -i raw_images/ \
    -o data/iris_contact/
```

| Argument | Default | Description |
|---|---|---|
| `-i` / `--input` | *(required)* | Directory of raw input images |
| `-o` / `--output` | *(required)* | Output root (`train/` and `val/` created inside) |
| `-c` / `--config` | `configs/data_prep.yaml` | Data-prep config |
| `-d` / `--debug` | off | Save mask visualisations for inspection |

**Step 2 — Crop images to iris bounding box**

Choose one of two methods:

*Option A — Mask-based (no extra model needed):*
```bash
python utils/do_cropping.py \
    -i data/iris_contact/ \
    -o data/iris_contact/
```

*Option B — YOLO-based (requires a YOLO iris-detection checkpoint):*
```bash
python utils/yolo_crop.py \
    -i data/iris_contact/ \
    -o data/iris_contact/ \
    --yolo_model path/to/yolo_iris.pt
```

After Step 2 your data directory will have the structure expected by all training scripts:

```
data/iris_contact/
    train/
        images/      full-resolution originals (480×640)
        masks/       3-channel ROI masks (PNG)
        cropped/     256×256 iris crops for Stage-1
    val/
        images/
        masks/
        cropped/
```

---

## Training

### Stage 1 — DenseNet-121 (patterned vs. non-patterned)

```bash
python train_stage1.py \
    --config   configs/stage1.yaml \
    --save_dir experiments/stage1
```

Fine-tune from the D-NetPAD pretrained checkpoint (recommended):
```bash
python train_stage1.py \
    --config     configs/stage1.yaml \
    --save_dir   experiments/stage1 \
    --model_ckpt path/to/D-NetPAD_Model.pth
```

### Stage 2 — ConvNeXt-Base + MGSA (clear vs. normal)

```bash
# ConvNeXt-Base with MGSA (paper configuration)
python train_stage2.py \
    --config   configs/stage2_convnext.yaml \
    --save_dir experiments/stage2_convnext

# ResNet-101 with MGSA
python train_stage2.py \
    --config   configs/stage2_resnet.yaml \
    --save_dir experiments/stage2_resnet

# Disable attention: set  attn_type: null  in the config
```

Training uses **subject-stratified 5-fold cross-validation** (`StratifiedGroupKFold`) to prevent identity leakage. Five checkpoints are saved — one per fold.

#### Attention configuration

| Config key | Options | Effect |
|---|---|---|
| `attn_type` | `spatial` / `multihead` / `null` | Attention module; `null` = plain classifier |
| `attn_stage` | `0`–`3` | Which ConvNeXt stage receives the mask |
| `num_heads` | integer | Heads for `multihead` only |
| `se_reduction` | integer | SE bottleneck reduction ratio |

---

## Evaluation

### Evaluate a single Stage-2 fold

```bash
python eval_stage2.py \
    --config     configs/stage2_convnext.yaml \
    --model_ckpt experiments/stage2_convnext/best_model_fold_1.pth \
    --test_data  data/iris_contact/val \
    --save_dir   experiments/eval_stage2
```

### End-to-end two-stage pipeline (5-fold ensemble)

```bash
python eval_two_stage.py \
    --config   configs/eval_two_stage.yaml \
    --ckpt1    weights/IITDCogent/stage1/DesNet121_best.pth \
    --ckpt2    weights/IITDCogent/stage2/best_model_fold_1.pth \
              weights/IITDCogent/stage2/best_model_fold_2.pth \
              weights/IITDCogent/stage2/best_model_fold_3.pth \
              weights/IITDCogent/stage2/best_model_fold_4.pth \
              weights/IITDCogent/stage2/best_model_fold_5.pth \
    --save_dir experiments/eval_two_stage
```

### With automatic data preparation (raw images → results in one step)

```bash
python eval_two_stage_with_data_prep.py \
    --config     configs/eval_two_stage.yaml \
    --ckpt1      weights/IITDCogent/stage1/DesNet121_best.pth \
    --ckpt2      weights/IITDCogent/stage2/best_model_fold_1.pth \
    --images_dir raw_images/ \
    --save_dir   experiments/eval_two_stage
```

---

## Inference on New Images

```bash
# Stage-1 only (patterned vs. non-patterned)
python infer_stage1.py \
    --config     configs/stage1.yaml \
    --model_ckpt weights/IITDCogent/stage1/DesNet121_best.pth \
    --image_path path/to/images/ \
    --save_dir   outputs/

# Stage-2 only (clear vs. normal, on pre-screened non-patterned images)
python infer_stage2.py \
    --config      configs/stage2_convnext.yaml \
    --model_ckpt  weights/IITDCogent/stage2/best_model_fold_1.pth \
    --image_path  path/to/images/ \
    --data-config configs/data_prep.yaml \
    --save_dir    outputs/
```

Both scripts output `results.csv` with columns: `Image Name`, `Prediction`, `Mask Success`.

---

## Citation

If you use this code or weights in your research, please cite:
Accepted: 2026 IEEE International Joint Conference on Biometrics (IJCB)
```bibtex
@article{farmanifard2026detecting,
  title={Detecting Clear Contact Lenses for Iris Recognition: A Two-Stage Mask-Guided Attention Approach},
  author={Farmanifard, Parisa and Ross, Arun},
  journal={arXiv preprint arXiv:2608.08977},
  year={2026}
}
```

---

## License

This project is released for research and non-commercial use only.
The D-NetPAD pre-trained weights are subject to their original licence terms.
