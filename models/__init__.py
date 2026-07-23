"""
Models package for deep-lens-classification.

Available models:
  - AttentionResNet  (ResNet-18/34/50/101 + multi-head ROI attention + SE blocks)
  - AttentionConvNeXt (ConvNeXt-tiny/small/base/large + optional mask-guided attention)

Use `build_stage2_model(cfg)` to instantiate either model from a config dict/DotMap.
"""

from models.attention_resnet import AttentionResNet, SEBlock
from models.convnext_net import AttentionConvNeXt
from models.efficientnet_net import EfficientNetStage2
from models.efficientnet_freq import EfficientNetFreq
from models.model_factory import build_stage2_model

__all__ = [
    "AttentionResNet",
    "SEBlock",
    "AttentionConvNeXt",
    "EfficientNetStage2",
    "EfficientNetFreq",
    "build_stage2_model",
]
