"""
convnext_net.py - ConvNeXt backbone for Stage-2 classification.

Builds on torchvision's ConvNeXt implementation.
Optionally extends it with mask-guided attention so the model can focus on the
iris / pupil region of interest.


Architecture summary
  ConvNeXt stem + 4 stages

  (Optional) after a chosen stage, apply mask-guided attention:
       - 'spatial'   : sigmoid(conv1x1(features)) * upsampled_roi_mask
       - 'multihead' : multi-head ROI gates concatenated + projected back
       - channel gate (SE): always added after spatial / multihead

  Global average pool

  Dropout -> Linear head

Usage
  # without attention (plain ConvNeXt fine-tuning)
  model = AttentionConvNeXt(convnext_type="convnext_base", num_classes=2)
  out   = model(img)

  # with spatial attention at stage 2
  model = AttentionConvNeXt(convnext_type="convnext_base", num_classes=2,
                             attn_stage=2, attn_type="spatial")
  out   = model(img, roi_mask)

  # with multi-head attention at stage 2
  model = AttentionConvNeXt(convnext_type="convnext_base", num_classes=2,
                             attn_stage=2, attn_type="multihead")
  out   = model(img, roi_mask)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.attention import SEBlock, build_attention


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AttentionConvNeXt(nn.Module):
    """
    ConvNeXt with optional mask-guided attention.

    Args:
        convnext_type: backbone variant.
                       Options: 'convnext_tiny' | 'convnext_small' |
                                'convnext_base' | 'convnext_large'
        num_classes:   number of output classes (default 2).
        attn_stage:    ConvNeXt stage (0-indexed, 0-3) after which to inject
                       mask-guided attention.  Pass None to disable.
        attn_type:     attention module to use at ``attn_stage``.
                       Options: 'spatial' | 'multihead' | None
                       Ignored when ``attn_stage`` is None.
        dropout_prob:  dropout probability before the classification head.
        num_heads:     number of heads for 'multihead' attention (default 4).
        se_reduction:  reduction ratio for the SE channel-attention block
                       (only used when attn_stage is not None).
    """

    # Output channels per stage for each ConvNeXt variant
    # ConvNeXt uses 4 stages; channel counts follow the paper.
    _STAGE_CHANNELS: dict[str, list[int]] = {
        "convnext_tiny":  [96,  192,  384,  768],
        "convnext_small": [96,  192,  384,  768],
        "convnext_base":  [128, 256,  512,  1024],
        "convnext_large": [192, 384,  768,  1536],
    }

    def __init__(
        self,
        convnext_type: str = "convnext_base",
        num_classes: int = 2,
        attn_stage: int | None = None,
        attn_type: str | None = "spatial",
        dropout_prob: float = 0.5,
        num_heads: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()

        _builders: dict[str, tuple] = {
            "convnext_tiny":  (models.convnext_tiny,  ConvNeXt_Tiny_Weights.DEFAULT),
            "convnext_small": (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT),
            "convnext_base":  (models.convnext_base,  ConvNeXt_Base_Weights.DEFAULT),
            "convnext_large": (models.convnext_large, ConvNeXt_Large_Weights.DEFAULT),
        }
        if convnext_type not in _builders:
            raise ValueError(
                f"Invalid convnext_type '{convnext_type}'. "
                f"Choose from: {list(_builders.keys())}"
            )

        self.convnext_type = convnext_type
        self.num_classes   = num_classes
        self.dropout_prob  = dropout_prob
        self.attn_stage: int | None = attn_stage

        builder, weights = _builders[convnext_type]
        backbone = builder(weights=weights)

        # torchvision's ConvNeXt is structured as:
        #   backbone.features  - Sequential of 8 blocks (stem + 4 stages × 2)
        #   backbone.avgpool
        #   backbone.classifier  - Sequential(LayerNorm, Flatten, Linear)
        #
        # Stages are at indices 1, 3, 5, 7; downsampling layers at 0, 2, 4, 6.
        self.stem   = backbone.features[0]   # stem (downsampling)
        self.stage0 = backbone.features[1]   # stage 0
        self.down1  = backbone.features[2]   # downsampling
        self.stage1 = backbone.features[3]   # stage 1
        self.down2  = backbone.features[4]   # downsampling
        self.stage2 = backbone.features[5]   # stage 2
        self.down3  = backbone.features[6]   # downsampling
        self.stage3 = backbone.features[7]   # stage 3

        self.avgpool = backbone.avgpool

        # Replace the original classifier
        in_features = backbone.classifier[-1].in_features
        self.dropout = nn.Dropout(p=dropout_prob)
        self.head = nn.Linear(in_features, num_classes)

        # Attention modules (built only if both attn_stage and attn_type are set)
        self.roi_attention: nn.Module | None = None
        self.se_block: nn.Module | None = None
        if self.attn_stage is not None:
            if not (0 <= self.attn_stage <= 3):
                raise ValueError(
                    f"attn_stage must be 0, 1, 2, or 3 (got {self.attn_stage})."
                )
            ch = self._STAGE_CHANNELS[convnext_type][self.attn_stage]
            self.roi_attention = build_attention(
                attn_type, channels=ch, num_heads=num_heads
            )
            if self.roi_attention is not None:
                self.se_block = SEBlock(ch, reduction=se_reduction)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.roi_attention is None:
            return self._forward_plain(x)
        assert roi_mask is not None, (
            "roi_mask must be provided when attn_stage / attn_type are set."
        )
        return self._forward_with_attn(x, roi_mask)

    def _forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage0(x)
        x = self.down1(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)

    def _forward_with_attn(
        self, x: torch.Tensor, roi_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.stem(x)

        for idx, (down, stage) in enumerate(
            [(None,       self.stage0),
             (self.down1, self.stage1),
             (self.down2, self.stage2),
             (self.down3, self.stage3)]
        ):
            if down is not None:
                x = down(x)
            x = stage(x)

            if idx == self.attn_stage:
                x = self.roi_attention(x, roi_mask)
                x = self.se_block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # smoke test
    img  = torch.randn(2, 3, 224, 224)
    mask = (torch.rand(2, 1, 224, 224) > 0.5).float()

    for variant in ["convnext_tiny", "convnext_base"]:
        for attn_stage, attn_type in [(None, None), (2, "spatial"), (2, "multihead")]:
            m = AttentionConvNeXt(
                convnext_type=variant, num_classes=2,
                attn_stage=attn_stage, attn_type=attn_type,
            )
            out = m(img, mask) if attn_stage is not None else m(img)
            print(f"{variant}  attn_stage={attn_stage}  attn_type={attn_type}  -> {out.shape}")
