"""
attention_resnet.py - Attention-enhanced ResNet for Stage-2 classification.
"""

from __future__ import annotations
import sys
import os
import torch
import torch.nn as nn
from torchvision import models


# add parent directory to sys.path so we can import from models.attention
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import SEBlock, build_attention


class AttentionResNet(nn.Module):
    """
    Attention-enhanced ResNet for Stage-2 (clear vs. normal) classification.

    Args:
        resnet_type:   backbone variant.
                       Options: 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101'
        num_classes:   number of output classes (default 2).
        attn_layer:    which ResNet layer receives ROI attention.
                       Options: 'layer1' | 'layer2' | 'layer3' | 'layer4' | None
                       Pass None or '' to disable attention entirely.
        attn_type:     attention module to use at ``attn_layer``.
                       Options: 'spatial' | 'multihead' | None
                       Ignored when ``attn_layer`` is None.
        dropout_prob:  dropout probability before the final FC (default 0.5).
        num_heads:     number of heads for 'multihead' attention (default 4).
        se_reduction:  SE channel-reduction ratio (default 16).
    """

    _CHANNELS = {
        "resnet18":  {"layer1": 64,  "layer2": 128, "layer3": 256,  "layer4": 512},
        "resnet34":  {"layer1": 64,  "layer2": 128, "layer3": 256,  "layer4": 512},
        "resnet50":  {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        "resnet101": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    }

    def __init__(
        self,
        resnet_type: str = "resnet101",
        num_classes: int = 2,
        attn_layer: str | None = "layer2",
        attn_type: str | None = "multihead",
        dropout_prob: float = 0.5,
        num_heads: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.resnet_type  = resnet_type
        self.num_classes  = num_classes
        self.dropout_prob = dropout_prob
        self.attn_layer   = attn_layer or None   # '' -> None

        _weight_map = {
            "resnet18":  (models.resnet18,  models.ResNet18_Weights.DEFAULT),
            "resnet34":  (models.resnet34,  models.ResNet34_Weights.DEFAULT),
            "resnet50":  (models.resnet50,  models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        }
        if resnet_type not in _weight_map:
            raise ValueError(
                f"Invalid resnet_type '{resnet_type}'. "
                f"Choose from: {list(_weight_map.keys())}"
            )
        builder, weights = _weight_map[resnet_type]
        self.resnet = builder(weights=weights)

        # Build attention modules only when a layer is specified
        self.roi_attention: nn.Module | None = None
        self.se_block: nn.Module | None = None
        if self.attn_layer:
            ch = self._CHANNELS[resnet_type][self.attn_layer]
            self.roi_attention = build_attention(
                attn_type, channels=ch, num_heads=num_heads
            )
            if self.roi_attention is not None:
                self.se_block = SEBlock(ch, reduction=se_reduction)

        in_features = self.resnet.fc.in_features
        self.dropout = nn.Dropout(p=dropout_prob)
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(in_features, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.roi_attention is None:
            return self.resnet(x)
        assert roi_mask is not None, (
            "roi_mask must be provided when attn_layer / attn_type are set."
        )
        return self._forward_with_attn(x, roi_mask)

    def _forward_with_attn(
        self, x: torch.Tensor, roi_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self.resnet, name)(x)
            if name == self.attn_layer:
                x = self.roi_attention(x, roi_mask)
                x = self.se_block(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        return x


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img  = torch.randn(1, 3, 224, 224)
    mask = (torch.rand(1, 1, 224, 224) > 0.5).float()

    for attn_type in ("spatial", "multihead", None):
        model = AttentionResNet(
            resnet_type="resnet101",
            num_classes=2,
            attn_layer="layer2" if attn_type else None,
            attn_type=attn_type,
        )
        out = model(img, mask) if attn_type else model(img)
        print(f"resnet101  attn_type={attn_type}  -> {out.shape}")
