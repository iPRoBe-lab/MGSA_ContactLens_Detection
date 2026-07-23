"""
efficientnet_net.py - EfficientNetV2-S wrapper for Stage-2 lens classification.

Uses the pretrained EfficientNetV2-S backbone (21M params) which is much better
suited for fine-grained classification on small datasets than ConvNeXt-Base (89M).

The model accepts (x, roi_mask=None) to stay compatible with the existing training
loop; roi_mask is ignored because the cropped iris images already focus on the
discriminative region.

Two-stage fine-tuning is supported:
  model.freeze_backbone()   — freeze all layers except classifier
  model.unfreeze_backbone() — unfreeze all layers for full fine-tuning
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


class EfficientNetStage2(nn.Module):
    """
    EfficientNetV2-S for Stage-2 (Normal vs Clear) classification.

    Args:
        num_classes:  number of output classes (default 2).
        dropout_prob: dropout probability in the classifier head (default 0.3).
    """

    def __init__(self, num_classes: int = 2, dropout_prob: float = 0.3):
        super().__init__()

        backbone = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Keep the full feature extractor
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Replace classifier head
        in_features = backbone.classifier[-1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward  (roi_mask accepted but ignored — cropped images already
    #           focus on the iris region)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        roi_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Two-stage fine-tuning helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers; only the classifier head trains."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """
        Return two parameter groups for differential learning rates.

        Args:
            backbone_lr: LR for pretrained backbone layers.
            head_lr:     LR for the randomly-initialized classifier head.
            weight_decay: applied to both groups.

        Returns:
            List of parameter-group dicts suitable for torch.optim.
        """
        backbone_params = list(self.features.parameters())
        head_params     = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr,
             "weight_decay": weight_decay},
            {"params": head_params,     "lr": head_lr,
             "weight_decay": weight_decay},
        ]


if __name__ == "__main__":
    # Smoke test
    img = torch.randn(4, 3, 224, 224)
    m = EfficientNetStage2(num_classes=2)
    print("Params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    out = m(img)
    print("Output shape:", out.shape)

    m.freeze_backbone()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {trainable / 1e6:.2f}M")

    m.unfreeze_backbone()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable / 1e6:.2f}M")
