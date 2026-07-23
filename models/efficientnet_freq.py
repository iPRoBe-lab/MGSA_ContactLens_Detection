"""
efficientnet_freq.py - EfficientNetV2-S with Frequency Domain Branch for Stage-2.

Key insight: Clear contact lenses alter the high-frequency texture of the iris
(subtle refractive changes, surface reflections). A standard CNN trained on spatial
features struggles to detect these. Adding an explicit frequency-domain branch
(FFT magnitude spectrum) lets the model directly reason about these artifacts.

Architecture
------------
  Input: cropped iris (3×224×224, grayscale replicated to 3 channels)

  Spatial branch: EfficientNetV2-S backbone → 1280-d features
  Freq   branch: log(1 + |FFT(mean_gray)|) → small CNN → 128-d features

  Concat → LayerNorm → Dropout → Linear(1408 → 2)

The frequency branch uses only the mean of the 3 channels (equivalent to the
original grayscale iris) so it works regardless of how the RGB was produced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (EfficientNet_V2_S_Weights,
                                 EfficientNet_V2_M_Weights)


class FreqBranch(nn.Module):
    """
    Multi-scale FFT branch: processes log-magnitude spectra at 3 resolutions.

    Clear lenses create frequency artifacts at different spatial scales:
    - Full resolution (224): fine iris texture changes
    - Half resolution (112): mid-scale surface reflections
    - Quarter resolution (56): coarse lens boundary effects

    Each scale is processed by a shared lightweight CNN, then concatenated.

    Input : (B, 3, H, W)
    Output: (B, freq_dim)
    """

    def __init__(self, inp_size: int = 224, freq_dim: int = 128):
        super().__init__()
        self.freq_dim = freq_dim
        per_scale = freq_dim // 3  # ~42 per scale
        extra = freq_dim - per_scale * 3

        def _scale_net(out_dim):
            return nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(3),
                nn.Flatten(),
                nn.Linear(64 * 9, out_dim),
                nn.GELU(),
            )

        self.net_full    = _scale_net(per_scale + extra)
        self.net_half    = _scale_net(per_scale)
        self.net_quarter = _scale_net(per_scale)

    def _fft_mag(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute centred log-magnitude FFT, normalised to [0,1]."""
        f   = torch.fft.fftshift(torch.fft.fft2(gray))
        mag = torch.log1p(f.abs())
        lo  = mag.flatten(2).min(dim=2)[0].view(-1, 1, 1, 1)
        hi  = mag.flatten(2).max(dim=2)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
        return (mag - lo) / hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)

        # Full resolution
        f1 = self.net_full(self._fft_mag(gray))

        # Half resolution
        g2 = F.avg_pool2d(gray, 2)
        f2 = self.net_half(self._fft_mag(g2))

        # Quarter resolution
        g3 = F.avg_pool2d(gray, 4)
        f3 = self.net_quarter(self._fft_mag(g3))

        return torch.cat([f1, f2, f3], dim=1)       # (B, freq_dim)


class EfficientNetFreq(nn.Module):
    """
    EfficientNetV2-S + Frequency Domain Branch for Stage-2 lens classification.

    Combines:
      - Spatial features from pretrained EfficientNetV2-S
      - Frequency features from a lightweight FFT branch (trained from scratch)

    The frequency branch is very small (< 0.5M params) so it does not
    dominate training — the spatial branch learns the coarse features and the
    frequency branch learns the subtle artifact signatures.

    Supports the same freeze_backbone / unfreeze_backbone / get_parameter_groups
    interface as EfficientNetStage2 so the training loop is identical.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_prob: float = 0.3,
        freq_dim: int = 128,
        inp_size: int = 224,
        variant: str = "s",  # "s" = V2-S (21M), "m" = V2-M (54M)
    ):
        super().__init__()

        if variant == "m":
            backbone = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        else:
            backbone = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        spatial_dim   = backbone.classifier[-1].in_features   # 1280

        self.freq_branch = FreqBranch(inp_size=inp_size, freq_dim=freq_dim)

        fused_dim = spatial_dim + freq_dim  # 1280 + 128 = 1408
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(p=dropout_prob),
            nn.Linear(fused_dim, num_classes),
        )

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised fused embedding (for SupCon loss)."""
        sp = self.features(x)
        sp = self.avgpool(sp)
        sp = torch.flatten(sp, 1)
        fr = self.freq_branch(x)
        emb = torch.cat([sp, fr], dim=1)
        return nn.functional.normalize(emb, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        roi_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        # Spatial branch
        sp = self.features(x)
        sp = self.avgpool(sp)
        sp = torch.flatten(sp, 1)                # (B, 1280)

        # Frequency branch
        fr = self.freq_branch(x)                 # (B, 128)

        return self.head(torch.cat([sp, fr], dim=1))

    # ------------------------------------------------------------------
    # Two-stage fine-tuning (same API as EfficientNetStage2)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Backbone: backbone_lr.  Freq branch + head: head_lr."""
        return [
            {"params": list(self.features.parameters()),
             "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": list(self.freq_branch.parameters()) +
                       list(self.head.parameters()),
             "lr": head_lr, "weight_decay": weight_decay},
        ]


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    m = EfficientNetFreq()
    print("Params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    out = m(x)
    print("Output:", out.shape)
