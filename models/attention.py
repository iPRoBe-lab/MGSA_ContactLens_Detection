"""
attention.py -  attention modules for Stage-2 classifiers.

Both AttentionResNet and AttentionConvNeXt import from here so the
implementations stay in one place.

Modules
-------
SEBlock
    Squeeze-and-Excitation channel recalibration.
    Always applied after an attention block.

MaskGuidedSpatialAttention  (attn_type='spatial')
    A single conv1x1 gate multiplied by the upsampled ROI mask.
    Lightweight; good default when mask quality is high.

MultiHeadROIAttention  (attn_type='multihead')
    Each of N heads independently gates the features with the ROI mask,
    then all heads are concatenated and projected back.
    More expressive; matches the original AttentionResNet design.

Usage
-----
    from models.attention import build_attention, SEBlock

    attn = build_attention('spatial',   channels=512)
    attn = build_attention('multihead', channels=512, num_heads=4)
    attn = build_attention(None,        channels=512)   # -> None
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SEBlock
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration.

    Args:
        channels:  number of input (and output) channels.
        reduction: bottleneck reduction ratio (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(B, C)
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y)))).view(B, C, 1, 1)
        return x * y


# ---------------------------------------------------------------------------
# Spatial attention
# ---------------------------------------------------------------------------

class MaskGuidedSpatialAttention(nn.Module):
    """Spatial attention gated by an upsampled ROI mask.

    Given feature map ``x`` (B, C, H, W) and ``roi_mask`` (B, 1, H0, W0):

    1. Compute a spatial gate: ``G = sigmoid(conv1x1(x))``  -> (B, 1, H, W)
    2. Upsample mask to feature-map resolution.
    3. Return ``x * G * mask``.

    Args:
        channels: number of input channels (used by the 1×1 gate conv).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(x))                           # (B,1,H,W)
        mask = F.interpolate(roi_mask, size=x.shape[-2:], mode="nearest")
        return x * gate * mask


# ---------------------------------------------------------------------------
# Multi-head ROI attention
# ---------------------------------------------------------------------------

class MultiHeadROIAttention(nn.Module):
    """Multi-head ROI attention.

    Each head projects the feature map to ``channels // num_heads`` dimensions,
    gates with the upsampled ROI mask, then all head outputs are concatenated
    and projected back to ``channels``.

    Args:
        channels:  number of input (and output) channels.
        num_heads: number of parallel attention heads (default 4).
                   ``channels`` must be divisible by ``num_heads``.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        self.heads = nn.ModuleList([
            nn.Conv2d(channels, channels // num_heads, kernel_size=1)
            for _ in range(num_heads)
        ])
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(roi_mask, size=x.shape[-2:], mode="nearest")
        head_outputs = [torch.sigmoid(h(x)) * mask for h in self.heads]
        attention = torch.cat(head_outputs, dim=1)   # (B, C, H, W)
        return self.output_conv(attention * x)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_ATTN_TYPES = ("spatial", "multihead")


def build_attention(
    attn_type: str | None,
    channels: int,
    num_heads: int = 4,
) -> nn.Module | None:
    """Instantiate an attention module by name, or return None.

    Args:
        attn_type: ``'spatial'``, ``'multihead'``, or ``None`` / ``''``.
        channels:  feature-map channel count.
        num_heads: number of heads (only used for ``'multihead'``).

    Returns:
        An ``nn.Module`` or ``None`` if attention is disabled.

    Raises:
        ValueError: if ``attn_type`` is not one of the supported values.
    """
    if not attn_type:
        return None
    attn_type = attn_type.lower().strip()
    if attn_type == "spatial":
        return MaskGuidedSpatialAttention(channels)
    if attn_type == "multihead":
        return MultiHeadROIAttention(channels, num_heads=num_heads)
    raise ValueError(
        f"Unknown attn_type '{attn_type}'. Choose from: {_ATTN_TYPES} or None."
    )
