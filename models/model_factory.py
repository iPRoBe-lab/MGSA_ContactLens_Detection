"""
model_factory.py - Single entry-point to build any Stage-2 model from a config.

Config requirements
Both AttentionResNet and AttentionConvNeXt share the same config keys where
possible. The `backbone` key selects which family to instantiate.

  backbone: resnet   # or: convnext
  num_classes: 2

  # ── ResNet-specific ──────────────────────────────────────────────────
  resnet_type: resnet101          # resnet18 | resnet34 | resnet50 | resnet101
  attention_layer: layer2         # layer1|layer2|layer3|layer4 | '' (disabled)

  # ── ConvNeXt-specific ────────────────────────────────────────────────
  convnext_type: convnext_base    # convnext_tiny|small|base|large
  attn_stage: 2                   # 0-3 | null (disabled)

  # ── Shared ───────────────────────────────────────────────────────────
  attn_type:    multihead         # spatial | multihead | null (disables attention)
  dropout_prob: 0.5
  num_heads:    4                 # only used when attn_type: multihead
  se_reduction: 16
"""

from __future__ import annotations

from typing import Any

from models.attention_resnet import AttentionResNet
from models.convnext_net import AttentionConvNeXt
from models.efficientnet_net import EfficientNetStage2
from models.efficientnet_freq import EfficientNetFreq


def build_stage2_model(cfg: Any) -> AttentionResNet | AttentionConvNeXt | EfficientNetStage2 | EfficientNetFreq:
    """
    Build a Stage-2 classifier from a config object or dict.

    Accepts both dict and DotMap (any object whose attributes are also
    accessible via ``cfg.key`` or ``cfg['key']``).

    Args:
        cfg: Config object. Must contain at minimum:
               backbone      - 'resnet' or 'convnext'
               num_classes   - int
             Plus backbone-specific keys (see module docstring).

    Returns:
        An nn.Module ready for training/evaluation.
    """
    # Support both dict and attribute-access objects (DotMap, Namespace, …)
    def _get(key: str, default=None):
        try:
            v = cfg[key]
        except (KeyError, TypeError):
            v = getattr(cfg, key, default)
        # DotMap returns an empty DotMap for missing keys - treat as default
        if hasattr(v, "__class__") and v.__class__.__name__ == "DotMap" and not dict(v):
            return default
        return v if v is not None else default

    backbone     = str(_get("backbone", "resnet")).lower().strip()
    num_classes  = int(_get("num_classes", 2))
    dropout_prob = float(_get("dropout_prob", 0.5))
    num_heads    = int(_get("num_heads", 4))
    se_reduction = int(_get("se_reduction", 16))

    # attn_type is shared; 'null' / '' / None all disable attention
    raw_attn_type = _get("attn_type", None)
    attn_type = (
        None if raw_attn_type in (None, "", "null")
        else str(raw_attn_type).lower().strip()
    )

    if backbone == "resnet":
        resnet_type = _get("resnet_type", "resnet101")
        attn_layer  = _get("attention_layer", "layer2") or None
        return AttentionResNet(
            resnet_type=resnet_type,
            num_classes=num_classes,
            attn_layer=attn_layer,
            attn_type=attn_type,
            dropout_prob=dropout_prob,
            num_heads=num_heads,
            se_reduction=se_reduction,
        )

    elif backbone == "convnext":
        convnext_type = _get("convnext_type", "convnext_base")
        raw_stage     = _get("attn_stage", None)
        attn_stage    = int(raw_stage) if raw_stage not in (None, "", "null") else None
        return AttentionConvNeXt(
            convnext_type=convnext_type,
            num_classes=num_classes,
            attn_stage=attn_stage,
            attn_type=attn_type,
            dropout_prob=dropout_prob,
            num_heads=num_heads,
            se_reduction=se_reduction,
        )

    elif backbone == "efficientnet":
        return EfficientNetStage2(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )

    elif backbone == "efficientnet_freq":
        inp_size = _get("inp_dim", 224)
        if isinstance(inp_size, (list, tuple)):
            inp_size = inp_size[0]
        freq_dim = int(_get("freq_dim", 128))
        variant  = str(_get("efficientnet_variant", "s")).lower()
        return EfficientNetFreq(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            freq_dim=freq_dim,
            inp_size=int(inp_size),
            variant=variant,
        )

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose 'resnet', 'convnext', or 'efficientnet'."
        )
