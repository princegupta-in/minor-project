"""Loss functions for class-imbalanced forest cover segmentation.

Forest classes (dense_forest, open_forest) are typically under-represented
compared to agriculture and built-up in Deoghar — Focal Loss addresses this.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017) for per-pixel segmentation.

    Downweights easy examples, focusing training on hard/minority class pixels.
    Particularly effective for forest cover where class imbalance is high.

    Args:
        gamma: focusing parameter (2.0 recommended)
        weight: per-class weights tensor (optional)
        ignore_index: label value to ignore in loss computation
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), target: (B, H, W)
        ce   = F.cross_entropy(logits, target, weight=self.weight,
                                ignore_index=self.ignore_index, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def class_weights_from_counts(counts: torch.Tensor, smoothing: float = 1e-3) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to n_classes.

    Usage:
        counts = torch.tensor([water_px, dense_forest_px, ..., barren_px], dtype=torch.float)
        weights = class_weights_from_counts(counts)
        loss_fn = FocalLoss(weight=weights.to(device))
    """
    freq = counts.float() / counts.sum().clamp(min=1)
    w    = 1.0 / (freq + smoothing)
    w    = w * (len(w) / w.sum())
    return w
