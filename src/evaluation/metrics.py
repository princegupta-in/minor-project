"""Segmentation metrics for Deoghar Forest Cover Analysis.

Includes:
  - Per-class IoU, F1, Precision, Recall
  - Mean IoU (mIoU), Overall Accuracy (OA)
  - Cohen's Kappa
  - Pontius & Millones (2011) Quantity / Allocation disagreement
  - Fuzzy Kappa (proximity-weighted)
  - Forest-specific: combined dense+open forest IoU
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# Deoghar project class IDs for forest cover
FOREST_CLASSES = [1, 2]   # dense_forest=1, open_forest=2


class StreamingConfusion:
    """Accumulates a confusion matrix over batches without storing predictions."""

    def __init__(self, n_classes: int, ignore_index: Optional[int] = None) -> None:
        self.n_classes    = n_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self) -> None:
        self.cm[:] = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred   = pred.ravel()
        target = target.ravel()
        valid  = (target >= 0) & (target < self.n_classes)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx  = target * self.n_classes + pred
        binc = np.bincount(idx, minlength=self.n_classes ** 2)
        self.cm += binc.reshape(self.n_classes, self.n_classes)

    def compute(self) -> Dict:
        cm    = self.cm.astype(np.float64)
        diag  = np.diag(cm)
        rows  = cm.sum(axis=1)   # GT totals per class
        cols  = cm.sum(axis=0)   # Pred totals per class
        total = cm.sum()

        union = rows + cols - diag
        iou   = np.where(union > 0, diag / union, 0.0)
        miou  = float(np.nanmean(iou))

        precision = np.where(cols > 0, diag / cols, 0.0)
        recall    = np.where(rows > 0, diag / rows, 0.0)
        f1        = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

        oa    = float(diag.sum() / max(total, 1))
        pe    = float((rows * cols).sum() / max(total ** 2, 1))
        kappa = float((oa - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0

        # Pontius disagreements
        p          = cm / max(total, 1)
        p_row      = p.sum(axis=1)
        p_col      = p.sum(axis=0)
        quantity   = 0.5 * float(np.abs(p_row - p_col).sum())
        allocation = max((1 - oa) - quantity, 0.0)

        # Forest-specific IoU (dense + open forest combined)
        forest_px_gt   = rows[FOREST_CLASSES].sum()
        forest_px_pred = cols[FOREST_CLASSES].sum()
        forest_correct = diag[FOREST_CLASSES].sum()
        forest_union   = forest_px_gt + forest_px_pred - forest_correct
        forest_iou     = float(forest_correct / forest_union) if forest_union > 0 else 0.0

        return {
            "miou":                   miou,
            "forest_iou":             forest_iou,
            "oa":                     oa,
            "kappa":                  kappa,
            "quantity_disagreement":  quantity,
            "allocation_disagreement": allocation,
            "iou_per_class":          iou.tolist(),
            "f1_per_class":           f1.tolist(),
            "precision_per_class":    precision.tolist(),
            "recall_per_class":       recall.tolist(),
            "confusion":              cm.astype(int).tolist(),
        }


def fuzzy_kappa(pred: np.ndarray, target: np.ndarray, n_classes: int, radius: int = 1) -> float:
    """Proximity-weighted kappa: a prediction within `radius` pixels of the
    correct class in GT is partially credited. Approx. van Vliet et al. formula.
    """
    from scipy.ndimage import maximum_filter
    correct = np.zeros_like(target, dtype=np.float64)
    for c in range(n_classes):
        gt_mask = (target == c).astype(np.uint8)
        dilated = maximum_filter(gt_mask, size=2 * radius + 1)
        correct[(pred == c) & (dilated > 0)] = 1.0
    oa = correct.mean()
    # Standard kappa adjustment
    cm    = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid = (target >= 0) & (target < n_classes)
    idx   = target[valid] * n_classes + pred[valid]
    cm   += np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    rows  = cm.sum(axis=1)
    cols  = cm.sum(axis=0)
    total = cm.sum()
    pe    = (rows * cols).sum() / max(total ** 2, 1)
    return float((oa - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0
