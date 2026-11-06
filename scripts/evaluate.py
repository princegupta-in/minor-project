"""Evaluate a trained model via back-testing against held-out label years.

Reports per-class IoU, mIoU, OA, Kappa, forest-specific IoU, and
writes a confusion matrix figure.

Usage:
    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/unet_convlstm_v1_best.pt \\
        --config configs/model_unet_convlstm.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LULCSequenceDataset, split_files_block
from src.models import build_model
from src.evaluation.metrics import StreamingConfusion, fuzzy_kappa
from src.visualization.maps import LULC_NAMES

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",  type=Path, required=True)
    p.add_argument("--config",      type=Path, required=True)
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--out",         type=Path, default=Path("outputs/figures"))
    return p.parse_args()


@torch.no_grad()
def main():
    args     = parse_args()
    cfg      = yaml.safe_load(open(args.config))
    data_cfg = yaml.safe_load(open(args.data_config))
    args.out.mkdir(parents=True, exist_ok=True)

    ds = LULCSequenceDataset(
        data_dir=data_cfg["processed_dir"],
        seq_length=cfg["seq_length"],
        target_offset=cfg.get("target_offset", 1),
    )
    with np.load(ds.files[0]) as first_f:
        actual_in_c = first_f["frames"].shape[1]
    cfg["model_kwargs"]["in_channels"] = actual_in_c

    model = build_model(cfg["model"], **cfg["model_kwargs"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    _, _, test_files = split_files_block(ds.files)
    test_set = set(test_files)
    test_idx = [i for i, (fi, _) in enumerate(ds.index) if ds.files[fi] in test_set]

    n_classes = cfg["n_classes"]
    cm        = StreamingConfusion(n_classes)
    fk_scores = []

    print(f"[evaluate] {len(test_idx)} test windows | checkpoint: {args.checkpoint.name}")
    for idx in test_idx:
        x, y = ds[idx]
        x = x.unsqueeze(0).to(dev)   # (1, T, C, H, W)
        logits = model(x)
        pred   = logits.argmax(dim=1).cpu().numpy()[0]   # (H, W)
        gt     = y.numpy()                                # (H, W)
        cm.update(pred, gt)
        fk_scores.append(fuzzy_kappa(pred, gt, n_classes))

    metrics = cm.compute()
    metrics["fuzzy_kappa_mean"] = float(np.mean(fk_scores))

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  mIoU          : {metrics['miou']:.4f}")
    print(f"  Forest IoU    : {metrics['forest_iou']:.4f}")
    print(f"  OA            : {metrics['oa']:.4f}")
    print(f"  Kappa         : {metrics['kappa']:.4f}")
    print(f"  Fuzzy Kappa   : {metrics['fuzzy_kappa_mean']:.4f}")
    print(f"{'='*55}")
    print("\nPer-class IoU:")
    for cls, iou in zip(LULC_NAMES[:n_classes], metrics["iou_per_class"]):
        print(f"  {cls:<16} {iou:.4f}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics_out = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    metrics_out["iou_per_class"] = dict(zip(LULC_NAMES[:n_classes], metrics["iou_per_class"]))
    with open(args.out / "eval_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Confusion matrix figure ────────────────────────────────────────────────
    cm_arr = np.array(metrics["confusion"], dtype=np.float64)
    row_sum = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_arr / row_sum, 0.0)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LULC_NAMES[:n_classes], yticklabels=LULC_NAMES[:n_classes], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Normalised Confusion Matrix — Deoghar Test Set", fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[evaluate] results → {args.out}")


if __name__ == "__main__":
    main()
