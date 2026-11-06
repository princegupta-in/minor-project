"""Recursively forecast forest cover for Deoghar, 2026–2035.

Uses a trained model to autoregressively predict future LULC maps.
Supports MC Dropout uncertainty quantification.

Usage:
    python scripts/predict_future.py \\
        --checkpoint outputs/checkpoints/unet_convlstm_v1_best.pt \\
        --config configs/model_unet_convlstm.yaml \\
        --years 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \\
        --mc-dropout-passes 20 \\
        --out outputs/predictions/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LULCSequenceDataset
from src.models import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config",     type=Path, required=True)
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--years", nargs="+", type=int,
                   default=list(range(2026, 2036)))
    p.add_argument("--out",    type=Path, default=Path("outputs/predictions"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--mc-dropout-passes", type=int, default=20,
                   help="MC Dropout forward passes for uncertainty (0 = disabled)")
    return p.parse_args()


def enable_dropout(m: torch.nn.Module) -> None:
    """Re-enable Dropout layers in eval mode (for MC Dropout)."""
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d)):
            mod.train()


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    x: torch.Tensor,
    mc_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass(es) and return (argmax label map, entropy map)."""
    if mc_passes <= 0:
        logits = model(x)
        prob   = torch.softmax(logits, dim=1)
        ent    = -(prob * torch.log(prob.clamp(min=1e-9))).sum(dim=1)
        return prob.argmax(dim=1).cpu().numpy(), ent.cpu().numpy()

    # MC Dropout
    model.eval()
    enable_dropout(model)
    probs = []
    for _ in range(mc_passes):
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1))
    p   = torch.stack(probs).mean(dim=0)
    ent = -(p * torch.log(p.clamp(min=1e-9))).sum(dim=1)
    return p.argmax(dim=1).cpu().numpy(), ent.cpu().numpy()


def main() -> None:
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
    print(f"[predict] forecasting {args.years} | {len(ds.files)} tiles | "
          f"mc_passes={args.mc_dropout_passes}")

    for tile_idx, fp in enumerate(ds.files):
        with np.load(fp) as f:
            frames = f["frames"]   # (T_total, C, H, W)

        recent = frames[-cfg["seq_length"]:].copy()
        x      = torch.from_numpy(recent).float().unsqueeze(0).to(dev)   # (1,T,C,H,W)

        year_labels, year_ents = [], []
        for year in args.years:
            label, ent = predict_one(model, x, args.mc_dropout_passes)
            year_labels.append(label[0])
            year_ents.append(ent[0])
            # Autoregressive roll: drop oldest frame, append last frame copy as proxy
            new_frame = recent[-1:].copy()
            recent    = np.concatenate([recent[1:], new_frame], axis=0)
            x         = torch.from_numpy(recent).float().unsqueeze(0).to(dev)

        np.savez_compressed(
            args.out / f"{fp.stem}_forecast.npz",
            years=np.array(args.years),
            labels=np.stack(year_labels).astype(np.int8),
            entropies=np.stack(year_ents).astype(np.float32),
        )

        if tile_idx % 50 == 0:
            print(f"  tile {tile_idx + 1}/{len(ds.files)}")

    print(f"[predict] forecasts written to {args.out}")


if __name__ == "__main__":
    main()
