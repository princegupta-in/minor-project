"""Train entry point — Deoghar Forest Cover Analysis.

Usage:
    python scripts/train.py --config configs/model_convlstm.yaml
    python scripts/train.py --config configs/model_unet_convlstm.yaml --debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LULCSequenceDataset, default_train_transform, split_files_block
from src.models import build_model
from src.training.losses import FocalLoss
from src.training.trainer import Trainer, TrainConfig


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      type=Path, required=True)
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--debug", action="store_true", help="Use 8 train + 4 val tiles for smoke-test")
    p.add_argument("--run-name",    default=None)
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    cfg      = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    run_name = args.run_name or cfg.get("name", "run")
    print(f"[train] run={run_name}  device={args.device}")
    print(f"[train] model={cfg['model']}  n_classes={cfg['n_classes']}")

    # Dataset
    full_ds = LULCSequenceDataset(
        data_dir=data_cfg["processed_dir"],
        seq_length=cfg["seq_length"],
        target_offset=cfg.get("target_offset", 1),
        transform=default_train_transform(),
    )

    # Spatial block split (avoids spatial leakage)
    train_files, val_files, _ = split_files_block(full_ds.files)
    train_set = set(train_files)
    val_set   = set(val_files)
    train_idx = [i for i, (fi, _) in enumerate(full_ds.index) if full_ds.files[fi] in train_set]
    val_idx   = [i for i, (fi, _) in enumerate(full_ds.index) if full_ds.files[fi] in val_set]

    if args.debug:
        train_idx, val_idx = train_idx[:8], val_idx[:4]

    train_ds     = Subset(full_ds, train_idx)
    val_ds       = Subset(full_ds, val_idx)
    print(f"[train] train={len(train_idx)} windows  val={len(val_idx)} windows")

    num_workers = 0 if sys.platform == "win32" else cfg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Model & input channel validation
    sample_x, _ = full_ds[0]
    actual_in_channels = sample_x.shape[1]
    if cfg["model_kwargs"].get("in_channels") != actual_in_channels:
        print(f"[train] Auto-aligning in_channels: {cfg['model_kwargs'].get('in_channels')} -> {actual_in_channels}")
        cfg["model_kwargs"]["in_channels"] = actual_in_channels

    model    = build_model(cfg["model"], **cfg["model_kwargs"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,}")

    # Loss with NoData (255) ignored
    loss_fn = (
        FocalLoss(gamma=cfg.get("focal_gamma", 2.0), ignore_index=255)
        if cfg.get("loss", "ce") == "focal"
        else torch.nn.CrossEntropyLoss(ignore_index=255)
    )

    train_cfg = TrainConfig(
        epochs=2 if args.debug else cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
        batch_size=cfg["batch_size"],
        num_workers=0 if sys.platform == "win32" else cfg.get("num_workers", 2),
        grad_accum_steps=cfg.get("grad_accum_steps", 1),
        amp=cfg.get("amp", True),
        ckpt_dir=cfg.get("ckpt_dir", "outputs/checkpoints"),
        log_dir=cfg.get("log_dir", "outputs/runs"),
        save_every=cfg.get("save_every", 5),
        early_stop_patience=cfg.get("early_stop_patience", 10),
    )

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        n_classes=cfg["n_classes"],
        config=train_cfg, device=args.device, run_name=run_name,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
