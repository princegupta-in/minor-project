"""Lightweight training loop with TensorBoard logging.
Adapted from ShivekRanjan/bengaluru-lulc-forecast- for the Deoghar project.
Adds forest-specific metric (forest_iou) to checkpoint metadata.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..evaluation.metrics import StreamingConfusion


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    num_workers: int = 2
    grad_accum_steps: int = 4
    amp: bool = True
    ckpt_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/runs"
    save_every: int = 5
    early_stop_patience: int = 12


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        n_classes: int,
        config: TrainConfig,
        device: str = "cuda",
        run_name: str = "run",
    ) -> None:
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model        = model.to(self.device)
        self.loss_fn      = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.n_classes    = n_classes
        self.config       = config
        self.run_name     = run_name

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")

        self.ckpt_dir = Path(config.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(Path(config.log_dir) / run_name))

        self.best_val_iou    = -1.0
        self.no_improve      = 0

    def _step(self, batch):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=self.config.amp and self.device.type == "cuda"):
            logits = self.model(x)
            loss   = self.loss_fn(logits, y)
        return {"loss": loss, "logits": logits.detach(), "y": y}

    def fit(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(epoch)
            val_met    = self._validate(epoch) if self.val_loader else {}
            self.scheduler.step()

            print(
                f"epoch {epoch:>3d}  train_loss={train_loss:.4f}"
                + (f"  val_miou={val_met.get('miou', 0):.3f}"
                   f"  val_forest_iou={val_met.get('forest_iou', 0):.3f}" if val_met else "")
                + f"  {time.time() - t0:.1f}s"
            )

            cur_iou  = val_met.get("miou", -1.0)
            improved = cur_iou > self.best_val_iou
            if improved:
                self.best_val_iou = cur_iou
                self.no_improve   = 0
                self._save("best.pt", epoch, val_met)
            else:
                self.no_improve += 1

            if epoch % self.config.save_every == 0:
                self._save(f"epoch_{epoch:03d}.pt", epoch, val_met)

            if self.no_improve >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={self.config.early_stop_patience})")
                break

        self.writer.close()

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        self.optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            out  = self._step(batch)
            loss = out["loss"] / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()
            if (i + 1) % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total += loss.item() * self.config.grad_accum_steps * batch[0].size(0)
            n     += batch[0].size(0)
            pbar.set_postfix({"loss": total / max(n, 1)})
        avg = total / max(n, 1)
        self.writer.add_scalar("train/loss", avg, epoch)
        return avg

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        cm   = StreamingConfusion(self.n_classes)
        loss_sum, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc=f"val {epoch}", leave=False):
            out   = self._step(batch)
            preds = out["logits"].argmax(dim=1)
            cm.update(preds.flatten().cpu().numpy(), out["y"].flatten().cpu().numpy())
            loss_sum += out["loss"].item() * batch[0].size(0)
            n        += batch[0].size(0)
        met           = cm.compute()
        met["loss"]   = loss_sum / max(n, 1)
        for k, v in met.items():
            if isinstance(v, float):
                self.writer.add_scalar(f"val/{k}", v, epoch)
        return met

    def _save(self, name: str, epoch: int, metrics: dict) -> None:
        path = self.ckpt_dir / f"{self.run_name}_{name}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "config": asdict(self.config),
        }, path)
