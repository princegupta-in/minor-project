"""PyTorch Dataset for tiled multitemporal LULC sequences — Deoghar project.

Each item is a spatial tile with a time series:
    x: (T, C, H, W) float32  — input frames (bands + indices + static)
    y: (H, W) int64           — next-year label map

On-disk format (one .npz per tile):
    frames: (T_total, C, H, W)  float32
    labels: (T_total, H, W)     int16
    years:  (T_total,)          int
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LULCSequenceDataset(Dataset):
    """Multitemporal LULC tile dataset.

    Args:
        data_dir: directory of .npz tile files
        seq_length: number of historical frames in input X
        target_offset: how many years ahead to predict (1 = next year)
        transform: optional callable applied to (x, y) tensors
    """

    def __init__(
        self,
        data_dir: Path | str,
        seq_length: int = 8,
        target_offset: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_dir     = Path(data_dir)
        self.files        = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz tiles in {data_dir}")
        self.seq_length   = seq_length
        self.target_offset = target_offset
        self.transform    = transform

        # Build sliding-window index over (file, start_t) pairs
        self.index: List[Tuple[int, int]] = []
        for fi, fp in enumerate(self.files):
            with np.load(fp) as f:
                T = f["frames"].shape[0]
            max_start = T - seq_length - target_offset
            for s in range(0, max_start + 1):
                self.index.append((fi, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, t0 = self.index[idx]
        with np.load(self.files[file_idx]) as f:
            frames = f["frames"][t0 : t0 + self.seq_length]                          # (T, C, H, W)
            label  = f["labels"][t0 + self.seq_length + self.target_offset - 1]      # (H, W)
        x = torch.from_numpy(frames).float()
        y = torch.from_numpy(label.astype(np.int64)).long()
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y


def split_files_block(
    files: List[Path],
    val_frac:  float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Spatially-blocked train/val/test split by tile column index.

    Tiles with the same column index stay together to avoid spatial leakage.
    Assumes filenames like 'tile_00050_00128.npz' (row_col encoding).
    """
    by_col: dict[int, List[Path]] = {}
    for fp in files:
        try:
            col = int(fp.stem.split("_")[-1])
        except ValueError:
            col = 0
        by_col.setdefault(col, []).append(fp)

    cols = sorted(by_col.keys())
    n       = len(cols)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    test_cols = set(cols[-n_test:])
    val_cols  = set(cols[-(n_test + n_val):-n_test])

    train, val, test = [], [], []
    for col, fps in by_col.items():
        if   col in test_cols: test.extend(fps)
        elif col in val_cols:  val.extend(fps)
        else:                  train.extend(fps)
    return train, val, test


def default_train_transform():
    """Random horizontal + vertical flip augmentation for (x, y) tensor pairs."""
    def transform(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])
        return x, y
    return transform
