"""Preprocess raw GEE GeoTIFFs into training-ready .npz tiles.

Adapted for Deoghar forest cover analysis:
  - 6 LULC classes (forest-focused)
  - EPSG:32645 reference CRS (UTM Zone 45N)
  - Memory-safe windowed streaming: processes tiles without overloading RAM.
  - Stacks: 10 S2 bands/indices + 2 static terrain bands (DEM, Slope) = 12 channels.

Steps:
  1. Open datasets and verify matching spatial dimensions and CRS.
  2. Estimate per-channel global mean and std for z-score normalization.
  3. Window-read each tile across all years (T=10, C=12, H=128, W=128).
  4. Save each tile to data/processed/tile_XXXXX_XXXXX.npz.
  5. Save grid_meta.json and normalization.json.

Usage:
    python scripts/preprocess.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_tiles(H: int, W: int, tile_size: int, overlap: int = 0):
    """Yield (r0, c0, r1, c1) tile corners with optional overlap."""
    stride = tile_size - overlap
    for r0 in range(0, H - tile_size + 1, stride):
        for c0 in range(0, W - tile_size + 1, stride):
            yield r0, c0, r0 + tile_size, c0 + tile_size


def compute_normalization_stats_windowed(
    s2_readers: list,
    static_reader,
    H: int,
    W: int,
    tile_size: int = 128,
    n_samples: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global mean and std per channel by sampling spatial tiles across the AOI."""
    np.random.seed(42)
    sample_rows = np.random.randint(0, H - tile_size, size=n_samples)
    sample_cols = np.random.randint(0, W - tile_size, size=n_samples)

    channel_vals = None

    for r, c in zip(sample_rows, sample_cols):
        win = Window(col_off=c, row_off=r, width=tile_size, height=tile_size)
        static_block = static_reader.read(window=win) if static_reader else None  # (2, H, W)

        year_blocks = []
        for src in s2_readers:
            s2_block = src.read(window=win)  # (10, H, W)
            if static_block is not None:
                block = np.concatenate([s2_block, static_block], axis=0)  # (12, H, W)
            else:
                block = s2_block
            year_blocks.append(block)

        stack = np.stack(year_blocks, axis=0)  # (T, C, H, W)
        if np.isnan(stack).mean() > 0.3:
            continue

        flat = stack.transpose(1, 0, 2, 3).reshape(stack.shape[1], -1)  # (C, T*H*W)
        if channel_vals is None:
            channel_vals = [[] for _ in range(flat.shape[0])]
        for ch in range(flat.shape[0]):
            valid_px = flat[ch][~np.isnan(flat[ch])]
            if len(valid_px) > 0:
                channel_vals[ch].append(valid_px[::10])  # subsample for speed

    means = np.array([np.mean(np.concatenate(v)) if v else 0.0 for v in channel_vals], dtype=np.float32)
    stds = np.array([np.std(np.concatenate(v)) if v else 1.0 for v in channel_vals], dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


def main() -> None:
    if rasterio is None:
        raise ImportError("rasterio is required: pip install rasterio")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = p.parse_args()

    cfg         = load_config(args.config)
    raw_dir     = Path(cfg["raw_dir"])
    label_dir   = Path(cfg["label_dir"])
    static_path = Path(cfg["static_path"])
    out_dir     = Path(cfg["processed_dir"])
    years       = list(range(cfg["years"]["start"], cfg["years"]["end"] + 1))
    tile_size   = cfg["tile_size"]
    overlap     = cfg.get("tile_overlap", 0)

    print(f"[preprocess] Years: {years[0]}..{years[-1]} ({len(years)} years)")
    print(f"[preprocess] Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px")
    print(f"[preprocess] Target CRS: {cfg['crs']} @ {cfg['target_resolution_m']}m")

    # Match year files
    s2_files = {}
    label_files = {}
    for y in years:
        s2_cand = sorted(raw_dir.glob(f"*{y}*.tif*"))
        lbl_cand = sorted(label_dir.glob(f"*{y}*.tif*"))
        if s2_cand and lbl_cand:
            s2_files[y] = s2_cand[0]
            label_files[y] = lbl_cand[0]
        else:
            print(f"  [warn] Missing S2 or Label for year {y}, skipping")

    valid_years = sorted(list(s2_files.keys()))
    if not valid_years:
        print("[preprocess] ERROR: No matching S2 and label files found. Check paths.")
        sys.exit(1)

    print(f"[preprocess] Found {len(valid_years)} synchronized annual raster pairs.")

    # Open readers
    s2_readers = [rasterio.open(s2_files[y]) for y in valid_years]
    lbl_readers = [rasterio.open(label_files[y]) for y in valid_years]
    static_reader = rasterio.open(static_path) if static_path.exists() else None

    ref_prof = s2_readers[0].profile
    H, W = ref_prof["height"], ref_prof["width"]
    s2_channels = s2_readers[0].count
    static_channels = static_reader.count if static_reader else 0
    total_channels = s2_channels + static_channels

    print(f"[preprocess] Grid Dimensions: {H} rows x {W} cols")
    print(f"[preprocess] Channel stack: {s2_channels} S2 bands/indices + {static_channels} static = {total_channels} channels")

    # Compute normalization stats
    print("[preprocess] Computing per-channel normalization statistics...")
    means, stds = compute_normalization_stats_windowed(
        s2_readers, static_reader, H, W, tile_size=tile_size, n_samples=60
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir.parent / "normalization.json"
    with open(stats_path, "w") as f:
        json.dump({
            "mean": means.tolist(),
            "std": stds.tolist(),
            "years": valid_years,
            "n_channels": int(total_channels),
        }, f, indent=2)
    print(f"  [info] Saved normalization stats -> {stats_path}")

    grid_meta_path = out_dir / "grid_meta.json"
    with open(grid_meta_path, "w") as f:
        json.dump({
            "crs": ref_prof["crs"].to_string() if ref_prof.get("crs") else "EPSG:32645",
            "transform": list(ref_prof["transform"]) if ref_prof.get("transform") else None,
            "height": H,
            "width": W,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "years": valid_years,
            "n_channels": total_channels,
        }, f, indent=2)
    print(f"  [info] Saved grid metadata -> {grid_meta_path}")

    # Generate tiles
    tile_coords = list(iter_tiles(H, W, tile_size, overlap))
    print(f"[preprocess] Tiling grid into {len(tile_coords)} patches using memory-safe streaming...")

    n_saved = 0
    n_skipped = 0

    for r0, c0, r1, c1 in tqdm(tile_coords, desc="Tiling"):
        win = Window(col_off=c0, row_off=r0, width=tile_size, height=tile_size)

        # Read static terrain for this tile
        static_tile = static_reader.read(window=win).astype(np.float32) if static_reader else None

        tile_frames = []
        tile_labels = []

        for y_idx in range(len(valid_years)):
            s2_tile = s2_readers[y_idx].read(window=win).astype(np.float32)
            lbl_tile = lbl_readers[y_idx].read(1, window=win).astype(np.int16)

            if static_tile is not None:
                frame = np.concatenate([s2_tile, static_tile], axis=0)
            else:
                frame = s2_tile

            tile_frames.append(frame)
            tile_labels.append(lbl_tile)

        frames_arr = np.stack(tile_frames, axis=0)  # (T, C, H, W)
        labels_arr = np.stack(tile_labels, axis=0)  # (T, H, W)

        # Skip tiles outside AOI / with excessive NaNs
        if np.isnan(frames_arr).mean() > 0.4:
            n_skipped += 1
            continue

        # Clean NaNs and normalize
        frames_arr = np.nan_to_num(frames_arr, nan=0.0)
        frames_arr = (frames_arr - means[None, :, None, None]) / stds[None, :, None, None]

        tile_id = f"tile_{r0:05d}_{c0:05d}"
        np.savez_compressed(
            out_dir / f"{tile_id}.npz",
            frames=frames_arr.astype(np.float32),
            labels=labels_arr.astype(np.int16),
            years=np.array(valid_years, dtype=np.int32),
        )
        n_saved += 1

    # Close readers
    for r in s2_readers:
        r.close()
    for r in lbl_readers:
        r.close()
    if static_reader:
        static_reader.close()

    print(f"\n[preprocess] Finished successfully!")
    print(f"  Saved tiles : {n_saved} -> {out_dir}")
    print(f"  Skipped off-AOI: {n_skipped}")
    print(f"  Shape per tile: ({len(valid_years)}, {total_channels}, {tile_size}, {tile_size})")


if __name__ == "__main__":
    main()
