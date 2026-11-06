"""
Reconstruct full-district GeoTIFFs from preprocessed .npz tiles for visual QGIS audit.

This allows you to verify that the labels after preprocessing:
1. Have no spatial distortions, missing seams, or tile edge artifacts.
2. Maintain correct class IDs (0..5: Water, Dense Forest, Open Forest, Ag, Built-up, Barren).
3. Align with raw satellite images in QGIS.

Usage:
    python scripts/verify_preprocessed_in_qgis.py --year 2024
    python scripts/verify_preprocessed_in_qgis.py --all-years
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    rasterio = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/maps"))
    p.add_argument("--year", type=int, default=None, help="Specific year to export (default: most recent)")
    p.add_argument("--all-years", action="store_true", help="Export GeoTIFFs for all processed years")
    return p.parse_args()


def main():
    if rasterio is None:
        print("rasterio is required: pip install rasterio", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    meta_path = args.processed_dir / "grid_meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run 'python scripts/preprocess.py' first.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    H, W = meta["height"], meta["width"]
    years = meta["years"]
    crs = meta["crs"]
    transform = Affine(*meta["transform"][:6]) if meta.get("transform") else None

    target_years = years if args.all_years else ([args.year] if args.year else [years[-1]])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(args.processed_dir.glob("tile_*.npz"))
    if not tile_files:
        print(f"No .npz tiles found in {args.processed_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tile_files)} tiles. Reconstructing full rasters for years: {target_years}")

    for year in target_years:
        if year not in years:
            print(f"[warn] Year {year} not in processed years {years}. Skipping.")
            continue

        year_idx = years.index(year)
        full_labels = np.full((H, W), fill_value=255, dtype=np.uint8)
        coverage_count = np.zeros((H, W), dtype=np.uint8)

        for tf in tile_files:
            # Filename format: tile_{r0}_{c0}.npz
            parts = tf.stem.split("_")
            r0, c0 = int(parts[1]), int(parts[2])
            with np.load(tf) as data:
                lbl_tile = data["labels"][year_idx]  # (H_tile, W_tile)
                th, tw = lbl_tile.shape
                r1, c1 = r0 + th, c0 + tw
                full_labels[r0:r1, c0:c1] = lbl_tile
                coverage_count[r0:r1, c0:c1] += 1

        uncovered = (coverage_count == 0).sum()
        total_pixels = H * W
        coverage_pct = 100.0 * (total_pixels - uncovered) / total_pixels

        # Compute summary stats
        unique_classes, counts = np.unique(full_labels[full_labels != 255], return_counts=True)
        class_dist = dict(zip(unique_classes.tolist(), counts.tolist()))

        out_tif = args.out_dir / f"preprocessed_labels_{year}.tif"
        profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "nodata": 255,
            "compress": "lzw",
        }

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(full_labels, 1)

        print(f"\n[OK] Wrote GeoTIFF: {out_tif}")
        print(f"     Spatial Coverage: {coverage_pct:.1f}%")
        print(f"     Class Distribution: {class_dist}")

    print("\n" + "=" * 65)
    print("HOW TO VERIFY IN QGIS:")
    print("1. Open QGIS and create a new project.")
    print("2. Add your base map (OpenStreetMap or Google Satellite via QuickMapServices plugin).")
    print(f"3. Drag and drop the reconstructed file: {args.out_dir / f'preprocessed_labels_{target_years[-1]}.tif'}")
    print("4. Right-click the layer in QGIS -> Properties -> Symbology.")
    print("5. Click 'Style' (bottom left) -> 'Load Style...'.")
    print("6. Select: outputs/maps/deoghar_lulc_style.qml")
    print("7. Inspect the Dark Green (Dense Forest) and Light Green (Open Forest) areas against the hill ranges (Trikut, Jasidih, Madhupur)!")
    print("=" * 65)


if __name__ == "__main__":
    main()

