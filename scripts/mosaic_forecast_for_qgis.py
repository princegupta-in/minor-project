"""
Mosaic tile-level future forecasts (.npz) into full-district GeoTIFFs for QGIS visualization.

Generates:
  1. outputs/predictions/forest_forecast_<year>.tif (LULC label map)
  2. outputs/predictions/uncertainty_entropy_<year>.tif (MC Dropout uncertainty)
  3. outputs/predictions/forest_change_<start>_to_<year>.tif (Loss / Stable / Regrowth)

Usage:
    python scripts/mosaic_forecast_for_qgis.py
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
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/predictions/maps"))
    return p.parse_args()


def main():
    if rasterio is None:
        print("rasterio is required: pip install rasterio", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    meta_path = args.processed_dir / "grid_meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    H, W = meta["height"], meta["width"]
    crs = meta["crs"]
    transform = Affine(*meta["transform"][:6]) if meta.get("transform") else None

    forecast_files = sorted(args.pred_dir.glob("tile_*_forecast.npz"))
    if not forecast_files:
        print(f"No forecast files found in {args.pred_dir}. Run scripts/predict_future.py first.", file=sys.stderr)
        sys.exit(1)

    # Read years from first forecast file
    with np.load(forecast_files[0]) as data:
        years = data["years"].tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mosaicking future forecasts for years {years}...")

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

    profile_entropy = profile.copy()
    profile_entropy.update(dtype="float32", nodata=-1.0)

    for y_idx, year in enumerate(years):
        full_labels = np.full((H, W), fill_value=255, dtype=np.uint8)
        full_entropy = np.full((H, W), fill_value=-1.0, dtype=np.float32)

        for ff in forecast_files:
            # Filename: tile_{r0}_{c0}_forecast.npz
            parts = ff.stem.replace("_forecast", "").split("_")
            r0, c0 = int(parts[1]), int(parts[2])

            with np.load(ff) as data:
                lbl = data["labels"][y_idx]
                ent = data["entropies"][y_idx]
                th, tw = lbl.shape
                full_labels[r0:r0 + th, c0:c0 + tw] = lbl
                full_entropy[r0:r0 + th, c0:c0 + tw] = ent

        # Export LULC prediction GeoTIFF
        out_tif = args.out_dir / f"deoghar_forecast_{year}.tif"
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(full_labels, 1)

        # Export Uncertainty (Entropy) GeoTIFF
        out_ent_tif = args.out_dir / f"deoghar_uncertainty_{year}.tif"
        with rasterio.open(out_ent_tif, "w", **profile_entropy) as dst:
            dst.write(full_entropy, 1)

        print(f"  [OK] Exported {out_tif.name} and {out_ent_tif.name}")

    print("\nForecast GeoTIFFs ready! Open in QGIS and apply 'outputs/maps/deoghar_lulc_style.qml'.")


if __name__ == "__main__":
    main()

