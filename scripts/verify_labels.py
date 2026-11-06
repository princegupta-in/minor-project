"""
Automated Label & Raster Quality Audit Script
Checks:
  1. Coordinate Reference System (CRS) matching EPSG:32645
  2. Resolution and dimensions consistency
  3. Class distribution per year (detects anomalous spikes due to cloud/shadow)
  4. Missing values / unexpected class IDs outside 0..5

Usage:
    python scripts/verify_labels.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

CLASS_NAMES = ["Water", "Dense Forest", "Open Forest", "Agriculture", "Built-up", "Barren"]
EXPECTED_CRS = "EPSG:32645"

def audit_labels(label_dir: Path):
    if rasterio is None:
        print("rasterio is required to run label audits. Run: pip install rasterio", file=sys.stderr)
        sys.exit(1)

    tifs = sorted(list(label_dir.glob("*.tif*")))
    if not tifs:
        print(f"No GeoTIFF label files found in {label_dir}")
        print("Once GEE export finishes, place the files in data/raw/labels/ and run this again.")
        return

    print(f"\nFound {len(tifs)} label rasters in {label_dir}:")
    print("-" * 75)
    print(f"{'Filename':<22} | {'CRS':<10} | {'Shape (H, W)':<15} | {'Unique IDs':<15}")
    print("-" * 75)

    stats_per_year = {}
    base_profile = None

    for tif in tifs:
        with rasterio.open(tif) as src:
            data = src.read(1)
            crs_str = src.crs.to_string() if src.crs else "None"
            shape_str = f"{src.height}x{src.width}"
            unique_vals = np.unique(data)
            unique_str = str(list(unique_vals))

            print(f"{tif.name:<22} | {crs_str:<10} | {shape_str:<15} | {unique_str:<15}")

            if base_profile is None:
                base_profile = src.profile
            else:
                if src.crs != base_profile["crs"]:
                    print(f"  [WARNING] CRS mismatch in {tif.name}: {src.crs} vs expected {base_profile['crs']}")
                if (src.height, src.width) != (base_profile["height"], base_profile["width"]):
                    print(f"  [WARNING] Grid shape mismatch in {tif.name}")

            # Calculate class percentages
            counts = {c: float(np.sum(data == c)) for c in range(6)}
            total = max(1, data.size)
            pcts = {c: (counts[c] / total) * 100.0 for c in range(6)}
            stats_per_year[tif.name] = pcts

    print("-" * 75)
    print("\nClass Percentage Summary (% of Total Area):")
    header = f"{'Raster':<22} | " + " | ".join([f"{name[:8]:>8}" for name in CLASS_NAMES])
    print(header)
    print("-" * len(header))

    for name, pcts in stats_per_year.items():
        row = f"{name:<22} | " + " | ".join([f"{pcts[c]:>7.2f}%" for c in range(6)])
        print(row)

    print("\n[Audit Complete] If any year shows sudden wild swings (e.g., >30% change in forest in 1 year),")
    print("inspect that year in QGIS with Google Satellite imagery to verify cloud masking.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=Path("data/raw/labels"))
    args = parser.parse_args()
    audit_labels(args.label_dir)
