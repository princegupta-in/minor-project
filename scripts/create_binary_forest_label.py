"""
Generate binary Forest vs. Non-Forest ground-truth labels directly from
raw Sentinel-2 imagery using calibrated NDVI thresholding and water masking.

Approved Formulation:
  1. Forest Canopy Condition: NDVI >= 0.43
  2. Water Filter: NDWI < 0.0 (excludes rivers, reservoirs, water bodies)
  3. Binary Output: 1 = Forest, 0 = Non-Forest

Usage:
    python scripts/create_binary_forest_label.py --year 2025
    python scripts/create_binary_forest_label.py --all-years
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import rasterio
    from scipy.ndimage import binary_opening
except ImportError:
    rasterio = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2025, help="Single year to process (default: 2025)")
    p.add_argument("--all-years", action="store_true", help="Process all available s2_*.tif images")
    p.add_argument("--s2-dir", type=Path, default=Path("data/raw/s2"))
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/labels"),
                   help="Directory to save label rasters (default: data/raw/labels)")
    p.add_argument("--maps-dir", type=Path, default=Path("outputs/maps"),
                   help="Directory to save QGIS-ready maps (default: outputs/maps)")
    p.add_argument("--ndvi-threshold", type=float, default=0.43,
                   help="NDVI cutoff for forest canopy (approved: 0.43)")
    p.add_argument("--ndwi-max", type=float, default=0.0,
                   help="Maximum NDWI allowed (excludes water bodies)")
    p.add_argument("--clean-speckle", action="store_true",
                   help="Apply morphological 3x3 filter to remove isolated 1-pixel noise")
    return p.parse_args()


def process_year(s2_file: Path, year: int, args) -> None:
    print(f"\n" + "-" * 70)
    print(f"Processing Year: {year}")
    print(f"  Input Sentinel-2: {s2_file}")
    print(f"  NDVI Threshold  : >= {args.ndvi_threshold}")
    print(f"  Water Filter    : NDWI < {args.ndwi_max}")

    with rasterio.open(s2_file) as src:
        prof = src.profile.copy()
        descs = [d.lower() if d else "" for d in src.descriptions]

        if "ndvi" in descs and "ndwi" in descs:
            ndvi_idx = descs.index("ndvi") + 1
            ndwi_idx = descs.index("ndwi") + 1
            ndvi = src.read(ndvi_idx)
            ndwi = src.read(ndwi_idx)
        else:
            red = src.read(3).astype(np.float32)
            green = src.read(2).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            denom_ndvi = nir + red
            denom_ndvi[denom_ndvi == 0] = 1e-6
            ndvi = (nir - red) / denom_ndvi

            denom_ndwi = green + nir
            denom_ndwi[denom_ndwi == 0] = 1e-6
            ndwi = (green - nir) / denom_ndwi

        valid_mask = ~np.isnan(ndvi)

    # Core Binary Forest Rule:
    forest_mask = (ndvi >= args.ndvi_threshold) & (ndwi < args.ndwi_max) & valid_mask

    if args.clean_speckle:
        forest_mask = binary_opening(forest_mask, structure=np.ones((3, 3)))

    binary_raster = np.zeros(ndvi.shape, dtype=np.uint8)
    binary_raster[forest_mask] = 1
    binary_raster[~valid_mask] = 255  # nodata

    total_valid = np.sum(valid_mask)
    forest_px   = np.sum(binary_raster == 1)
    forest_km2  = forest_px * 0.0001
    total_km2   = total_valid * 0.0001
    forest_pct  = (forest_px / total_valid) * 100.0

    print(f"  Total Valid Area : {total_km2:,.2f} km²")
    print(f"  Forest Area (1)  : {forest_km2:,.2f} km² ({forest_pct:.2f}% of study area)")
    print(f"  Non-Forest (0)   : {total_km2 - forest_km2:,.2f} km² ({100.0 - forest_pct:.2f}%)")

    prof.update(
        count=1,
        dtype=rasterio.uint8,
        nodata=255,
        compress="lzw"
    )

    # Save to data/raw/labels/labels_<year>.tif (for training pipeline)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = args.out_dir / f"labels_{year}.tif"
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(binary_raster, 1)
    print(f"  [OK] Saved training label -> {out_tif}")

    # Also save to outputs/maps/forest_binary_<year>.tif (for QGIS inspection)
    args.maps_dir.mkdir(parents=True, exist_ok=True)
    maps_tif = args.maps_dir / f"forest_binary_{year}.tif"
    with rasterio.open(maps_tif, "w", **prof) as dst:
        dst.write(binary_raster, 1)
    print(f"  [OK] Saved QGIS map       -> {maps_tif}")


def main():
    if rasterio is None:
        raise ImportError("rasterio and scipy are required: pip install rasterio scipy")

    args = parse_args()
    if args.all_years:
        s2_files = sorted(list(args.s2_dir.glob("s2_*.tif")))
        if not s2_files:
            print(f"No s2_*.tif files found in {args.s2_dir}")
            return
        print(f"Processing all {len(s2_files)} years found in {args.s2_dir}...")
        for sf in s2_files:
            # extract year from s2_YYYY.tif
            parts = sf.stem.split("_")
            year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2025
            process_year(sf, year, args)
    else:
        sf = args.s2_dir / f"s2_{args.year}.tif"
        if not sf.exists():
            raise FileNotFoundError(f"File not found: {sf}")
        process_year(sf, args.year, args)

    print("\n" + "=" * 70)
    print("[Complete] All binary labels generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
