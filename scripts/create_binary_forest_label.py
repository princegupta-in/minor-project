"""
Generate binary Forest vs. Non-Forest ground-truth labels directly from
raw Sentinel-2 imagery using calibrated index thresholding, MNDWI water gating,
and OpenStreetMap (OSM) structural waveguide road exclusion.

Formulation:
  1. Forest Canopy Condition: NDVI >= 0.43
  2. Enhanced Water Filter   : MNDWI < 0.0 (Green vs. SWIR1; cleanly eliminates water bodies)
  3. OSM Structural Waveguide: Excludes buffered road corridors (highways, primary, secondary roads)
  4. Contiguous Canopy Filter: Minimum patch size >= 4 pixels (400 m²; suppresses single-tree noise)
  5. Binary Output           : 1 = Forest, 0 = Non-Forest, 255 = NoData

Usage:
    python scripts/create_binary_forest_label.py --year 2025
    python scripts/create_binary_forest_label.py --all-years
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform
    from scipy.ndimage import binary_dilation, binary_opening, label
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
                   help="NDVI cutoff for forest canopy (default: 0.43)")
    p.add_argument("--use-mndwi", action="store_true", default=True,
                   help="Use MNDWI (Green vs SWIR1) for robust water exclusion (default: True)")
    p.add_argument("--mndwi-max", type=float, default=0.0,
                   help="Maximum MNDWI allowed (excludes water bodies, default: 0.0)")
    p.add_argument("--osm-file", type=Path, default=Path("data/raw/osm_deoghar.json"),
                   help="Path to OSM JSON file containing road/waterway features")
    p.add_argument("--use-osm-roads", action="store_true", default=True,
                   help="Exclude OSM road corridors to prevent canopy bleeding across roads (default: True)")
    p.add_argument("--no-osm-roads", dest="use_osm_roads", action="store_false",
                   help="Disable OSM road corridor exclusion")
    p.add_argument("--road-buffer-px", type=int, default=1,
                   help="Dilation radius in pixels for roads (1 px = ~30m total corridor width)")
    p.add_argument("--min-patch-pixels", type=int, default=4,
                   help="Minimum contiguous forest patch size in pixels (default: 4 = 400 m²)")
    return p.parse_args()


_ROAD_MASK_CACHE: dict[tuple, np.ndarray] = {}


def get_road_corridor_mask(osm_path: Path, crs, trans, shape, buffer_px: int) -> np.ndarray | None:
    """Rasterize and dilate OSM road ways into a binary mask on the raster grid."""
    cache_key = (str(osm_path), str(crs), shape, buffer_px)
    if cache_key in _ROAD_MASK_CACHE:
        return _ROAD_MASK_CACHE[cache_key]

    if not osm_path.exists():
        print(f"  [Notice] OSM vector file not found at {osm_path}. Skipping road waveguide.")
        return None

    with open(osm_path, encoding="utf-8") as f:
        osm_data = json.load(f)

    lines = []
    for el in osm_data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el and "highway" in el.get("tags", {}):
            lons = [p["lon"] for p in el["geometry"]]
            lats = [p["lat"] for p in el["geometry"]]
            xs, ys = transform("EPSG:4326", crs, lons, lats)
            coords = list(zip(xs, ys))
            if len(coords) >= 2:
                lines.append(({"type": "LineString", "coordinates": coords}, 1))

    if not lines:
        print("  [Notice] No road lines found in OSM data. Skipping road waveguide.")
        return None

    road_raster = rasterize(lines, out_shape=shape, transform=trans, fill=0, dtype=np.uint8)
    if buffer_px > 0:
        road_mask = binary_dilation(road_raster, iterations=buffer_px)
    else:
        road_mask = road_raster > 0

    _ROAD_MASK_CACHE[cache_key] = road_mask
    print(f"  [Waveguide] OSM road network active: {len(lines)} ways, {np.sum(road_mask):,} corridor pixels masked")
    return road_mask


def filter_min_patch_size(binary_mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove connected components smaller than min_pixels."""
    if min_pixels <= 1:
        return binary_mask
    labeled, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    keep = sizes >= min_pixels
    return keep[labeled]


def process_year(s2_file: Path, year: int, args) -> None:
    print(f"\n" + "-" * 70)
    print(f"Processing Year: {year}")
    print(f"  Input Sentinel-2: {s2_file}")
    print(f"  NDVI Threshold  : >= {args.ndvi_threshold}")
    print(f"  Water Filter    : MNDWI < {args.mndwi_max}" if args.use_mndwi else "NDWI < 0.0")

    with rasterio.open(s2_file) as src:
        prof = src.profile.copy()
        crs = src.crs
        trans = src.transform
        shape = src.shape
        descs = [d.lower() if d else "" for d in src.descriptions]

        # Read green, red, nir, swir1
        green = src.read(2).astype(np.float32)
        red   = src.read(3).astype(np.float32)
        nir   = src.read(4).astype(np.float32)

        # NDVI: (nir - red) / (nir + red)
        if "ndvi" in descs:
            ndvi = src.read(descs.index("ndvi") + 1)
        else:
            denom_ndvi = nir + red
            denom_ndvi[denom_ndvi == 0] = 1e-6
            ndvi = (nir - red) / denom_ndvi

        # Water filtering: MNDWI (B3 green vs B11 swir1) or standard NDWI (B3 green vs B8 nir)
        if args.use_mndwi:
            swir1 = src.read(6).astype(np.float32)
            denom_w = green + swir1
            denom_w[denom_w == 0] = 1e-6
            water_index = (green - swir1) / denom_w
        else:
            denom_w = green + nir
            denom_w[denom_w == 0] = 1e-6
            water_index = (green - nir) / denom_w

        valid_mask = ~np.isnan(ndvi)

    # 1. Spectral Classification (NDVI & Water Filter)
    forest_mask = (ndvi >= args.ndvi_threshold) & (water_index < args.mndwi_max) & valid_mask

    # 2. OSM Road Waveguide Exclusion
    if args.use_osm_roads:
        road_mask = get_road_corridor_mask(args.osm_file, crs, trans, shape, args.road_buffer_px)
        if road_mask is not None:
            pre_px = np.sum(forest_mask)
            forest_mask = forest_mask & (~road_mask)
            road_bleed_px = pre_px - np.sum(forest_mask)
            print(f"  [Waveguide] Excluded road bleeding: {road_bleed_px:,} pixels ({road_bleed_px * 0.0001:.2f} km²)")

    # 3. Minimum Contiguous Canopy Filter
    if args.min_patch_pixels > 1:
        pre_clean = np.sum(forest_mask)
        forest_mask = filter_min_patch_size(forest_mask, args.min_patch_pixels)
        speckle_px = pre_clean - np.sum(forest_mask)
        print(f"  [Cohesion] Filtered tiny speckles (< {args.min_patch_pixels} px): {speckle_px:,} pixels ({speckle_px * 0.0001:.2f} km²)")

    binary_raster = np.zeros(shape, dtype=np.uint8)
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
    try:
        with rasterio.open(maps_tif, "w", **prof) as dst:
            dst.write(binary_raster, 1)
        print(f"  [OK] Saved QGIS map       -> {maps_tif}")
    except PermissionError:
        maps_tif_alt = args.maps_dir / f"forest_binary_{year}_improved.tif"
        print(f"  [Notice] {maps_tif.name} is locked in QGIS. Saving to {maps_tif_alt.name}")
        with rasterio.open(maps_tif_alt, "w", **prof) as dst:
            dst.write(binary_raster, 1)
        print(f"  [OK] Saved QGIS map       -> {maps_tif_alt}")


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
