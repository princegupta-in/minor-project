"""
Export SRTM DEM (elevation) and derived Slope for the Deoghar AOI.

Produces a 2-band GeoTIFF:
  Band 1: Elevation in meters (SRTM 30m, resampled to 10m)
  Band 2: Slope in degrees

Destination: Google Drive folder or directly to data/raw/static/
CRS: EPSG:32645 (UTM Zone 45N)

Usage:
    python scripts/gee_export_dem.py --drive-folder deoghar_forest_static
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ee


def init_ee(project: str | None = None) -> None:
    import os
    project = project or os.environ.get("GEE_PROJECT")
    if not project:
        print("Set GEE_PROJECT environment variable or pass --project.", file=sys.stderr)
        sys.exit(1)
    ee.Initialize(project=project)


def load_aoi(aoi_path: Path) -> ee.Geometry:
    with open(aoi_path) as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        coords = gj["features"][0]["geometry"]
    elif gj.get("type") == "Feature":
        coords = gj["geometry"]
    else:
        coords = gj
    return ee.Geometry(coords)


def get_dem_and_slope(aoi: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    return dem.addBands(slope).toFloat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aoi", type=Path, default=Path("data/aoi/deoghar.geojson"))
    p.add_argument("--drive-folder", default="deoghar_forest_static")
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--project", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_ee(project=args.project)

    aoi = load_aoi(args.aoi)
    static_img = get_dem_and_slope(aoi)

    name = "static_dem_slope"
    task = ee.batch.Export.image.toDrive(
        image=static_img,
        description=name,
        folder=args.drive_folder,
        fileNamePrefix=name,
        region=aoi,
        scale=args.scale,
        crs="EPSG:32645",
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Submitted static DEM+Slope export (Task ID: {task.id})")
    print(f"Monitor at https://code.earthengine.google.com/tasks")
    print(f"Save output as 'data/raw/static/static_dem_slope.tif'")


if __name__ == "__main__":
    main()

