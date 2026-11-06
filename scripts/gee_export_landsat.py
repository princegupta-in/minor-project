"""
Export Landsat-8/9 annual median composites for the Deoghar AOI.

Used as a fallback/supplement for years where Sentinel-2 coverage is sparse
(Sentinel-2 archive starts 2015; Landsat-8 starts 2013).

CRS: EPSG:32645 (UTM Zone 45N)

Usage:
    python scripts/gee_export_landsat.py --years 2013 2025 \\
        --drive-folder deoghar_forest_landsat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ee


# Landsat-8/9 OLI surface reflectance bands
LS_BANDS  = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
LS_RENAME = ["blue",  "green", "red",   "nir",   "swir1", "swir2"]


def init_ee(project: str | None = None) -> None:
    import os
    project = project or os.environ.get("GEE_PROJECT")
    if not project:
        print("Set GEE_PROJECT env var or pass --project.", file=sys.stderr)
        sys.exit(1)
    ee.Initialize(project=project)


def load_aoi(path: Path) -> ee.Geometry:
    with open(path) as f:
        gj = json.load(f)
    gj = gj.get("features", [gj])[0].get("geometry", gj)
    return ee.Geometry(gj)


def mask_landsat_clouds(img: ee.Image) -> ee.Image:
    """QA_PIXEL bit-mask cloud/shadow masking."""
    qa = img.select("QA_PIXEL")
    cloud        = qa.bitwiseAnd(1 << 3).neq(0)
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    return img.updateMask(cloud.Or(cloud_shadow).Not())


def add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    nbr  = img.normalizedDifference(["nir", "swir2"]).rename("nbr")
    ndwi = img.normalizedDifference(["green", "nir"]).rename("ndwi")
    return img.addBands([ndvi, nbr, ndwi])


def annual_composite_landsat(year: int, aoi: ee.Geometry) -> ee.Image:
    start = f"{year}-01-01"
    end   = f"{year + 1}-01-01"
    # Merge Landsat 8 and 9 collections
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi).filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 70))
        .map(mask_landsat_clouds)
        .select(LS_BANDS + ["QA_PIXEL"], LS_RENAME + ["QA_PIXEL"])
    )
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi).filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 70))
        .map(mask_landsat_clouds)
        .select(LS_BANDS + ["QA_PIXEL"], LS_RENAME + ["QA_PIXEL"])
    )
    merged = l8.merge(l9).select(LS_RENAME)
    composite = merged.median().clip(aoi)
    composite = add_indices(composite)
    # Scale to [0, 1] from 16-bit integer
    composite = composite.multiply(0.0000275).add(-0.2)
    return composite.toFloat().set("year", year)


def export_image(img, name, aoi, drive_folder, scale=30):
    task = ee.batch.Export.image.toDrive(
        image=img, description=name, folder=drive_folder,
        fileNamePrefix=name, region=aoi, scale=scale,
        crs="EPSG:32645", maxPixels=int(1e10), fileFormat="GeoTIFF",
    )
    task.start()
    return task


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs=2, type=int, default=[2013, 2025], metavar=("START", "END"))
    p.add_argument("--aoi", type=Path, default=Path("data/aoi/deoghar.geojson"))
    p.add_argument("--drive-folder", default="deoghar_forest_landsat")
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--project", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    init_ee(args.project)
    aoi = load_aoi(args.aoi)
    start, end = args.years
    print(f"Submitting Landsat-8/9 exports {start}..{end}")
    for year in range(start, end + 1):
        img  = annual_composite_landsat(year, aoi)
        name = f"landsat_{year}"
        task = export_image(img, name, aoi, args.drive_folder, args.scale)
        print(f"  {name}  task_id={task.id}")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
