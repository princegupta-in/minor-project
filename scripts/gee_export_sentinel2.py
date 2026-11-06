"""
Export annual Sentinel-2 L2A median composites for the Deoghar AOI.

Adapted from ShivekRanjan/bengaluru-lulc-forecast- with the following changes:
  - AOI: Deoghar district, Jharkhand
  - CRS: EPSG:32645  (UTM Zone 45N — correct for Jharkhand)
  - Added NBR (Normalized Burn Ratio) for forest disturbance detection
  - Cloud masking tweaked for higher cloud cover tolerance (monsoon region)
  - Optional bi-seasonal export (Kharif + Rabi windows)

Workflow:
  1. Authenticate once:  earthengine authenticate
  2. Place Deoghar boundary at data/aoi/deoghar.geojson
  3. Run:
       python scripts/gee_export_sentinel2.py --years 2015 2025 \\
           --drive-folder deoghar_forest_s2
  4. Sync exported GeoTIFFs from Google Drive to data/raw/s2/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ee


# Sentinel-2 L2A bands used (10m + 20m resampled to 10m)
S2_BANDS  = ["B2",   "B3",    "B4",  "B8",  "B8A",  "B11",  "B12"]
S2_RENAME = ["blue", "green", "red", "nir", "nir8a", "swir1", "swir2"]


def init_ee(project: str | None = None) -> None:
    import os
    project = project or os.environ.get("GEE_PROJECT")
    if not project:
        print(
            "Earth Engine requires a Cloud project ID. "
            "Set the GEE_PROJECT env var or pass --project.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        ee.Initialize(project=project)
    except Exception:
        print("EE init failed. Run: earthengine authenticate", file=sys.stderr)
        raise


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


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """SCL-based cloud/shadow mask. Tolerant threshold for monsoon imagery."""
    scl = img.select("SCL")
    bad = (
        scl.eq(3)   # cloud shadow
        .Or(scl.eq(8))   # cloud medium prob
        .Or(scl.eq(9))   # cloud high prob
        .Or(scl.eq(10))  # thin cirrus
    )
    return img.updateMask(bad.Not())


def add_indices(img: ee.Image) -> ee.Image:
    """Append NDVI, NBR, NDWI spectral indices.

    NBR (Normalized Burn Ratio) is especially useful for detecting
    forest disturbance, fire scars, and logging clearcuts.
    """
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    nbr  = img.normalizedDifference(["nir", "swir2"]).rename("nbr")   # forest health
    ndwi = img.normalizedDifference(["green", "nir"]).rename("ndwi")
    return img.addBands([ndvi, nbr, ndwi])


def annual_composite(
    year: int,
    aoi: ee.Geometry,
    start_month: int = 1,
    end_month: int = 12,
) -> ee.Image:
    """Median composite for a calendar year (or seasonal window) + indices."""
    if end_month >= start_month:
        start = f"{year}-{start_month:02d}-01"
        end   = f"{year}-{end_month:02d}-28" if end_month in (2,) else f"{year}-{end_month:02d}-30"
    else:
        # Rabi wraps year boundary (Nov–Mar)
        start = f"{year}-{start_month:02d}-01"
        end   = f"{year + 1}-{end_month:02d}-28"

    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))  # relaxed for monsoon
        .map(mask_s2_clouds)
        .select(S2_BANDS, S2_RENAME)
    )
    composite = coll.median().clip(aoi)
    composite = add_indices(composite)
    composite = composite.set("year", year)
    return composite.toFloat()


def export_image(
    img: ee.Image,
    name: str,
    aoi: ee.Geometry,
    drive_folder: str,
    scale: int = 10,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=drive_folder,
        fileNamePrefix=name,
        region=aoi,
        scale=scale,
        crs="EPSG:32645",   # UTM Zone 45N for Jharkhand
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs=2, type=int, default=[2015, 2025], metavar=("START", "END"))
    p.add_argument("--aoi", type=Path, default=Path("data/aoi/deoghar.geojson"))
    p.add_argument("--out", type=Path, default=Path("data/raw/s2"))
    p.add_argument("--drive-folder", default="deoghar_forest_s2")
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--project", default=None)
    p.add_argument(
        "--seasonal", action="store_true",
        help="Export two composites per year (Kharif Jun-Oct + Rabi Nov-Mar)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_ee(project=args.project)

    if not args.aoi.exists():
        print(
            f"AOI not found at {args.aoi}.\n"
            "Download Deoghar district boundary from:\n"
            "  https://bhuvan-app1.nrsc.gov.in/bhuvan2d/bhuvan/bhuvan2d.php\n"
            "  or https://data.gov.in (India Administrative Boundary)",
            file=sys.stderr,
        )
        sys.exit(1)

    aoi = load_aoi(args.aoi)
    args.out.mkdir(parents=True, exist_ok=True)

    start, end = args.years
    print(f"Submitting Sentinel-2 exports {start}..{end} -> Drive folder '{args.drive_folder}'")

    tasks = []
    for year in range(start, end + 1):
        if args.seasonal:
            for season, (sm, em) in {"kharif": (6, 10), "rabi": (11, 3)}.items():
                img  = annual_composite(year, aoi, sm, em)
                name = f"s2_{year}_{season}"
                task = export_image(img, name, aoi, args.drive_folder, args.scale)
                tasks.append((year, season, task))
                print(f"  submitted {name}  task_id={task.id}")
        else:
            img  = annual_composite(year, aoi)
            name = f"s2_{year}"
            task = export_image(img, name, aoi, args.drive_folder, args.scale)
            tasks.append((year, "annual", task))
            print(f"  submitted {name}  task_id={task.id}")

    print()
    print("All tasks submitted. Monitor at https://code.earthengine.google.com/tasks")
    print(f"After completion, sync Drive folder '{args.drive_folder}' into {args.out}")


if __name__ == "__main__":
    main()
