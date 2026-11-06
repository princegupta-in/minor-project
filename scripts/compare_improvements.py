import json
import rasterio
from rasterio.warp import transform
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_opening
import numpy as np

s2_file = "data/raw/s2/s2_2025.tif"
osm_file = "data/raw/osm_deoghar.json"

with open(osm_file) as f:
    osm = json.load(f)

with rasterio.open(s2_file) as src:
    crs = src.crs
    trans = src.transform
    shape = src.shape
    
    green = src.read(2).astype(np.float32)
    red   = src.read(3).astype(np.float32)
    nir   = src.read(4).astype(np.float32)
    swir1 = src.read(6).astype(np.float32)
    
    denom_ndvi = nir + red
    denom_ndvi[denom_ndvi == 0] = 1e-6
    ndvi = (nir - red) / denom_ndvi
    
    denom_ndwi = green + nir
    denom_ndwi[denom_ndwi == 0] = 1e-6
    ndwi = (green - nir) / denom_ndwi

    denom_mndwi = green + swir1
    denom_mndwi[denom_mndwi == 0] = 1e-6
    mndwi = (green - swir1) / denom_mndwi
    
    valid_mask = ~np.isnan(ndvi)

# 1. Baseline
base_mask = (ndvi >= 0.43) & (ndwi < 0.0) & valid_mask
base_km2 = np.sum(base_mask) * 0.0001
print(f"1. Baseline (NDVI >= 0.43 & NDWI < 0.0): {base_km2:,.2f} km² ({np.sum(base_mask)/np.sum(valid_mask)*100:.2f}%)")

# 2. Extract Road vectors from OSM
road_lines = []
for el in osm.get("elements", []):
    if el.get("type") == "way" and "geometry" in el and "highway" in el.get("tags", {}):
        lons = [p["lon"] for p in el["geometry"]]
        lats = [p["lat"] for p in el["geometry"]]
        xs, ys = transform("EPSG:4326", crs, lons, lats)
        coords = list(zip(xs, ys))
        if len(coords) >= 2:
            road_lines.append(({"type": "LineString", "coordinates": coords}, 1))

road_raster = rasterize(road_lines, out_shape=shape, transform=trans, fill=0, dtype=np.uint8)
# Dilation by 1 pixel (30m wide road corridor)
road_corridor_1 = binary_dilation(road_raster, iterations=1)
road_corridor_2 = binary_dilation(road_raster, iterations=2) # 50m wide

print(f"  Road centerline pixels: {np.sum(road_raster):,}")
print(f"  Road 30m corridor pixels: {np.sum(road_corridor_1):,}")
print(f"  Road 50m corridor pixels: {np.sum(road_corridor_2):,}")

# 3. Baseline + Road Cutout (30m corridor)
base_no_roads_1 = base_mask & (~road_corridor_1)
print(f"2. Baseline + 30m Road Exclusion: {np.sum(base_no_roads_1)*0.0001:,.2f} km² (Road false-forest removed: {np.sum(base_mask & road_corridor_1)*0.0001:.2f} km²)")

# 4. Baseline + MNDWI < 0.0 + Road Cutout
mndwi_no_roads = (ndvi >= 0.43) & (mndwi < 0.0) & (~road_corridor_1) & valid_mask
print(f"3. + MNDWI < 0.0 & Road Exclusion: {np.sum(mndwi_no_roads)*0.0001:,.2f} km²")

# 5. + Clean Speckle (remove isolated 1-2 px noise)
clean_mask = binary_opening(mndwi_no_roads, structure=np.ones((3, 3)))
print(f"4. + Speckle / Isolated Pixel Opening: {np.sum(clean_mask)*0.0001:,.2f} km² ({np.sum(clean_mask)/np.sum(valid_mask)*100:.2f}%)")
from scipy.ndimage import label
import numpy as np

# Let's test connected component filtering vs morphological opening
mask = (mndwi_no_roads).astype(np.uint8)

# Connected components
labeled, num_features = label(mask)
print(f"Total separate forest patches detected: {num_features:,}")

# Patch sizes
sizes = np.bincount(labeled.ravel())
sizes[0] = 0  # ignore background

for min_px in [2, 4, 6, 9]:
    keep = sizes >= min_px
    filtered = keep[labeled]
    f_km2 = np.sum(filtered) * 0.0001
    f_pct = (np.sum(filtered) / np.sum(valid_mask)) * 100
    print(f"  Min patch size >= {min_px:2d} pixels ({min_px*100:4d} m²): {f_km2:,.2f} km² ({f_pct:.2f}%)")
