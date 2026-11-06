import rasterio
import numpy as np

s2_file = "data/raw/s2/s2_2025.tif"
with rasterio.open(s2_file) as src:
    print("Raster shape:", src.shape)
    descs = list(src.descriptions)
    print("Descriptions:", descs)
    
    # Read bands needed
    blue  = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    red   = src.read(3).astype(np.float32)
    nir   = src.read(4).astype(np.float32)
    swir1 = src.read(6).astype(np.float32)
    
    # Indices
    ndvi = (nir - red) / np.where((nir + red) == 0, 1e-6, (nir + red))
    ndwi = (green - nir) / np.where((green + nir) == 0, 1e-6, (green + nir))
    mndwi = (green - swir1) / np.where((green + swir1) == 0, 1e-6, (green + swir1))
    ndbi  = (swir1 - nir) / np.where((swir1 + nir) == 0, 1e-6, (swir1 + nir))
    
    valid = ~np.isnan(ndvi)
    
    print("\n--- Statistics over valid pixels ---")
    print(f"NDVI  min={np.nanmin(ndvi):.2f}, mean={np.nanmean(ndvi):.2f}, max={np.nanmax(ndvi):.2f}")
    print(f"NDWI  min={np.nanmin(ndwi):.2f}, mean={np.nanmean(ndwi):.2f}, max={np.nanmax(ndwi):.2f}")
    print(f"MNDWI min={np.nanmin(mndwi):.2f}, mean={np.nanmean(mndwi):.2f}, max={np.nanmax(mndwi):.2f}")
    print(f"NDBI  min={np.nanmin(ndbi):.2f}, mean={np.nanmean(ndbi):.2f}, max={np.nanmax(ndbi):.2f}")

    # Baseline: NDVI >= 0.43 & NDWI < 0.0
    base_mask = (ndvi >= 0.43) & (ndwi < 0.0) & valid
    print(f"\nBaseline (NDVI >= 0.43 & NDWI < 0.0):")
    print(f"  Pixels: {np.sum(base_mask):,}, Area: {np.sum(base_mask)*0.0001:,.2f} km² ({np.sum(base_mask)/np.sum(valid)*100:.2f}%)")
    
    # With MNDWI < 0.0 instead of NDWI
    mndwi_mask = (ndvi >= 0.43) & (mndwi < 0.0) & valid
    print(f"With MNDWI < 0.0:")
    print(f"  Pixels: {np.sum(mndwi_mask):,}, Area: {np.sum(mndwi_mask)*0.0001:,.2f} km² ({np.sum(mndwi_mask)/np.sum(valid)*100:.2f}%)")
    
    # Built-up exclusion with NDBI: forest must have NDBI < -0.05 (excluding paved roads/impervious)
    # Real forest has strongly negative NDBI because NIR >> SWIR1
    forest_no_roads = (ndvi >= 0.43) & (mndwi < 0.0) & (ndbi < -0.05) & valid
    print(f"With MNDWI < 0.0 & NDBI < -0.05 (Built-up/Road suppression):")
    print(f"  Pixels: {np.sum(forest_no_roads):,}, Area: {np.sum(forest_no_roads)*0.0001:,.2f} km² ({np.sum(forest_no_roads)/np.sum(valid)*100:.2f}%)")

