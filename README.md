# Deoghar Forest Cover Change Detection & Prediction

**Deoghar, Jharkhand, 2015–2035** — ConvLSTM and U-Net + ConvLSTM models trained on multitemporal
Sentinel-2 and Landsat-8/9 imagery to detect historical forest cover change and forecast
forest-area distribution through 2035, with CA-Markov and Random Forest baselines.

> 📊 Adapted from [ShivekRanjan/bengaluru-lulc-forecast-](https://github.com/ShivekRanjan/bengaluru-lulc-forecast-)
> (Bengaluru LULC Forecast) with modifications for forest-focused analysis in Jharkhand.

---

## Objective

1. **Detection** — Quantify year-by-year forest cover change in Deoghar district from 2015 to present.
2. **Attribution** — Identify hotspots of deforestation vs. afforestation/regrowth.
3. **Prediction** — Forecast spatial forest cover distribution for 2026–2035.

---

## Study Area

| Property | Value |
|---|---|
| District | Deoghar, Jharkhand |
| State | Jharkhand, India |
| Approx. AOI | Lon 86.55°–87.10° E, Lat 24.20°–24.65° N |
| CRS | EPSG:32645 (UTM Zone 45N) |
| Primary biome | Sal (Shorea robusta) forest, dry deciduous forest |

---

## LULC Classes (Forest-Focused)

| ID | Class | Description |
|----|-------------|------------------------------------------|
| 0 | water | Rivers, reservoirs, seasonal water bodies |
| 1 | dense_forest | Canopy cover > 40% (primarily Sal forest)|
| 2 | open_forest | Canopy cover 10–40%, scrub, regrowth |
| 3 | agriculture | Cropland, paddy, horticulture |
| 4 | built_up | Settlements, roads, urban |
| 5 | barren | Rocky outcrops, degraded land, mining |

---

## Project Structure

```
.
├── configs/                   # YAML configs for data, models, training
│   ├── data.yaml
│   ├── model_convlstm.yaml
│   └── model_unet_convlstm.yaml
├── data/
│   ├── aoi/                   # Deoghar AOI GeoJSON / shapefile
│   ├── raw/                   # GEE exports (untouched TIFs)
│   ├── interim/               # Cloud-masked, composited rasters
│   └── processed/             # Tiled .npz tensors (T, C, H, W) + labels
├── notebooks/                 # Exploration, GEE prep, results viz
│   ├── 01_gee_exploration.ipynb
│   ├── 02_eda_and_ndvi_trends.ipynb
│   └── 03_results_visualization.ipynb
├── scripts/                   # CLI entry points
│   ├── gee_export_sentinel2.py
│   ├── gee_export_landsat.py
│   ├── gee_export_labels.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_future.py
├── src/
│   ├── data/                  # Dataset, transforms, tiling
│   ├── models/                # ConvLSTM, UNetConvLSTM, baselines
│   ├── training/              # Trainer, losses
│   ├── evaluation/            # Metrics (IoU, F1, Kappa, FuzzyKappa)
│   ├── visualization/         # Maps, animations, NDVI trends
│   └── utils/                 # Config loading, logging, geo helpers
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── figures/
├── tests/
├── requirements.txt
└── ROADMAP.md
```

---

## Quick Start

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine

```bash
earthengine authenticate
# Set your GEE Cloud project:
set GEE_PROJECT=your-gcp-project-id   # Windows
# export GEE_PROJECT=your-gcp-project-id   # Linux/Mac
```

### 3. Export imagery from GEE

Place `data/aoi/deoghar.geojson` (Deoghar district boundary), then:

```bash
python scripts/gee_export_sentinel2.py --years 2015 2025 --out data/raw/s2
python scripts/gee_export_landsat.py   --years 2015 2025 --out data/raw/landsat
python scripts/gee_export_labels.py    --years 2015 2025 --out data/raw/labels
```

Exports go to your Google Drive. Sync back into `data/raw/`.

### 4. Preprocess

```bash
python scripts/preprocess.py --config configs/data.yaml
```

Produces tiled `.npz` patches in `data/processed/`.

### 5. Train

```bash
# Debug run (small tiles, batch=1) — fits local GPU with ≥4GB VRAM
python scripts/train.py --config configs/model_convlstm.yaml --device cuda --debug

# Full training (recommended on Colab/Kaggle T4)
python scripts/train.py --config configs/model_unet_convlstm.yaml --device cuda
```

### 6. Predict 2026–2035

```bash
python scripts/predict_future.py \
    --checkpoint outputs/checkpoints/unet_convlstm_v1_best.pt \
    --config configs/model_unet_convlstm.yaml \
    --years 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
    --mc-dropout-passes 20 \
    --out outputs/predictions/
```

### 7. Evaluate (back-test against held-out years)

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/unet_convlstm_v1_best.pt \
    --config configs/model_unet_convlstm.yaml
```

---

## Key Differences from Reference (Bengaluru LULC)

| Aspect | Bengaluru Project | Deoghar Project |
|---|---|---|
| Focus | General 5-class LULC | Forest-focused 6-class (dense + open forest) |
| AOI | Bengaluru Metro (urban) | Deoghar district (rural/forest) |
| CRS | EPSG:32643 (UTM 43N) | EPSG:32645 (UTM 45N) |
| Sentinel-2 CRS | UTM 43N | UTM 45N |
| Classes | water, veg×2, built-up, barren | water, forest×2, agriculture, built-up, barren |
| Key index | NDBI (urbanisation) | NDVI + NBR (forest health & burn) |
| Prediction horizon | 2030 | 2035 |
| Monsoon handling | Standard | Bi-seasonal compositing (Rabi + Kharif) |

---

## Compute Strategy

| Stage | Recommended environment |
|---|---|
| GEE export, preprocessing | Local PC (8GB RAM+) |
| Debug training | Local PC (GTX 1650 / RTX 3050, batch=1, patch=64) |
| Full training | Google Colab Pro / Kaggle (T4 16GB) |
| Evaluation, visualization | Local PC |

---

## References

- Shi et al. (2015) — ConvLSTM: Convolutional LSTM for spatiotemporal prediction
- Lin et al. (2017) — Focal Loss for dense object detection
- ShivekRanjan et al. — [bengaluru-lulc-forecast](https://github.com/ShivekRanjan/bengaluru-lulc-forecast-) (reference codebase)
- FSI (Forest Survey of India) — India State of Forest Report 2023
- Pontius & Millones (2011) — Death to kappa
