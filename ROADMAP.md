# Deoghar Forest Cover Analysis — Project Roadmap

## Phase 0: Setup & Data Acquisition (Week 1–2)

- [ ] Set up Python virtual environment and install dependencies
- [ ] Sign up for / verify Google Earth Engine access
- [ ] Obtain Deoghar district boundary GeoJSON from:
  - Bhuvan (NRSC): https://bhuvan-app1.nrsc.gov.in
  - data.gov.in: India Administrative Boundary
  - GADM: https://gadm.org/download_country.html (India, Level 2)
- [ ] Save boundary to `data/aoi/deoghar.geojson`
- [ ] Create GEE training polygons asset for 6 LULC classes
  (digitise ~50 polygons/class from high-resolution imagery)
- [ ] Export Sentinel-2 annual composites 2015–2025:
  `python scripts/gee_export_sentinel2.py --years 2015 2025`
- [ ] Export Landsat-8/9 composites 2013–2025 (for pre-Sentinel baseline):
  `python scripts/gee_export_landsat.py --years 2013 2025`
- [ ] Export LULC labels via RF classifier:
  `python scripts/gee_export_labels.py --years 2015 2025 --training-asset <asset-path>`
- [ ] Export DEM + slope static raster for UTM Zone 45N AOI
- [ ] Sync all GeoTIFFs from Google Drive to `data/raw/`

## Phase 1: Preprocessing & EDA (Week 2–3)

- [ ] Run preprocessing pipeline:
  `python scripts/preprocess.py --config configs/data.yaml`
- [ ] Open `notebooks/01_gee_exploration.ipynb` — explore band histograms,
  cloud cover stats, seasonal variability
- [ ] Open `notebooks/02_eda_and_ndvi_trends.ipynb` — time-series NDVI,
  NBR, class distribution, per-year forest area estimates
- [ ] Verify tile quality: check for NaN tiles, label class balance
- [ ] Compute and save class frequency counts for weighted loss

## Phase 2: Baseline Models (Week 3–4)

- [ ] Run CA-Markov baseline:
  ```python
  from src.models.baselines import CAMarkov
  model = CAMarkov(n_classes=6).fit(labels_stack)
  pred_2026 = model.predict(labels_stack[-1])
  ```
- [ ] Run Linear Extrapolation baseline
- [ ] Run Random Forest baseline
- [ ] Evaluate all baselines on test tiles (mIoU, OA, Kappa, forest IoU)
- [ ] Save baseline confusion matrices to `outputs/figures/`

## Phase 3: Deep Learning — ConvLSTM (Week 4–5)

- [ ] Debug run (local GPU):
  `python scripts/train.py --config configs/model_convlstm.yaml --debug --device cuda`
- [ ] Full training (Colab/Kaggle T4 recommended):
  `python scripts/train.py --config configs/model_convlstm.yaml --device cuda`
- [ ] Monitor with TensorBoard:
  `tensorboard --logdir outputs/runs`
- [ ] Evaluate:
  `python scripts/evaluate.py --checkpoint outputs/checkpoints/convlstm_v1_best.pt --config configs/model_convlstm.yaml`

## Phase 4: Deep Learning — U-Net + ConvLSTM (Week 5–6)

- [ ] Full training:
  `python scripts/train.py --config configs/model_unet_convlstm.yaml --device cuda`
- [ ] Evaluate on test set; compare against ConvLSTM and baselines
- [ ] Tune hyperparameters if forest IoU < 0.60

## Phase 5: Future Prediction 2026–2035 (Week 6–7)

- [ ] Predict with best checkpoint:
  ```bash
  python scripts/predict_future.py \
      --checkpoint outputs/checkpoints/unet_convlstm_v1_best.pt \
      --config configs/model_unet_convlstm.yaml \
      --years 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
      --mc-dropout-passes 20
  ```
- [ ] Mosaic tile-level predictions into full-district rasters
- [ ] Generate forest change maps (loss/gain hotspots) per year
- [ ] Compute forest area trend curve 2015–2035

## Phase 6: Visualization & Reporting (Week 7–8)

- [ ] Open `notebooks/03_results_visualization.ipynb`
- [ ] Generate before-after maps (2015 vs 2030, 2015 vs 2035)
- [ ] Generate transition probability matrix
- [ ] Generate uncertainty maps (MC Dropout entropy)
- [ ] Plot NDVI trend time series for major forest patches
- [ ] Write final report comparing all models
- [ ] Identify deforestation hotspots for policy recommendations

---

## Key Decision Points

| Decision | Options | Recommended |
|---|---|---|
| Primary satellite | Sentinel-2 vs Landsat | Sentinel-2 (10m res) |
| Model | ConvLSTM vs UNet+ConvLSTM | UNet+ConvLSTM (best mIoU) |
| Compositing | Annual vs bi-seasonal | Annual (simpler, less cloud) |
| Uncertainty | None vs MC Dropout | MC Dropout (20 passes) |
| Training compute | Local vs Colab | Colab T4 for full training |

---

## Data Volume Estimate

| Item | Size estimate |
|---|---|
| Sentinel-2 GeoTIFFs (11 years × 9 bands) | ~2–4 GB |
| Landsat GeoTIFFs (13 years × 9 bands) | ~500 MB |
| Label GeoTIFFs (11 years) | ~100 MB |
| Processed .npz tiles | ~1–3 GB |
| Model checkpoints | ~50 MB |
