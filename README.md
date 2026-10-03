# ☀️ Sydney Urban Heat & Solar Forecaster

> **A Microclimate & Electrification Simulator for COP31 "Build for 2035"**  
> Advancing two COP31 Tracks: **Resilient Cities & Buildings** (cooling urban heat islands) and **Electrification** (rooftop solar optimization).

---

## 🌟 Overview & Capabilities

1. **Satellite Land Surface Temperature (LST) Extraction:** Streams Landsat 9 thermal infrared (Band 10, LWIR11) and optical bands across Western Sydney (Parramatta).
2. **Spectral & Canopy Feature Analytics:** Computes calibrated NDVI (Vegetation), NDBI (Built-up Density), and Broadband Albedo.
3. **Microclimate ML Regressor (XGBoost):** Learns the microclimate physics:
   $$T_{\text{surface}} = f(\text{NDVI}, \text{NDBI}, \text{Albedo})$$
   Enables real-time simulation of urban greening and cool roofs (e.g., +25% tree canopy cools surface by **2.5°C – 4.5°C**).
4. **Solar Geometry & Shading Engine:** Ingests OpenStreetMap building footprints, estimates roof geometries, and subtracts tree shadows across daylight hours (`pvlib` solar altitude/azimuth) to compute unshaded solar panel capacity ($\text{kWp}$ & $\text{MWh/year}$).
5. **DINOv3 Foundation Model Integration:** Ready for Meta's `dinov3_vitl16_chmv2` (Canopy Height Maps) and `dino.txt` (Zero-Shot Solar Segmentation), especially on high-VRAM GPUs like RTX 4090.
6. **Clean Streamlit Integration:** Exposes a high-level Python API (`HeatSolarAPI`) for frontend teammates with zero GIS headache.

---

## 📁 Repository Structure

```
sydney-heat-solar/
├── config.py                 # Central configurations (BBox, CRS, thresholds, API constants)
├── api.py                    # Clean interface consumed by Streamlit frontend
├── app.py                    # Complete interactive Streamlit dashboard
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── ingest.py             # STAC Planetary Computer Landsat 9 streaming & caching
│   └── osm.py                # OpenStreetMap building vector loader & height estimator
│
├── features/
│   ├── spectral.py           # NDVI, NDBI, Albedo, LST formulas
│   ├── dinov3_features.py    # DINOv3 SAT-493M dense patch embeddings
│   ├── dinov3_segmentation.py# dino.txt zero-shot semantic segmentation (solar/roofs)
│   └── chmv2_canopy.py       # CHMv2 canopy height predictor (1m resolution)
│
├── analysis/
│   ├── solar_geometry.py     # pvlib sun position & 2D shadow vector projection
│   ├── heatmap_model.py      # XGBoost microclimate regression & intervention engine
│   └── intervention.py       # Multi-variable scenario simulator & KPI aggregator
│
├── cache/                    # Pre-computed assets for lightning-fast demo loading
│   ├── landsat_bands.nc      # Cached satellite bands
│   ├── buildings.geojson     # OSM building footprints
│   ├── xgb_model.joblib      # Trained microclimate regressor
│   ├── heatmap_baseline.npy  # Baseline temperature grid
│   └── solar_capacity.csv    # Building solar generation metrics
│
└── scripts/
    ├── 01_download_data.py   # Ingest or synthesize Landsat & OSM footprints
    ├── 02_train_model.py     # Fit & validate XGBoost microclimate regressor
    ├── 03_cache_results.py   # Pre-render scenarios for instant UI response
    └── 04_run_dinov3_inference.py # GPU inference for CHMv2 & dino.txt (RTX 4090)
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Run Pipeline

```bash
cd sydney-heat-solar

# Step 1: Ingest Satellite and Building Footprints
python scripts/01_download_data.py

# Step 2: Train XGBoost Microclimate Model (takes ~2 seconds)
python scripts/02_train_model.py

# Step 3: Pre-compute & Cache All Demo Matrices
python scripts/03_cache_results.py

# Step 4 (Optional on GPU / RTX 4090): Run DINOv3 Canopy & Solar Segmentation
python scripts/04_run_dinov3_inference.py --mode all
```

### 2. Launch the Streamlit Web Dashboard

```bash
streamlit run app.py
```
Open `http://localhost:8501` to view the interactive map, sliders, and live thermal delta simulator.

---

## 💻 For Frontend Teammates: Consuming `api.py`

Your frontend teammate only needs `api.py`. Here is how they interact with it:

```python
from api import HeatSolarAPI

# 1. Initialize and load pre-cached data (instant startup)
api = HeatSolarAPI()
api.load_cached_data()

# 2. Get baseline heatmap coordinates & temperature array for Folium/Pydeck
lats, lons, baseline_temp = api.get_baseline_heatmap()

# 3. Simulate an intervention when user moves sliders:
result = api.simulate(
    tree_pct=25.0,        # +25% tree canopy
    solar_pct=35.0,       # 35% rooftop solar target
    cool_roof_albedo=0.05 # Cool roof coatings
)

# 4. Read KPI metrics for metric cards:
print(f"Average Cooling: -{result.avg_cooling_celsius:.2f} °C")
print(f"Max Local Spot Cooling: -{result.max_cooling_celsius:.2f} °C")
print(f"Total Solar Potential: {result.total_solar_mw:.1f} MWp")
print(f"Annual CO2 Offset: {result.co2_offset_tonnes_year:,.0f} tonnes/year")
print(f"Hotspots (>40°C) reduced from {result.hotspot_count_before} to {result.hotspot_count_after}")

# 5. Get the new forecasted temperature grid to redraw the heatmap:
new_heatmap_matrix = result.predicted_temp_map
```

---

## 🖥️ Moving to RTX 4090 Desktop (24 GB VRAM)

1. Copy the `sydney-heat-solar/` folder to your RTX 4090 desktop.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the DINOv3 inference script to take advantage of 24 GB VRAM:
   ```bash
   python scripts/04_run_dinov3_inference.py --mode all
   ```
4. Copy the resulting `cache/` files back to the laptop if presenting locally, or run the demo directly on the desktop!
