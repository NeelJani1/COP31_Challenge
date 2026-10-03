"""Clean API for the Streamlit frontend.

Usage in Streamlit:
    from api import HeatSolarAPI

    api = HeatSolarAPI()
    api.load_cached_data()  # Load pre-computed data

    # When user moves a slider:
    result = api.simulate(tree_pct=20, solar_pct=30)
    result.avg_cooling_celsius  # -> -2.1
    result.total_solar_mw       # -> 14.2
    result.baseline_temp_map    # -> 2D numpy array for heatmap
    result.predicted_temp_map   # -> 2D numpy array for new heatmap
"""
import numpy as np
import geopandas as gpd
import xarray as xr
from pathlib import Path

from config import Config
from analysis.heatmap_model import MicroclimateModel
from analysis.intervention import InterventionResult, run_intervention


class HeatSolarAPI:
    """Main API for the Sydney Urban Heat & Solar Forecaster.

    Designed for your Streamlit teammate to consume.
    Two modes:
    1. Live mode: Downloads satellite data + runs full pipeline
    2. Cached mode: Loads pre-computed results (for demo speed)
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.features: dict[str, xr.DataArray] = {}
        self.buildings: gpd.GeoDataFrame = None
        self.model: MicroclimateModel = None
        self._loaded = False

    # === Data Loading ===

    def load_cached_data(self) -> "HeatSolarAPI":
        """Load all pre-computed data from cache. Fast startup for demo."""
        from data.osm import load_cached_buildings
        from data.ingest import load_cached_bands
        from features.spectral import compute_all_features

        # Load satellite bands
        data = load_cached_bands(self.cfg)
        self.features = compute_all_features(data, self.cfg)

        # Load buildings with solar capacity
        self.buildings = load_cached_buildings(self.cfg)

        # Compute solar capacity if not already present
        if "peak_kw" not in self.buildings.columns:
            from analysis.solar_geometry import compute_simple_solar_capacity
            self.buildings = compute_simple_solar_capacity(
                self.buildings,
                self.features["ndvi"].values.squeeze(),
                self.cfg,
            )

        # Load trained model
        self.model = MicroclimateModel(self.cfg)
        self.model.load()

        self._loaded = True
        return self

    def load_live_data(self) -> "HeatSolarAPI":
        """Download fresh satellite data and train model (requires internet)."""
        from data.ingest import search_landsat, load_bands, cache_bands
        from data.osm import load_buildings, cache_buildings
        from features.spectral import compute_all_features
        from analysis.solar_geometry import compute_simple_solar_capacity

        # Download satellite data
        print("[API] Searching for Landsat 9 scenes...")
        items = search_landsat(self.cfg)
        print(f"[API] Found {len(items)} scenes, loading bands...")
        data = load_bands(self.cfg, items)
        cache_bands(self.cfg, data)

        # Compute features
        print("[API] Computing spectral features...")
        self.features = compute_all_features(data, self.cfg)

        # Download buildings
        print("[API] Downloading building footprints...")
        self.buildings = load_buildings(self.cfg)
        cache_buildings(self.cfg, self.buildings)

        # Compute solar capacity
        self.buildings = compute_simple_solar_capacity(
            self.buildings,
            self.features["ndvi"].values.squeeze(),
            self.cfg,
        )

        # Train model
        print("[API] Training XGBoost microclimate model...")
        self.model = MicroclimateModel(self.cfg)
        metrics = self.model.train(self.features)
        self.model.save()
        print(f"[API] Model trained: MAE={metrics['mae']:.2f}°C, R²={metrics['r2']:.3f}")

        self._loaded = True
        return self

    # === Core Simulation ===

    def simulate(
        self,
        tree_pct: float = 20.0,
        solar_pct: float = 30.0,
        cool_roof_albedo: float = 0.0,
    ) -> InterventionResult:
        """Run intervention simulation.

        This is the main function called when the user moves a slider.

        Args:
            tree_pct: Target tree canopy increase (0-100%)
            solar_pct: Rooftop solar coverage target (0-100%)
            cool_roof_albedo: Cool roof albedo increase (0-0.3)

        Returns:
            InterventionResult with all metrics and map data
        """
        if not self._loaded:
            raise RuntimeError(
                "Call load_cached_data() or load_live_data() first."
            )

        return run_intervention(
            features=self.features,
            buildings=self.buildings,
            model=self.model,
            cfg=self.cfg,
            tree_canopy_increase_pct=tree_pct,
            solar_coverage_pct=solar_pct,
            cool_roof_albedo_increase=cool_roof_albedo,
        )

    # === Convenience Getters for Frontend ===

    def get_baseline_heatmap(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the baseline temperature heatmap.

        Returns:
            (lat_grid, lon_grid, temp_2d) for Folium rendering
        """
        lst = self.features["lst_celsius"]
        temp = lst.values.squeeze()
        if hasattr(lst, "y") and hasattr(lst, "x"):
            return lst.y.values, lst.x.values, temp
        h, w = temp.shape[-2:]
        lats = np.linspace(self.cfg.BBOX[3], self.cfg.BBOX[1], h)
        lons = np.linspace(self.cfg.BBOX[0], self.cfg.BBOX[2], w)
        return lats, lons, temp

    def get_buildings_geojson(self) -> dict:
        """Get buildings as GeoJSON dict for Folium rendering."""
        if self.buildings is None:
            raise RuntimeError("Buildings not loaded.")
        return self.buildings.__geo_interface__

    def get_ndvi_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get NDVI vegetation map for overlay visualization."""
        ndvi = self.features["ndvi"]
        arr = ndvi.values.squeeze()
        if hasattr(ndvi, "y") and hasattr(ndvi, "x"):
            return ndvi.y.values, ndvi.x.values, arr
        h, w = arr.shape[-2:]
        lats = np.linspace(self.cfg.BBOX[3], self.cfg.BBOX[1], h)
        lons = np.linspace(self.cfg.BBOX[0], self.cfg.BBOX[2], w)
        return lats, lons, arr

    def get_model_metrics(self) -> dict:
        """Get XGBoost model performance metrics."""
        if self.model is None:
            return {}
        return self.model.metrics

    def train_model(self) -> dict:
        """Train (or retrain) the XGBoost model on current data."""
        self.model = MicroclimateModel(self.cfg)
        metrics = self.model.train(self.features)
        self.model.save()
        return metrics
