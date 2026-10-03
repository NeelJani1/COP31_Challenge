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

    def load_cached_data(self, allow_fallback: bool = False) -> "HeatSolarAPI":
        """Load all pre-computed data from cache. Fast startup for demo."""
        from data.osm import load_cached_buildings
        from data.ingest import load_cached_bands
        from features.spectral import compute_all_features

        # Load satellite bands
        try:
            data = load_cached_bands(self.cfg, allow_mock=allow_fallback)
        except FileNotFoundError:
            if allow_fallback:
                return self.load_live_data(allow_fallback=True)
            raise
        self.features = compute_all_features(data, self.cfg)

        # Load buildings with solar capacity
        try:
            self.buildings = load_cached_buildings(self.cfg, allow_mock=allow_fallback)
        except FileNotFoundError:
            if allow_fallback:
                from data.osm import create_synthetic_buildings, cache_buildings
                self.buildings = create_synthetic_buildings(self.cfg)
                cache_buildings(self.cfg, self.buildings)
            else:
                raise

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
        try:
            self.model.load()
        except FileNotFoundError:
            if allow_fallback:
                self.model.train(self.features)
                self.model.save()
            else:
                raise

        self._loaded = True
        return self

    def load_live_data(self, allow_fallback: bool = True) -> "HeatSolarAPI":
        """Download fresh satellite data and train model (requires internet)."""
        from data.ingest import search_landsat, load_bands, cache_bands, create_synthetic_landsat
        from data.osm import load_buildings, cache_buildings, create_synthetic_buildings
        from features.spectral import compute_all_features
        from analysis.solar_geometry import compute_simple_solar_capacity

        data = None
        try:
            # Download satellite data
            print("[API] Searching for Landsat 9 scenes...")
            items = search_landsat(self.cfg)
            if not items:
                raise RuntimeError("No Landsat scenes found.")
            print(f"[API] Found {len(items)} scenes, loading bands...")
            data = load_bands(self.cfg, items)
            cache_bands(self.cfg, data)
        except Exception as e:
            if allow_fallback:
                print(f"[API] Live satellite download failed ({e}), generating synthetic data...")
                data = create_synthetic_landsat(self.cfg)
                cache_bands(self.cfg, data)
            else:
                raise

        # Compute features
        print("[API] Computing spectral features...")
        self.features = compute_all_features(data, self.cfg)

        # Download buildings
        try:
            print("[API] Downloading building footprints...")
            self.buildings = load_buildings(self.cfg)
            cache_buildings(self.cfg, self.buildings)
        except Exception as e:
            if allow_fallback:
                print(f"[API] OSM download failed ({e}), generating synthetic buildings...")
                self.buildings = create_synthetic_buildings(self.cfg)
                cache_buildings(self.cfg, self.buildings)
            else:
                raise

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

    def _extract_coords_and_array(
        self, data_array: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract lat, lon, and 2D numpy array from an xarray DataArray."""
        arr = np.atleast_2d(np.squeeze(data_array.values))
        if hasattr(data_array, "y") and hasattr(data_array, "x"):
            return np.asarray(data_array.y.values), np.asarray(data_array.x.values), arr
        elif hasattr(data_array, "latitude") and hasattr(data_array, "longitude"):
            return np.asarray(data_array.latitude.values), np.asarray(data_array.longitude.values), arr
        elif hasattr(data_array, "lat") and hasattr(data_array, "lon"):
            return np.asarray(data_array.lat.values), np.asarray(data_array.lon.values), arr
        h, w = arr.shape[-2:]
        lats = np.linspace(self.cfg.BBOX[3], self.cfg.BBOX[1], h)
        lons = np.linspace(self.cfg.BBOX[0], self.cfg.BBOX[2], w)
        return lats, lons, arr

    def get_baseline_heatmap(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the baseline temperature heatmap.

        Returns:
            (lat_grid, lon_grid, temp_2d) for Folium rendering
        """
        if "lst_celsius" not in self.features:
            raise RuntimeError("Data not loaded. Call load_cached_data() first.")
        return self._extract_coords_and_array(self.features["lst_celsius"])

    def get_buildings_geojson(self) -> dict:
        """Get buildings as GeoJSON dict for Folium rendering."""
        if self.buildings is None:
            raise RuntimeError("Buildings not loaded.")
        if self.buildings.empty or "geometry" not in self.buildings.columns:
            return {"type": "FeatureCollection", "features": []}
        cols_to_keep = [
            "geometry",
            "roof_area_m2",
            "height_m",
            "building",
            "unshaded_area_m2",
            "peak_kw",
            "annual_kwh",
            "n_panels",
        ]
        cols_present = [c for c in cols_to_keep if c in self.buildings.columns]
        valid_bldgs = self.buildings[cols_present]
        valid_bldgs = valid_bldgs[
            valid_bldgs.geometry.notna() & (~valid_bldgs.geometry.is_empty)
        ]
        if valid_bldgs.empty:
            return {"type": "FeatureCollection", "features": []}
        return valid_bldgs.__geo_interface__

    def get_ndvi_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get NDVI vegetation map for overlay visualization."""
        if "ndvi" not in self.features:
            raise RuntimeError("Data not loaded. Call load_cached_data() first.")
        return self._extract_coords_and_array(self.features["ndvi"])

    def get_canopy_heights(self) -> tuple[np.ndarray, np.ndarray]:
        """Get precomputed or computed canopy heights and tree mask."""
        height_path = Path(self.cfg.CACHE_DIR) / "canopy_heights.npy"
        mask_path = Path(self.cfg.CACHE_DIR) / "tree_mask.npy"
        temp_shape = None
        if "lst_celsius" in self.features:
            temp_shape = np.atleast_2d(
                np.squeeze(self.features["lst_celsius"].values)
            ).shape

        if height_path.exists() and mask_path.exists():
            h_arr = np.load(str(height_path))
            m_arr = np.load(str(mask_path))
            if temp_shape is None or h_arr.shape == temp_shape:
                return h_arr, m_arr

        from features.chmv2_canopy import CHMv2Predictor

        if "ndvi" in self.features:
            heights = CHMv2Predictor.predict_canopy_height_from_ndvi(
                self.features["ndvi"].values.squeeze()
            )
            tree_mask = heights > 3.0
            return heights, tree_mask
        raise RuntimeError("No canopy height data available.")

    def get_solar_mask(self) -> np.ndarray:
        """Get existing solar panel mask from cache or spectral estimation."""
        mask_path = Path(self.cfg.CACHE_DIR) / "solar_panel_mask.npy"
        temp_shape = None
        if "lst_celsius" in self.features:
            temp_shape = np.atleast_2d(
                np.squeeze(self.features["lst_celsius"].values)
            ).shape

        if mask_path.exists():
            s_arr = np.load(str(mask_path))
            if temp_shape is None or s_arr.shape == temp_shape:
                return s_arr

        if "ndbi" in self.features and "ndvi" in self.features:
            from features.dinov3_segmentation import ZeroShotSegmentor

            res = ZeroShotSegmentor.detect_spectral_fallback(
                self.features["ndvi"].values.squeeze(),
                self.features["ndbi"].values.squeeze(),
            )
            return res["solar_mask"]
        raise RuntimeError("No solar mask data available.")

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
