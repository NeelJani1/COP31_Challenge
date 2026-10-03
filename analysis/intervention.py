"""Intervention simulator — the core of the demo.

Takes user-controlled parameters (tree coverage %, solar panel target %)
and computes the resulting changes in temperature and energy production.
This is what makes the sliders work in the Streamlit UI.
"""
import numpy as np
import geopandas as gpd
import xarray as xr
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


@dataclass
class InterventionResult:
    """Results from running an intervention simulation."""

    # Temperature
    baseline_temp_map: np.ndarray        # 2D baseline temperatures
    predicted_temp_map: np.ndarray       # 2D after intervention
    avg_cooling_celsius: float           # Average temperature reduction
    max_cooling_celsius: float           # Maximum cooling at hottest spots
    hotspot_count_before: int            # Pixels > 40 deg C before
    hotspot_count_after: int             # Pixels > 40 deg C after

    # Solar
    total_solar_capacity_kw: float       # Total peak kW
    total_annual_kwh: float              # Annual energy production
    n_buildings_with_solar: int          # Buildings that can host panels
    total_panels: int                    # Total panel count

    # Coordinates for frontend map rendering
    lat_grid: np.ndarray                 # Latitude coordinates
    lon_grid: np.ndarray                 # Longitude coordinates

    # Summary metrics for dashboard cards
    @property
    def total_solar_mw(self) -> float:
        return self.total_solar_capacity_kw / 1000

    @property
    def cooling_summary(self) -> str:
        if self.avg_cooling_celsius >= 0:
            return f"-{self.avg_cooling_celsius:.1f}°C average"
        return f"+{abs(self.avg_cooling_celsius):.1f}°C average"

    @property
    def solar_summary(self) -> str:
        return f"{self.total_solar_mw:.1f} MW peak capacity"

    @property
    def co2_offset_tonnes_year(self) -> float:
        """Estimated CO2 offset from solar (0.79 kg CO2/kWh for NSW grid)."""
        return self.total_annual_kwh * 0.79 / 1000


def run_intervention(
    features: dict[str, xr.DataArray],
    buildings: gpd.GeoDataFrame,
    model,  # MicroclimateModel
    cfg: Config,
    tree_canopy_increase_pct: float = 20.0,
    solar_coverage_pct: float = 30.0,
    cool_roof_albedo_increase: float = 0.0,
) -> InterventionResult:
    """Run a full intervention simulation.

    This is the main function the Streamlit frontend calls when
    a user moves a slider.

    Args:
        features: Dict of xarray spectral feature arrays
        buildings: GeoDataFrame of building footprints with solar capacity
        model: Trained MicroclimateModel
        cfg: Config
        tree_canopy_increase_pct: % increase in tree coverage (0-100)
        solar_coverage_pct: % of suitable roof area to cover (0-100)
        cool_roof_albedo_increase: Albedo boost from cool roofs (0-0.3)

    Returns:
        InterventionResult with all metrics and map data
    """
    # Localized intervention physics:
    # Urban greening cools streets and corridors where trees are actually planted,
    # avoiding water bodies (Parramatta River) and areas with already mature dense canopy.
    ndvi_base = features["ndvi"].values if hasattr(features["ndvi"], "values") else np.asarray(features["ndvi"])
    ndbi_base = features["ndbi"].values if hasattr(features["ndbi"], "values") else np.asarray(features["ndbi"])

    # Water identification: NDVI < 0.05 & NDBI < -0.15 (do not plant trees in the river)
    is_water = (ndvi_base < 0.05) & (ndbi_base < -0.15)
    # Mature forest / dense existing canopy: limited headroom to add more trees
    mature_forest = np.clip((ndvi_base - 0.50) / 0.25, 0.0, 1.0)
    # Tree planting suitability (0.0 to 1.0):
    # Maximum along residential streets, asphalt heat corridors, and road verges
    tree_suitability = np.clip(1.0 - mature_forest, 0.0, 1.0) * (~is_water).astype(float)

    # Cool roof suitability: targets built-up impervious surfaces (high NDBI), zero on water
    built_suitability = np.clip((ndbi_base + 0.10) / 0.40, 0.0, 1.0) * (~is_water).astype(float)
    cool_roof_suitability = built_suitability * (~is_water).astype(float)

    # Convert percentage to localized NDVI/NDBI deltas
    delta_ndvi = (tree_canopy_increase_pct / 100.0) * 0.70 * tree_suitability
    delta_ndbi_tree = -(tree_canopy_increase_pct / 100.0) * 0.35 * tree_suitability
    delta_ndbi_cool = -0.15 * cool_roof_albedo_increase * cool_roof_suitability
    delta_ndbi = delta_ndbi_tree + delta_ndbi_cool
    delta_albedo = cool_roof_albedo_increase * cool_roof_suitability

    # Run temperature simulation
    baseline_temp, predicted_temp, avg_cooling = model.simulate_intervention(
        features,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi,
        delta_albedo=delta_albedo,
    )

    # Hotspot analysis (>40 deg C threshold)
    hotspot_threshold = 40.0
    hotspots_before = int(np.nansum(baseline_temp > hotspot_threshold))
    hotspots_after = int(np.nansum(predicted_temp > hotspot_threshold))

    # Solar capacity calculation
    solar_factor = float(np.clip(solar_coverage_pct / 100.0, 0.0, 1.0))
    if buildings is None or len(buildings) == 0 or solar_factor <= 0.0:
        total_kw = 0.0
        total_kwh = 0.0
        n_solar_buildings = 0
        total_panels = 0
    elif "peak_kw" in buildings.columns:
        total_kw = float(buildings["peak_kw"].sum() * solar_factor)
        total_kwh = float(buildings["annual_kwh"].sum() * solar_factor)
        panels_per_bldg = (buildings["n_panels"] * solar_factor).astype(int)
        total_panels = int(panels_per_bldg.sum())
        n_solar_buildings = int((panels_per_bldg > 0).sum())
    elif "roof_area_m2" in buildings.columns:
        # Fallback: estimate from roof area
        total_roof = float(buildings["roof_area_m2"].sum())
        usable = total_roof * 0.7 * solar_factor  # 70% usable
        usable_per_bldg = buildings["roof_area_m2"] * 0.7 * solar_factor
        panels_per_bldg = (usable_per_bldg / cfg.SOLAR_PANEL_AREA_M2).astype(int)
        total_panels = int(panels_per_bldg.sum())
        total_kw = total_panels * cfg.SOLAR_PANEL_WATT / 1000
        total_kwh = total_kw * cfg.SUN_HOURS_PER_DAY * 365
        n_solar_buildings = int((panels_per_bldg > 0).sum())
    else:
        total_kw = 0.0
        total_kwh = 0.0
        n_solar_buildings = 0
        total_panels = 0

    # Extract coordinate grids for map rendering
    lst = features["lst_celsius"]
    if hasattr(lst, "y") and hasattr(lst, "x"):
        lat_grid = lst.y.values
        lon_grid = lst.x.values
    elif hasattr(lst, "latitude") and hasattr(lst, "longitude"):
        lat_grid = lst.latitude.values
        lon_grid = lst.longitude.values
    else:
        h, w = baseline_temp.shape[-2:]
        lat_grid = np.linspace(cfg.BBOX[3], cfg.BBOX[1], h)
        lon_grid = np.linspace(cfg.BBOX[0], cfg.BBOX[2], w)

    valid_diff = (baseline_temp - predicted_temp)[~np.isnan(baseline_temp - predicted_temp)]
    max_cooling = float(np.max(valid_diff)) if len(valid_diff) > 0 else 0.0

    return InterventionResult(
        baseline_temp_map=np.atleast_2d(np.squeeze(baseline_temp)),
        predicted_temp_map=np.atleast_2d(np.squeeze(predicted_temp)),
        avg_cooling_celsius=avg_cooling,
        max_cooling_celsius=max_cooling,
        hotspot_count_before=hotspots_before,
        hotspot_count_after=hotspots_after,
        total_solar_capacity_kw=total_kw,
        total_annual_kwh=total_kwh,
        n_buildings_with_solar=n_solar_buildings,
        total_panels=total_panels,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )
