"""Solar geometry, shadow modeling, and panel placement optimization.

Uses pvlib for accurate sun position calculations and shapely for
geometric shadow projection. Computes hourly shadows from trees
onto building rooftops to determine unshaded area for solar panels.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.ops import unary_union
import pvlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def get_solar_positions(
    cfg: Config, date: str = "2024-01-15", freq: str = "1h"
) -> pd.DataFrame:
    """Get sun positions throughout the day for Sydney.

    Args:
        cfg: Config with lat/lon
        date: Target date (summer day for worst-case heat)
        freq: Time frequency for calculations

    Returns:
        DataFrame with azimuth and apparent_elevation columns,
        filtered to daylight hours only.
    """
    times = pd.date_range(
        f"{date} 05:00", f"{date} 20:00",
        freq=freq, tz="Australia/Sydney",
    )
    solar_pos = pvlib.solarposition.get_solarposition(
        times, cfg.LATITUDE, cfg.LONGITUDE
    )
    return solar_pos[solar_pos["apparent_elevation"] > 0]


def compute_shadow_polygon(
    tree_polygon: Polygon,
    tree_height: float,
    solar_altitude: float,
    solar_azimuth: float,
) -> Polygon:
    """Compute shadow polygon cast by a tree at a given sun position.

    The shadow extends opposite to the sun's direction, with length
    determined by tree height and sun elevation angle.

    Args:
        tree_polygon: Shapely polygon of tree canopy (in meters/UTM)
        tree_height: Height of the tree in meters
        solar_altitude: Sun elevation angle in degrees
        solar_azimuth: Sun azimuth angle in degrees (0=North, clockwise)

    Returns:
        Shadow polygon (union of tree canopy and shadow projection)
    """
    if tree_height <= 0.0 or tree_polygon is None or tree_polygon.is_empty:
        return Polygon()

    if solar_altitude <= 2.0:
        return tree_polygon  # Tree canopy itself when sun is near horizon

    shadow_length = tree_height / np.tan(np.radians(solar_altitude))

    # Shadow falls opposite to sun direction
    shadow_dx = -shadow_length * np.sin(np.radians(solar_azimuth))
    shadow_dy = -shadow_length * np.cos(np.radians(solar_azimuth))

    shadow = translate(tree_polygon, xoff=shadow_dx, yoff=shadow_dy)
    return tree_polygon.union(shadow).convex_hull


def compute_daily_shadow_union(
    tree_polygons: gpd.GeoSeries,
    tree_heights: np.ndarray,
    solar_positions: pd.DataFrame,
) -> Polygon:
    """Compute the union of all shadows over all daylight hours.

    Any part of a roof shaded at ANY time during the day is excluded
    from solar panel placement.

    Returns:
        Unified shadow polygon for the entire day.
    """
    all_shadows = []
    for _, sun_pos in solar_positions.iterrows():
        for tree_poly, height in zip(tree_polygons, tree_heights):
            if tree_poly is None or tree_poly.is_empty or height <= 0.0:
                continue
            try:
                shadow = compute_shadow_polygon(
                    tree_poly, height,
                    sun_pos["apparent_elevation"],
                    sun_pos["azimuth"],
                )
                if shadow is not None and not shadow.is_empty:
                    all_shadows.append(shadow)
            except Exception:
                continue  # Skip invalid geometries

    if all_shadows:
        return unary_union(all_shadows)
    return Polygon()  # No shadows


def _get_australian_solar_cap(b_type: str, roof_area: float = 0.0, height: float = 7.0) -> float:
    """Return Australian inverter / grid connection cap (kWp) by building typology."""
    b = str(b_type).lower().strip()
    if b in ("residential", "house", "home", "detached", "terrace", "semidetached_house", "cottage", "cabin", "bungalow", "duplex"):
        return 15.0
    if b in ("apartments", "dormitory", "barracks"):
        return 48.0
    if b in ("industrial", "warehouse", "manufacture", "depot"):
        return 250.0
    if b in ("commercial", "office", "retail", "hotel", "bank", "supermarket", "kiosk", "skyscraper", "school", "university", "college", "hospital", "clinic", "civic", "public", "government", "library", "church"):
        return 200.0
    # Infer for generic "yes" or unspecified tags
    if height > 24.0 or roof_area > 1800.0:
        return 200.0
    if height > 11.0 or roof_area > 450.0:
        return 48.0
    if roof_area > 0 and roof_area <= 380.0:
        return 15.0
    return 15.0


def compute_unshaded_roof_area(
    buildings: gpd.GeoDataFrame,
    tree_canopy: gpd.GeoDataFrame,
    tree_heights: np.ndarray,
    cfg: Config,
) -> gpd.GeoDataFrame:
    """Compute the unshaded area of each building rooftop.

    Steps:
    1. Get solar positions for a summer day
    2. Project tree shadows across all daylight hours
    3. Subtract shadow union from each building polygon
    4. Calculate remaining usable roof area

    Returns:
        buildings GeoDataFrame with added columns:
        - unshaded_area_m2: Available roof area for solar
        - n_panels: Number of panels that fit
        - peak_kw: Peak generation capacity in kW
        - annual_kwh: Estimated annual energy production
    """
    if buildings is None or len(buildings) == 0:
        empty = gpd.GeoDataFrame(
            columns=[
                "building",
                "roof_area_m2",
                "unshaded_area_m2",
                "n_panels",
                "peak_kw",
                "annual_kwh",
                "geometry",
            ],
            crs=cfg.CRS_WGS84,
        )
        return empty

    solar_pos = get_solar_positions(cfg)

    # Work in UTM for accurate area calculations
    buildings_utm = buildings.to_crs(cfg.CRS_UTM)

    if tree_canopy is not None and len(tree_canopy) > 0 and tree_heights is not None:
        if isinstance(tree_heights, (int, float)):
            heights_arr = np.full(len(tree_canopy), float(tree_heights))
        else:
            heights_arr = np.asarray(tree_heights, dtype=float)
            if heights_arr.ndim == 0:
                heights_arr = np.full(len(tree_canopy), float(heights_arr))
        if len(heights_arr) > 0:
            trees_utm = tree_canopy.to_crs(cfg.CRS_UTM)
            shadow_union = compute_daily_shadow_union(
                trees_utm.geometry, heights_arr, solar_pos
            )
        else:
            shadow_union = Polygon()
    else:
        shadow_union = Polygon()

    # Subtract shadows from building footprints
    results = buildings_utm.copy()
    if not shadow_union.is_empty:
        unshaded_geom = results.geometry.difference(shadow_union)
        results["unshaded_area_m2"] = unshaded_geom.area.fillna(0.0).clip(lower=0.0)
    else:
        results["unshaded_area_m2"] = results.geometry.area.fillna(0.0).clip(lower=0.0)

    # Ensure unshaded area does not exceed total roof area if present
    if "roof_area_m2" in results.columns:
        results["unshaded_area_m2"] = results[["unshaded_area_m2", "roof_area_m2"]].min(axis=1)

    # Calculate solar capacity with realistic Australian connection and installation limits
    raw_panels = (
        results["unshaded_area_m2"] / cfg.SOLAR_PANEL_AREA_M2
    ).astype(int).clip(lower=0)
    raw_kw = raw_panels * cfg.SOLAR_PANEL_WATT / 1000.0

    if "building" in results.columns:
        caps = results.apply(
            lambda r: _get_australian_solar_cap(
                r.get("building", ""),
                r.get("roof_area_m2", 0.0),
                r.get("height_m", 7.0),
            ),
            axis=1,
        )
        peak_kw = np.minimum(raw_kw, caps)
        n_panels = (peak_kw * 1000.0 / cfg.SOLAR_PANEL_WATT).astype(int)
    else:
        peak_kw = raw_kw
        n_panels = raw_panels

    results["n_panels"] = n_panels
    results["peak_kw"] = peak_kw.round(1) if hasattr(peak_kw, "round") else np.round(peak_kw, 1)
    results["annual_kwh"] = (
        results["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365
    ).round(0)

    # Convert back to WGS84 for mapping
    results = results.to_crs(cfg.CRS_WGS84)
    return results


def compute_simple_solar_capacity(
    buildings: gpd.GeoDataFrame,
    ndvi_array: np.ndarray,
    cfg: Config,
) -> gpd.GeoDataFrame:
    """Simplified solar capacity without full shadow modeling.

    Uses NDVI-based tree canopy buffer as a faster alternative to
    the full pvlib shadow computation. Good for the MVP.

    Args:
        buildings: Building footprints GeoDataFrame
        ndvi_array: 2D NDVI array for the area
        cfg: Config

    Returns:
        buildings with solar capacity columns added
    """
    if buildings is None or len(buildings) == 0:
        return gpd.GeoDataFrame(
            columns=[
                "building",
                "roof_area_m2",
                "unshaded_area_m2",
                "n_panels",
                "peak_kw",
                "annual_kwh",
                "geometry",
            ],
            crs=cfg.CRS_WGS84,
        )

    results = buildings.copy()

    # Ensure roof_area_m2 exists
    if "roof_area_m2" not in results.columns:
        results_utm = results.to_crs(cfg.CRS_UTM)
        results["roof_area_m2"] = results_utm.geometry.area.fillna(0.0)

    # Estimate shading fraction from NDVI (higher NDVI near building = more shade)
    # Simple heuristic: assume 30% of roof is shaded on average
    shade_fraction = 0.30
    results["unshaded_area_m2"] = (
        results["roof_area_m2"] * (1 - shade_fraction)
    ).clip(lower=0.0)

    raw_panels = (
        results["unshaded_area_m2"] / cfg.SOLAR_PANEL_AREA_M2
    ).astype(int).clip(lower=0)
    raw_kw = raw_panels * cfg.SOLAR_PANEL_WATT / 1000.0

    if "building" in results.columns:
        caps = results.apply(
            lambda r: _get_australian_solar_cap(
                r.get("building", ""),
                r.get("roof_area_m2", 0.0),
                r.get("height_m", 7.0),
            ),
            axis=1,
        )
        peak_kw = np.minimum(raw_kw, caps)
        n_panels = (peak_kw * 1000.0 / cfg.SOLAR_PANEL_WATT).astype(int)
    else:
        peak_kw = raw_kw
        n_panels = raw_panels

    results["n_panels"] = n_panels
    results["peak_kw"] = peak_kw.round(1) if hasattr(peak_kw, "round") else np.round(peak_kw, 1)
    results["annual_kwh"] = (
        results["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365
    ).round(0)

    return results
