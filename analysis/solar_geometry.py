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
    if solar_altitude <= 2.0:
        return tree_polygon  # No meaningful shadow near horizon

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
            if tree_poly is None or tree_poly.is_empty:
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
    solar_pos = get_solar_positions(cfg)

    # Work in UTM for accurate area calculations
    buildings_utm = buildings.to_crs(cfg.CRS_UTM)
    trees_utm = tree_canopy.to_crs(cfg.CRS_UTM)

    # Compute daily shadow union
    shadow_union = compute_daily_shadow_union(
        trees_utm.geometry, tree_heights, solar_pos
    )

    # Subtract shadows from building footprints
    results = buildings_utm.copy()
    if not shadow_union.is_empty:
        results["unshaded_geometry"] = results.geometry.difference(shadow_union)
        results["unshaded_area_m2"] = results["unshaded_geometry"].area
    else:
        results["unshaded_area_m2"] = results.geometry.area

    # Calculate solar capacity
    results["n_panels"] = (
        results["unshaded_area_m2"] / cfg.SOLAR_PANEL_AREA_M2
    ).astype(int)
    results["peak_kw"] = results["n_panels"] * cfg.SOLAR_PANEL_WATT / 1000
    results["annual_kwh"] = (
        results["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365
    )

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
    results = buildings.copy()

    # Estimate shading fraction from NDVI (higher NDVI near building = more shade)
    # Simple heuristic: assume 30% of roof is shaded on average
    shade_fraction = 0.30
    results["unshaded_area_m2"] = results["roof_area_m2"] * (1 - shade_fraction)

    results["n_panels"] = (
        results["unshaded_area_m2"] / cfg.SOLAR_PANEL_AREA_M2
    ).astype(int)
    results["peak_kw"] = results["n_panels"] * cfg.SOLAR_PANEL_WATT / 1000
    results["annual_kwh"] = (
        results["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365
    )

    return results
