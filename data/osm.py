"""Building footprint data from OpenStreetMap via osmnx.

Downloads building polygons for the target area and computes roof
areas in square meters using UTM projection.
"""
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def _extract_height(row, default_height: float = 7.0) -> float:
    """Extract or estimate building height from OSM tags."""
    # Check explicit height tag
    if "height" in row.index and pd.notna(row.get("height")):
        try:
            val = str(row["height"]).replace("m", "").replace("M", "").strip()
            return float(val)
        except (ValueError, TypeError):
            pass
    # Estimate from levels (assume 3.2m per floor)
    if "building:levels" in row.index and pd.notna(row.get("building:levels")):
        try:
            return float(row["building:levels"]) * 3.2
        except (ValueError, TypeError):
            pass
    # Type-based defaults
    b_type = str(row.get("building", "")).lower()
    if b_type in ("apartments", "commercial", "office", "retail"):
        return 16.0
    return default_height


def load_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Download building footprints for the target suburb.

    Returns a GeoDataFrame in WGS84 with computed roof_area_m2 and
    height_m columns.
    """
    west, south, east, north = cfg.BBOX
    buildings = ox.features_from_bbox(
        bbox=(north, south, east, west),
        tags={"building": True},
    )
    # Keep only valid polygon geometries
    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    buildings = buildings[buildings.geometry.is_valid & (~buildings.geometry.is_empty)].copy()

    # Compute building heights
    buildings["height_m"] = buildings.apply(_extract_height, axis=1)

    # Project to UTM for area calculations
    buildings_utm = buildings.to_crs(cfg.CRS_UTM)
    areas = buildings_utm.geometry.area
    buildings["roof_area_m2"] = areas.fillna(0.0)

    # Filter out tiny artifacts (< 20 sqm)
    buildings = buildings[buildings["roof_area_m2"] >= 20].copy()

    return buildings


def create_synthetic_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Generate realistic synthetic building footprints across the target area."""
    from shapely.geometry import box

    rng = np.random.RandomState(101)
    min_lon, min_lat, max_lon, max_lat = cfg.BBOX

    polys = []
    heights = []
    btypes = []

    # Create grid of simulated parcels
    step_x = 0.0018  # ~160m
    step_y = 0.0015  # ~160m

    for lon in np.arange(min_lon + 0.002, max_lon - 0.002, step_x):
        for lat in np.arange(min_lat + 0.002, max_lat - 0.002, step_y):
            dist_cbd = np.sqrt((lon - cfg.LONGITUDE)**2 + (lat - cfg.LATITUDE)**2)

            if dist_cbd < 0.008:
                dx = rng.uniform(0.0006, 0.0012)
                dy = rng.uniform(0.0005, 0.0010)
                h = rng.uniform(18.0, 55.0)
                btype = "commercial"
            else:
                dx = rng.uniform(0.0003, 0.0007)
                dy = rng.uniform(0.0003, 0.0006)
                h = rng.uniform(5.5, 9.0)
                btype = "residential"

            p_box = box(lon, lat, lon + dx, lat + dy)
            polys.append(p_box)
            heights.append(round(float(h), 1))
            btypes.append(btype)

    gdf = gpd.GeoDataFrame(
        {
            "building": btypes,
            "height_m": heights,
            "geometry": polys,
        },
        crs=cfg.CRS_WGS84,
    )

    gdf_utm = gdf.to_crs(cfg.CRS_UTM)
    gdf["roof_area_m2"] = gdf_utm.geometry.area.round(1)
    gdf = gdf[gdf["roof_area_m2"] >= 35].reset_index(drop=True)
    return gdf


def cache_buildings(cfg: Config, buildings: gpd.GeoDataFrame) -> Path:
    """Cache building footprints as GeoJSON."""
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fpath = cache_path / "buildings.geojson"
    # Keep only serializable columns
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
    cols_present = [c for c in cols_to_keep if c in buildings.columns]
    buildings[cols_present].to_file(str(fpath), driver="GeoJSON")
    return fpath


def load_cached_buildings(cfg: Config, allow_mock: bool = False) -> gpd.GeoDataFrame:
    """Load cached building footprints, optionally falling back to synthetic."""
    fpath = Path(cfg.CACHE_DIR) / "buildings.geojson"
    if fpath.exists():
        return gpd.read_file(str(fpath))
    if allow_mock:
        bldgs = create_synthetic_buildings(cfg)
        cache_buildings(cfg, bldgs)
        return bldgs
    raise FileNotFoundError(
        f"No cached buildings at {fpath}. Run 01_download_data.py first."
    )
