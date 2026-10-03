"""Building footprint data from OpenStreetMap via osmnx.

Downloads building polygons for the target area and computes roof
areas in square meters using UTM projection.
"""
import geopandas as gpd
import osmnx as ox
import pandas as pd
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


def cache_buildings(cfg: Config, buildings: gpd.GeoDataFrame) -> Path:
    """Cache building footprints as GeoJSON."""
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fpath = cache_path / "buildings.geojson"
    # Keep only serializable columns
    cols_to_keep = ["geometry", "roof_area_m2", "height_m", "building"]
    cols_present = [c for c in cols_to_keep if c in buildings.columns]
    buildings[cols_present].to_file(str(fpath), driver="GeoJSON")
    return fpath


def load_cached_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Load cached building footprints."""
    fpath = Path(cfg.CACHE_DIR) / "buildings.geojson"
    if fpath.exists():
        return gpd.read_file(str(fpath))
    raise FileNotFoundError(
        f"No cached buildings at {fpath}. Run 01_download_data.py first."
    )
