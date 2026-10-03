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


def _standardize_building_type(row) -> str:
    """Classify OSM building tag and dimensions into standardized typology."""
    b = str(row.get("building", "")).lower().strip()
    area = float(row.get("roof_area_m2", 0.0))
    height = float(row.get("height_m", 7.0))
    if b in ("house", "residential", "detached", "terrace", "semidetached_house", "cottage", "cabin", "bungalow", "duplex"):
        return "residential"
    if b in ("apartments", "dormitory", "barracks"):
        return "apartments"
    if b in ("commercial", "retail", "office", "hotel", "bank", "supermarket", "kiosk", "skyscraper"):
        return "commercial"
    if b in ("industrial", "warehouse", "manufacture", "depot"):
        return "industrial"
    if b in ("school", "university", "college", "kindergarten", "hospital", "clinic", "civic", "public", "government", "library", "church", "cathedral", "chapel"):
        return "commercial"
    if height > 24.0 or area > 1800.0:
        return "commercial"
    if height > 11.0 or area > 450.0:
        return "apartments"
    return "residential"


def load_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Download building footprints for the target suburb.

    Returns a GeoDataFrame in WGS84 with computed roof_area_m2, height_m,
    standardized building type, and Australian-calibrated solar capacity metrics.
    """
    west, south, east, north = cfg.BBOX
    # OSMnx 2.x expects (left, bottom, right, top) = (west, south, east, north)
    buildings = ox.features_from_bbox(
        bbox=(west, south, east, north),
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
    buildings["roof_area_m2"] = areas.fillna(0.0).round(1)

    # Filter out tiny artifacts (< 20 sqm)
    buildings = buildings[buildings["roof_area_m2"] >= 20].copy()

    # Standardize building classification
    buildings["building"] = buildings.apply(_standardize_building_type, axis=1)

    # Compute realistic solar installation metrics
    # Australian limits: residential 5-15 kWp, apartments 48 kWp, commercial 50-200 kWp
    n_panels = []
    peak_kw = []
    unshaded_m2 = []
    for _, row in buildings.iterrows():
        b_type = str(row.get("building", "")).lower()
        area = float(row["roof_area_m2"])
        if b_type == "residential":
            usable = min(area * 0.28, 55.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 13.2), 1)
        elif b_type == "apartments":
            usable = min(area * 0.35, 180.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 48.0), 1)
        elif b_type in ("commercial", "office"):
            usable = min(area * 0.40, 600.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 160.0), 1)
        else:  # industrial
            usable = min(area * 0.45, 900.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 220.0), 1)

        unshaded_m2.append(round(panels * cfg.SOLAR_PANEL_AREA_M2, 1))
        n_panels.append(panels)
        peak_kw.append(kw)

    buildings["unshaded_area_m2"] = unshaded_m2
    buildings["n_panels"] = n_panels
    buildings["peak_kw"] = peak_kw
    buildings["annual_kwh"] = (buildings["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365).round(0)

    cols = ["geometry", "roof_area_m2", "height_m", "building", "unshaded_area_m2", "peak_kw", "annual_kwh", "n_panels"]
    return buildings[cols].reset_index(drop=True)


def create_synthetic_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Generate realistic, real-world aligned building footprints across Parramatta.

    Simulates:
    - Parramatta CBD commercial high-rises & retail along Church St / Macquarie St / Argyle St
    - Westfield Parramatta and Parramatta Square towers
    - North Parramatta residential homes and walk-up apartments along street corridors
    - Westmead Health Precinct (hospital complexes) and residential blocks
    - Harris Park / South Parramatta character cottages
    - Camellia / Rosehill industrial warehouses
    - Strict avoidance of Parramatta River corridor and Parramatta Park
    """
    from shapely.geometry import box, Polygon, LineString, Point
    from shapely.affinity import rotate

    rng = np.random.RandomState(101)
    min_lon, min_lat, max_lon, max_lat = cfg.BBOX

    m_per_deg_lat = 110900.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(cfg.LATITUDE))

    # Parramatta River centerline polyline for exclusion
    river_line = LineString([
        (150.9800, -33.8065),
        (150.9870, -33.8080),
        (150.9930, -33.8105),
        (150.9985, -33.8126),
        (151.0040, -33.8130),
        (151.0100, -33.8142),
        (151.0150, -33.8152),
        (151.0200, -33.8162),
    ])

    # Parramatta Park boundary polygon for exclusion
    parramatta_park_poly = Polygon([
        (150.9922, -33.8198),
        (150.9922, -33.8130),
        (150.9960, -33.8115),
        (151.0000, -33.8122),
        (151.0016, -33.8140),
        (151.0016, -33.8198),
        (150.9970, -33.8202),
    ])

    polys = []
    heights = []
    btypes = []

    def make_poly(lon, lat, width_m, length_m, angle_deg=0.0, shape="rect"):
        if shape == "L":
            p1 = box(-width_m / 2, -length_m / 2, width_m / 2, length_m / 4)
            p2 = box(-width_m / 2, length_m / 4, 0, length_m / 2)
            poly = p1.union(p2)
        elif shape == "T":
            p1 = box(-width_m / 2, -length_m / 4, width_m / 2, length_m / 4)
            p2 = box(-width_m / 4, length_m / 4, width_m / 4, length_m / 2)
            poly = p1.union(p2)
        else:
            poly = box(-width_m / 2, -length_m / 2, width_m / 2, length_m / 2)

        if angle_deg != 0.0:
            poly = rotate(poly, angle_deg, origin=(0, 0))

        coords = np.array(poly.exterior.coords)
        lon_coords = lon + coords[:, 0] / m_per_deg_lon
        lat_coords = lat + coords[:, 1] / m_per_deg_lat
        return Polygon(zip(lon_coords, lat_coords))

    def add_building(lon, lat, w, l, h, btype, angle=0.0, shape="rect"):
        p = make_poly(lon, lat, w, l, angle, shape)
        pt = Point(lon, lat)
        # Avoid river corridor (~45m buffer = ~0.00045 deg)
        if river_line.distance(pt) < 0.00045:
            return False
        # Avoid Parramatta Park
        if parramatta_park_poly.contains(pt):
            return False
        # Avoid bounding box borders
        if not (min_lon + 0.0006 <= lon <= max_lon - 0.0006 and min_lat + 0.0006 <= lat <= max_lat - 0.0006):
            return False

        polys.append(p)
        heights.append(round(float(h), 1))
        btypes.append(btype)
        return True

    # 1. Parramatta CBD: Iconic Landmarks & High Rises
    # Westfield Parramatta complex (multi-tiered retail rooftop)
    add_building(151.0028, -33.8182, 58, 68, 35.0, "commercial", angle=-4.0, shape="rect")
    add_building(151.0038, -33.8190, 52, 62, 38.0, "commercial", angle=-4.0, shape="rect")
    add_building(151.0045, -33.8184, 46, 56, 32.0, "commercial", angle=-4.0, shape="rect")
    # Parramatta Square precinct (commercial office towers & civic hub)
    add_building(151.0062, -33.8162, 38, 42, 120.0, "commercial", angle=-5.0, shape="rect")
    add_building(151.0073, -33.8160, 44, 46, 138.0, "commercial", angle=-5.0, shape="rect")
    add_building(151.0084, -33.8163, 40, 40, 110.0, "commercial", angle=-5.0, shape="rect")
    add_building(151.0055, -33.8160, 32, 30, 28.0, "commercial", angle=-5.0, shape="rect")

    # CBD street network commercial towers (Macquarie St, George St, Argyle St, Campbell St)
    for street_lat in [-33.8146, -33.8160, -33.8175, -33.8195]:
        for street_lon in np.arange(151.0015, 151.0110, 0.00075):
            for offset_m in [-15, 15]:
                lat = street_lat + offset_m / m_per_deg_lat
                lon = street_lon + rng.uniform(-0.0001, 0.0001)
                w = rng.uniform(22, 34)
                l = rng.uniform(25, 42)
                h = rng.uniform(28, 75)
                bt = rng.choice(["commercial", "office", "apartments"], p=[0.4, 0.4, 0.2])
                sh = rng.choice(["rect", "L", "T"], p=[0.7, 0.2, 0.1])
                add_building(lon, lat, w, l, h, bt, angle=-4.5, shape=sh)

    # 2. North Parramatta Residential Street Grid (Grose, Ross, Albert, Iron, Fennell streets)
    for s_lat in [-33.8095, -33.8075, -33.8055, -33.8035, -33.8015]:
        for s_lon in np.arange(150.9950, 151.0190, 0.00035):
            for side in [-12, 12]:
                lat = s_lat + side / m_per_deg_lat
                lon = s_lon + rng.uniform(-0.00005, 0.00005)
                if rng.rand() < 0.12:
                    # Walk-up apartments
                    w, l, h = rng.uniform(18, 24), rng.uniform(22, 28), rng.uniform(9.5, 13.5)
                    bt, sh = "apartments", "rect"
                else:
                    # Single-family residential home
                    w, l, h = rng.uniform(11, 15), rng.uniform(14, 19), rng.uniform(5.5, 8.0)
                    bt = "residential"
                    sh = "L" if rng.rand() < 0.3 else "rect"
                add_building(lon, lat, w, l, h, bt, angle=1.0, shape=sh)

    # 3. Westmead Precinct: Hospital Campus & Residential Blocks
    # Westmead Hospital campus
    for h_lat in np.arange(-33.8065, -33.8015, 0.0010):
        for h_lon in np.arange(150.9840, 150.9905, 0.0012):
            w, l, h = rng.uniform(34, 48), rng.uniform(42, 65), rng.uniform(22, 42)
            add_building(h_lon, h_lat, w, l, h, "commercial", angle=5.0, shape="rect")

    # Westmead suburban homes
    for s_lat in np.arange(-33.8180, -33.8075, 0.0015):
        for s_lon in np.arange(150.9815, 150.9915, 0.0004):
            for side in [-12, 12]:
                lat = s_lat + side / m_per_deg_lat
                lon = s_lon + rng.uniform(-0.00005, 0.00005)
                w, l, h = rng.uniform(11, 15), rng.uniform(14, 19), rng.uniform(5.5, 7.8)
                sh = "L" if rng.rand() < 0.3 else "rect"
                add_building(lon, lat, w, l, h, "residential", angle=0.0, shape=sh)

    # 4. Harris Park / South Parramatta Character Residential
    for s_lat in np.arange(-33.8285, -33.8210, 0.0014):
        for s_lon in np.arange(151.0020, 151.0150, 0.0004):
            for side in [-12, 12]:
                lat = s_lat + side / m_per_deg_lat
                lon = s_lon + rng.uniform(-0.00005, 0.00005)
                w, l, h = rng.uniform(11, 14), rng.uniform(13, 18), rng.uniform(5.5, 7.5)
                sh = "L" if rng.rand() < 0.25 else "rect"
                add_building(lon, lat, w, l, h, "residential", angle=0.0, shape=sh)

    # 5. Camellia / Rosehill Industrial Logistics Warehouses
    for ind_lat in np.arange(-33.8280, -33.8170, 0.0018):
        for ind_lon in np.arange(151.0125, 151.0195, 0.0015):
            w, l, h = rng.uniform(32, 52), rng.uniform(45, 75), rng.uniform(8.5, 14.0)
            sh = "L" if rng.rand() < 0.2 else "rect"
            add_building(ind_lon, ind_lat, w, l, h, "industrial", angle=2.0, shape=sh)

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

    # Compute realistic solar installation metrics
    # Australian limits: residential 5-15 kWp, commercial 50-200 kWp
    n_panels = []
    peak_kw = []
    unshaded_m2 = []
    for _, row in gdf.iterrows():
        b_type = str(row.get("building", "")).lower()
        area = float(row["roof_area_m2"])
        if b_type == "residential":
            usable = min(area * 0.28, 55.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 13.2), 1)
        elif b_type == "apartments":
            usable = min(area * 0.35, 180.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 48.0), 1)
        elif b_type in ("commercial", "office"):
            usable = min(area * 0.40, 600.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 160.0), 1)
        else:  # industrial
            usable = min(area * 0.45, 900.0)
            panels = int(usable / cfg.SOLAR_PANEL_AREA_M2)
            kw = round(min(panels * cfg.SOLAR_PANEL_WATT / 1000.0, 220.0), 1)

        unshaded_m2.append(round(panels * cfg.SOLAR_PANEL_AREA_M2, 1))
        n_panels.append(panels)
        peak_kw.append(kw)

    gdf["unshaded_area_m2"] = unshaded_m2
    gdf["n_panels"] = n_panels
    gdf["peak_kw"] = peak_kw
    gdf["annual_kwh"] = (gdf["peak_kw"] * cfg.SUN_HOURS_PER_DAY * 365).round(0)

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
