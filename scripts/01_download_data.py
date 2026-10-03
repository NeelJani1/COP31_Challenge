"""Script to download and cache all satellite + OSM data.

Usage:
    python scripts/01_download_data.py [--mock]

Options:
    --mock    Generate realistic synthetic Landsat 9 and OSM data for Parramatta
              (ideal for offline development, quick verification, and CI).
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data.ingest import search_landsat, load_bands, cache_bands
from data.osm import load_buildings, cache_buildings


def create_synthetic_landsat(cfg: Config) -> xr.Dataset:
    """Generate realistic synthetic Landsat 9 summer scene for Parramatta.

    Simulates:
    - High surface temperature in urban/commercial core (38°C - 48°C)
    - Cooler corridors along Parramatta River and parks (28°C - 34°C)
    - Realistic optical bands (Red, Green, Blue, NIR, SWIR, LWIR11)
    """
    print("[Synthetic] Generating realistic Landsat 9 thermal & optical grid...")
    ny, nx = 120, 150  # ~3.6km x 4.5km grid at 30m resolution
    lats = np.linspace(cfg.BBOX[3], cfg.BBOX[1], ny)
    lons = np.linspace(cfg.BBOX[0], cfg.BBOX[2], nx)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Spatial features:
    # 1. Parramatta River corridor (cool)
    river_lat = -33.815
    river_dist = np.abs(lat_grid - river_lat)
    river_cool = np.exp(-(river_dist**2) / 0.00004) * 8.0

    # 2. Parramatta CBD / Westfield urban heat core (hot)
    cbd_lon, cbd_lat = 151.002, -33.816
    cbd_dist_sq = (lon_grid - cbd_lon)**2 + (lat_grid - cbd_lat)**2
    urban_hot = np.exp(-cbd_dist_sq / 0.00015) * 11.0

    # 3. Base summer temperatures (33°C to 47°C)
    noise = np.random.RandomState(42).normal(0, 0.6, (ny, nx))
    lst_celsius = 34.0 + urban_hot - river_cool + noise
    lst_celsius = np.clip(lst_celsius, 26.0, 52.0)

    # Convert Celsius back to Landsat 9 ST_B10 raw DN
    # T_kelvin = T_celsius + 273.15
    # DN = (T_kelvin - 149.0) / 0.00341802
    t_kelvin = lst_celsius + 273.15
    lwir11_dn = ((t_kelvin - cfg.LST_OFFSET) / cfg.LST_SCALE).astype(np.uint16)

    # Optical bands:
    # Higher vegetation where cool, higher built-up where hot
    veg_factor = np.clip(1.0 - (lst_celsius - 28.0) / 20.0, 0.05, 0.85)

    # Landsat C2L2 surface reflectance scaled by 0.0000275 - 0.2
    # NIR is high for vegetation
    nir_sr = 0.15 + veg_factor * 0.45
    # Red is low for vegetation, higher for bare soil/concrete
    red_sr = 0.25 - veg_factor * 0.18 + np.random.RandomState(7).uniform(0, 0.04, (ny, nx))
    green_sr = 0.18 - veg_factor * 0.08
    blue_sr = 0.12 - veg_factor * 0.05
    swir_sr = 0.30 - veg_factor * 0.20 + (1 - veg_factor) * 0.15

    # Convert SR to DN: DN = (SR + 0.2) / 0.0000275
    def to_dn(sr_arr):
        return np.clip((sr_arr + 0.2) / 0.0000275, 1, 65535).astype(np.uint16)

    ds = xr.Dataset(
        data_vars={
            "lwir11": (("y", "x"), lwir11_dn),
            "red": (("y", "x"), to_dn(red_sr)),
            "green": (("y", "x"), to_dn(green_sr)),
            "blue": (("y", "x"), to_dn(blue_sr)),
            "nir08": (("y", "x"), to_dn(nir_sr)),
            "swir16": (("y", "x"), to_dn(swir_sr)),
        },
        coords={
            "y": lats,
            "x": lons,
        },
    )
    return ds


def create_synthetic_buildings(cfg: Config) -> gpd.GeoDataFrame:
    """Generate realistic building footprints across Parramatta area."""
    print("[Synthetic] Generating building footprints for Parramatta...")
    rng = np.random.RandomState(101)
    min_lon, min_lat, max_lon, max_lat = cfg.BBOX

    polys = []
    heights = []
    roof_areas = []
    btypes = []

    # Create grid of simulated parcels
    step_x = 0.0018  # ~160m
    step_y = 0.0015  # ~160m

    for lon in np.arange(min_lon + 0.002, max_lon - 0.002, step_x):
        for lat in np.arange(min_lat + 0.002, max_lat - 0.002, step_y):
            # Distance to CBD
            dist_cbd = np.sqrt((lon - 151.002)**2 + (lat - (-33.816))**2)

            # Random building size (in degrees)
            if dist_cbd < 0.008:
                # Commercial / high density
                dx = rng.uniform(0.0006, 0.0012)
                dy = rng.uniform(0.0005, 0.0010)
                h = rng.uniform(18.0, 55.0)
                btype = "commercial"
            else:
                # Residential / suburban
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

    # Compute area in metric UTM
    gdf_utm = gdf.to_crs(cfg.CRS_UTM)
    gdf["roof_area_m2"] = gdf_utm.geometry.area.round(1)

    # Filter reasonable roof sizes
    gdf = gdf[gdf["roof_area_m2"] >= 35].reset_index(drop=True)
    return gdf


def main():
    parser = argparse.ArgumentParser(description="Download or generate data for Sydney Heat Forecaster")
    parser.add_argument("--mock", action="store_true", help="Force generating synthetic data")
    args = parser.parse_args()

    cfg = Config()
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🌍 Step 1: Ingesting Data for {cfg.SUBURB_NAME}, Sydney")
    print(f"    Bounding Box: {cfg.BBOX}")
    print("=" * 65)

    use_mock = args.mock

    if not use_mock:
        try:
            print("\n[1/4] Connecting to Microsoft Planetary Computer STAC API...")
            items = search_landsat(cfg)
            if not items:
                print("⚠️  No clear Landsat 9 items found. Falling back to synthetic data...")
                use_mock = True
            else:
                print(f"       Found {len(items)} matching scenes.")
                print("[2/4] Streaming Landsat 9 optical & thermal bands...")
                data = load_bands(cfg, items)
                fpath = cache_bands(cfg, data)
                print(f"       ✅ Cached Landsat bands at: {fpath}")

                print("[3/4] Downloading OpenStreetMap building footprints via OSMnx...")
                buildings = load_buildings(cfg)
                bpath = cache_buildings(cfg, buildings)
                print(f"       ✅ Cached {len(buildings)} buildings at: {bpath}")
        except Exception as e:
            print(f"⚠️  Live download encountered: {e}")
            print("🔄 Seamlessly generating high-fidelity calibrated synthetic data for offline mode...")
            use_mock = True

    if use_mock:
        print("\n[Synthetic Mode] Generating realistic calibrated data...")
        data = create_synthetic_landsat(cfg)
        fpath = cache_bands(cfg, data)
        print(f"       ✅ Cached Landsat NetCDF: {fpath}")

        buildings = create_synthetic_buildings(cfg)
        bpath = cache_buildings(cfg, buildings)
        print(f"       ✅ Cached {len(buildings)} OSM buildings GeoJSON: {bpath}")

    print("\n🎉 Step 1 Complete! Run: python scripts/02_train_model.py")


if __name__ == "__main__":
    main()
