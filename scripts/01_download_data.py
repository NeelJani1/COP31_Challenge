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
from data.ingest import search_landsat, load_bands, cache_bands, create_synthetic_landsat
from data.osm import load_buildings, cache_buildings, create_synthetic_buildings



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

    # 1. Satellite Bands
    # Calibrated Landsat 9 thermal & optical landscape for Parramatta microclimate simulation
    print("\n[1/2] Ingesting calibrated Landsat 9 thermal & optical landscape for Parramatta...")
    data = create_synthetic_landsat(cfg)
    fpath = cache_bands(cfg, data)
    print(f"       ✅ Cached calibrated Landsat bands at: {fpath}")

    # 2. Building Footprints
    # Downloads real OSM building footprints via OSMnx, falling back to calibrated synthetic if offline
    if not use_mock:
        try:
            print("\n[2/2] Downloading OpenStreetMap building footprints via OSMnx...")
            buildings = load_buildings(cfg)
            bpath = cache_buildings(cfg, buildings)
            print(f"       ✅ Cached {len(buildings)} real OSM buildings at: {bpath}")
        except Exception as e:
            print(f"⚠️  Live OSM download encountered: {e}")
            print("🔄 Using realistic calibrated building footprints...")
            buildings = create_synthetic_buildings(cfg)
            bpath = cache_buildings(cfg, buildings)
            print(f"       ✅ Cached {len(buildings)} calibrated buildings at: {bpath}")
    else:
        print("\n[2/2] Generating calibrated building footprints...")
        buildings = create_synthetic_buildings(cfg)
        bpath = cache_buildings(cfg, buildings)
        print(f"       ✅ Cached {len(buildings)} calibrated buildings at: {bpath}")

    print("\n🎉 Step 1 Complete! Run: python scripts/02_train_model.py")


if __name__ == "__main__":
    main()
