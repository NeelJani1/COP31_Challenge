import pytest
import numpy as np
import json
from config import Config
from api import HeatSolarAPI


def test_api_cached_workflow():
    api = HeatSolarAPI()
    api.load_cached_data()

    assert api._loaded
    assert "lst_celsius" in api.features
    assert api.buildings is not None

    # Test baseline heatmap
    lats, lons, temp = api.get_baseline_heatmap()
    assert temp.ndim == 2
    assert len(lats) == temp.shape[0]
    assert len(lons) == temp.shape[1]

    # Test GeoJSON serialization
    geojson = api.get_buildings_geojson()
    assert isinstance(geojson, dict)
    assert geojson["type"] == "FeatureCollection"
    # Ensure it dumps to valid JSON without error
    dumped = json.dumps(geojson)
    assert len(dumped) > 100

    # Test NDVI map
    ndvi_lats, ndvi_lons, ndvi_arr = api.get_ndvi_map()
    assert ndvi_arr.shape == temp.shape

    # Test canopy heights and solar mask
    canopy_h, tree_m = api.get_canopy_heights()
    assert canopy_h.shape == temp.shape
    solar_m = api.get_solar_mask()
    assert solar_m.shape == temp.shape

    # Test simulation
    result = api.simulate(tree_pct=30.0, solar_pct=40.0, cool_roof_albedo=0.08)
    assert result.avg_cooling_celsius > 0.0
    assert result.total_solar_mw > 0.0
    assert result.total_panels > 0


def test_api_not_loaded_error():
    api = HeatSolarAPI()
    with pytest.raises(RuntimeError):
        api.simulate()


def test_api_empty_buildings_geojson():
    import geopandas as gpd
    api = HeatSolarAPI()
    # Test completely empty GeoDataFrame
    api.buildings = gpd.GeoDataFrame()
    geojson = api.get_buildings_geojson()
    assert geojson == {"type": "FeatureCollection", "features": []}
    dumped = json.dumps(geojson, allow_nan=False)
    assert dumped == '{"type": "FeatureCollection", "features": []}'

    # Test empty GeoDataFrame with geometry column
    api.buildings = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
    geojson2 = api.get_buildings_geojson()
    assert geojson2 == {"type": "FeatureCollection", "features": []}
    dumped2 = json.dumps(geojson2, allow_nan=False)
    assert dumped2 == '{"type": "FeatureCollection", "features": []}'


def test_api_latitude_longitude_coords():
    import xarray as xr
    api = HeatSolarAPI()
    custom_lats = np.array([-33.811, -33.812, -33.813])
    custom_lons = np.array([150.991, 150.992, 150.993, 150.994])
    arr = np.ones((3, 4)) * 32.0

    api.features = {
        "lst_celsius": xr.DataArray(arr, coords={"latitude": custom_lats, "longitude": custom_lons}, dims=["latitude", "longitude"]),
        "ndvi": xr.DataArray(arr * 0.01, coords={"latitude": custom_lats, "longitude": custom_lons}, dims=["latitude", "longitude"]),
    }
    lats, lons, temp = api.get_baseline_heatmap()
    assert np.allclose(lats, custom_lats)
    assert np.allclose(lons, custom_lons)
    assert temp.shape == (3, 4)

    nlats, nlons, narr = api.get_ndvi_map()
    assert np.allclose(nlats, custom_lats)
    assert np.allclose(nlons, custom_lons)


def test_create_synthetic_buildings_and_landsat():
    from data.osm import create_synthetic_buildings
    from data.ingest import create_synthetic_landsat
    cfg = Config()
    bldgs = create_synthetic_buildings(cfg)
    assert len(bldgs) > 50
    assert "roof_area_m2" in bldgs.columns
    assert "geometry" in bldgs.columns

    ds = create_synthetic_landsat(cfg)
    assert "red" in ds
    assert "nir08" in ds
    assert "lwir11" in ds


