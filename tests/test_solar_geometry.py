import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from config import Config
from analysis.solar_geometry import (
    get_solar_positions,
    compute_shadow_polygon,
    compute_daily_shadow_union,
    compute_unshaded_roof_area,
    compute_simple_solar_capacity,
)


@pytest.fixture
def cfg():
    return Config()


def test_get_solar_positions(cfg):
    solar_pos = get_solar_positions(cfg, date="2024-01-15", freq="2h")
    assert not solar_pos.empty
    assert "apparent_elevation" in solar_pos.columns
    assert "azimuth" in solar_pos.columns
    assert (solar_pos["apparent_elevation"] > 0).all()


def test_compute_shadow_polygon():
    # 5m x 5m canopy polygon
    tree_poly = box(0, 0, 5, 5)
    height = 10.0
    solar_alt = 45.0  # tan(45) = 1.0 => shadow length = 10m
    solar_az = 180.0  # Sun from South => shadow extends North (dy > 0)

    shadow = compute_shadow_polygon(tree_poly, height, solar_alt, solar_az)
    assert shadow is not None
    assert shadow.is_valid
    assert shadow.area > tree_poly.area

    # Near horizon or night (altitude <= 2)
    night_shadow = compute_shadow_polygon(tree_poly, height, 1.0, solar_az)
    assert night_shadow == tree_poly

    # Zero or negative tree height casts no shadow
    zero_shadow = compute_shadow_polygon(tree_poly, 0.0, solar_alt, solar_az)
    assert zero_shadow.is_empty
    neg_shadow = compute_shadow_polygon(tree_poly, -5.0, solar_alt, solar_az)
    assert neg_shadow.is_empty


def test_compute_daily_shadow_union(cfg):
    tree_polys = gpd.GeoSeries([box(0, 0, 4, 4), box(10, 10, 14, 14)])
    heights = np.array([8.0, 10.0])
    solar_pos = get_solar_positions(cfg, date="2024-01-15", freq="4h")

    shadow_union = compute_daily_shadow_union(tree_polys, heights, solar_pos)
    assert shadow_union.is_valid
    assert not shadow_union.is_empty


def test_compute_unshaded_roof_area(cfg):
    # Create building and nearby tree
    # Coordinates in WGS84 for Parramatta (~151.00, -33.81)
    bldgs = gpd.GeoDataFrame(
        {
            "building": ["residential"],
            "roof_area_m2": [150.0],
            "geometry": [box(151.000, -33.815, 151.0002, -33.8148)],
        },
        crs=cfg.CRS_WGS84,
    )
    trees = gpd.GeoDataFrame(
        {
            "geometry": [box(151.0001, -33.8149, 151.00015, -33.81485)],
        },
        crs=cfg.CRS_WGS84,
    )
    heights = np.array([8.0])

    res = compute_unshaded_roof_area(bldgs, trees, heights, cfg)
    assert "unshaded_area_m2" in res.columns
    assert "n_panels" in res.columns
    assert "peak_kw" in res.columns
    assert "annual_kwh" in res.columns
    assert "unshaded_geometry" not in res.columns  # Ensure non-serializable column is omitted

    # Check non-negative and bounded
    assert (res["unshaded_area_m2"] >= 0).all()
    assert (res["unshaded_area_m2"] <= res["roof_area_m2"]).all()
    assert (res["peak_kw"] >= 0).all()


def test_compute_unshaded_roof_area_empty(cfg):
    empty_bldgs = gpd.GeoDataFrame(columns=["geometry"], crs=cfg.CRS_WGS84)
    res = compute_unshaded_roof_area(empty_bldgs, None, None, cfg)
    assert len(res) == 0
    assert "unshaded_area_m2" in res.columns


def test_compute_simple_solar_capacity(cfg):
    bldgs = gpd.GeoDataFrame(
        {
            "roof_area_m2": [100.0, 200.0],
            "geometry": [
                box(151.0, -33.81, 151.001, -33.809),
                box(151.002, -33.81, 151.003, -33.809),
            ],
        },
        crs=cfg.CRS_WGS84,
    )
    ndvi = np.array([[0.2, 0.5], [0.1, 0.4]])

    res = compute_simple_solar_capacity(bldgs, ndvi, cfg)
    assert len(res) == 2
    assert (res["unshaded_area_m2"] == 0.7 * res["roof_area_m2"]).all()
    assert (res["n_panels"] > 0).all()
    assert (res["peak_kw"] > 0).all()
    assert (res["annual_kwh"] > 0).all()


def test_compute_unshaded_roof_area_scalar_height(cfg):
    bldgs = gpd.GeoDataFrame(
        {
            "roof_area_m2": [150.0],
            "geometry": [box(151.000, -33.815, 151.0002, -33.8148)],
        },
        crs=cfg.CRS_WGS84,
    )
    trees = gpd.GeoDataFrame(
        {
            "geometry": [box(151.0001, -33.8149, 151.00015, -33.81485)],
        },
        crs=cfg.CRS_WGS84,
    )
    # Scalar float tree height
    res = compute_unshaded_roof_area(bldgs, trees, 8.0, cfg)
    assert len(res) == 1
    assert "unshaded_area_m2" in res.columns
    assert res["unshaded_area_m2"].iloc[0] <= 150.0

