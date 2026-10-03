import pytest
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from config import Config
from analysis.heatmap_model import MicroclimateModel
from analysis.intervention import run_intervention, InterventionResult


@pytest.fixture
def model_and_features():
    cfg = Config()
    rng = np.random.RandomState(42)
    ny, nx = 15, 15
    ndvi = rng.uniform(0.1, 0.6, (ny, nx))
    ndbi = rng.uniform(-0.3, 0.4, (ny, nx))
    albedo = rng.uniform(0.1, 0.3, (ny, nx))
    lst = 36.0 + 12.0 * ndbi - 10.0 * ndvi + rng.normal(0, 0.1, (ny, nx))

    features = {
        "ndvi": xr.DataArray(ndvi, dims=("y", "x"), coords={"y": np.linspace(-33.80, -33.83, ny), "x": np.linspace(150.98, 151.02, nx)}),
        "ndbi": xr.DataArray(ndbi, dims=("y", "x")),
        "albedo": xr.DataArray(albedo, dims=("y", "x")),
        "lst_celsius": xr.DataArray(lst, dims=("y", "x"), coords={"y": np.linspace(-33.80, -33.83, ny), "x": np.linspace(150.98, 151.02, nx)}),
    }

    model = MicroclimateModel(cfg)
    model.train(features)
    return cfg, model, features


@pytest.fixture
def sample_buildings(model_and_features):
    cfg, _, _ = model_and_features
    bldgs = gpd.GeoDataFrame(
        {
            "building": ["house", "office"],
            "roof_area_m2": [120.0, 500.0],
            "peak_kw": [24.0, 100.0],
            "annual_kwh": [48000.0, 200000.0],
            "n_panels": [60, 250],
            "geometry": [
                box(150.99, -33.82, 150.992, -33.818),
                box(151.00, -33.82, 151.005, -33.818),
            ],
        },
        crs=cfg.CRS_WGS84,
    )
    return bldgs


def test_run_intervention_basic(model_and_features, sample_buildings):
    cfg, model, features = model_and_features
    res = run_intervention(
        features=features,
        buildings=sample_buildings,
        model=model,
        cfg=cfg,
        tree_canopy_increase_pct=25.0,
        solar_coverage_pct=50.0,
        cool_roof_albedo_increase=0.05,
    )

    assert isinstance(res, InterventionResult)
    assert res.baseline_temp_map.ndim == 2
    assert res.predicted_temp_map.ndim == 2
    assert res.avg_cooling_celsius > 0.0
    assert res.max_cooling_celsius >= res.avg_cooling_celsius
    assert res.total_solar_capacity_kw == (24.0 + 100.0) * 0.5
    assert res.total_solar_mw == res.total_solar_capacity_kw / 1000.0
    assert res.total_panels == int((60 + 250) * 0.5)
    assert res.co2_offset_tonnes_year > 0


def test_run_intervention_zero_sliders(model_and_features, sample_buildings):
    cfg, model, features = model_and_features
    res = run_intervention(
        features=features,
        buildings=sample_buildings,
        model=model,
        cfg=cfg,
        tree_canopy_increase_pct=0.0,
        solar_coverage_pct=0.0,
        cool_roof_albedo_increase=0.0,
    )

    assert np.isclose(res.avg_cooling_celsius, 0.0, atol=1e-3)
    assert res.total_solar_capacity_kw == 0.0
    assert res.total_panels == 0
    assert res.n_buildings_with_solar == 0


def test_run_intervention_empty_buildings(model_and_features):
    cfg, model, features = model_and_features
    res = run_intervention(
        features=features,
        buildings=None,
        model=model,
        cfg=cfg,
        tree_canopy_increase_pct=20.0,
        solar_coverage_pct=30.0,
    )
    assert res.total_solar_capacity_kw == 0.0
    assert res.n_buildings_with_solar == 0
    assert res.avg_cooling_celsius > 0.0


def test_run_intervention_solar_clamping(model_and_features, sample_buildings):
    cfg, model, features = model_and_features
    # >100% should clamp to 100%
    res_high = run_intervention(
        features=features,
        buildings=sample_buildings,
        model=model,
        cfg=cfg,
        solar_coverage_pct=150.0,
    )
    res_100 = run_intervention(
        features=features,
        buildings=sample_buildings,
        model=model,
        cfg=cfg,
        solar_coverage_pct=100.0,
    )
    assert res_high.total_solar_capacity_kw == res_100.total_solar_capacity_kw
    assert res_high.total_panels == res_100.total_panels

    # <0% should clamp to 0%
    res_neg = run_intervention(
        features=features,
        buildings=sample_buildings,
        model=model,
        cfg=cfg,
        solar_coverage_pct=-25.0,
    )
    assert res_neg.total_solar_capacity_kw == 0.0
    assert res_neg.n_buildings_with_solar == 0


def test_cooling_summary_formatting():
    res_cool = InterventionResult(
        baseline_temp_map=np.zeros((2, 2)),
        predicted_temp_map=np.zeros((2, 2)),
        avg_cooling_celsius=2.34,
        max_cooling_celsius=3.5,
        hotspot_count_before=5,
        hotspot_count_after=1,
        total_solar_capacity_kw=100.0,
        total_annual_kwh=100000.0,
        n_buildings_with_solar=10,
        total_panels=250,
        lat_grid=np.zeros(2),
        lon_grid=np.zeros(2),
    )
    assert res_cool.cooling_summary == "-2.3°C average"

    # Warming scenario (negative cooling)
    res_warm = InterventionResult(
        baseline_temp_map=np.zeros((2, 2)),
        predicted_temp_map=np.zeros((2, 2)),
        avg_cooling_celsius=-1.5,
        max_cooling_celsius=-0.8,
        hotspot_count_before=1,
        hotspot_count_after=5,
        total_solar_capacity_kw=0.0,
        total_annual_kwh=0.0,
        n_buildings_with_solar=0,
        total_panels=0,
        lat_grid=np.zeros(2),
        lon_grid=np.zeros(2),
    )
    assert res_warm.cooling_summary == "+1.5°C average"

