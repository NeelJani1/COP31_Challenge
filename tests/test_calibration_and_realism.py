"""Rigorous tests for geographic accuracy, physical realism, and calibration."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from config import Config
from data.osm import create_synthetic_buildings
from data.ingest import create_synthetic_landsat
from features.spectral import compute_all_features
from analysis.heatmap_model import MicroclimateModel
from analysis.intervention import run_intervention


@pytest.fixture
def cfg():
    return Config()


def test_building_geographic_accuracy_and_exclusions(cfg):
    bldgs = create_synthetic_buildings(cfg)
    assert len(bldgs) >= 300, "Should generate a realistic sample across Parramatta"

    # Verify geometries are valid
    assert bldgs.geometry.is_valid.all(), "All building geometries must be valid"
    assert (~bldgs.geometry.is_empty).all(), "No building geometry should be empty"

    # Parramatta River centerline
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

    # Parramatta Park interior
    park_poly = Polygon([
        (150.9930, -33.8190),
        (150.9930, -33.8135),
        (150.9960, -33.8120),
        (151.0000, -33.8125),
        (151.0010, -33.8140),
        (151.0010, -33.8190),
        (150.9970, -33.8198),
    ])

    # Check river avoidance (no centroids within 35m / ~0.00035 deg)
    for _, b in bldgs.iterrows():
        cent = b.geometry.centroid
        assert river_line.distance(cent) > 0.00030, (
            f"Building at {cent} inappropriately overlaps Parramatta River"
        )
        assert not park_poly.contains(cent), (
            f"Building at {cent} inappropriately placed inside Parramatta Park"
        )

    # Check residential vs commercial building sizing
    res = bldgs[bldgs["building"] == "residential"]
    assert len(res) > 200, "Must have substantial residential coverage"
    # Australian suburban homes: mean ~150-250 m2
    assert 120.0 <= res["roof_area_m2"].mean() <= 280.0
    assert (res["roof_area_m2"] < 400.0).all()
    assert (res["height_m"] <= 10.0).all()

    # Commercial buildings
    comm = bldgs[bldgs["building"].isin(["commercial", "office"])]
    assert len(comm) > 30, "Must have commercial core along Church/Macquarie"
    assert comm["roof_area_m2"].mean() >= 500.0
    assert comm["height_m"].max() >= 80.0  # Parramatta Square towers


def test_solar_connection_limits(cfg):
    bldgs = create_synthetic_buildings(cfg)

    # Residential solar: Australian connection limit <= 15 kWp
    res = bldgs[bldgs["building"] == "residential"]
    assert (res["peak_kw"] <= 15.0).all(), "Residential solar must not exceed 15 kWp connection limit"
    assert (res["peak_kw"] >= 3.0).all(), "Residential solar should have realistic min generation"

    # Commercial solar: Australian commercial rooftop systems typically 50 - 200 kWp
    comm = bldgs[bldgs["building"].isin(["commercial", "office"])]
    assert (comm["peak_kw"] <= 200.0).all(), "Commercial solar should respect commercial scale"
    assert comm["peak_kw"].max() >= 50.0

    # Total generation should be physically realistic for ~1,500 buildings (20 - 60 MW)
    total_mw = bldgs["peak_kw"].sum() / 1000.0
    assert 20.0 <= total_mw <= 60.0, f"Total solar capacity {total_mw:.1f} MW out of realistic range"


def test_thermal_and_spectral_calibration(cfg):
    ds = create_synthetic_landsat(cfg)
    features = compute_all_features(ds, cfg)

    lst = features["lst_celsius"].values
    ndvi = features["ndvi"].values
    ndbi = features["ndbi"].values

    # Check full range
    assert lst.min() >= 20.0
    assert lst.max() <= 49.0

    # Check Parramatta River water: cool (22 - 25°C), low NDVI, low NDBI
    # River coordinate sample near Charles St / Lennox Bridge (-33.8130, 151.0040)
    lats = features["lst_celsius"].y.values
    lons = features["lst_celsius"].x.values

    # Find river point
    lat_idx = np.argmin(np.abs(lats - (-33.8130)))
    lon_idx = np.argmin(np.abs(lons - 151.0040))
    river_temp = lst[lat_idx, lon_idx]
    assert 22.0 <= river_temp <= 25.5, f"River temperature {river_temp:.1f}°C not cool"
    assert ndvi[lat_idx, lon_idx] < 0.20, "River NDVI must be low"
    assert ndbi[lat_idx, lon_idx] < -0.10, "River NDBI must be negative"

    # Check Parramatta Park: cool oasis (26 - 30°C), high NDVI (>0.55)
    # Park coordinate sample (-33.8155, 150.9968)
    p_lat_idx = np.argmin(np.abs(lats - (-33.8155)))
    p_lon_idx = np.argmin(np.abs(lons - 150.9968))
    park_temp = lst[p_lat_idx, p_lon_idx]
    assert 25.5 <= park_temp <= 30.5, f"Park temperature {park_temp:.1f}°C out of range"
    assert ndvi[p_lat_idx, p_lon_idx] > 0.50, f"Park NDVI {ndvi[p_lat_idx, p_lon_idx]:.2f} must be > 0.50"

    # Check CBD Heat Island: hot (39 - 46°C), high NDBI (>0.25)
    # CBD center (-33.8168, 151.0055)
    cbd_lat_idx = np.argmin(np.abs(lats - (-33.8168)))
    cbd_lon_idx = np.argmin(np.abs(lons - 151.0055))
    cbd_temp = lst[cbd_lat_idx, cbd_lon_idx]
    assert 39.0 <= cbd_temp <= 46.5, f"CBD temperature {cbd_temp:.1f}°C not a heat island"
    assert ndbi[cbd_lat_idx, cbd_lon_idx] > 0.25, "CBD NDBI must be high"


def test_localized_intervention_physics(cfg):
    ds = create_synthetic_landsat(cfg)
    features = compute_all_features(ds, cfg)
    bldgs = create_synthetic_buildings(cfg)

    model = MicroclimateModel(cfg)
    model.train(features)

    res = run_intervention(features, bldgs, model, cfg, tree_canopy_increase_pct=25.0, solar_coverage_pct=30.0)

    # Localized cooling: max cooling should be significantly higher than average cooling
    assert res.avg_cooling_celsius > 0.5
    assert res.max_cooling_celsius > res.avg_cooling_celsius * 1.5, (
        "Cooling should be localized to corridors, not a uniform blur"
    )

    # River water pixel should experience minimal/zero cooling compared to land
    lats = features["lst_celsius"].y.values
    lons = features["lst_celsius"].x.values
    lat_idx = np.argmin(np.abs(lats - (-33.8130)))
    lon_idx = np.argmin(np.abs(lons - 151.0040))
    river_diff = res.baseline_temp_map[lat_idx, lon_idx] - res.predicted_temp_map[lat_idx, lon_idx]
    assert river_diff < res.avg_cooling_celsius, "Water pixels should not receive tree cooling"


def test_cool_roof_albedo_cooling_physics(cfg):
    ds = create_synthetic_landsat(cfg)
    features = compute_all_features(ds, cfg)
    bldgs = create_synthetic_buildings(cfg)

    model = MicroclimateModel(cfg)
    model.train(features)

    # Cool roof albedo increase alone
    res = run_intervention(
        features, bldgs, model, cfg,
        tree_canopy_increase_pct=0.0,
        solar_coverage_pct=0.0,
        cool_roof_albedo_increase=0.15,
    )
    # Must produce tangible urban cooling
    assert res.avg_cooling_celsius > 0.4, f"Cool roof cooling {res.avg_cooling_celsius:.2f}°C too low"
    assert res.max_cooling_celsius > 1.5, f"Max cool roof localized cooling {res.max_cooling_celsius:.2f}°C too low"

    # Water should experience zero cool roof effect
    lats = features["lst_celsius"].y.values
    lons = features["lst_celsius"].x.values
    lat_idx = np.argmin(np.abs(lats - (-33.8130)))
    lon_idx = np.argmin(np.abs(lons - 151.0040))
    river_diff = res.baseline_temp_map[lat_idx, lon_idx] - res.predicted_temp_map[lat_idx, lon_idx]
    assert np.isclose(river_diff, 0.0, atol=0.05), "River should not receive cool roof coating"


def test_cached_osm_buildings_authenticity(cfg):
    from data.osm import load_cached_buildings
    bldgs = load_cached_buildings(cfg, allow_mock=True)
    assert len(bldgs) >= 300, "Should have substantial building coverage"
    assert bldgs.geometry.is_valid.all()

    # Bounding box bounds check
    min_lon, min_lat, max_lon, max_lat = cfg.BBOX
    bounds = bldgs.total_bounds
    assert bounds[0] >= min_lon - 0.005
    assert bounds[1] >= min_lat - 0.005
    assert bounds[2] <= max_lon + 0.005
    assert bounds[3] <= max_lat + 0.005

    # Check connection limits on cached buildings
    res = bldgs[bldgs["building"] == "residential"]
    if len(res) > 0:
        assert (res["peak_kw"] <= 15.0).all(), "Residential solar must not exceed 15 kWp connection limit"
    comm = bldgs[bldgs["building"].isin(["commercial", "office"])]
    if len(comm) > 0:
        assert (comm["peak_kw"] <= 200.0).all(), "Commercial solar must not exceed 200 kWp limit"
