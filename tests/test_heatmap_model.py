import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from config import Config
from analysis.heatmap_model import MicroclimateModel


@pytest.fixture
def dummy_features():
    rng = np.random.RandomState(42)
    ny, nx = 20, 20
    ndvi = rng.uniform(0.1, 0.7, (ny, nx))
    ndbi = rng.uniform(-0.4, 0.3, (ny, nx))
    albedo = rng.uniform(0.1, 0.3, (ny, nx))
    # Simulated temperature physics: higher ndbi -> hotter, higher ndvi -> cooler
    lst = 35.0 + 15.0 * ndbi - 8.0 * ndvi - 5.0 * albedo + rng.normal(0, 0.2, (ny, nx))

    return {
        "ndvi": xr.DataArray(ndvi, dims=("y", "x")),
        "ndbi": xr.DataArray(ndbi, dims=("y", "x")),
        "albedo": xr.DataArray(albedo, dims=("y", "x")),
        "lst_celsius": xr.DataArray(lst, dims=("y", "x")),
    }


def test_model_training_and_metrics(dummy_features):
    cfg = Config()
    model = MicroclimateModel(cfg)
    metrics = model.train(dummy_features)

    assert model.is_fitted
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["mae"] < 1.0  # Should fit well on synthetic linear data
    assert metrics["r2"] > 0.85
    assert metrics["n_samples"] == 400


def test_model_predict(dummy_features):
    cfg = Config()
    model = MicroclimateModel(cfg)
    model.train(dummy_features)

    ndvi = dummy_features["ndvi"].values
    ndbi = dummy_features["ndbi"].values
    albedo = dummy_features["albedo"].values

    pred = model.predict(ndvi, ndbi, albedo)
    assert pred.shape == ndvi.shape
    assert not np.isnan(pred).any()


def test_model_predict_with_nans(dummy_features):
    cfg = Config()
    model = MicroclimateModel(cfg)
    model.train(dummy_features)

    ndvi = dummy_features["ndvi"].values.copy()
    ndbi = dummy_features["ndbi"].values.copy()
    albedo = dummy_features["albedo"].values.copy()

    ndvi[0, 0] = np.nan
    pred = model.predict(ndvi, ndbi, albedo)
    assert np.isnan(pred[0, 0])
    assert not np.isnan(pred[0, 1])


def test_model_shape_mismatch(dummy_features):
    cfg = Config()
    model = MicroclimateModel(cfg)
    model.train(dummy_features)

    with pytest.raises(ValueError):
        model.predict(np.zeros((5, 5)), np.zeros((6, 6)), np.zeros((5, 5)))


def test_simulate_intervention(dummy_features):
    cfg = Config()
    model = MicroclimateModel(cfg)
    model.train(dummy_features)

    # Greening: more trees (+0.2 NDVI, -0.1 NDBI)
    base, pred, cooling = model.simulate_intervention(
        dummy_features,
        delta_ndvi=0.2,
        delta_ndbi=-0.1,
        delta_albedo=0.05,
    )
    assert base.shape == pred.shape
    assert cooling > 0.0  # Greening should cool the microclimate


def test_model_save_and_load(dummy_features, tmp_path):
    cfg = Config()
    model = MicroclimateModel(cfg)
    metrics = model.train(dummy_features)

    save_file = str(tmp_path / "model.joblib")
    model.save(save_file)

    loaded_model = MicroclimateModel(cfg)
    loaded_model.load(save_file)
    assert loaded_model.is_fitted
    assert loaded_model.metrics["mae"] == metrics["mae"]
