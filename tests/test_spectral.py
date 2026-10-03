import pytest
import numpy as np
import xarray as xr
from config import Config
from features.spectral import (
    compute_ndvi,
    compute_ndbi,
    compute_albedo,
    compute_lst_celsius,
    compute_all_features,
)


@pytest.fixture
def sample_dataset():
    # Shape: (2, 2)
    # NIR high, Red low -> high NDVI
    # SWIR high, NIR low -> high NDBI
    # DN values for ST_B10: ~44000 gives ~300K -> ~27°C
    data = xr.Dataset(
        data_vars={
            "red": (("y", "x"), np.array([[1000.0, 5000.0], [3000.0, 0.0]])),
            "green": (("y", "x"), np.array([[1200.0, 4000.0], [2500.0, 0.0]])),
            "blue": (("y", "x"), np.array([[800.0, 3000.0], [2000.0, 0.0]])),
            "nir08": (("y", "x"), np.array([[9000.0, 2000.0], [3000.0, 0.0]])),
            "swir16": (("y", "x"), np.array([[3000.0, 8000.0], [1000.0, 0.0]])),
            "lwir11": (("y", "x"), np.array([[44163.0, 45000.0], [0.0, 50000.0]])),
        },
        coords={
            "y": [1.0, 2.0],
            "x": [10.0, 20.0],
        },
    )
    return data


def test_compute_ndvi(sample_dataset):
    ndvi = compute_ndvi(sample_dataset)
    assert ndvi.shape == (2, 2)
    # At (0,0): nir=9000, red=1000 => (9000-1000)/(9000+1000) = 8000/10000 = 0.8
    assert np.isclose(ndvi.values[0, 0], 0.8, atol=1e-3)
    # Range check
    assert (ndvi.values >= -1.0).all()
    assert (ndvi.values <= 1.0).all()


def test_compute_ndbi(sample_dataset):
    ndbi = compute_ndbi(sample_dataset)
    assert ndbi.shape == (2, 2)
    # At (0,1): swir=8000, nir=2000 => (8000-2000)/(8000+2000) = 6000/10000 = 0.6
    assert np.isclose(ndbi.values[0, 1], 0.6, atol=1e-3)
    assert (ndbi.values >= -1.0).all()
    assert (ndbi.values <= 1.0).all()


def test_compute_albedo(sample_dataset):
    albedo = compute_albedo(sample_dataset)
    assert albedo.shape == (2, 2)
    assert (albedo.values >= 0.0).all()
    assert (albedo.values <= 1.0).all()


def test_compute_lst_celsius(sample_dataset):
    cfg = Config()
    lst = compute_lst_celsius(sample_dataset, cfg)
    assert lst.shape == (2, 2)
    # At (1, 0), lwir11 is 0.0 (nodata) -> should be NaN
    assert np.isnan(lst.values[1, 0])
    # At (0, 0), 44163 * 0.00341802 + 149.0 = 150.95 + 149.0 = 299.95K => 26.8°C
    assert 20.0 < lst.values[0, 0] < 35.0


def test_compute_all_features(sample_dataset):
    cfg = Config()
    features = compute_all_features(sample_dataset, cfg)
    assert set(features.keys()) == {"ndvi", "ndbi", "albedo", "lst_celsius"}
    for k, v in features.items():
        assert isinstance(v, xr.DataArray)
        assert v.shape == (2, 2)
