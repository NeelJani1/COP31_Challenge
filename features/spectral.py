"""Spectral index calculation from Landsat 9 bands.

Computes NDVI, NDBI, Albedo, and Land Surface Temperature (LST)
from Landsat 9 Collection 2 Level 2 surface reflectance and
surface temperature products.
"""
import numpy as np
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def compute_ndvi(data: xr.Dataset) -> xr.DataArray:
    """Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to +1. Trees typically > 0.45.
    """
    nir = data["nir08"].astype(float)
    red = data["red"].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi.clip(-1, 1)


def compute_ndbi(data: xr.Dataset) -> xr.DataArray:
    """Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    High values = dense urban. Low = vegetation/water.
    """
    swir = data["swir16"].astype(float)
    nir = data["nir08"].astype(float)
    ndbi = (swir - nir) / (swir + nir + 1e-10)
    return ndbi.clip(-1, 1)


def compute_albedo(data: xr.Dataset) -> xr.DataArray:
    """Simplified broadband albedo from visible bands.

    Uses Liang (2001) approximation with Landsat C2L2 reflectance scaling.
    Higher albedo = more reflective surface = cooler.
    """
    red = data["red"].astype(float)
    green = data["green"].astype(float)
    blue = data["blue"].astype(float)
    nir = data["nir08"].astype(float)
    # Liang (2001) simplified albedo approximation
    albedo = 0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * green
    # Normalize: Landsat C2L2 surface reflectance scaling
    albedo = albedo * 0.0000275 - 0.2
    return albedo.clip(0, 1)


def compute_lst_celsius(data: xr.Dataset, cfg: Config) -> xr.DataArray:
    """Convert Landsat thermal band to surface temperature in Celsius.

    Landsat 9 Collection 2 Level 2 Surface Temperature formula:
    T_kelvin = DN * 0.00341802 + 149.0
    T_celsius = T_kelvin - 273.15
    """
    lwir = data["lwir11"].astype(float)
    # Mask nodata (DN == 0)
    valid = lwir > 0
    temp_kelvin = xr.where(valid, lwir * cfg.LST_SCALE + cfg.LST_OFFSET, np.nan)
    temp_celsius = temp_kelvin - 273.15
    # Mask unreasonable values
    temp_celsius = temp_celsius.where(
        (temp_celsius > 0) & (temp_celsius < 70), np.nan
    )
    return temp_celsius


def compute_all_features(
    data: xr.Dataset, cfg: Config
) -> dict[str, xr.DataArray]:
    """Compute all spectral features in one call.

    Returns:
        Dict with keys: 'ndvi', 'ndbi', 'albedo', 'lst_celsius'
    """
    return {
        "ndvi": compute_ndvi(data),
        "ndbi": compute_ndbi(data),
        "albedo": compute_albedo(data),
        "lst_celsius": compute_lst_celsius(data, cfg),
    }
