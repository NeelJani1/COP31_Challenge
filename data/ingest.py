"""Satellite data ingestion from Microsoft Planetary Computer.

Streams Landsat 9 Collection 2 Level 2 data (surface reflectance + surface
temperature) from the Planetary Computer STAC catalog. Caches locally as
NetCDF for fast reloading during the demo.
"""
import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
import odc.stac
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def connect_stac(cfg: Config) -> pystac_client.Client:
    """Connect to Planetary Computer STAC catalog."""
    return pystac_client.Client.open(
        cfg.STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_landsat(cfg: Config, catalog=None) -> list:
    """Search for cloud-free Landsat 9 scenes over the target area.

    Returns items sorted by cloud cover (clearest first).
    """
    if catalog is None:
        catalog = connect_stac(cfg)

    search = catalog.search(
        collections=[cfg.LANDSAT_COLLECTION],
        bbox=cfg.BBOX,
        datetime=cfg.DATE_RANGE,
        query={"eo:cloud_cover": {"lt": cfg.MAX_CLOUD_COVER}},
    )
    items = list(search.items())
    # Sort by cloud cover, take clearest
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    return items


def load_bands(
    cfg: Config,
    items: list = None,
    bands: tuple[str, ...] = ("red", "green", "blue", "nir08", "swir16", "lwir11"),
) -> xr.Dataset:
    """Load specific Landsat bands as an xarray Dataset.

    Streams data directly from Planetary Computer into memory.
    Uses the clearest (first) scene from the item list.
    """
    if items is None:
        items = search_landsat(cfg)

    if not items:
        raise RuntimeError(
            f"No Landsat scenes found for bbox={cfg.BBOX}, "
            f"date_range={cfg.DATE_RANGE}, cloud_cover<{cfg.MAX_CLOUD_COVER}"
        )

    data = odc.stac.load(
        items[:1],  # Use the best (clearest) scene
        bands=list(bands),
        bbox=cfg.BBOX,
        resolution=cfg.RESOLUTION,
    )
    return data


def create_synthetic_landsat(cfg: Config) -> xr.Dataset:
    """Generate realistic synthetic Landsat 9 summer scene for Parramatta.

    Simulates:
    - High surface temperature in urban/commercial core (38°C - 48°C)
    - Cooler corridors along Parramatta River and parks (28°C - 34°C)
    - Realistic optical bands (Red, Green, Blue, NIR, SWIR, LWIR11)
    """
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
    t_kelvin = lst_celsius + 273.15
    lwir11_dn = ((t_kelvin - cfg.LST_OFFSET) / cfg.LST_SCALE).astype(np.uint16)

    # Optical bands:
    # Higher vegetation where cool, higher built-up where hot
    veg_factor = np.clip(1.0 - (lst_celsius - 28.0) / 20.0, 0.05, 0.85)

    nir_sr = 0.15 + veg_factor * 0.45
    red_sr = 0.25 - veg_factor * 0.18 + np.random.RandomState(7).uniform(0, 0.04, (ny, nx))
    green_sr = 0.18 - veg_factor * 0.08
    blue_sr = 0.12 - veg_factor * 0.05
    swir_sr = 0.30 - veg_factor * 0.20 + (1 - veg_factor) * 0.15

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


def cache_bands(cfg: Config, data: xr.Dataset) -> Path:
    """Cache the loaded bands as a NetCDF file."""
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fpath = cache_path / "landsat_bands.nc"
    data.to_netcdf(str(fpath))
    return fpath


def load_cached_bands(cfg: Config, allow_mock: bool = False) -> xr.Dataset:
    """Load cached bands from disk, optionally falling back to synthetic."""
    fpath = Path(cfg.CACHE_DIR) / "landsat_bands.nc"
    if fpath.exists():
        with xr.open_dataset(str(fpath)) as ds:
            return ds.load()
    if allow_mock:
        ds = create_synthetic_landsat(cfg)
        cache_bands(cfg, ds)
        return ds
    raise FileNotFoundError(f"No cached data at {fpath}. Run 01_download_data.py first.")
