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


def cache_bands(cfg: Config, data: xr.Dataset) -> Path:
    """Cache the loaded bands as a NetCDF file."""
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    fpath = cache_path / "landsat_bands.nc"
    data.to_netcdf(str(fpath))
    return fpath


def load_cached_bands(cfg: Config) -> xr.Dataset:
    """Load cached bands from disk."""
    fpath = Path(cfg.CACHE_DIR) / "landsat_bands.nc"
    if fpath.exists():
        return xr.open_dataset(str(fpath))
    raise FileNotFoundError(f"No cached data at {fpath}. Run 01_download_data.py first.")
