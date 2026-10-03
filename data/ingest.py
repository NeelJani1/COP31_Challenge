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
    """Generate realistic calibrated Landsat 9 summer scene for Parramatta.

    Accurately simulates:
    - Parramatta River corridor: cool water (22°C - 25°C), low NDVI, very low NDBI
    - Parramatta Park & green corridors: cool oasis (26°C - 30°C), high NDVI (>0.55), low NDBI
    - Parramatta CBD (Church St / Macquarie St / Westfield): intense heat island (39°C - 45°C), high NDBI, low NDVI
    - Camellia / Rosehill industrial precinct: intense heat island (41°C - 46°C), high NDBI, low NDVI
    - Suburban residential street grid with tree-lined corridors (33°C - 37°C)
    """
    ny, nx = 120, 150  # ~3.6km x 4.5km grid at 30m resolution
    lats = np.linspace(cfg.BBOX[3], cfg.BBOX[1], ny)
    lons = np.linspace(cfg.BBOX[0], cfg.BBOX[2], nx)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    m_per_deg_lat = 110900.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(cfg.LATITUDE))

    # Real Parramatta River polyline
    river_pts = np.array([
        [150.9800, -33.8065],
        [150.9870, -33.8080],
        [150.9930, -33.8105],
        [150.9985, -33.8126],
        [151.0040, -33.8130],
        [151.0100, -33.8142],
        [151.0150, -33.8152],
        [151.0200, -33.8162],
    ])

    p_x = lon_grid[..., None]
    p_y = lat_grid[..., None]
    a_x, a_y = river_pts[:-1, 0], river_pts[:-1, 1]
    b_x, b_y = river_pts[1:, 0], river_pts[1:, 1]

    ab_x = (b_x - a_x) * m_per_deg_lon
    ab_y = (b_y - a_y) * m_per_deg_lat
    ab_len_sq = ab_x**2 + ab_y**2
    ap_x = (p_x - a_x) * m_per_deg_lon
    ap_y = (p_y - a_y) * m_per_deg_lat

    t = np.clip((ap_x * ab_x + ap_y * ab_y) / (ab_len_sq + 1e-6), 0.0, 1.0)
    closest_x = a_x + t * (b_x - a_x)
    closest_y = a_y + t * (b_y - a_y)
    dist_to_river_m = np.min(
        np.sqrt(((p_x - closest_x) * m_per_deg_lon)**2 + ((p_y - closest_y) * m_per_deg_lat)**2),
        axis=-1,
    )

    water_weight = np.clip(1.0 - np.maximum(0.0, dist_to_river_m - 35.0) / 30.0, 0.0, 1.0)
    riverbank_weight = np.clip(1.0 - np.abs(dist_to_river_m - 80.0) / 45.0, 0.0, 1.0)

    # Parramatta Park
    dx_park = (lon_grid - 150.9968) * m_per_deg_lon
    dy_park = (lat_grid - (-33.8155)) * m_per_deg_lat
    park_weight = np.clip(1.0 - ((dx_park / 440.0)**2 + (dy_park / 460.0)**2), 0.0, 1.0) * (1.0 - water_weight)

    # Other parks
    prince_alfred = np.clip(
        1.0 - (((lon_grid - 151.0040) * m_per_deg_lon / 120.0)**2 + ((lat_grid - (-33.8105)) * m_per_deg_lat / 90.0)**2),
        0.0, 1.0,
    )
    ollie_webb = np.clip(
        1.0 - (((lon_grid - 150.9940) * m_per_deg_lon / 160.0)**2 + ((lat_grid - (-33.8240)) * m_per_deg_lat / 120.0)**2),
        0.0, 1.0,
    )
    green_reserves = np.maximum(prince_alfred, ollie_webb) * (1.0 - water_weight)
    total_park_weight = np.maximum(park_weight, green_reserves)

    # Parramatta CBD core
    dx_cbd = (lon_grid - 151.0055) * m_per_deg_lon
    dy_cbd = (lat_grid - (-33.8168)) * m_per_deg_lat
    cbd_weight = np.clip(1.0 - ((dx_cbd / 460.0)**2 + (dy_cbd / 400.0)**2), 0.0, 1.0) * (1.0 - water_weight) * (1.0 - total_park_weight)

    # Camellia / Rosehill Industrial core
    dx_ind = (lon_grid - 151.0160) * m_per_deg_lon
    dy_ind = (lat_grid - (-33.8220)) * m_per_deg_lat
    ind_weight = np.clip(1.0 - ((dx_ind / 400.0)**2 + (dy_ind / 520.0)**2), 0.0, 1.0) * (1.0 - water_weight)

    # Residential street texture
    rng = np.random.RandomState(42)
    street_grid = (np.sin(lons * 2000.0)[None, :] * np.cos(lats * 2200.0)[:, None]) * 0.5 + 0.5
    noise = rng.normal(0, 0.4, (ny, nx))

    # Calibrated Albedo
    albedo = (
        0.16 * (1.0 - water_weight - total_park_weight - cbd_weight - ind_weight)
        + 0.06 * water_weight
        + 0.18 * total_park_weight
        + 0.12 * cbd_weight
        + 0.11 * ind_weight
        + rng.normal(0, 0.015, (ny, nx))
    )
    albedo = np.clip(albedo, 0.05, 0.35)

    # Calibrated LST (Celsius) physically coupled to solar absorption and surface albedo
    lst_celsius = (
        35.0 * (1.0 - water_weight - total_park_weight - cbd_weight - ind_weight)
        + 23.5 * water_weight
        + 27.5 * total_park_weight
        + (40.5 + 4.5 * cbd_weight) * cbd_weight
        + (41.5 + 4.5 * ind_weight) * ind_weight
        - 2.5 * riverbank_weight * (1.0 - water_weight)
        - 1.5 * (street_grid * (1.0 - cbd_weight - ind_weight - water_weight))
        - 16.0 * (albedo - 0.15) * (1.0 - water_weight)
        + noise
    )
    lst_celsius = np.clip(lst_celsius, 22.0, 48.0)

    # Calibrated NDVI
    ndvi = (
        0.34 * (1.0 - water_weight - total_park_weight - cbd_weight - ind_weight)
        - 0.08 * water_weight
        + 0.65 * total_park_weight
        + 0.14 * cbd_weight
        + 0.11 * ind_weight
        + 0.20 * riverbank_weight * (1.0 - water_weight)
        + 0.08 * (street_grid * (1.0 - cbd_weight - ind_weight - water_weight))
        + rng.normal(0, 0.02, (ny, nx))
    )
    ndvi = np.clip(ndvi, -0.2, 0.85)

    # Calibrated NDBI
    ndbi = (
        0.12 * (1.0 - water_weight - total_park_weight - cbd_weight - ind_weight)
        - 0.28 * water_weight
        - 0.18 * total_park_weight
        + 0.38 * cbd_weight
        + 0.42 * ind_weight
        - 0.08 * riverbank_weight * (1.0 - water_weight)
        - 0.04 * (street_grid * (1.0 - cbd_weight - ind_weight - water_weight))
        + rng.normal(0, 0.02, (ny, nx))
    )
    ndbi = np.clip(ndbi, -0.4, 0.6)

    # Convert Celsius to Landsat 9 ST_B10 raw DN
    t_kelvin = lst_celsius + 273.15
    lwir11_dn = np.round((t_kelvin - cfg.LST_OFFSET) / cfg.LST_SCALE).astype(np.uint16)

    # Optical bands: mathematically inverted to yield precise NDVI, NDBI, and Albedo
    base = 12000.0
    nir_dn = (base * (1.0 + ndvi) / 2.0).clip(100, 65535).astype(np.uint16)
    red_dn = (base * (1.0 - ndvi) / 2.0).clip(100, 65535).astype(np.uint16)
    swir_dn = (nir_dn * (1.0 + ndbi) / (1.0 - ndbi + 1e-6)).clip(100, 65535).astype(np.uint16)

    target_sum = (albedo + 0.2) / 0.0000275
    rem = target_sum - 0.130 * red_dn - 0.373 * nir_dn
    bg = np.clip(rem / (0.356 + 0.085), 100, 65535).astype(np.uint16)
    blue_dn = (bg * 0.9).clip(100, 65535).astype(np.uint16)
    green_dn = (bg * 1.1).clip(100, 65535).astype(np.uint16)

    ds = xr.Dataset(
        data_vars={
            "lwir11": (("y", "x"), lwir11_dn),
            "red": (("y", "x"), red_dn),
            "green": (("y", "x"), green_dn),
            "blue": (("y", "x"), blue_dn),
            "nir08": (("y", "x"), nir_dn),
            "swir16": (("y", "x"), swir_dn),
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
