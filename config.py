"""Central configuration for Sydney Urban Heat & Solar Forecaster."""
import os
from dataclasses import dataclass, field

# Ensure matplotlib uses writable tmp directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")



@dataclass
class Config:
    # === Target Area ===
    SUBURB_NAME: str = "Parramatta"
    # Bounding box: [west, south, east, north]
    BBOX: list[float] = field(
        default_factory=lambda: [150.98, -33.83, 151.02, -33.80]
    )
    LATITUDE: float = -33.815
    LONGITUDE: float = 151.005
    CRS_UTM: str = "EPSG:32756"  # UTM Zone 56S (Sydney)
    CRS_WGS84: str = "EPSG:4326"

    # === Satellite Data ===
    STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    LANDSAT_COLLECTION: str = "landsat-c2-l2"
    DATE_RANGE: str = "2024-01-01/2024-02-28"  # Summer for max heat
    MAX_CLOUD_COVER: int = 10
    RESOLUTION: int = 30  # meters

    # === Thresholds ===
    NDVI_TREE_THRESHOLD: float = 0.45  # NDVI > this = tree canopy
    TREE_SHADE_BUFFER_M: float = 5.0  # Shadow buffer around trees
    SOLAR_PANEL_AREA_M2: float = 1.7  # Standard residential panel
    SOLAR_PANEL_WATT: int = 400  # Watts per panel
    SUN_HOURS_PER_DAY: float = 5.5  # Average peak sun hours Sydney

    # === DINOv3 ===
    DINOV3_REPO: str = "/home/neel/dinov3-main/dinov3-main"
    DINOV3_SAT_HF_MODEL: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    CHMV2_HF_MODEL: str = "facebook/dinov3-vitl16-chmv2-dpt-head"

    # === Cache Paths ===
    CACHE_DIR: str = "cache"

    # === LST Conversion (Landsat 9 Collection 2 Level 2) ===
    LST_SCALE: float = 0.00341802
    LST_OFFSET: float = 149.0
