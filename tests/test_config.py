import pytest
from config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.SUBURB_NAME == "Parramatta"
    assert len(cfg.BBOX) == 4
    # BBOX is [west, south, east, north]
    west, south, east, north = cfg.BBOX
    assert west < east
    assert south < north
    assert -90 <= south <= 90
    assert -90 <= north <= 90
    assert -180 <= west <= 180
    assert -180 <= east <= 180


def test_config_solar_constants():
    cfg = Config()
    assert cfg.SOLAR_PANEL_AREA_M2 > 0
    assert cfg.SOLAR_PANEL_WATT > 0
    assert cfg.SUN_HOURS_PER_DAY > 0
    assert cfg.NDVI_TREE_THRESHOLD > 0
