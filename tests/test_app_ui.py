"""Automated UI and stability tests for Streamlit application."""
import pytest
import sys
import ctypes
from streamlit.testing.v1 import AppTest


def test_freelist_patch_active():
    """Verify that the CPython 3.14 freelist is neutralized."""
    if sys.version_info[:2] != (3, 14):
        pytest.skip("Test specifically targets Python 3.14 freelist")
    pyapi = ctypes.PyDLL(None)
    pyapi.PyThreadState_Get.restype = ctypes.c_void_p
    tstate = pyapi.PyThreadState_Get()
    assert tstate, "Failed to get PyThreadState"
    interp = ctypes.c_void_p.from_address(tstate + 16).value
    assert interp, "Failed to get PyInterpreterState"
    head = ctypes.c_void_p.from_address(interp + 0x2d88).value or 0
    count = ctypes.c_long.from_address(interp + 0x2d90).value
    assert head == 0, f"Freelist head must be NULL to avoid segfaults, got {hex(head)}"
    assert count == 1, f"Freelist count must be non-zero (1) to force PyMem_Free, got {count}"


def test_app_initial_render():
    """Verify app.py renders without uncaught exceptions."""
    at = AppTest.from_file("../app.py")
    at.run()
    assert not at.exception, f"App threw exception: {at.exception}"
    assert len(at.slider) >= 4
    assert len(at.checkbox) >= 2


def test_app_slider_and_toggle_interactions():
    """Simulate moving every slider and toggling every checkbox/radio."""
    at = AppTest.from_file("../app.py")
    at.run()
    assert not at.exception

    # Test tree canopy slider
    at.slider[0].set_value(50).run()
    assert not at.exception

    # Test solar coverage slider
    at.slider[1].set_value(75).run()
    assert not at.exception

    # Test cool roof slider
    at.slider[2].set_value(0.15).run()
    assert not at.exception

    # Test overlay opacity slider
    at.slider[3].set_value(0.40).run()
    assert not at.exception

    # Toggle building polygons
    at.checkbox[1].set_value(True).run()
    assert not at.exception

    # Toggle heatmap off then on
    at.checkbox[0].set_value(False).run()
    assert not at.exception
    at.checkbox[0].set_value(True).run()
    assert not at.exception

    # Radio view change
    at.radio[0].set_value("Cooling Difference (Delta)").run()
    assert not at.exception
    at.radio[0].set_value("Baseline (Observed)").run()
    assert not at.exception

    # Basemap selectbox change
    at.selectbox[1].set_value("OpenStreetMap (Standard)").run()
    assert not at.exception
    at.selectbox[1].set_value("Topographic (OpenTopoMap)").run()
    assert not at.exception


def test_app_extreme_slider_boundaries():
    """Test slider boundary extremes (0% and 100%)."""
    at = AppTest.from_file("../app.py")
    at.run()
    assert not at.exception

    # 0% tree, 0% solar, 0.0 cool roof
    at.slider[0].set_value(0).run()
    at.slider[1].set_value(0).run()
    at.slider[2].set_value(0.0).run()
    assert not at.exception

    # Max tree, max solar, max cool roof
    at.slider[0].set_value(80).run()
    at.slider[1].set_value(90).run()
    at.slider[2].set_value(0.25).run()
    assert not at.exception
