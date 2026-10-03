import pytest
import numpy as np
from PIL import Image
import torch
from features.dinov3_features import DINOv3SATExtractor, _check_model_available
from features.chmv2_canopy import CHMv2Predictor
from features.dinov3_segmentation import ZeroShotSegmentor, URBAN_CLASSES


def test_dinov3_check_model_available():
    # Known model in local HuggingFace cache
    res = _check_model_available("facebook/dinov3-vits16-pretrain-lvd1689m")
    assert res is True
    # Non-existent model
    res_fake = _check_model_available("nonexistent_model_xyz_12345")
    assert res_fake is False


def test_dinov3_feature_extractor_cpu_or_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv3SATExtractor(device=device)
    extractor.load_model("facebook/dinov3-vits16-pretrain-lvd1689m", local_files_only=True)

    img = Image.new("RGB", (224, 224), color=(80, 120, 160))
    feats = extractor.extract_features(img)
    assert feats.ndim == 3
    # 224 / 16 = 14 patches per axis
    assert feats.shape[0] == 14
    assert feats.shape[1] == 14
    assert feats.shape[2] == 384  # ViT-S embedding dim


def test_chmv2_fallback_from_ndvi():
    ndvi = np.array([[0.1, 0.4], [0.6, 0.8]])
    heights = CHMv2Predictor.predict_canopy_height_from_ndvi(ndvi, threshold=0.35)
    assert heights[0, 0] == 0.0  # below threshold
    assert heights[0, 1] > 0.0  # above threshold
    assert heights[1, 1] > heights[1, 0]  # higher NDVI -> taller canopy
    predictor = CHMv2Predictor()
    mask = predictor.get_tree_mask(heights, min_height=3.0)
    assert mask.dtype == bool


def test_segmentor_spectral_fallback():
    ndvi = np.array([[0.1, 0.5], [0.3, 0.6]])
    ndbi = np.array([[0.2, -0.2], [0.1, -0.3]])

    res = ZeroShotSegmentor.detect_spectral_fallback(ndvi, ndbi)
    assert "segmentation_map" in res
    assert "solar_mask" in res
    assert "tree_mask" in res
    assert res["segmentation_map"].shape == ndvi.shape


def test_segmentor_spectral_fallback_nans():
    ndvi = np.array([[np.nan, 0.5], [0.3, 0.6]])
    ndbi = np.array([[0.2, np.nan], [0.1, -0.3]])

    res = ZeroShotSegmentor.detect_spectral_fallback(ndvi, ndbi)
    assert res["segmentation_map"][0, 0] == 255  # NoData
    assert res["segmentation_map"][0, 1] == 255  # NoData
    assert not res["solar_mask"][0, 0]
    assert not res["solar_mask"][0, 1]

