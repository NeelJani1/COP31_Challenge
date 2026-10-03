"""Canopy Height Maps v2 (CHMv2) using DINOv3 SAT backbone.

Predicts tree canopy heights from satellite imagery. This is a killer
feature for the hackathon — it directly gives us vegetation height data
for shadow modeling and urban cooling analysis.

Best run on RTX 4090 (24GB VRAM). Results cached as .npy files.

Reference: https://arxiv.org/abs/2603.06382
"""
import torch
import numpy as np
from PIL import Image


class CHMv2Predictor:
    """Predict canopy heights from satellite imagery using CHMv2."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(
        self, model_name: str = "facebook/dinov3-vitl16-chmv2-dpt-head", local_files_only: bool = False
    ):
        """Load the CHMv2 model from HuggingFace or local cache."""
        from transformers import (
            AutoModelForDepthEstimation,
            AutoImageProcessor,
        )

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name, local_files_only=True)
        except Exception:
            if local_files_only:
                raise
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict_canopy_height(self, image: Image.Image) -> np.ndarray:
        """Predict canopy height map from a satellite image.

        Args:
            image: RGB satellite image (PIL Image)

        Returns:
            2D numpy array of canopy heights in meters.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get height map
        height_map = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["predicted_depth"]

        return height_map.cpu().numpy()

    def predict_from_array(self, rgb_array: np.ndarray) -> np.ndarray:
        """Predict canopy heights from an RGB numpy array (H, W, 3)."""
        image = Image.fromarray(rgb_array.astype(np.uint8))
        return self.predict_canopy_height(image)

    def get_tree_mask(
        self, height_map: np.ndarray, min_height: float = 3.0
    ) -> np.ndarray:
        """Get binary mask of areas with trees (height > threshold).

        Args:
            height_map: 2D canopy height array in meters
            min_height: Minimum height to classify as tree (default 3m)

        Returns:
            Boolean mask where True = tree present
        """
        return height_map > min_height
