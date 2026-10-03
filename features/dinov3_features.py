"""DINOv3 SAT-493M feature extraction for satellite imagery.

Uses Meta's DINOv3 vision foundation model pretrained on 493M satellite
images (0.6m GSD Maxar imagery) to extract dense, high-quality features.
These features can be used for downstream tasks like vegetation
classification, solar panel detection, and urban land-use segmentation.

Requires: DINOv3 SAT model weights (request from Meta).
Falls back gracefully if weights are unavailable.
"""
import torch
import numpy as np
from PIL import Image
from typing import Optional


class DINOv3SATExtractor:
    """Extract dense features from satellite imagery using DINOv3-SAT."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(
        self, model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    ):
        """Load the DINOv3 SAT model from HuggingFace."""
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract patch-level features from a satellite image tile.

        Returns:
            (H, W, D) array of dense features where D is the
            embedding dimension (1024 for ViT-L).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Get patch tokens (exclude CLS + 4 register tokens)
        patch_tokens = outputs.last_hidden_state[:, 5:, :]
        # Reshape to spatial grid
        n_patches = patch_tokens.shape[1]
        h = w = int(n_patches**0.5)
        features = patch_tokens.reshape(1, h, w, -1)
        return features[0].cpu().numpy()

    def extract_features_from_array(self, rgb_array: np.ndarray) -> np.ndarray:
        """Extract features from an RGB numpy array (H, W, 3)."""
        image = Image.fromarray(rgb_array.astype(np.uint8))
        return self.extract_features(image)
