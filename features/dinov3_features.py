"""DINOv3 SAT-493M feature extraction for satellite imagery.

Uses Meta's DINOv3 vision foundation model pretrained on satellite / vision
imagery to extract dense, high-quality features. These features can be used
for downstream tasks like vegetation classification, solar panel detection,
and urban land-use segmentation.

Requires: DINOv3 model weights (can use locally cached models).
Falls back gracefully if weights are unavailable.
"""
import torch
import numpy as np
from PIL import Image
from typing import Optional


def _check_model_available(model_name: str) -> bool:
    """Check if a HuggingFace model or local cached checkpoint is accessible."""
    try:
        from transformers import AutoModel
        try:
            AutoModel.from_pretrained(model_name, local_files_only=True)
            return True
        except Exception:
            pass
        AutoModel.from_pretrained(model_name)
        return True
    except Exception:
        return False


class DINOv3SATExtractor:
    """Extract dense features from satellite imagery using DINOv3."""

    def __init__(self, device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.model = None
        self.processor = None
        self.transform = None

    def load_model(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        local_files_only: bool = False,
    ):
        """Load the DINOv3 model from local cache or HuggingFace."""
        from transformers import AutoImageProcessor, AutoModel
        from torchvision.transforms import v2

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name, local_files_only=True
            )
        except Exception:
            if not local_files_only:
                try:
                    self.processor = AutoImageProcessor.from_pretrained(model_name)
                except Exception:
                    self.processor = None
            else:
                self.processor = None

        if self.processor is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((224, 224), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

        try:
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            if local_files_only:
                raise
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract patch-level features from a satellite image tile.

        Returns:
            (H, W, D) array of dense features where D is the embedding dimension.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt").to(
                self.device
            )
            pixel_values = (
                inputs["pixel_values"]
                if isinstance(inputs, dict)
                else inputs.pixel_values
            )
        else:
            pixel_values = self.transform(image).unsqueeze(0).to(self.device, torch.float32)

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)

        # Get patch tokens (exclude CLS + register tokens)
        num_registers = getattr(self.model.config, "num_register_tokens", 4)
        num_prefix = 1 + num_registers
        patch_tokens = outputs.last_hidden_state[:, num_prefix:, :]
        # Reshape to spatial grid
        n_patches = patch_tokens.shape[1]
        h = int(np.round(n_patches**0.5))
        if h * h == n_patches:
            features = patch_tokens.reshape(1, h, h, -1)
        else:
            features = patch_tokens.unsqueeze(1)
        return features[0].cpu().numpy()

    def extract_features_from_array(self, rgb_array: np.ndarray) -> np.ndarray:
        """Extract features from an RGB numpy array (H, W, 3)."""
        image = Image.fromarray(rgb_array.astype(np.uint8))
        return self.extract_features(image)
