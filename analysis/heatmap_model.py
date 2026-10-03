"""XGBoost microclimate model for urban heat prediction.

Learns the relationship: T_surface = f(NDVI, NDBI, Albedo)
from actual Landsat thermal data, then predicts temperature
changes when NDVI/NDBI are modified (tree planting, solar panels).
"""
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class MicroclimateModel:
    """XGBoost model predicting surface temperature from spectral features."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )
        self.feature_names = ["ndvi", "ndbi", "albedo"]
        self.is_fitted = False
        self.metrics: dict = {}

    def prepare_training_data(
        self, features: dict[str, xr.DataArray]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Flatten satellite arrays into a training DataFrame.

        Args:
            features: dict with keys 'ndvi', 'ndbi', 'albedo', 'lst_celsius'

        Returns:
            (X features DataFrame, y temperature Series)
        """
        for req in self.feature_names + ["lst_celsius"]:
            if req not in features:
                raise KeyError(f"Missing required feature '{req}' in features dict")

        data = {}
        for name in self.feature_names:
            arr = features[name].values.flatten() if hasattr(features[name], "values") else np.asarray(features[name]).flatten()
            data[name] = arr
        lst_arr = features["lst_celsius"].values.flatten() if hasattr(features["lst_celsius"], "values") else np.asarray(features["lst_celsius"]).flatten()
        data["lst_celsius"] = lst_arr

        df = pd.DataFrame(data).dropna()
        if len(df) < 10:
            raise ValueError(
                f"Insufficient valid training samples ({len(df)}). "
                "Ensure satellite data contains at least 10 non-NaN pixels."
            )

        X = df[self.feature_names]
        y = df["lst_celsius"]
        return X, y

    def train(self, features: dict[str, xr.DataArray]) -> dict:
        """Train the XGBoost model on satellite data.

        Returns:
            dict with training metrics (MAE, R-squared, feature importances)
        """
        X, y = self.prepare_training_data(features)

        test_size = 0.2 if len(X) >= 50 else 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Validation metrics
        y_pred = self.model.predict(X_val)
        self.metrics = {
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "r2": float(r2_score(y_val, y_pred)),
            "feature_importances": dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            ),
            "n_samples": len(X),
            "temp_range": (float(y.min()), float(y.max())),
        }
        return self.metrics

    def predict(
        self, ndvi: np.ndarray, ndbi: np.ndarray, albedo: np.ndarray
    ) -> np.ndarray:
        """Predict surface temperature from spectral features.

        All inputs must have the same shape. Returns predicted
        temperature in Celsius with the same shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        ndvi_arr = np.asarray(ndvi)
        ndbi_arr = np.asarray(ndbi)
        albedo_arr = np.asarray(albedo)

        if not (ndvi_arr.shape == ndbi_arr.shape == albedo_arr.shape):
            raise ValueError(
                f"Shape mismatch: ndvi {ndvi_arr.shape}, ndbi {ndbi_arr.shape}, albedo {albedo_arr.shape}"
            )

        shape = ndvi_arr.shape
        X = pd.DataFrame(
            {
                "ndvi": ndvi_arr.flatten(),
                "ndbi": ndbi_arr.flatten(),
                "albedo": albedo_arr.flatten(),
            }
        )

        # Handle NaN pixels
        mask = X.notna().all(axis=1)
        result = np.full(len(X), np.nan)
        if mask.any():
            result[mask] = self.model.predict(X[mask])

        return result.reshape(shape)

    def simulate_intervention(
        self,
        features: dict[str, xr.DataArray],
        delta_ndvi: float = 0.0,
        delta_ndbi: float = 0.0,
        delta_albedo: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Simulate the effect of an intervention on surface temperature.

        Args:
            features: Current spectral feature arrays
            delta_ndvi: Change in NDVI (e.g., +0.2 for more trees)
            delta_ndbi: Change in NDBI (e.g., -0.1 for greening)
            delta_albedo: Change in albedo (e.g., +0.1 for cool roofs)

        Returns:
            (baseline_temp, predicted_temp, avg_cooling_celsius)
        """
        ndvi_base = features["ndvi"].values if hasattr(features["ndvi"], "values") else np.asarray(features["ndvi"])
        ndbi_base = features["ndbi"].values if hasattr(features["ndbi"], "values") else np.asarray(features["ndbi"])
        albedo_base = features["albedo"].values if hasattr(features["albedo"], "values") else np.asarray(features["albedo"])

        ndvi = np.clip(ndvi_base + delta_ndvi, -1, 1)
        ndbi = np.clip(ndbi_base + delta_ndbi, -1, 1)
        albedo = np.clip(albedo_base + delta_albedo, 0, 1)

        baseline = self.predict(ndvi_base, ndbi_base, albedo_base)
        predicted = self.predict(ndvi, ndbi, albedo)

        valid_diff = (baseline - predicted)[~np.isnan(baseline - predicted)]
        avg_cooling = float(np.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
        return baseline, predicted, avg_cooling

    def save(self, path: str | None = None):
        """Save trained model to disk."""
        if path is None:
            path = str(Path(self.cfg.CACHE_DIR) / "xgb_model.joblib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        joblib.dump(
            self.metrics, path.replace(".joblib", "_metrics.joblib")
        )

    def load(self, path: str | None = None):
        """Load trained model from disk."""
        if path is None:
            path = str(Path(self.cfg.CACHE_DIR) / "xgb_model.joblib")
        self.model = joblib.load(path)
        if hasattr(self.model, "set_params"):
            try:
                self.model.set_params(n_jobs=1)
            except Exception:
                pass
        metrics_path = path.replace(".joblib", "_metrics.joblib")
        if Path(metrics_path).exists():
            self.metrics = joblib.load(metrics_path)
        self.is_fitted = True
