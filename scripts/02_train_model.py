"""Train the XGBoost microclimate model on cached satellite data."""
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data.ingest import load_cached_bands
from features.spectral import compute_all_features
from analysis.heatmap_model import MicroclimateModel


def main():
    cfg = Config()

    print("=" * 65)
    print(f"🧠 Step 2: Training XGBoost Microclimate Regressor")
    print(f"    Target physics: LST = f(NDVI, NDBI, Albedo)")
    print("=" * 65)

    print("\n[1/3] Loading cached satellite bands...")
    data = load_cached_bands(cfg)
    print(f"       Variables: {list(data.data_vars.keys())}")

    print("\n[2/3] Computing spectral indices (NDVI, NDBI, Albedo, LST)...")
    features = compute_all_features(data, cfg)

    for name, arr in features.items():
        v = arr.values
        valid_v = v[~np.isnan(v)]
        if len(valid_v) > 0:
            print(f"       • {name:12s}: shape={v.shape} | range=[{valid_v.min():.3f}, {valid_v.max():.3f}] | mean={valid_v.mean():.3f}")
        else:
            print(f"       • {name:12s}: shape={v.shape} (all NaN)")

    print("\n[3/3] Fitting XGBoost Regressor and evaluating on 20% holdout...")
    model = MicroclimateModel(cfg)
    metrics = model.train(features)
    model.save()

    print("\n" + "-" * 50)
    print(f"🎯 Validation Results:")
    print(f"   • Mean Absolute Error (MAE): {metrics['mae']:.2f} °C")
    print(f"   • Coefficient of Det (R²):  {metrics['r2']:.3f}")
    print(f"   • Samples Trained On:       {metrics['n_samples']:,} pixels")
    print(f"   • Temperature Span:         {metrics['temp_range'][0]:.1f}°C to {metrics['temp_range'][1]:.1f}°C")
    print(f"   • Feature Importances:      {metrics['feature_importances']}")
    print("-" * 50)
    print("💾 Model & metrics saved to cache/xgb_model.joblib")
    print("\n🎉 Step 2 Complete! Run: python scripts/03_cache_results.py")


if __name__ == "__main__":
    main()
