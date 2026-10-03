"""Pre-compute baseline simulations and cache all results for lightning-fast demo loading."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from api import HeatSolarAPI


def main():
    cfg = Config()
    cache_path = Path(cfg.CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("⚡ Step 3: Pre-computing & Caching Demo Assets")
    print("=" * 65)

    print("\n[1/4] Initializing API and loading cached data...")
    api = HeatSolarAPI(cfg)
    api.load_cached_data()

    print("\n[2/4] Caching baseline temperature matrix and coordinates...")
    lats, lons, baseline_temp = api.get_baseline_heatmap()
    np.save(str(cache_path / "baseline_lats.npy"), lats)
    np.save(str(cache_path / "baseline_lons.npy"), lons)
    np.save(str(cache_path / "heatmap_baseline.npy"), baseline_temp)
    print(f"       ✅ Heatmap size: {baseline_temp.shape}, Mean Temp: {np.nanmean(baseline_temp):.1f}°C")

    print("\n[3/4] Caching building solar capacities...")
    solar_df = pd.DataFrame({
        "roof_area_m2": api.buildings["roof_area_m2"],
        "unshaded_area_m2": api.buildings["unshaded_area_m2"],
        "n_panels": api.buildings["n_panels"],
        "peak_kw": api.buildings["peak_kw"],
        "annual_kwh": api.buildings["annual_kwh"],
    })
    solar_csv_path = cache_path / "solar_capacity.csv"
    solar_df.to_csv(str(solar_csv_path), index=False)
    print(f"       ✅ Solar summary for {len(solar_df)} buildings saved to {solar_csv_path}")
    print(f"       • Total Potential Peak Capacity: {solar_df['peak_kw'].sum() / 1000:.2f} MW")
    print(f"       • Total Potential Clean Generation: {solar_df['annual_kwh'].sum() / 1e6:.2f} GWh/year")

    print("\n[4/4] Testing scenario simulations (Tree +20%, Solar +30%)...")
    res = api.simulate(tree_pct=20.0, solar_pct=30.0, cool_roof_albedo=0.05)
    print(f"       • Average Cooling: {res.avg_cooling_celsius:.2f} °C")
    print(f"       • Max Localized Cooling: {res.max_cooling_celsius:.2f} °C")
    print(f"       • Hotspots (>40°C) reduced: {res.hotspot_count_before} -> {res.hotspot_count_after}")
    print(f"       • Solar Capacity deployed: {res.total_solar_mw:.2f} MW")
    print(f"       • Annual CO2 emissions offset: {res.co2_offset_tonnes_year:.1f} tonnes/year")

    print("\n🎉 Step 3 Complete! Your backend is fully calibrated and pre-cached.")
    print("🚀 You can now launch: streamlit run app.py")


if __name__ == "__main__":
    main()
