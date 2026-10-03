"""Sydney Urban Heat & Solar Forecaster — Streamlit Web Application.

Run:
    streamlit run app.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend — prevents segfault on thread reentry
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image

from config import Config
from api import HeatSolarAPI

# Page configuration
st.set_page_config(
    page_title="Sydney Urban Heat & Solar Forecaster | Build for 2035",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished hackathon presentation
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3648;
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #00e5ff;
    }
    .metric-lbl {
        font-size: 13px;
        color: #9aa5b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_api() -> HeatSolarAPI:
    """Initialize API and load cached assets."""
    api = HeatSolarAPI()
    try:
        api.load_cached_data()
    except Exception:
        # If cache is missing, train live or from mock data
        api.load_live_data()
    return api


def main():
    st.title("☀️ Sydney Urban Heat & Solar Forecaster")
    st.caption("Addressing COP31 Priorities: **Resilient Cities & Buildings** and **Electrification**")

    api = get_api()

    # Sidebar Controls
    with st.sidebar:
        st.header("🕹️ Intervention Controls")
        st.markdown("Simulate green infrastructure and rooftop solar retrofits across Western Sydney.")

        suburb = st.selectbox("Target Suburb", [api.cfg.SUBURB_NAME, "Penrith (Coming Soon)", "Blacktown (Coming Soon)"])

        st.subheader("1. Urban Greening")
        tree_target = st.slider("Tree Canopy Increase (%)", min_value=0, max_value=80, value=25, step=5,
                                help="Simulates planting trees in heat-stressed streets and corridors.")

        st.subheader("2. Rooftop Electrification")
        solar_target = st.slider("Rooftop Solar Coverage (%)", min_value=0, max_value=90, value=35, step=5,
                                 help="Percentage of unshaded building roof area equipped with solar PV.")

        st.subheader("3. Cool Roofs (High Albedo)")
        cool_roof = st.slider("Cool Roof Albedo Boost", min_value=0.0, max_value=0.25, value=0.05, step=0.01,
                              help="Reflective coatings applied to commercial and industrial roofs.")

        st.markdown("---")
        st.subheader("🗺️ Map Overlays")
        basemap_style = st.selectbox(
            "Basemap Style",
            ["Satellite (Esri World Imagery)", "OpenStreetMap (Standard)", "Topographic (OpenTopoMap)"],
            index=0,
            help="Choose basemap tile provider. Completely free, no API key required.",
        )
        show_heatmap = st.checkbox("Show Microclimate Heatmap", value=True)
        heatmap_mode = st.radio("Heatmap View", ["After Intervention", "Baseline (Observed)", "Cooling Difference (Delta)"], index=0)
        show_buildings = st.checkbox("Show Building Vectors", value=False)
        heatmap_opacity = st.slider("Overlay Opacity", 0.2, 0.9, 0.65)

    # Run the simulation through backend API
    result = api.simulate(
        tree_pct=tree_target,
        solar_pct=solar_target,
        cool_roof_albedo=cool_roof,
    )

    # Top KPI Metrics Dashboard
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Average Cooling",
            value=f"-{result.avg_cooling_celsius:.2f} °C",
            delta=f"-{result.max_cooling_celsius:.1f} °C Peak Spot",
            delta_color="normal",
        )
    with m2:
        st.metric(
            label="Peak Surface Temp",
            value=f"{np.nanmax(result.predicted_temp_map):.1f} °C",
            delta=f"-{np.nanmax(result.baseline_temp_map) - np.nanmax(result.predicted_temp_map):.1f} °C",
            delta_color="inverse",
        )
    with m3:
        st.metric(
            label="Hotspots (>40°C)",
            value=f"{result.hotspot_count_after:,} px",
            delta=f"{result.hotspot_count_after - result.hotspot_count_before:,} px",
            delta_color="inverse",
        )
    with m4:
        st.metric(
            label="Solar Generation Capacity",
            value=f"{result.total_solar_mw:.1f} MWp",
            delta=f"{result.total_panels:,} Panels",
        )
    with m5:
        st.metric(
            label="Annual CO₂ Offset",
            value=f"{result.co2_offset_tonnes_year:,.0f} t/yr",
            delta=f"{result.total_annual_kwh / 1e6:.1f} GWh/yr",
        )

    st.markdown("---")

    # Main Visual Tabs
    tab_map, tab_analytics, tab_explain = st.tabs(["🗺️ Interactive Simulator", "📊 Impact Analytics", "🧠 ML Physics & DINOv3"])

    with tab_map:
        col_map, col_info = st.columns([3, 1])

        with col_map:
            # Center of map
            center_lat, center_lon = api.cfg.LATITUDE, api.cfg.LONGITUDE

            if basemap_style == "Satellite (Esri World Imagery)":
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=14,
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri World Imagery",
                    control_scale=True,
                )
            elif basemap_style == "Topographic (OpenTopoMap)":
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=14,
                    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                    attr="OpenTopoMap",
                    control_scale=True,
                )
            else:
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=14,
                    tiles="OpenStreetMap",
                    control_scale=True,
                )

            # Render Raster Temperature Overlay
            if show_heatmap:
                if heatmap_mode == "Baseline (Observed)":
                    render_arr = result.baseline_temp_map
                    vmin, vmax = 28.0, 48.0
                    cmap_name = "inferno"
                    legend_caption = "Baseline Land Surface Temperature (°C)"
                elif heatmap_mode == "After Intervention":
                    render_arr = result.predicted_temp_map
                    vmin, vmax = 28.0, 48.0
                    cmap_name = "inferno"
                    legend_caption = "Forecasted Land Surface Temperature (°C)"
                else:  # Cooling Difference
                    render_arr = result.baseline_temp_map - result.predicted_temp_map
                    vmin, vmax = 0.0, 4.0
                    cmap_name = "YlGnBu"
                    legend_caption = "Cooling Impact (°C Drop)"

                # Map array to RGBA
                render_arr = np.atleast_2d(np.squeeze(render_arr))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap(cmap_name)
                rgba = cmap(norm(np.nan_to_num(render_arr, nan=vmin)))
                rgba[..., 3] = np.where(np.isnan(render_arr), 0.0, heatmap_opacity)

                # Convert to PNG base64
                img = Image.fromarray((rgba * 255).astype(np.uint8))
                buff = io.BytesIO()
                img.save(buff, format="PNG")
                img_b64 = f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

                # Bounds
                bounds = [
                    [float(np.min(result.lat_grid)), float(np.min(result.lon_grid))],
                    [float(np.max(result.lat_grid)), float(np.max(result.lon_grid))],
                ]

                folium.raster_layers.ImageOverlay(
                    image=img_b64,
                    bounds=bounds,
                    opacity=heatmap_opacity,
                    name="Temperature Heatmap",
                    interactive=True,
                ).add_to(m)

                # Automatically frame the heatmap bounds
                m.fit_bounds(bounds)

                # Legend
                colormap = cm.LinearColormap(
                    colors=[mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 10)],
                    vmin=vmin,
                    vmax=vmax,
                    caption=legend_caption,
                )
                colormap.add_to(m)

            # Optional Building Polygons
            if show_buildings and api.buildings is not None:
                # Add simplified polygons for performance
                sample_bldgs = api.buildings.iloc[:120]  # sample top 120 for smooth rendering
                for _, b in sample_bldgs.iterrows():
                    kw = b.get("peak_kw", 0) * (solar_target / 100.0)
                    folium.GeoJson(
                        b.geometry.__geo_interface__,
                        style_function=lambda x, _kw=kw: {
                            "fillColor": "#00e5ff" if _kw > 5 else "#ff9100",
                            "color": "#ffffff",
                            "weight": 0.8,
                            "fillOpacity": 0.5,
                        },
                        tooltip=f"Roof: {b.get('roof_area_m2', 0):.0f} m² | Potential: {kw:.1f} kWp",
                    ).add_to(m)

            # Render map smoothly without triggering rerun cycles on scroll
            components.html(m._repr_html_(), height=620)

        with col_info:
            st.markdown("### 🔍 Live Site Intelligence")
            st.info(f"**Bounding Box:** `{api.cfg.BBOX}`\n\n**Satellite Source:** Landsat 9 (30m thermal & optical) via Planetary Computer STAC.")

            st.markdown("#### ⚡ Clean Energy Impact")
            st.write(f"• **Suitable Buildings:** {result.n_buildings_with_solar:,}")
            st.write(f"• **Installed Solar:** {result.total_solar_capacity_kw:,.0f} kWp")
            st.write(f"• **Homes Powered Approx:** {int(result.total_annual_kwh / 6500):,} homes")

            st.markdown("#### 🌿 Thermal Resilience")
            st.write(f"• **Avg Surface Cooling:** {result.avg_cooling_celsius:.2f} °C")
            st.write(f"• **Severe Hotspot Relief:** {result.hotspot_count_before - result.hotspot_count_after} pixels")

    with tab_analytics:
        st.subheader("Temperature Distributions: Baseline vs Intervention")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            b_valid = result.baseline_temp_map.flatten()
            p_valid = result.predicted_temp_map.flatten()
            ax.hist(b_valid[~np.isnan(b_valid)], bins=30, alpha=0.6, color="salmon", label="Baseline (Observed)")
            ax.hist(p_valid[~np.isnan(p_valid)], bins=30, alpha=0.6, color="lightseagreen", label="After Intervention")
            ax.set_xlabel("Surface Temperature (°C)")
            ax.set_ylabel("Pixel Frequency")
            ax.legend()
            ax.set_title("Shift Toward Cooler Microclimate")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.markdown("#### Key Takeaways for City Planners")
            st.markdown(
                f"""
                - **Heat Island Mitigation:** Adding **{tree_target}%** tree canopy directly cools ambient street temperatures by up to **{result.max_cooling_celsius:.1f}°C**.
                - **Unshaded Solar Optimization:** Geometric shadow exclusion ensures solar panels are positioned only on rooftops that remain unshaded throughout daylight hours.
                - **Dual Win for COP31:** Achieves both building resilience (lowering AC demand) and clean power electrification on the same urban footprint.
                """
            )

    with tab_explain:
        st.subheader("Model Architecture & Feature Attribution")
        metrics = api.get_model_metrics()
        if metrics:
            st.write(f"**Model Type:** `XGBRegressor` trained on real Landsat 9 calibrated pixels ({metrics.get('n_samples', 0):,} data points).")
            st.write(f"**Validation Performance:** MAE = `{metrics.get('mae', 0):.2f}°C` | R² = `{metrics.get('r2', 0):.3f}`")

            if "feature_importances" in metrics:
                st.write("**Feature Importances:**")
                st.bar_chart(metrics["feature_importances"])

        st.markdown(
            """
            #### Foundation Model Integration (DINOv3 SAT & CHMv2)
            - **Canopy Height Mapping:** Employs Meta's `dinov3_vitl16_chmv2` backbone to estimate 1-meter tree canopy heights from satellite imagery.
            - **Shadow Vector Tracing:** Intersects solar altitude and azimuth (`pvlib`) with tree height masks to model hourly shading on residential rooftops.
            - **Zero-Shot Segmentation:** Can segment solar PV panels and cool roof materials with zero human training annotations.
            """
        )


if __name__ == "__main__":
    main()
