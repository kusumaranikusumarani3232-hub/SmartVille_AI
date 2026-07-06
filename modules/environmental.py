"""
SmartVille AI — Module 2: Environmental Analytics
===================================================
Deep-dive environmental analytics: AQI, PM, noise, water quality, maps,
ML-powered AQI prediction, SHAP explainability, and anomaly detection.
All computation is LOCAL. Gemini is NOT called here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    CHART_PALETTE,
    CITY_CENTER,
    DISTRICT_META,
    DISTRICTS,
    aqi_category,
    noise_category,
    water_category,
)
from ml.explainability import (
    compute_shap_values,
    plot_feature_importance,
    plot_shap_waterfall,
)
from ml.predict import predict_aqi
from utils.data_loader import load_uploaded_env
from utils.ui_components import (
    aqi_gauge_html,
    apply_chart_theme,
    metric_card,
    prediction_card,
    section_header,
    alert_card,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_environmental(
    env_df: pd.DataFrame,
    district: str,
    start_date,
    end_date,
) -> None:
    """Render the Environmental Analytics tab."""

    # ── Filter ────────────────────────────────────────────────────────────
    filtered = env_df.copy()
    filtered = filtered[
        (filtered["date"] >= pd.Timestamp(start_date)) &
        (filtered["date"] <= pd.Timestamp(end_date))
    ]
    if district != "All Districts":
        filtered = filtered[filtered["district"] == district]

    if len(filtered) == 0:
        st.warning("No environmental data for selected filters.")
        return

    _render_status_banner(filtered, district)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_env_metrics(filtered)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_aqi_trend(filtered, district)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_map_and_breakdown(env_df, start_date, end_date)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_ml_prediction(district, start_date)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_anomaly_detection(filtered)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_csv_upload()


# ─────────────────────────────────────────────────────────────────────────────
# Status Banner
# ─────────────────────────────────────────────────────────────────────────────

def _render_status_banner(filtered: pd.DataFrame, district: str) -> None:
    # Most recent AQI available
    recent = filtered.sort_values("date").groupby("district").last().reset_index()
    avg_aqi = float(recent["aqi"].mean()) if len(recent) > 0 else 0.0

    aqi_gauge_html(avg_aqi, district)


# ─────────────────────────────────────────────────────────────────────────────
# Env Metric Cards
# ─────────────────────────────────────────────────────────────────────────────

def _render_env_metrics(filtered: pd.DataFrame) -> None:
    section_header("📊 Environmental Indicators", "Averaged over selected period and district")

    avg_pm25  = filtered["pm25"].mean()
    avg_pm10  = filtered["pm10"].mean()
    avg_noise = filtered["noise_db"].mean()
    avg_water = filtered["water_quality"].mean()
    avg_wind  = filtered["wind_speed"].mean()

    _, noise_color = noise_category(avg_noise)
    _, water_color = water_category(avg_water)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        pm25_class = "danger" if avg_pm25 > 55 else "warning" if avg_pm25 > 25 else "success"
        metric_card("🔵", f"{avg_pm25:.1f}", "PM2.5 (µg/m³)", card_class=pm25_class)
    with c2:
        pm10_class = "danger" if avg_pm10 > 150 else "warning" if avg_pm10 > 75 else "success"
        metric_card("🟤", f"{avg_pm10:.1f}", "PM10 (µg/m³)", card_class=pm10_class)
    with c3:
        noise_class = "danger" if avg_noise > 70 else "warning" if avg_noise > 60 else "success"
        metric_card("🔊", f"{avg_noise:.1f}", "Noise Level (dB)", card_class=noise_class)
    with c4:
        water_class = "success" if avg_water >= 80 else "warning" if avg_water >= 65 else "danger"
        metric_card("💧", f"{avg_water:.1f}", "Water Quality Index", card_class=water_class)
    with c5:
        metric_card("💨", f"{avg_wind:.1f}", "Wind Speed (km/h)", card_class="info")


# ─────────────────────────────────────────────────────────────────────────────
# AQI Trend Chart
# ─────────────────────────────────────────────────────────────────────────────

def _render_aqi_trend(filtered: pd.DataFrame, district: str) -> None:
    section_header("📉 AQI Time Series", "Daily air quality with rolling averages and threshold bands")

    view = st.radio(
        "View", ["All Districts", "By District", "PM2.5 vs PM10"],
        horizontal=True, key="env_view",
        label_visibility="collapsed",
    )

    if view == "All Districts" or district != "All Districts":
        daily = filtered.groupby("date")[["aqi", "pm25", "pm10"]].mean().reset_index()
        daily["aqi_7d"] = daily["aqi"].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["aqi"],
            name="Daily AQI", mode="lines",
            line=dict(color="rgba(108,99,255,0.4)", width=1.2),
            fill="tozeroy", fillcolor="rgba(108,99,255,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["aqi_7d"],
            name="7-Day Avg", mode="lines",
            line=dict(color="#6c63ff", width=2.8),
        ))
        # Threshold bands
        fig.add_hrect(y0=0,   y1=50,  fillcolor="rgba(0,230,118,0.05)",  line_width=0, annotation_text="Good")
        fig.add_hrect(y0=50,  y1=100, fillcolor="rgba(255,238,88,0.04)", line_width=0, annotation_text="Moderate")
        fig.add_hrect(y0=100, y1=150, fillcolor="rgba(255,152,0,0.04)",  line_width=0, annotation_text="Sensitive")
        fig.add_hrect(y0=150, y1=500, fillcolor="rgba(255,71,87,0.05)",  line_width=0, annotation_text="Unhealthy")

        fig = apply_chart_theme(fig, title="Air Quality Index Over Time", height=380)
        fig.update_layout(yaxis_title="AQI", xaxis_title="")

    elif view == "By District":
        dist_daily = filtered.groupby(["date", "district"])["aqi"].mean().reset_index()
        colors = {d: DISTRICT_META[d]["color"] for d in DISTRICTS}
        fig = px.line(
            dist_daily, x="date", y="aqi", color="district",
            color_discrete_map=colors,
            labels={"date": "", "aqi": "AQI", "district": "District"},
        )
        fig = apply_chart_theme(fig, title="AQI by District", height=380)

    else:  # PM comparison
        daily = filtered.groupby("date")[["pm25", "pm10"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["pm25"], name="PM2.5",
                                 line=dict(color="#6c63ff", width=2.5)))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["pm10"], name="PM10",
                                 line=dict(color="#00d4ff", width=2.5)))
        fig.add_hline(y=25,  line=dict(color="#6c63ff", dash="dash", width=1.2),
                      annotation_text="PM2.5 WHO limit (25)", annotation_font=dict(color="#6c63ff", size=10))
        fig.add_hline(y=50,  line=dict(color="#00d4ff", dash="dash", width=1.2),
                      annotation_text="PM10 WHO limit (50)",  annotation_font=dict(color="#00d4ff", size=10))
        fig = apply_chart_theme(fig, title="Particulate Matter Trends", height=380)
        fig.update_layout(yaxis_title="µg/m³")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Map + Box Plot
# ─────────────────────────────────────────────────────────────────────────────

def _render_map_and_breakdown(env_df: pd.DataFrame, start_date, end_date) -> None:
    section_header("🗺️ Geographic Distribution", "AQI heatmap across SmartVille districts")

    col_map, col_box = st.columns([3, 2])

    with col_map:
        # Aggregate AQI per district per day → use latest point per district for map
        env_filt = env_df[
            (env_df["date"] >= pd.Timestamp(start_date)) &
            (env_df["date"] <= pd.Timestamp(end_date))
        ]
        map_data = env_filt.groupby("district").agg(
            aqi=("aqi", "mean"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        ).reset_index()
        map_data["aqi_label"], map_data["color"] = zip(
            *map_data["aqi"].apply(aqi_category)
        )
        map_data["district_label"] = map_data["district"].map(
            lambda d: DISTRICT_META[d]["label"]
        )

        fig_map = px.scatter_mapbox(
            map_data,
            lat="lat", lon="lon",
            size="aqi",
            color="aqi",
            color_continuous_scale=["#00e676", "#ffee58", "#ff9800", "#ff4757", "#9c27b0"],
            range_color=[0, 200],
            hover_name="district_label",
            hover_data={"aqi": True, "aqi_label": True, "lat": False, "lon": False},
            size_max=45,
            zoom=11,
            center=CITY_CENTER,
            mapbox_style="open-street-map",
            labels={"aqi": "AQI"},
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=380,
            coloraxis_colorbar=dict(
                title="AQI",
                tickfont=dict(color="#94a3b8"),
                title_font=dict(color="#94a3b8"),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    with col_box:
        env_filt = env_df[
            (env_df["date"] >= pd.Timestamp(start_date)) &
            (env_df["date"] <= pd.Timestamp(end_date))
        ]
        d_colors = [DISTRICT_META[d]["color"] for d in DISTRICTS]
        fig_box = go.Figure()
        for dist, color in zip(DISTRICTS, d_colors):
            data = env_filt[env_filt["district"] == dist]["aqi"].dropna()
            if len(data) > 0:
                fig_box.add_trace(go.Box(
                    y=data, name=dist,
                    marker=dict(color=color, opacity=0.75),
                    boxmean=True, line=dict(color=color),
                ))
        fig_box = apply_chart_theme(fig_box, title="AQI Distribution by District", height=380)
        fig_box.update_layout(yaxis_title="AQI", showlegend=False)
        fig_box.add_hline(y=100, line=dict(color="#ff4757", dash="dash", width=1.2))
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# ML Prediction Section
# ─────────────────────────────────────────────────────────────────────────────

def _render_ml_prediction(district: str, start_date) -> None:
    section_header(
        "🤖 ML-Powered AQI Prediction",
        "Local RandomForest model — adjust inputs to predict air quality",
    )

    pred_month = getattr(start_date, "month", date.today().month)
    pred_dow   = getattr(start_date, "weekday", lambda: date.today().weekday())()

    col_inputs, col_result = st.columns([2, 1])

    with col_inputs:
        c1, c2 = st.columns(2)
        with c1:
            pm25      = st.slider("PM2.5 (µg/m³)", 5.0, 150.0, 35.0, 0.5, key="pred_pm25")
            pm10      = st.slider("PM10 (µg/m³)",  10.0, 240.0, 65.0, 1.0, key="pred_pm10")
            temp      = st.slider("Temperature (°C)", 2.0, 40.0, 18.0, 0.5, key="pred_temp")
        with c2:
            humidity  = st.slider("Humidity (%)",    20.0, 98.0, 60.0, 1.0, key="pred_hum")
            wind      = st.slider("Wind Speed (km/h)", 0.0, 60.0, 15.0, 0.5, key="pred_wind")
            noise     = st.slider("Noise Level (dB)", 38.0, 95.0, 62.0, 0.5, key="pred_noise")

        pred_district = st.selectbox(
            "District", DISTRICTS,
            index=DISTRICTS.index(district) if district in DISTRICTS else 0,
            key="pred_district",
        )

        predict_btn = st.button("🔮 Predict AQI", type="primary", key="predict_aqi_btn")

    with col_result:
        if predict_btn or st.session_state.get("last_aqi_pred") is not None:
            if predict_btn:
                result = predict_aqi(
                    pm25=pm25, pm10=pm10, temperature=temp,
                    humidity=humidity, wind_speed=wind, noise_db=noise,
                    month=pred_month, day_of_week=pred_dow, district=pred_district,
                )
                st.session_state["last_aqi_pred"] = result
            else:
                result = st.session_state.get("last_aqi_pred")

            if result:
                aqi_val = result["predicted_aqi"]
                label, color = aqi_category(aqi_val)
                prediction_card(
                    "Predicted AQI",
                    f"{aqi_val:.0f}",
                    label,
                    color=color,
                )

                # Feature importance chart
                st.markdown("**📊 Feature Importance (XAI)**")
                fig_imp = plot_feature_importance(result["feature_importance"])
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            else:
                st.error("Model not loaded. Run the app once to auto-train models.")
        else:
            st.markdown("""
            <div class="info-card">
              <span style="font-size:2rem">🔮</span>
              <h3>AQI Predictor</h3>
              <p>Adjust the sliders and click <strong>Predict AQI</strong> to see the model's forecast.</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def _render_anomaly_detection(filtered: pd.DataFrame) -> None:
    section_header("⚠️ Anomaly Detection", "Days where conditions exceeded safe thresholds")

    anomaly_aqi   = filtered[filtered["aqi"] > 150].copy()
    anomaly_noise = filtered[filtered["noise_db"] > 75].copy()
    anomaly_water = filtered[filtered["water_quality"] < 60].copy()

    col1, col2, col3 = st.columns(3)

    for col, adf, icon, label, color in [
        (col1, anomaly_aqi,   "🌫️", "AQI > 150 (Unhealthy)", "#ff4757"),
        (col2, anomaly_noise, "🔊", "Noise > 75 dB",          "#ffb347"),
        (col3, anomaly_water, "💧", "Water Quality < 60",     "#00d4ff"),
    ]:
        with col:
            st.markdown(f"**{icon} {label}**")
            if len(adf) == 0:
                alert_card("✅ No anomalies", "All readings within safe range.", "success")
            else:
                st.markdown(
                    f'<span class="badge badge-high">{len(adf)} anomalous readings</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                show = adf[["date", "district", "aqi" if "aqi" in adf.columns else "noise_db"]].head(5)
                st.dataframe(show.rename(columns=str.title), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Upload
# ─────────────────────────────────────────────────────────────────────────────

def _render_csv_upload() -> None:
    with st.expander("📁 Upload Custom Environmental Data (CSV)", expanded=False):
        st.markdown(
            "Upload a CSV with at least `date` and `aqi` columns. "
            "Optional: `pm25`, `pm10`, `temperature`, `humidity`, `noise_db`, `water_quality`, `district`."
        )
        uploaded = st.file_uploader("Choose CSV", type=["csv"], key="env_upload")
        if uploaded is not None:
            result = load_uploaded_env(uploaded)
            if result is not None:
                clean_df, warnings = result
                for w in warnings:
                    st.warning(w)
                st.success(f"✅ Loaded {len(clean_df):,} rows successfully.")
                st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)

                if "aqi" in clean_df.columns:
                    daily = clean_df.groupby("date")["aqi"].mean().reset_index()
                    fig = go.Figure(go.Scatter(
                        x=daily["date"], y=daily["aqi"],
                        mode="lines+markers",
                        line=dict(color="#6c63ff", width=2.5),
                        marker=dict(size=5),
                    ))
                    fig = apply_chart_theme(fig, "Uploaded Data — AQI Trend", height=280)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
