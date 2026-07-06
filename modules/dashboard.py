"""
SmartVille AI — Module 1: Community Dashboard
================================================
Provides an at-a-glance overview of SmartVille's health across all 4 pillars.
All computation is LOCAL (Pandas/Plotly). Gemini is NOT called here.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    CHART_PALETTE,
    COMPLAINT_CATEGORIES,
    DISTRICT_META,
    DISTRICTS,
    PRIORITY_COLORS,
    STATUS_COLORS,
    aqi_category,
)
from utils.ui_components import (
    alert_card,
    apply_chart_theme,
    metric_card,
    section_header,
    status_badge,
    priority_badge,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(
    env_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    district: str,
    start_date,
    end_date,
    role: str,
) -> None:
    """Render the Community Dashboard tab."""

    # ── Filter data ────────────────────────────────────────────────────────
    env = env_df.copy()
    comp = complaints_df.copy()

    env  = env[(env["date"] >= pd.Timestamp(start_date)) & (env["date"] <= pd.Timestamp(end_date))]
    comp = comp[(comp["date"] >= pd.Timestamp(start_date)) & (comp["date"] <= pd.Timestamp(end_date))]

    if district != "All Districts":
        env  = env[env["district"] == district]
        comp = comp[comp["district"] == district]

    _render_kpis(env, comp)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_trend_charts(env, comp, district)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_district_overview(env_df, complaints_df, start_date, end_date)
    st.markdown("<br>", unsafe_allow_html=True)

    _render_alerts(env, comp)

    if role == "Administrator":
        st.markdown("<br>", unsafe_allow_html=True)
        _render_admin_analytics(comp)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpis(env: pd.DataFrame, comp: pd.DataFrame) -> None:
    section_header("📊 Key Performance Indicators", "SmartVille at a glance")

    # Compute KPIs
    total_comp  = len(comp)
    avg_aqi     = env["aqi"].mean() if len(env) > 0 else 0
    resolution  = (
        comp["status"].isin(["Resolved", "Closed"]).mean() * 100
        if len(comp) > 0 else 0
    )
    critical    = int(
        ((comp["priority"] == "Critical") & ~comp["status"].isin(["Resolved", "Closed"])).sum()
    ) if len(comp) > 0 else 0

    open_issues = int((comp["status"] == "Open").sum()) if len(comp) > 0 else 0

    # Delta vs previous period (compare first vs second half of date range)
    half = len(comp) // 2
    if half > 2:
        prev_res = comp.iloc[:half]["status"].isin(["Resolved", "Closed"]).mean() * 100
        curr_res = comp.iloc[half:]["status"].isin(["Resolved", "Closed"]).mean() * 100
        res_delta = curr_res - prev_res
    else:
        res_delta = None

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("📢", f"{total_comp:,}", "Total Complaints", card_class="primary")
    with c2:
        aqi_class = "success" if avg_aqi <= 50 else "warning" if avg_aqi <= 100 else "danger"
        metric_card("🌬️", f"{avg_aqi:.0f}", "Avg Air Quality Index", card_class=aqi_class)
    with c3:
        metric_card(
            "✅", f"{resolution:.1f}%", "Resolution Rate",
            delta=res_delta, delta_label="vs prior",
            card_class="success" if resolution >= 60 else "warning",
        )
    with c4:
        metric_card(
            "🚨", str(critical), "Critical Open Issues",
            card_class="danger" if critical > 0 else "success",
        )
    with c5:
        metric_card("📭", str(open_issues), "Open Complaints", card_class="warning")


# ─────────────────────────────────────────────────────────────────────────────
# Trend Charts
# ─────────────────────────────────────────────────────────────────────────────

def _render_trend_charts(env: pd.DataFrame, comp: pd.DataFrame, district: str) -> None:
    section_header("📈 Trends Overview", "Complaints vs environmental conditions over time")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Weekly complaint trend by category
        if len(comp) > 0:
            comp_copy = comp.copy()
            comp_copy["week"] = comp_copy["date"].dt.to_period("W").dt.start_time
            weekly = (
                comp_copy.groupby(["week", "category"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                weekly,
                x="week", y="count", color="category",
                color_discrete_sequence=CHART_PALETTE,
                labels={"week": "Week", "count": "Complaints", "category": "Category"},
            )
            fig = apply_chart_theme(fig, title="Weekly Complaints by Category", height=340)
            fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No complaint data for selected range.")

    with col_right:
        # AQI rolling average
        if len(env) > 0:
            env_copy = env.copy()
            env_daily = env_copy.groupby("date")["aqi"].mean().reset_index()
            env_daily["aqi_7d"] = env_daily["aqi"].rolling(7, min_periods=1).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=env_daily["date"], y=env_daily["aqi"],
                mode="lines", name="Daily AQI",
                line=dict(color="rgba(108,99,255,0.35)", width=1),
                fill="tozeroy",
                fillcolor="rgba(108,99,255,0.06)",
            ))
            fig2.add_trace(go.Scatter(
                x=env_daily["date"], y=env_daily["aqi_7d"],
                mode="lines", name="7-Day Avg",
                line=dict(color="#6c63ff", width=2.5),
            ))
            # AQI threshold bands
            fig2.add_hrect(y0=0,   y1=50,  fillcolor="rgba(0,230,118,0.04)",  line_width=0)
            fig2.add_hrect(y0=50,  y1=100, fillcolor="rgba(255,238,88,0.04)", line_width=0)
            fig2.add_hrect(y0=100, y1=500, fillcolor="rgba(255,71,87,0.04)",  line_width=0)

            fig2 = apply_chart_theme(fig2, title="AQI Trend", height=340)
            fig2.update_layout(yaxis_title="AQI", xaxis_title="")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No environmental data for selected range.")


# ─────────────────────────────────────────────────────────────────────────────
# District Overview
# ─────────────────────────────────────────────────────────────────────────────

def _render_district_overview(
    env_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    start_date,
    end_date,
) -> None:
    section_header("🗺️ District Performance", "Comparing all districts side-by-side")

    env  = env_df[(env_df["date"] >= pd.Timestamp(start_date)) & (env_df["date"] <= pd.Timestamp(end_date))]
    comp = comp_df[(comp_df["date"] >= pd.Timestamp(start_date)) & (comp_df["date"] <= pd.Timestamp(end_date))]

    # District-level aggregations
    aqi_by_dist  = env.groupby("district")["aqi"].mean().reindex(DISTRICTS).fillna(0)
    comp_by_dist = comp.groupby("district").size().reindex(DISTRICTS).fillna(0)
    res_by_dist  = (
        comp.groupby("district")
        .apply(lambda g: g["status"].isin(["Resolved", "Closed"]).mean() * 100)
        .reindex(DISTRICTS)
        .fillna(0)
    )

    col1, col2 = st.columns(2)

    with col1:
        d_colors = [DISTRICT_META[d]["color"] for d in DISTRICTS]
        fig = go.Figure(go.Bar(
            x=list(aqi_by_dist.index),
            y=list(aqi_by_dist.values),
            marker=dict(color=d_colors, opacity=0.85),
            text=[f"{v:.0f}" for v in aqi_by_dist.values],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig = apply_chart_theme(fig, title="Average AQI by District", height=300)
        fig.update_yaxes(title="AQI")
        # Add threshold line
        fig.add_hline(y=100, line=dict(color="#ff4757", width=1.5, dash="dash"),
                      annotation_text="Unhealthy threshold", annotation_position="top right",
                      annotation_font=dict(color="#ff4757", size=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Total Complaints",
            x=list(comp_by_dist.index),
            y=list(comp_by_dist.values),
            marker=dict(color="rgba(108,99,255,0.7)"),
        ))
        fig2.add_trace(go.Scatter(
            name="Resolution Rate (%)",
            x=list(res_by_dist.index),
            y=list(res_by_dist.values),
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#00ff9d", width=2.5),
            marker=dict(size=7, color="#00ff9d"),
        ))
        fig2 = apply_chart_theme(fig2, title="Complaints & Resolution Rate by District", height=300)
        fig2.update_layout(
            yaxis=dict(title="Total Complaints"),
            yaxis2=dict(title="Resolution %", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)", range=[0, 110]),
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Alerts Feed
# ─────────────────────────────────────────────────────────────────────────────

def _render_alerts(env: pd.DataFrame, comp: pd.DataFrame) -> None:
    section_header("🔔 Active Alerts", "Environmental anomalies and critical citizen issues")

    col_env, col_comp = st.columns(2)

    with col_env:
        st.markdown("##### 🌿 Environmental Alerts")
        if len(env) > 0:
            recent_env = env[env["date"] >= env["date"].max() - timedelta(days=14)]
            bad_aqi    = recent_env[recent_env["aqi"] > 150].groupby("district")["aqi"].max()
            bad_noise  = recent_env[recent_env["noise_db"] > 70].groupby("district")["noise_db"].max()
            bad_water  = recent_env[recent_env["water_quality"] < 65].groupby("district")["water_quality"].min()

            any_alert = False
            for dist, val in bad_aqi.items():
                label, _ = aqi_category(val)
                alert_card(f"⚠️ {dist} District — AQI {val:.0f}",
                           f"Air quality is {label}. PM levels elevated.", "danger")
                any_alert = True
            for dist, val in bad_noise.items():
                alert_card(f"🔊 {dist} District — {val:.0f} dB",
                           "Noise exceeds safe residential limit (70 dB).", "warning")
                any_alert = True
            for dist, val in bad_water.items():
                alert_card(f"💧 {dist} District — WQI {val:.0f}",
                           "Water quality index below acceptable threshold.", "warning")
                any_alert = True
            if not any_alert:
                alert_card("✅ All Clear", "No environmental anomalies in the last 14 days.", "success")
        else:
            st.info("No environmental data.")

    with col_comp:
        st.markdown("##### 📢 Critical Complaint Alerts")
        if len(comp) > 0:
            critical = comp[
                (comp["priority"] == "Critical") &
                ~comp["status"].isin(["Resolved", "Closed"])
            ].sort_values("date", ascending=False).head(6)

            if len(critical) == 0:
                alert_card("✅ No Critical Issues", "All critical complaints have been resolved.", "success")
            else:
                for _, row in critical.iterrows():
                    desc = row["description"][:80] + "…" if len(row["description"]) > 80 else row["description"]
                    alert_card(
                        f"🚨 {row['district']} — {row['category']}",
                        f"{desc} (ID: {row['complaint_id']})",
                        "danger",
                    )
        else:
            st.info("No complaint data.")


# ─────────────────────────────────────────────────────────────────────────────
# Admin Analytics
# ─────────────────────────────────────────────────────────────────────────────

def _render_admin_analytics(comp: pd.DataFrame) -> None:
    section_header("🔐 Administrator Analytics", "Operational metrics — Administrator view only")

    if len(comp) == 0:
        st.info("No data in selected range.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        # Status pie
        status_counts = comp["status"].value_counts()
        colors = [STATUS_COLORS.get(s, "#64748b") for s in status_counts.index]
        fig = go.Figure(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color="#f1f5f9"),
        ))
        fig = apply_chart_theme(fig, title="Complaint Status Distribution", height=280, show_legend=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        # Avg resolution time by category
        resolved = comp.dropna(subset=["resolution_days"])
        if len(resolved) > 0:
            avg_res = resolved.groupby("category")["resolution_days"].mean().sort_values()
            fig2 = go.Figure(go.Bar(
                x=list(avg_res.values),
                y=list(avg_res.index),
                orientation="h",
                marker=dict(
                    color=[
                        "#ff4757" if v > 10 else "#ffb347" if v > 5 else "#00ff9d"
                        for v in avg_res.values
                    ],
                    opacity=0.85,
                ),
                text=[f"{v:.1f}d" for v in avg_res.values],
                textposition="outside",
            ))
            fig2 = apply_chart_theme(fig2, title="Avg Resolution Time (days)", height=280)
            fig2.update_xaxes(title="Days")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No resolved complaints yet.")

    with col3:
        # Priority breakdown
        prio_counts = comp["priority"].value_counts()
        p_colors = [PRIORITY_COLORS.get(p, "#64748b") for p in prio_counts.index]
        fig3 = go.Figure(go.Pie(
            labels=prio_counts.index,
            values=prio_counts.values,
            hole=0.55,
            marker=dict(colors=p_colors, line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color="#f1f5f9"),
        ))
        fig3 = apply_chart_theme(fig3, title="Priority Distribution", height=280)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
