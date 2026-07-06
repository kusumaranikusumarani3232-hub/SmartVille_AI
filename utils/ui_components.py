"""
SmartVille AI — Reusable UI Components
========================================
HTML/CSS-based components rendered via st.markdown(unsafe_allow_html=True).
All styling is defined in assets/custom.css.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from config.settings import PLOTLY_BASE, aqi_category, noise_category, water_category


# ─────────────────────────────────────────────────────────────────────────────
# Metric Cards
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(
    icon: str,
    value: str,
    label: str,
    delta: Optional[float] = None,
    delta_label: str = "vs prev period",
    card_class: str = "primary",
) -> None:
    """Render a glassmorphism metric card via st.markdown."""
    if delta is not None:
        direction = "pos" if delta >= 0 else "neg"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = (
            f'<div class="mc-delta {direction}">'
            f'{arrow} {abs(delta):.1f}% {delta_label}</div>'
        )
    else:
        delta_html = ""

    st.markdown(
        f"""
        <div class="metric-card {card_class}">
          <div class="top-bar"></div>
          <span class="mc-icon">{icon}</span>
          <div class="mc-value">{value}</div>
          <div class="mc-label">{label}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Badges
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_CLASS = {
    "Open": "open", "In Progress": "progress",
    "Resolved": "resolved", "Closed": "closed",
}
_PRIORITY_CLASS = {
    "Low": "low", "Medium": "medium", "High": "high", "Critical": "critical",
}


def status_badge(status: str) -> str:
    css = _STATUS_CLASS.get(status, "closed")
    return f'<span class="badge badge-{css}">{status}</span>'


def priority_badge(priority: str) -> str:
    css = _PRIORITY_CLASS.get(priority, "medium")
    return f'<span class="badge badge-{css}">{priority}</span>'


def role_badge(role: str) -> str:
    css = "citizen" if role == "Citizen" else "admin"
    icon = "👤" if role == "Citizen" else "🔐"
    return f'<span class="badge badge-{css}">{icon} {role}</span>'


def aqi_badge(value: float) -> str:
    label, color = aqi_category(value)
    if value <= 50:
        css = "good"
    elif value <= 100:
        css = "moderate"
    elif value <= 200:
        css = "unhealthy"
    else:
        css = "hazardous"
    return f'<span class="badge badge-{css}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Alert Cards
# ─────────────────────────────────────────────────────────────────────────────

def alert_card(
    title: str,
    body: str,
    kind: str = "info",  # "danger" | "warning" | "success" | "info" | "purple"
) -> None:
    st.markdown(
        f"""
        <div class="alert-card alert-{kind}">
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section Header
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
          <h2>{title}</h2>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AQI Gauge Display
# ─────────────────────────────────────────────────────────────────────────────

def aqi_gauge_html(value: float, district: str = "") -> None:
    label, color = aqi_category(value)
    subtitle = f"{district} District" if district and district != "All Districts" else "City Average"
    st.markdown(
        f"""
        <div class="aqi-gauge">
          <div class="aqi-number" style="color:{color};">{value:.0f}</div>
          <div class="aqi-category" style="color:{color};">{label}</div>
          <div class="aqi-subtitle">{subtitle} — Air Quality Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_chart_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 360,
    show_legend: bool = True,
) -> go.Figure:
    """Apply SmartVille dark theme to any Plotly figure."""
    layout = dict(PLOTLY_BASE)
    layout["height"] = height
    layout["showlegend"] = show_legend
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(color="#e2e8f0", size=14, family="Inter"),
            x=0.02,
        )
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gemini badge
# ─────────────────────────────────────────────────────────────────────────────

def gemini_badge() -> None:
    st.markdown(
        '<span class="gemini-badge">✨ Powered by Gemini 2.5 Flash</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result card
# ─────────────────────────────────────────────────────────────────────────────

def prediction_card(
    heading: str,
    value: str,
    label: str,
    confidence: Optional[float] = None,
    color: str = "#6c63ff",
) -> None:
    conf_html = ""
    if confidence is not None:
        conf_html = f'<div class="pred-conf">Confidence: {confidence:.1%}</div>'
    st.markdown(
        f"""
        <div class="pred-card">
          <h3>{heading}</h3>
          <div class="pred-value" style="color:{color};">{value}</div>
          <div class="pred-label">{label}</div>
          {conf_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main app header
# ─────────────────────────────────────────────────────────────────────────────

def app_header(role: str) -> None:
    st.markdown(
        f"""
        <div class="sv-header">
          <div class="role-pill">{role_badge(role)}</div>
          <h1>🏙️ SmartVille AI</h1>
          <p>Civic Intelligence Platform &nbsp;·&nbsp; Google Cloud AI Hackathon 2024</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
