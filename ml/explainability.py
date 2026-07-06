"""
SmartVille AI — Explainability Module
========================================
Provides visual explainability for ML predictions using:
  • Global: RandomForest feature_importances_ (bar chart)
  • Individual: SHAP TreeExplainer waterfall (with graceful fallback)

All computation is LOCAL — Gemini is not involved here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.ui_components import apply_chart_theme


# ─────────────────────────────────────────────────────────────────────────────
# Global Feature Importance (always available)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importances: list[dict], title: str = "Feature Importance") -> go.Figure:
    """
    Horizontal bar chart of feature importances from RandomForest.
    `importances` is a list of {"feature": str, "importance": float}.
    """
    top = sorted(importances, key=lambda x: x["importance"])[-10:]  # top 10
    features = [d["feature"] for d in top]
    values   = [d["importance"] for d in top]

    colors = [
        f"rgba(108, 99, 255, {0.4 + 0.6 * (v / max(values, default=1))})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig = apply_chart_theme(fig, title=title, height=320)
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SHAP — individual prediction explanation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
) -> Optional[dict]:
    """
    Compute SHAP values for a single prediction row.
    Returns {"base_value": float, "shap_values": list[float], "features": list[str]}
    or None if SHAP is not available.
    """
    try:
        import shap  # lazy import — not always available
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_row)

        # For regression, shap_values is shape (1, n_features)
        if isinstance(shap_vals, list):
            # Multi-class: take the class with highest predicted probability
            shap_vals = shap_vals[np.argmax([abs(sv).mean() for sv in shap_vals])]

        return {
            "base_value":  float(explainer.expected_value)
                           if not isinstance(explainer.expected_value, np.ndarray)
                           else float(explainer.expected_value[0]),
            "shap_values": shap_vals[0].tolist(),
            "features":    feature_names,
            "row_values":  X_row.iloc[0].tolist(),
        }
    except Exception:
        return None


def plot_shap_waterfall(shap_result: dict, prediction: float) -> go.Figure:
    """
    Render a SHAP waterfall chart using Plotly.
    Shows feature contributions to the final prediction.
    """
    feats  = shap_result["features"]
    vals   = shap_result["shap_values"]
    base   = shap_result["base_value"]

    # Sort by absolute SHAP value
    pairs = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:8]
    feats_sorted, vals_sorted = zip(*pairs) if pairs else ([], [])

    feats_sorted = list(feats_sorted)
    vals_sorted  = list(vals_sorted)

    colors = ["#00d4ff" if v >= 0 else "#ff4757" for v in vals_sorted]
    labels = [f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in vals_sorted]

    fig = go.Figure(go.Bar(
        x=vals_sorted,
        y=feats_sorted,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=labels,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))

    # Base value annotation
    fig.add_vline(
        x=0,
        line=dict(color="rgba(255,255,255,0.2)", width=1.5, dash="dash"),
    )

    fig = apply_chart_theme(
        fig,
        title=f"SHAP — Why did the model predict AQI = {prediction:.0f}?",
        height=320,
    )
    fig.update_layout(xaxis_title="SHAP Value (contribution to prediction)", yaxis_title="")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Probability Bars (complaint classifier)
# ─────────────────────────────────────────────────────────────────────────────

def plot_priority_probabilities(prob_dict: dict[str, float]) -> go.Figure:
    """Bar chart of predicted priority probabilities."""
    labels  = list(prob_dict.keys())
    values  = list(prob_dict.values())
    colors  = {
        "Low": "#64748b", "Medium": "#ffb347",
        "High": "#ff4757", "Critical": "#ff1744",
    }
    bar_colors = [colors.get(l, "#6c63ff") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[v * 100 for v in values],
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
    ))
    fig = apply_chart_theme(fig, title="Priority Prediction Confidence", height=280)
    fig.update_yaxes(range=[0, 110], title="Probability (%)")
    fig.update_xaxes(title="Priority Level")
    return fig
