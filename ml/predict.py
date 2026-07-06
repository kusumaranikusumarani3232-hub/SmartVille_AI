"""
SmartVille AI — ML Inference Layer
=====================================
Loads pre-trained models (once, via @st.cache_resource) and exposes
clean prediction functions used by the UI modules.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import AQI_MODEL_PATH, COMPLAINT_MODEL_PATH, PRIORITY_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Model Loaders — cached for the entire Streamlit session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_aqi_model() -> Optional[dict]:
    """Load AQI model metadata. Returns None if file missing."""
    if not os.path.exists(AQI_MODEL_PATH):
        return None
    import joblib
    return joblib.load(AQI_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def _load_complaint_model() -> Optional[dict]:
    """Load complaint classifier metadata. Returns None if file missing."""
    if not os.path.exists(COMPLAINT_MODEL_PATH):
        return None
    import joblib
    return joblib.load(COMPLAINT_MODEL_PATH)


def ensure_models_trained() -> None:
    """
    Called once at app startup. Trains models if .joblib files are missing.
    Shows a Streamlit spinner during first-time training (≈ 20-30 s).
    """
    aqi_missing = not os.path.exists(AQI_MODEL_PATH)
    comp_missing = not os.path.exists(COMPLAINT_MODEL_PATH)

    if aqi_missing or comp_missing:
        msg = "⚙️ First-time setup: training ML models (this takes ~30s)…"
        with st.spinner(msg):
            from ml.train_models import train_all  # lazy import
            train_all(verbose=False)

        # Clear cached None values so fresh models load
        _load_aqi_model.clear()
        _load_complaint_model.clear()


# ─────────────────────────────────────────────────────────────────────────────
# AQI Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_aqi(
    pm25: float,
    pm10: float,
    temperature: float,
    humidity: float,
    wind_speed: float,
    noise_db: float,
    month: int,
    day_of_week: int,
    district: str,
) -> Optional[dict]:
    """
    Predict AQI for given conditions.

    Returns dict with keys:
      - predicted_aqi : float
      - feature_importance : list[dict]  (name, importance)
    """
    meta = _load_aqi_model()
    if meta is None:
        return None

    model   = meta["model"]
    enc     = meta["district_encoder"]
    features = meta["features"]

    district_enc = enc.transform([[district]])[0][0]
    quarter = (month - 1) // 3 + 1

    row = pd.DataFrame([{
        "pm25":        pm25,
        "pm10":        pm10,
        "temperature": temperature,
        "humidity":    humidity,
        "wind_speed":  wind_speed,
        "noise_db":    noise_db,
        "month":       month,
        "day_of_week": day_of_week,
        "quarter":     quarter,
        "district_enc": district_enc,
    }])[features]

    pred = float(model.predict(row)[0])
    pred = max(0, round(pred, 1))

    importances = [
        {"feature": f.replace("_", " ").title(), "importance": round(float(imp), 4)}
        for f, imp in zip(features, model.feature_importances_)
    ]
    importances.sort(key=lambda x: x["importance"], reverse=True)

    return {"predicted_aqi": pred, "feature_importance": importances}


# ─────────────────────────────────────────────────────────────────────────────
# Complaint Priority Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_priority(
    description: str,
    category: str,
    district: str,
    hour: int = 12,
    day_of_week: int = 0,
) -> Optional[dict]:
    """
    Predict complaint priority.

    Returns dict with keys:
      - predicted_priority : str  (Low | Medium | High | Critical)
      - confidence         : float
      - probabilities      : dict[str, float]
      - color              : str (hex)
    """
    meta = _load_complaint_model()
    if meta is None:
        return None

    pipeline     = meta["pipeline"]
    label_enc    = meta["label_encoder"]
    text_feat    = meta["text_feature"]
    cat_feats    = meta["cat_features"]
    num_feats    = meta["num_features"]
    prio_levels  = meta["priority_levels"]

    row = pd.DataFrame([{
        text_feat:     description,
        "category":    category,
        "district":    district,
        "hour":        hour,
        "day_of_week": day_of_week,
    }])

    pred_enc   = pipeline.predict(row)[0]
    proba      = pipeline.predict_proba(row)[0]

    predicted  = prio_levels[int(pred_enc)]
    confidence = float(proba[int(pred_enc)])

    prob_dict = {prio_levels[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "predicted_priority": predicted,
        "confidence":         confidence,
        "probabilities":      prob_dict,
        "color":              PRIORITY_COLORS.get(predicted, "#64748b"),
    }
