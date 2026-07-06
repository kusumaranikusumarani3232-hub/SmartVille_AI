"""
SmartVille AI — ML Model Trainer
==================================
Trains and serialises two scikit-learn models:
  1. AQI Predictor       — RandomForestRegressor
  2. Complaint Classifier— RandomForestClassifier via ColumnTransformer pipeline

Run once; models are saved as .joblib files.
The app auto-trains if files are missing on first launch.

Usage (standalone):
    python ml/train_models.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Allow running as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    AQI_MODEL_PATH,
    COMPLAINT_MODEL_PATH,
    COMPLAINTS_DATA_PATH,
    ENV_DATA_PATH,
    ML_DIR,
    PRIORITY_LEVELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_data() -> None:
    """Auto-generate CSVs if missing (calls the data generator)."""
    if not os.path.exists(ENV_DATA_PATH) or not os.path.exists(COMPLAINTS_DATA_PATH):
        from data.generate_data import generate_environmental_data, generate_complaints_data  # type: ignore
        os.makedirs(os.path.dirname(ENV_DATA_PATH), exist_ok=True)
        generate_environmental_data().to_csv(ENV_DATA_PATH, index=False)
        generate_complaints_data().to_csv(COMPLAINTS_DATA_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — AQI Predictor
# ─────────────────────────────────────────────────────────────────────────────

def train_aqi_model(verbose: bool = True) -> dict:
    """
    Train a RandomForestRegressor to predict AQI from environmental features.
    Returns metadata dict with feature names and evaluation metrics.
    """
    df = pd.read_csv(ENV_DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["aqi"])

    # Feature engineering
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"]     = df["date"].dt.quarter

    # Encode district as ordinal
    district_enc = OrdinalEncoder(
        categories=[["North", "South", "East", "West", "Central", "Harbor"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df["district_enc"] = district_enc.fit_transform(df[["district"]])

    FEATURES = [
        "pm25", "pm10", "temperature", "humidity", "wind_speed",
        "noise_db", "month", "day_of_week", "quarter", "district_enc",
    ]
    TARGET = "aqi"

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    if verbose:
        print(f"  AQI model  - MAE: {mae:.2f}  |  R2: {r2:.4f}")

    # Save
    os.makedirs(ML_DIR, exist_ok=True)
    metadata = {
        "model": model,
        "features": FEATURES,
        "district_encoder": district_enc,
        "mae": mae,
        "r2": r2,
    }
    joblib.dump(metadata, AQI_MODEL_PATH, compress=3)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Complaint Priority Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_complaint_model(verbose: bool = True) -> dict:
    """
    Train a Pipeline (TF-IDF on description + OHE on category/district)
    with RandomForestClassifier to predict complaint priority.
    Returns metadata dict.
    """
    df = pd.read_csv(COMPLAINTS_DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["priority", "category", "district", "description"])
    df["description"] = df["description"].fillna("").astype(str)

    # Encode target
    label_enc = LabelEncoder()
    label_enc.classes_ = np.array(PRIORITY_LEVELS)  # fix order: Low=0 … Critical=3
    df["priority_encoded"] = label_enc.transform(df["priority"])

    # Add time features
    df["hour"]        = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour.fillna(12)
    df["day_of_week"] = df["date"].dt.dayofweek

    TEXT_FEAT = "description"
    CAT_FEATS = ["category", "district"]
    NUM_FEATS = ["hour", "day_of_week"]

    # ColumnTransformer — combines TF-IDF text with categorical ordinals
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=80, ngram_range=(1, 2)), TEXT_FEAT),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_FEATS,
            ),
            ("num", "passthrough", NUM_FEATS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X = df[[TEXT_FEAT] + CAT_FEATS + NUM_FEATS]
    y = df["priority_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if verbose:
        print("  Complaint classifier report:")
        print(
            classification_report(
                y_test, y_pred,
                target_names=PRIORITY_LEVELS,
                zero_division=0,
            )
        )

    os.makedirs(ML_DIR, exist_ok=True)
    metadata = {
        "pipeline": pipeline,
        "label_encoder": label_enc,
        "text_feature": TEXT_FEAT,
        "cat_features": CAT_FEATS,
        "num_features": NUM_FEATS,
        "priority_levels": PRIORITY_LEVELS,
    }
    joblib.dump(metadata, COMPLAINT_MODEL_PATH, compress=3)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_all(verbose: bool = True) -> None:
    """Train both models, skipping any that already exist."""
    _ensure_data()
    os.makedirs(ML_DIR, exist_ok=True)

    if os.path.exists(AQI_MODEL_PATH):
        if verbose:
            print("OK - AQI model already exists — skipping training.")
    else:
        if verbose:
            print("Training AQI predictor …")
        train_aqi_model(verbose=verbose)
        if verbose:
            print("OK - AQI model saved.")

    if os.path.exists(COMPLAINT_MODEL_PATH):
        if verbose:
            print("OK - Complaint classifier already exists — skipping training.")
    else:
        if verbose:
            print("Training complaint priority classifier …")
        train_complaint_model(verbose=verbose)
        if verbose:
            print("OK - Complaint classifier saved.")


if __name__ == "__main__":
    train_all(verbose=True)
