"""
SmartVille AI — Data Loader
============================
Handles loading, caching, and cleaning of SmartVille datasets.
Auto-generates demo data if CSV files are missing (self-contained for deployment).
"""

from __future__ import annotations

import os
import sys
import io
from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import ENV_DATA_PATH, COMPLAINTS_DATA_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_data_files() -> None:
    """Generate CSV files if they don't exist (first-time setup)."""
    if not os.path.exists(ENV_DATA_PATH) or not os.path.exists(COMPLAINTS_DATA_PATH):
        # Dynamically import generator so it's not in the hot path
        data_dir = os.path.dirname(ENV_DATA_PATH)
        sys.path.insert(0, data_dir)
        from generate_data import generate_environmental_data, generate_complaints_data  # type: ignore

        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(ENV_DATA_PATH):
            env_df = generate_environmental_data()
            env_df.to_csv(ENV_DATA_PATH, index=False)

        if not os.path.exists(COMPLAINTS_DATA_PATH):
            comp_df = generate_complaints_data()
            comp_df.to_csv(COMPLAINTS_DATA_PATH, index=False)


def _clean_env(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce environmental dataframe columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    numeric_cols = ["aqi", "pm25", "pm10", "temperature", "humidity",
                    "wind_speed", "noise_db", "water_quality", "green_cover",
                    "latitude", "longitude"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip to sane ranges
    if "aqi" in df.columns:
        df["aqi"] = df["aqi"].clip(0, 500)
    if "pm25" in df.columns:
        df["pm25"] = df["pm25"].clip(0, 200)
    if "pm10" in df.columns:
        df["pm10"] = df["pm10"].clip(0, 350)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0, 100)
    if "water_quality" in df.columns:
        df["water_quality"] = df["water_quality"].clip(0, 100)

    df = df.dropna(subset=["aqi"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _clean_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce complaints dataframe columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Normalise categoricals
    if "priority" in df.columns:
        valid_p = {"Low", "Medium", "High", "Critical"}
        df["priority"] = df["priority"].where(df["priority"].isin(valid_p), "Medium")
    if "status" in df.columns:
        valid_s = {"Open", "In Progress", "Resolved", "Closed"}
        df["status"] = df["status"].where(df["status"].isin(valid_s), "Open")

    # Ensure required columns exist
    for col in ["latitude", "longitude", "upvotes", "resolution_days"]:
        if col not in df.columns:
            df[col] = None

    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")
    df["upvotes"]         = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0).astype(int)
    df["latitude"]        = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"]       = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Streamlit cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_env_data() -> pd.DataFrame:
    """Load and clean environmental data. Generates it if missing."""
    _ensure_data_files()
    df = pd.read_csv(ENV_DATA_PATH)
    return _clean_env(df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_complaints_data() -> pd.DataFrame:
    """Load and clean complaints data. Generates it if missing."""
    _ensure_data_files()
    df = pd.read_csv(COMPLAINTS_DATA_PATH)
    return _clean_complaints(df)


def load_uploaded_env(uploaded_file) -> Optional[tuple[pd.DataFrame, list[str]]]:
    """
    Parse and validate a user-uploaded environmental CSV.
    Returns (cleaned_df, list_of_warnings) or None on fatal error.
    """
    required_cols = {"date", "aqi"}
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
            return None

        warnings: list[str] = []
        cleaned = _clean_env(df)
        if len(cleaned) < len(df):
            warnings.append(f"{len(df) - len(cleaned)} rows dropped due to invalid data.")

        return cleaned, warnings

    except Exception as exc:
        st.error(f"Failed to parse uploaded file: {exc}")
        return None


def filter_env(df: pd.DataFrame, district: str, start_date, end_date) -> pd.DataFrame:
    """Apply sidebar district + date-range filters to env dataframe."""
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    df = df[mask]
    if district != "All Districts":
        df = df[df["district"] == district]
    return df.copy()


def filter_complaints(df: pd.DataFrame, district: str, start_date, end_date) -> pd.DataFrame:
    """Apply sidebar district + date-range filters to complaints dataframe."""
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    df = df[mask]
    if district != "All Districts":
        df = df[df["district"] == district]
    return df.copy()
