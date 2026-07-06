"""
SmartVille AI — Synthetic Dataset Generator
============================================
Generates realistic environmental and complaint datasets for SmartVille,
a fictional smart city with 6 districts. Run once to produce CSV files.

Usage:
    python data/generate_data.py

Output:
    data/sample_environmental.csv  — 4 380 rows (6 districts × 730 days)
    data/sample_complaints.csv     — 2 000 citizen complaint records
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
SEED = 42
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)
N_COMPLAINTS = 2000

DISTRICTS = ["North", "South", "East", "West", "Central", "Harbor"]

# District centre coordinates (fictional SmartVille ~ Bay Area region)
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "North":   (37.820, -122.420),
    "South":   (37.720, -122.420),
    "East":    (37.770, -122.360),
    "West":    (37.770, -122.480),
    "Central": (37.770, -122.420),
    "Harbor":  (37.740, -122.390),
}

# Base AQI per district (industrial → residential scale)
BASE_AQI: dict[str, float] = {
    "North": 118, "South": 72, "East": 92,
    "West": 58,   "Central": 44, "Harbor": 68,
}

CATEGORIES = ["Roads", "Water Supply", "Garbage", "Noise", "Electricity", "Parks", "Safety"]

# Priority probability weights per category  [Low, Medium, High, Critical]
PRIORITY_WEIGHTS: dict[str, list[float]] = {
    "Roads":        [0.20, 0.40, 0.30, 0.10],
    "Water Supply": [0.10, 0.30, 0.40, 0.20],
    "Garbage":      [0.30, 0.40, 0.25, 0.05],
    "Noise":        [0.40, 0.40, 0.15, 0.05],
    "Electricity":  [0.10, 0.30, 0.40, 0.20],
    "Parks":        [0.50, 0.35, 0.10, 0.05],
    "Safety":       [0.05, 0.20, 0.40, 0.35],
}

STREETS = [
    "Main Street", "Oak Avenue", "Park Road", "Harbor Drive",
    "Central Boulevard", "Elm Street", "Pine Avenue", "Commerce Way",
    "Industrial Road", "Green Lane", "Waterfront Road", "City Square",
    "Maple Drive", "Sunrise Road", "Civic Center Way",
]

LANDMARKS = [
    "City Hall", "SmartVille Hospital", "Central Market", "Tech Park",
    "University Campus", "Sports Complex", "Bus Terminal", "Shopping Mall",
    "SmartVille School", "Community Center", "Police Station", "Fire Station",
    "Grand Library", "Convention Centre", "SmartVille Station",
]

DESCRIPTIONS: dict[str, list[str]] = {
    "Roads": [
        "Large pothole on {street} causing vehicle damage and safety risk.",
        "Road surface severely deteriorated near {landmark}; urgent repair needed.",
        "Street flooding due to blocked drainage on {street}.",
        "Damaged road markings near {landmark} creating a safety hazard.",
        "Bridge surface cracking on {street} near {district} area.",
        "Collapsed pavement on {street} blocking one lane of traffic.",
    ],
    "Water Supply": [
        "No water supply for {n} hours in {street} area.",
        "Visible pipe leakage causing water wastage near {landmark}.",
        "Brown discoloured water from taps on {street}.",
        "Critically low water pressure throughout {district} district.",
        "Main water pipeline burst near {landmark}; flooding the road.",
    ],
    "Garbage": [
        "Garbage not collected for {n} days on {street}.",
        "Overflowing bins near {landmark} attracting pests and rodents.",
        "Illegal dumping of construction waste spotted near {landmark}.",
        "Garbage pile blocking pedestrian footpath on {street}.",
        "Decomposing waste near {landmark} causing foul odour.",
    ],
    "Noise": [
        "Excessive construction noise at {landmark} continues after permitted hours.",
        "Loud music from commercial venue on {street} disturbing residents at night.",
        "Heavy industrial machinery noise from {district} factory during quiet hours.",
        "Night construction activity near {landmark} violating city noise ordinance.",
    ],
    "Electricity": [
        "Power outage on {street} for {n} hours with no update from utility.",
        "Streetlight not working near {landmark} creating a dark and unsafe area.",
        "Visible electrical sparks from exposed transformer near {street}.",
        "Frequent power fluctuations in {district} district damaging appliances.",
        "Exposed live wiring near {landmark} posing a serious electrocution risk.",
    ],
    "Parks": [
        "Broken playground equipment at {landmark} — hazardous for children.",
        "Damaged and unsafe park benches near {street}.",
        "Poor maintenance of {district} Community Park; overgrown grass.",
        "Overgrown vegetation blocking the main pathway at {landmark}.",
    ],
    "Safety": [
        "Dangerous intersection at {street} and {landmark} with no traffic signal.",
        "Poor street lighting on {street} creating safety concerns at night.",
        "Stray dogs attacking pedestrians near {landmark}.",
        "Abandoned vehicle blocking emergency access lane on {street}.",
        "Crumbling boundary wall near {landmark} poses imminent collapse risk.",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Environmental Data Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_environmental_data() -> pd.DataFrame:
    """Generate daily environmental readings for all districts."""
    rng = np.random.default_rng(SEED)
    records: list[dict] = []

    total_days = (END_DATE - START_DATE).days + 1
    dates = [START_DATE + timedelta(days=i) for i in range(total_days)]

    for d in dates:
        month = d.month
        # Seasonal temperature (°C) — peak ~25 °C in July, trough ~8 °C in Jan
        temp_mean = 8 + 17 * np.sin(np.pi * (month - 1) / 11)
        temp = float(rng.normal(temp_mean, 2.8))
        temp = round(np.clip(temp, 2, 38), 1)

        # Humidity inverse-correlated with temperature (Mediterranean-like)
        humidity = float(rng.normal(72 - temp * 0.9, 9))
        humidity = round(np.clip(humidity, 20, 98), 1)

        # Wind speed (higher in winter/spring)
        wind_mean = 18 + 8 * np.cos(2 * np.pi * (month - 4) / 12)
        wind_speed = round(float(abs(rng.normal(wind_mean, 6))), 1)

        # Seasonal AQI multiplier (worst Jan/Feb — inversions + heating)
        seasonal = 1.0 + 0.28 * np.cos(2 * np.pi * (month - 1) / 12)
        wind_factor = max(0.6, 1.0 - 0.008 * min(wind_speed, 60))

        for dist in DISTRICTS:
            base = BASE_AQI[dist]
            daily_noise = float(rng.normal(0, 14))
            aqi = base * seasonal * wind_factor + daily_noise
            aqi = round(np.clip(aqi, 8, 380), 1)

            pm25 = round(np.clip(aqi * 0.36 + float(rng.normal(0, 3)), 3, 145), 1)
            pm10 = round(np.clip(aqi * 0.56 + float(rng.normal(0, 5)), 6, 240), 1)

            # Noise (dB)
            noise_base = {"North": 72, "East": 68, "Central": 53, "Harbor": 62,
                          "South": 60, "West": 58}[dist]
            # Louder on weekdays
            weekday_bump = 3 if d.weekday() < 5 else -2
            noise_db = round(np.clip(float(rng.normal(noise_base + weekday_bump, 4.5)), 38, 95), 1)

            # Water quality index (0–100, higher = better)
            wq_base = {"Harbor": 63, "North": 67, "East": 76,
                       "South": 82, "West": 85, "Central": 92}[dist]
            water_quality = round(np.clip(float(rng.normal(wq_base, 4.5)), 38, 100), 1)

            # Green cover %
            gc_base = {"North": 14, "South": 34, "East": 19,
                       "West": 44, "Central": 64, "Harbor": 29}[dist]
            green_cover = round(np.clip(float(rng.normal(gc_base, 2.5)), 5, 80), 1)

            lat0, lon0 = DISTRICT_COORDS[dist]
            lat = round(lat0 + float(rng.normal(0, 0.004)), 6)
            lon = round(lon0 + float(rng.normal(0, 0.004)), 6)

            records.append({
                "date":          d.strftime("%Y-%m-%d"),
                "district":      dist,
                "aqi":           aqi,
                "pm25":          pm25,
                "pm10":          pm10,
                "temperature":   temp,
                "humidity":      humidity,
                "wind_speed":    wind_speed,
                "noise_db":      noise_db,
                "water_quality": water_quality,
                "green_cover":   green_cover,
                "latitude":      lat,
                "longitude":     lon,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Complaints Data Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_complaints_data() -> pd.DataFrame:
    """Generate citizen complaint records for SmartVille."""
    random.seed(SEED)
    np.random.seed(SEED)

    priorities = ["Low", "Medium", "High", "Critical"]
    statuses   = ["Open", "In Progress", "Resolved", "Closed"]

    # Complaint volume distribution per district (East/Downtown noisiest)
    district_weights = [0.20, 0.15, 0.26, 0.15, 0.09, 0.15]
    total_days = (END_DATE - START_DATE).days

    records: list[dict] = []

    for i in range(N_COMPLAINTS):
        rand_days = random.randint(0, total_days)
        date = START_DATE + timedelta(days=rand_days)
        hour = random.randint(7, 21)
        minute = random.randint(0, 59)

        district = random.choices(DISTRICTS, weights=district_weights)[0]
        category = random.choice(CATEGORIES)
        priority = random.choices(priorities, weights=PRIORITY_WEIGHTS[category])[0]

        # Status depends on how old the complaint is
        days_old = (datetime(2025, 1, 1) - date).days
        if days_old > 120:
            sw = [0.03, 0.12, 0.55, 0.30]
        elif days_old > 45:
            sw = [0.12, 0.28, 0.42, 0.18]
        else:
            sw = [0.42, 0.38, 0.15, 0.05]
        status = random.choices(statuses, weights=sw)[0]

        # Resolution time (only for Resolved/Closed)
        resolution_days: float | None = None
        if status in ("Resolved", "Closed"):
            scale = {"Critical": 1.5, "High": 4.5, "Medium": 9.0, "Low": 18.0}[priority]
            resolution_days = round(max(0.3, np.random.exponential(scale)), 1)

        # Build description
        tmpl_list = DESCRIPTIONS[category]
        tmpl = random.choice(tmpl_list)
        description = tmpl.format(
            street   = random.choice(STREETS),
            landmark = random.choice(LANDMARKS),
            district = district,
            n        = random.choice([2, 3, 4, 5, 6, 8, 12, 24]),
        )

        # Coordinates: district centre ± small jitter
        lat0, lon0 = DISTRICT_COORDS[district]
        lat = round(lat0 + np.random.normal(0, 0.018), 6)
        lon = round(lon0 + np.random.normal(0, 0.018), 6)

        # Upvotes — higher priority complaints attract more citizen attention
        upvote_mean = {"Low": 2, "Medium": 7, "High": 18, "Critical": 42}[priority]
        upvotes = max(0, int(np.random.exponential(upvote_mean)))

        records.append({
            "complaint_id":    f"SV-{date.year}-{i + 1:05d}",
            "date":            date.strftime("%Y-%m-%d"),
            "time":            f"{hour:02d}:{minute:02d}",
            "district":        district,
            "category":        category,
            "priority":        priority,
            "status":          status,
            "description":     description,
            "resolution_days": resolution_days,
            "latitude":        lat,
            "longitude":       lon,
            "upvotes":         upvotes,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    env_path        = os.path.join(out_dir, "sample_environmental.csv")
    complaints_path = os.path.join(out_dir, "sample_complaints.csv")

    print("Generating environmental data …", end=" ", flush=True)
    env_df = generate_environmental_data()
    env_df.to_csv(env_path, index=False)
    print(f"done  ({len(env_df):,} rows) → {env_path}")

    print("Generating complaints data …",    end=" ", flush=True)
    comp_df = generate_complaints_data()
    comp_df.to_csv(complaints_path, index=False)
    print(f"done  ({len(comp_df):,} rows) → {complaints_path}")

    print("\n✅  SmartVille datasets ready.")


if __name__ == "__main__":
    main()
