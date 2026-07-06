"""SmartVille AI — Application Settings & Constants"""
import os

# ── App Metadata ──────────────────────────────────────────────────────────────
APP_NAME        = "SmartVille AI"
APP_SUBTITLE    = "Civic Intelligence Platform"
APP_VERSION     = "1.0.0"
CITY_NAME       = "SmartVille"
HACKATHON_LABEL = "Google Cloud AI Hackathon 2024"

# ── Gemini Configuration ──────────────────────────────────────────────────────
GEMINI_MODEL        = "gemini-2.5-flash"
MAX_CHAT_HISTORY    = 6      # Trim history to last N turns before sending
MAX_CONTEXT_CHARS   = 1200   # Hard cap on data-context string sent to Gemini

# ── Geography ─────────────────────────────────────────────────────────────────
DISTRICTS = ["North", "South", "East", "West", "Central", "Harbor"]

DISTRICT_META = {
    "North":   {"label": "North (Industrial Zone)", "lat": 37.820, "lon": -122.420, "color": "#ff4757"},
    "South":   {"label": "South (Residential)",     "lat": 37.720, "lon": -122.420, "color": "#00ff9d"},
    "East":    {"label": "East (Downtown)",          "lat": 37.770, "lon": -122.360, "color": "#00d4ff"},
    "West":    {"label": "West (Suburban)",          "lat": 37.770, "lon": -122.480, "color": "#a78bfa"},
    "Central": {"label": "Central (Green Zone)",     "lat": 37.770, "lon": -122.420, "color": "#34d399"},
    "Harbor":  {"label": "Harbor (Waterfront)",      "lat": 37.740, "lon": -122.390, "color": "#ffb347"},
}

CITY_CENTER = {"lat": 37.760, "lon": -122.420}

# ── Complaint Definitions ─────────────────────────────────────────────────────
COMPLAINT_CATEGORIES = [
    "Roads", "Water Supply", "Garbage", "Noise",
    "Electricity", "Parks", "Safety",
]

CATEGORY_ICONS = {
    "Roads": "🛣️", "Water Supply": "💧", "Garbage": "🗑️",
    "Noise": "🔊", "Electricity": "⚡", "Parks": "🌳", "Safety": "🛡️",
}

PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_COLORS = {
    "Low": "#64748b", "Medium": "#ffb347",
    "High": "#ff4757", "Critical": "#ff1744",
}
PRIORITY_ICONS  = {
    "Low": "🟡", "Medium": "🟠", "High": "🔴", "Critical": "🚨",
}

STATUS_LEVELS = ["Open", "In Progress", "Resolved", "Closed"]
STATUS_COLORS = {
    "Open": "#ff4757", "In Progress": "#ffb347",
    "Resolved": "#00ff9d", "Closed": "#64748b",
}

# ── Environmental Thresholds ──────────────────────────────────────────────────
# AQI bands: (min, max, color, label)
AQI_BANDS = [
    (0,   50,  "#00e676", "Good"),
    (51,  100, "#ffee58", "Moderate"),
    (101, 150, "#ff9800", "Unhealthy for Sensitive Groups"),
    (151, 200, "#ff4757", "Unhealthy"),
    (201, 300, "#9c27b0", "Very Unhealthy"),
    (301, 500, "#b71c1c", "Hazardous"),
]

NOISE_BANDS = [
    (0,  55, "#00e676", "Safe"),
    (56, 65, "#ffb347", "Moderate"),
    (66, 75, "#ff9800", "Loud"),
    (76, 120,"#ff4757", "Very Loud"),
]

WATER_BANDS = [
    (90, 100, "#00e676", "Excellent"),
    (75, 89,  "#00d4ff", "Good"),
    (60, 74,  "#ffb347", "Fair"),
    (0,  59,  "#ff4757", "Poor"),
]

def aqi_category(value: float) -> tuple[str, str]:
    """Return (label, hex_color) for an AQI value."""
    for lo, hi, color, label in AQI_BANDS:
        if lo <= value <= hi:
            return label, color
    return "Hazardous", "#b71c1c"

def noise_category(value: float) -> tuple[str, str]:
    for lo, hi, color, label in NOISE_BANDS:
        if lo <= value <= hi:
            return label, color
    return "Very Loud", "#ff4757"

def water_category(value: float) -> tuple[str, str]:
    for lo, hi, color, label in WATER_BANDS:
        if lo <= value <= hi:
            return label, color
    return "Poor", "#ff4757"

# ── Chart Palette ─────────────────────────────────────────────────────────────
CHART_PALETTE = [
    "#6c63ff", "#00d4ff", "#00ff9d", "#ffb347",
    "#ff4757", "#a78bfa", "#34d399", "#f472b6",
]

PLOTLY_BASE = dict(
    template       = "plotly_dark",
    paper_bgcolor  = "rgba(0,0,0,0)",
    plot_bgcolor   = "rgba(0,0,0,0.04)",
    font           = dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin         = dict(l=20, r=20, t=50, b=30),
    colorway       = CHART_PALETTE,
    legend         = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                          borderwidth=1, font=dict(size=11)),
    xaxis          = dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False),
    yaxis          = dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False),
)

# ── File Paths ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(__file__))

DATA_DIR              = os.path.join(_BASE, "data")
ENV_DATA_PATH         = os.path.join(DATA_DIR, "sample_environmental.csv")
COMPLAINTS_DATA_PATH  = os.path.join(DATA_DIR, "sample_complaints.csv")

ML_DIR                = os.path.join(_BASE, "ml", "models")
AQI_MODEL_PATH        = os.path.join(ML_DIR, "aqi_predictor.joblib")
COMPLAINT_MODEL_PATH  = os.path.join(ML_DIR, "complaint_classifier.joblib")
