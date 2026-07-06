"""
SmartVille AI — Gemini Client
================================
Single cached Gemini instance with:
  • Response caching (same question = no API call)
  • Chat history trimming (max 6 turns)
  • Context compression (Pandas summaries, never raw DataFrames)
  • Graceful fallback when API key is missing or call fails
"""

from __future__ import annotations

import hashlib
from typing import Optional

import streamlit as st

from config.settings import GEMINI_MODEL, MAX_CHAT_HISTORY, MAX_CONTEXT_CHARS

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — concise, role-specific
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are SmartVille AI, a civic intelligence assistant for SmartVille's smart city platform.

Your role:
• Help city administrators understand environmental data and complaint patterns
• Provide clear, empathetic guidance to citizens about city services
• Give actionable, concise recommendations based ONLY on the data context provided

Rules:
• Base ALL insights on the data context provided before each message
• DO NOT invent statistics or make assumptions beyond the context
• Keep responses under 280 words; use bullet points for clarity
• Highlight urgency for critical or hazardous conditions
• You do NOT perform calculations — the platform already computed them for you

Tone: Professional, solution-focused, approachable."""


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """Safely read API key from Streamlit secrets, session state, or environment variables."""
    # 1. Try streamlit secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key and key != "YOUR_GEMINI_API_KEY_HERE":
            return key
    except Exception:
        pass

    # 2. Try session state
    if "GEMINI_API_KEY" in st.session_state:
        key = st.session_state["GEMINI_API_KEY"]
        if key:
            return key

    # 3. Try environment variables
    import os
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key

    return None


@st.cache_resource(show_spinner=False)
def _get_gemini_model():
    """
    Create and cache a single Gemini GenerativeModel for the session.
    Returns None if the API key is not configured.
    """
    key = _get_api_key()
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT,
            generation_config={
                "temperature": 0.55,
                "top_p": 0.9,
                "max_output_tokens": 512,
            },
        )
        return model
    except Exception as exc:
        st.warning(f"Gemini initialisation failed: {exc}")
        return None


def _cache_key(user_msg: str, context_str: str) -> str:
    """MD5 hash used to de-duplicate identical questions."""
    payload = f"{user_msg.strip().lower()}|{context_str[:300]}"
    return hashlib.md5(payload.encode()).hexdigest()


def _compress_context(data_context: dict) -> str:
    """
    Convert a dict of computed metrics to a compact string (≤ MAX_CONTEXT_CHARS).
    This is ALL that gets sent to Gemini — never raw DataFrames or CSV rows.
    """
    lines = [f"• {k}: {v}" for k, v in data_context.items()]
    text = "\n".join(lines)
    return text[:MAX_CONTEXT_CHARS]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Return True if a valid API key is available."""
    return _get_api_key() is not None


def generate_response(
    user_message: str,
    data_context: dict,
    chat_history: list[dict],
) -> tuple[str, bool]:
    """
    Generate a Gemini response.

    Parameters
    ----------
    user_message  : The user's question or request.
    data_context  : Dict of pre-computed metrics to inject as context.
                    Values should be primitives (str, int, float).
    chat_history  : List of {"role": "user"|"assistant", "content": str}.

    Returns
    -------
    (response_text, was_cached)
    """
    context_str = _compress_context(data_context)
    ckey = _cache_key(user_message, context_str)

    # ── Cache hit ──────────────────────────────────────────────────────────
    if "gemini_cache" not in st.session_state:
        st.session_state.gemini_cache = {}
    if ckey in st.session_state.gemini_cache:
        return st.session_state.gemini_cache[ckey], True

    # ── API key check ──────────────────────────────────────────────────────
    model = _get_gemini_model()
    if model is None:
        return (
            "⚠️ **Gemini API key not configured.**\n\n"
            "To enable the AI assistant:\n"
            "1. Get a free key at [aistudio.google.com](https://aistudio.google.com)\n"
            "2. Add `GEMINI_API_KEY = \"AIza...\"` to `.streamlit/secrets.toml`\n"
            "3. On Streamlit Cloud: App Settings → Secrets",
            False,
        )

    # ── Build prompt with compressed context ───────────────────────────────
    prompt = (
        f"**Current SmartVille Data Snapshot:**\n{context_str}\n\n"
        f"**Question:** {user_message}"
    )

    # ── Trim history to last MAX_CHAT_HISTORY turns ────────────────────────
    trimmed = chat_history[-MAX_CHAT_HISTORY:]
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in trimmed
    ]

    # ── Call Gemini ────────────────────────────────────────────────────────
    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(prompt)
        text = response.text.strip()
        st.session_state.gemini_cache[ckey] = text
        return text, False
    except Exception as exc:
        return f"⚠️ Gemini API error: {exc}", False


def build_data_context(env_df, complaints_df, district: str) -> dict:
    """
    Build a compact context dictionary from pre-computed Pandas aggregations.
    This is called LOCALLY — Gemini only receives the resulting ~500-char string.
    """
    import pandas as pd

    ctx: dict = {"city": "SmartVille", "district_filter": district}

    # ── Environmental summary ─────────────────────────────────────────────
    if env_df is not None and len(env_df) > 0:
        recent = env_df[env_df["date"] >= (env_df["date"].max() - pd.Timedelta(days=30))]
        ctx["avg_aqi_30d"]         = round(recent["aqi"].mean(), 1)
        ctx["max_aqi_30d"]         = round(recent["aqi"].max(), 1)
        ctx["avg_pm25_30d"]        = round(recent["pm25"].mean(), 1)
        ctx["avg_noise_db_30d"]    = round(recent["noise_db"].mean(), 1)
        ctx["avg_water_quality_30d"] = round(recent["water_quality"].mean(), 1)

        worst_district = (
            env_df.groupby("district")["aqi"].mean().idxmax()
            if "district" in env_df.columns and district == "All Districts"
            else district
        )
        ctx["worst_aqi_district"] = worst_district

    # ── Complaint summary ─────────────────────────────────────────────────
    if complaints_df is not None and len(complaints_df) > 0:
        ctx["total_complaints"] = len(complaints_df)
        ctx["open_complaints"]  = int((complaints_df["status"] == "Open").sum())
        ctx["critical_open"]    = int(
            ((complaints_df["priority"] == "Critical") &
             (~complaints_df["status"].isin(["Resolved", "Closed"]))).sum()
        )
        ctx["resolution_rate_pct"] = round(
            complaints_df["status"].isin(["Resolved", "Closed"]).mean() * 100, 1
        )
        top_cat = complaints_df["category"].value_counts().idxmax()
        ctx["top_complaint_category"] = top_cat

        resolved = complaints_df.dropna(subset=["resolution_days"])
        if len(resolved) > 0:
            ctx["avg_resolution_days"] = round(resolved["resolution_days"].mean(), 1)

    return ctx
