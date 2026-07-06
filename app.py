import streamlit as st
import pandas as pd
import datetime
import os

# Set page configuration first
st.set_page_config(
    page_title="SmartVille AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to safely retrieve secrets
def safe_get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# Load CSS custom stylesheet
css_path = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom CSS file not found at assets/custom.css")

from config.settings import DISTRICTS
from utils.data_loader import load_env_data, load_complaints_data
from utils.ui_components import app_header
from ml.predict import ensure_models_trained

# Ensure ML models are trained and synthetic data is generated
ensure_models_trained()

# Load clean datasets (cached under the hood)
env_df = load_env_data()
complaints_df = load_complaints_data()

# ── Session State Configuration ──────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = "Citizen"

# ── Sidebar Navigation & Filters ──────────────────────────────────────────────
st.sidebar.title("🏙️ SmartVille Control Panel")

# 1. Role Selection
role = st.sidebar.selectbox(
    "Select User Role",
    ["Citizen", "Administrator"],
    index=0 if st.session_state.role == "Citizen" else 1,
    key="role_select"
)
st.session_state.role = role

# 2. District Selection
district = st.sidebar.selectbox(
    "District Focus",
    ["All Districts"] + DISTRICTS,
    index=0
)

# 3. Date Filter
# The sample datasets cover from 2023-01-01 to 2024-12-31
min_date = datetime.date(2023, 1, 1)
max_date = datetime.date(2024, 12, 31)

date_range = st.sidebar.date_input(
    "Date Range Filter",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle cases where date_range doesn't return exactly 2 items yet
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# 4. Gemini API Key
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Settings")
gemini_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=safe_get_secret("GEMINI_API_KEY", "") or st.session_state.get("GEMINI_API_KEY", ""),
    placeholder="AIzaSy...",
    help="Get a key from Google AI Studio"
)
if gemini_key:
    st.session_state["GEMINI_API_KEY"] = gemini_key

# ── Main Application Header ────────────────────────────────────────────────────
app_header(st.session_state.role)

# ── Tabs Configuration ─────────────────────────────────────────────────────────
tab_names = ["📊 Dashboard", "🌬️ Environmental Analytics", "📢 Complaint Intelligence", "🤖 AI Assistant"]
t_dash, t_env, t_comp, t_ai = st.tabs(tab_names)

# Tab 1: Community Dashboard
with t_dash:
    from modules.dashboard import render_dashboard
    render_dashboard(
        env_df=env_df,
        complaints_df=complaints_df,
        district=district,
        start_date=start_date,
        end_date=end_date,
        role=st.session_state.role
    )

# Tab 2: Environmental Analytics
with t_env:
    from modules.environmental import render_environmental
    render_environmental(
        env_df=env_df,
        district=district,
        start_date=start_date,
        end_date=end_date
    )

# Tab 3: Complaint Intelligence
with t_comp:
    from modules.complaints import render_complaints
    render_complaints(
        complaints_df=complaints_df,
        district=district,
        start_date=start_date,
        end_date=end_date,
        role=st.session_state.role
    )

# Tab 4: AI Assistant
with t_ai:
    st.header("🤖 SmartVille AI Assistant")
    st.caption("Ask questions about local environmental metrics or complaints context.")
    
    from utils.gemini_client import is_configured, generate_response, build_data_context
    
    if not is_configured():
        st.warning("⚠️ **Gemini API key is not configured.**")
        st.info("Please enter your `GEMINI_API_KEY` in the sidebar control panel to unlock the AI Assistant.")
    else:
        # Clear chat history button
        if st.button("🧹 Clear Chat History", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()

        # Initialize session state chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User chat input
        user_query = st.chat_input("How can I help you improve SmartVille today?")
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Append user query to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Gather data context locally (compressed to stay below size limits)
            # Filter first to selected district and date range
            from utils.data_loader import filter_env, filter_complaints
            f_env = filter_env(env_df, district, start_date, end_date)
            f_comp = filter_complaints(complaints_df, district, start_date, end_date)
            
            data_context = build_data_context(f_env, f_comp, district)
            
            # Generate Gemini response
            with st.chat_message("assistant"):
                with st.spinner("AI is analyzing data snapshot..."):
                    response_text, was_cached = generate_response(
                        user_message=user_query,
                        data_context=data_context,
                        chat_history=st.session_state.chat_history[:-1]  # Exclude current question
                    )
                    st.markdown(response_text)
                    if was_cached:
                        st.caption("*(Cached response)*")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
