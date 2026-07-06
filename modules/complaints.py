"""
SmartVille AI — Module 3: Citizen Complaint Intelligence
=========================================================
Dual-view module:
  • Citizen View  — submit complaints, track status, view local feed
  • Admin View    — full analytics, map, ML priority prediction

All ML and analytics run LOCALLY. Gemini is NOT called here.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    CATEGORY_ICONS,
    CHART_PALETTE,
    CITY_CENTER,
    COMPLAINT_CATEGORIES,
    DISTRICT_META,
    DISTRICTS,
    PRIORITY_COLORS,
    PRIORITY_ICONS,
    PRIORITY_LEVELS,
    STATUS_COLORS,
)
from ml.explainability import plot_priority_probabilities
from ml.predict import predict_priority
from utils.database import get_all_complaints, get_complaint, insert_complaint, upvote_complaint
from utils.ui_components import (
    alert_card,
    apply_chart_theme,
    metric_card,
    prediction_card,
    priority_badge,
    section_header,
    status_badge,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_complaints(
    complaints_df: pd.DataFrame,
    district: str,
    start_date,
    end_date,
    role: str,
) -> None:
    """Render the Complaint Intelligence tab."""

    filtered = complaints_df.copy()
    filtered = filtered[
        (filtered["date"] >= pd.Timestamp(start_date)) &
        (filtered["date"] <= pd.Timestamp(end_date))
    ]
    if district != "All Districts":
        filtered = filtered[filtered["district"] == district]

    if role == "Citizen":
        _render_citizen_view(filtered, district)
    else:
        _render_admin_view(filtered, complaints_df)


# ─────────────────────────────────────────────────────────────────────────────
# Citizen View
# ─────────────────────────────────────────────────────────────────────────────

def _render_citizen_view(filtered: pd.DataFrame, district: str) -> None:
    col_form, col_track = st.columns([3, 2])

    with col_form:
        _render_complaint_form(district)

    with col_track:
        _render_complaint_tracker()
        st.markdown("---")
        _render_local_feed(filtered, district)


def _render_complaint_form(district: str) -> None:
    section_header("📝 Submit a Complaint", "Report an issue in your neighbourhood")

    with st.form("complaint_form", clear_on_submit=True):
        cat = st.selectbox(
            "Category",
            COMPLAINT_CATEGORIES,
            format_func=lambda c: f"{CATEGORY_ICONS.get(c, '')} {c}",
            key="form_cat",
        )
        dist = st.selectbox(
            "District",
            DISTRICTS,
            index=DISTRICTS.index(district) if district in DISTRICTS else 0,
            key="form_dist",
        )
        desc = st.text_area(
            "Describe the issue",
            placeholder="e.g. Large pothole on Oak Avenue causing vehicle damage…",
            height=120,
            key="form_desc",
        )
        submitted = st.form_submit_button("🚀 Submit Complaint", type="primary")

    if submitted:
        if not desc.strip():
            st.error("Please provide a description.")
        else:
            with st.spinner("Analysing priority…"):
                result = predict_priority(
                    description=desc,
                    category=cat,
                    district=dist,
                    hour=datetime.now().hour,
                    day_of_week=datetime.now().weekday(),
                )

            if result:
                pred_priority = result["predicted_priority"]
                confidence    = result["confidence"]
                color         = result["color"]
            else:
                pred_priority, confidence, color = "Medium", 0.7, "#ffb347"

            cid = insert_complaint(
                district=dist,
                category=cat,
                description=desc,
                priority_pred=pred_priority,
                priority_conf=confidence,
            )

            st.success(f"✅ Complaint submitted! Your tracking ID: **{cid}**")

            icon = PRIORITY_ICONS.get(pred_priority, "🟡")
            st.markdown(
                f"""
                <div class="pred-card">
                  <h3>AI-Predicted Priority</h3>
                  <div class="pred-value" style="color:{color};">{icon} {pred_priority}</div>
                  <div class="pred-label">Confidence: {confidence:.1%}</div>
                  <div class="pred-conf">Based on category, location, and description analysis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result and "probabilities" in result:
                fig = plot_priority_probabilities(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_complaint_tracker() -> None:
    st.markdown("#### 🔍 Track Your Complaint")
    cid_input = st.text_input("Enter Complaint ID", placeholder="SV-USR-2024-00001", key="track_id")
    if st.button("Track", key="track_btn"):
        if cid_input.strip():
            record = get_complaint(cid_input.strip())
            if record:
                _render_complaint_card(record)
            else:
                st.warning("Complaint ID not found. Check your tracking ID.")
        else:
            st.info("Enter a complaint ID to track.")


def _render_complaint_card(record: dict) -> None:
    icon  = PRIORITY_ICONS.get(record["priority_pred"], "🟡")
    color = PRIORITY_COLORS.get(record["priority_pred"], "#64748b")
    st.markdown(
        f"""
        <div class="info-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>{record['complaint_id']}</strong>
            <span class="badge badge-{record['status'].lower().replace(' ', '-')}">{record['status']}</span>
          </div>
          <div style="margin:0.5rem 0;font-size:0.8rem;color:var(--text-secondary);">
            {record['category']} · {record['district']} · {record['submitted_at'][:10]}
          </div>
          <p style="font-size:0.82rem;margin:0.5rem 0;">{record['description']}</p>
          <div>
            <span>Priority: </span>
            <strong style="color:{color};">{icon} {record['priority_pred']}</strong>
            <span style="font-size:0.72rem;color:var(--text-muted);margin-left:0.5rem;">
              ({record['priority_conf']:.1%} confidence)
            </span>
          </div>
          <div style="margin-top:0.4rem;font-size:0.72rem;color:var(--text-muted);">
            👍 {record['upvotes']} upvotes
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("👍 Upvote this complaint", key=f"upvote_{record['complaint_id']}"):
        upvote_complaint(record["complaint_id"])
        st.success("Upvoted!")


def _render_local_feed(filtered: pd.DataFrame, district: str) -> None:
    st.markdown(f"#### 📢 Recent Issues{f' — {district}' if district != 'All Districts' else ''}")
    recent = filtered.sort_values("date", ascending=False).head(6)
    if len(recent) == 0:
        st.info("No recent complaints in this area.")
        return
    for _, row in recent.iterrows():
        icon = PRIORITY_ICONS.get(row["priority"], "🟡")
        desc = row["description"][:70] + "…" if len(str(row["description"])) > 70 else row["description"]
        color = PRIORITY_COLORS.get(row["priority"], "#64748b")
        st.markdown(
            f"""
            <div class="alert-card alert-{'danger' if row['priority'] == 'Critical' else 'warning' if row['priority'] == 'High' else 'info'}">
              <h4>{icon} {row['category']} — {row['district']}</h4>
              <p>{desc} <span style="color:var(--text-muted)">(Status: {row['status']})</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Admin View
# ─────────────────────────────────────────────────────────────────────────────

def _render_admin_view(filtered: pd.DataFrame, full_df: pd.DataFrame) -> None:
    section_header("🔐 Complaint Intelligence Dashboard", "Full analytics for administrators")

    _render_admin_kpis(filtered)
    st.markdown("<br>", unsafe_allow_html=True)

    col_donut, col_prio = st.columns(2)
    with col_donut:
        _render_category_chart(filtered)
    with col_prio:
        _render_priority_trend(filtered)

    st.markdown("<br>", unsafe_allow_html=True)
    _render_complaint_map(filtered)

    st.markdown("<br>", unsafe_allow_html=True)
    col_box, col_heat = st.columns(2)
    with col_box:
        _render_resolution_time_chart(filtered)
    with col_heat:
        _render_district_category_heatmap(filtered)

    st.markdown("<br>", unsafe_allow_html=True)
    _render_admin_ml_tool()

    st.markdown("<br>", unsafe_allow_html=True)
    _render_data_table(filtered)


def _render_admin_kpis(comp: pd.DataFrame) -> None:
    total    = len(comp)
    open_c   = int((comp["status"] == "Open").sum())        if total else 0
    progress = int((comp["status"] == "In Progress").sum()) if total else 0
    resolved = int(comp["status"].isin(["Resolved","Closed"]).sum()) if total else 0
    avg_res  = round(comp["resolution_days"].dropna().mean(), 1) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("📢", f"{total:,}",    "Total Complaints",   card_class="primary")
    with c2: metric_card("📭", str(open_c),     "Open",               card_class="danger")
    with c3: metric_card("⚙️", str(progress),   "In Progress",        card_class="warning")
    with c4: metric_card("✅", str(resolved),    "Resolved / Closed",  card_class="success")
    with c5: metric_card("⏱️", f"{avg_res}d",   "Avg Resolution Time",card_class="info")


def _render_category_chart(comp: pd.DataFrame) -> None:
    if len(comp) == 0:
        return
    counts = comp["category"].value_counts()
    icons  = [CATEGORY_ICONS.get(c, "") for c in counts.index]
    labels = [f"{i} {c}" for i, c in zip(icons, counts.index)]
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values,
        hole=0.55,
        marker=dict(colors=CHART_PALETTE, line=dict(color="#0a0e1a", width=2)),
        textfont=dict(color="#f1f5f9"),
    ))
    fig = apply_chart_theme(fig, "Complaints by Category", height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_priority_trend(comp: pd.DataFrame) -> None:
    if len(comp) == 0:
        return
    comp_copy = comp.copy()
    comp_copy["month"] = comp_copy["date"].dt.to_period("M").dt.start_time
    monthly = (
        comp_copy.groupby(["month", "priority"])
        .size()
        .reset_index(name="count")
    )
    p_colors = {p: PRIORITY_COLORS[p] for p in PRIORITY_LEVELS}
    fig = px.bar(
        monthly, x="month", y="count", color="priority",
        color_discrete_map=p_colors,
        barmode="stack",
        labels={"month": "", "count": "Complaints", "priority": "Priority"},
    )
    fig = apply_chart_theme(fig, "Monthly Complaints by Priority", height=300)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_complaint_map(comp: pd.DataFrame) -> None:
    if len(comp) == 0 or "latitude" not in comp.columns:
        return
    map_data = comp.dropna(subset=["latitude", "longitude"]).copy()
    if len(map_data) == 0:
        return

    map_data["size"]       = map_data["upvotes"].fillna(1).clip(1, 50) + 5
    p_colors_map = {p: PRIORITY_COLORS[p] for p in PRIORITY_LEVELS}
    map_data["color_hex"]  = map_data["priority"].map(p_colors_map)

    fig = px.scatter_mapbox(
        map_data,
        lat="latitude", lon="longitude",
        color="priority",
        color_discrete_map=p_colors_map,
        size="size",
        size_max=20,
        hover_name="complaint_id",
        hover_data={
            "category": True, "district": True, "status": True,
            "description": True, "size": False,
            "latitude": False, "longitude": False,
        },
        zoom=11,
        center=CITY_CENTER,
        mapbox_style="open-street-map",
        labels={"priority": "Priority"},
        title="Complaint Hotspot Map",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#f1f5f9")),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_resolution_time_chart(comp: pd.DataFrame) -> None:
    resolved = comp.dropna(subset=["resolution_days"])
    if len(resolved) == 0:
        st.info("No resolved complaints yet.")
        return
    fig = go.Figure()
    for cat in COMPLAINT_CATEGORIES:
        data = resolved[resolved["category"] == cat]["resolution_days"].dropna()
        if len(data) > 0:
            fig.add_trace(go.Box(
                y=data, name=cat, boxmean=True,
                marker=dict(opacity=0.7),
            ))
    fig = apply_chart_theme(fig, "Resolution Time by Category (days)", height=320)
    fig.update_layout(yaxis_title="Days", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_district_category_heatmap(comp: pd.DataFrame) -> None:
    if len(comp) == 0:
        return
    pivot = (
        comp.groupby(["district", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(DISTRICTS, fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, "rgba(108,99,255,0.1)"], [0.5, "#6c63ff"], [1, "#ff4757"]],
        text=pivot.values.astype(str),
        texttemplate="%{text}",
        hovertemplate="District: %{y}<br>Category: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig = apply_chart_theme(fig, "Complaint Heatmap — District × Category", height=320)
    fig.update_layout(xaxis=dict(tickangle=-30))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_admin_ml_tool() -> None:
    section_header("🤖 ML Priority Predictor (Admin Tool)", "Predict complaint priority before assignment")

    with st.form("admin_ml_form"):
        c1, c2 = st.columns(2)
        with c1:
            adm_cat  = st.selectbox("Category", COMPLAINT_CATEGORIES, key="adm_cat")
            adm_dist = st.selectbox("District",  DISTRICTS,            key="adm_dist")
        with c2:
            adm_hour = st.slider("Hour of day", 0, 23, 10, key="adm_hour")
            adm_dow  = st.selectbox(
                "Day of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                key="adm_dow",
            )
        adm_desc = st.text_area("Complaint description", height=80, key="adm_desc")
        run_pred = st.form_submit_button("⚡ Predict Priority", type="primary")

    if run_pred and adm_desc.strip():
        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        result = predict_priority(
            description=adm_desc,
            category=adm_cat,
            district=adm_dist,
            hour=adm_hour,
            day_of_week=dow_map.get(adm_dow, 0),
        )
        if result:
            col_pred, col_chart = st.columns(2)
            with col_pred:
                prediction_card(
                    "Predicted Priority",
                    f"{PRIORITY_ICONS.get(result['predicted_priority'], '')} {result['predicted_priority']}",
                    f"Confidence: {result['confidence']:.1%}",
                    color=result["color"],
                )
            with col_chart:
                fig = plot_priority_probabilities(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.error("Model not loaded.")


def _render_data_table(comp: pd.DataFrame) -> None:
    with st.expander("📋 Full Complaint Table", expanded=False):
        cols = ["complaint_id", "date", "district", "category", "priority", "status",
                "description", "resolution_days", "upvotes"]
        show_cols = [c for c in cols if c in comp.columns]
        df_show = comp[show_cols].sort_values("date", ascending=False).head(100)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
