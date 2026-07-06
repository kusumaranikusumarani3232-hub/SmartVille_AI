"""
SmartVille AI — SQLite Database Manager
=========================================
Handles persistence of user-submitted complaints.
Uses temp directory so it works on both local dev and Streamlit Cloud.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


# SQLite lives in temp dir — survives the session, resets on app restart (Streamlit Cloud)
DB_PATH = os.path.join(tempfile.gettempdir(), "smartville_community.db")

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_complaints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id    TEXT    UNIQUE NOT NULL,
    submitted_at    TEXT    NOT NULL,
    district        TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    priority_pred   TEXT    NOT NULL,
    priority_conf   REAL    NOT NULL,
    status          TEXT    DEFAULT 'Open',
    upvotes         INTEGER DEFAULT 0
);
"""


@st.cache_resource(show_spinner=False)
def get_db_connection() -> sqlite3.Connection:
    """Return a cached, thread-safe SQLite connection. Created once per session."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_complaint_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COUNT(*) FROM user_complaints").fetchone()
    n = row[0] + 1
    year = datetime.now().year
    return f"SV-USR-{year}-{n:05d}"


def insert_complaint(
    district: str,
    category: str,
    description: str,
    priority_pred: str,
    priority_conf: float,
) -> str:
    """Insert a new user-submitted complaint. Returns the complaint ID."""
    conn = get_db_connection()
    cid = _next_complaint_id(conn)
    conn.execute(
        """INSERT OR IGNORE INTO user_complaints
           (complaint_id, submitted_at, district, category, description,
            priority_pred, priority_conf, status, upvotes)
           VALUES (?,?,?,?,?,?,?,'Open',0)""",
        (cid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         district, category, description, priority_pred, round(priority_conf, 3)),
    )
    conn.commit()
    return cid


def get_complaint(complaint_id: str) -> Optional[dict]:
    """Fetch a single complaint by ID (or None)."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM user_complaints WHERE complaint_id = ?", (complaint_id,)
    ).fetchone()
    return dict(row) if row else None


def get_all_complaints() -> pd.DataFrame:
    """Return all user-submitted complaints as a DataFrame."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM user_complaints ORDER BY submitted_at DESC"
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=[
            "id", "complaint_id", "submitted_at", "district", "category",
            "description", "priority_pred", "priority_conf", "status", "upvotes",
        ])
    return pd.DataFrame([dict(r) for r in rows])


def upvote_complaint(complaint_id: str) -> None:
    """Increment upvote count for a complaint."""
    conn = get_db_connection()
    conn.execute(
        "UPDATE user_complaints SET upvotes = upvotes + 1 WHERE complaint_id = ?",
        (complaint_id,),
    )
    conn.commit()
