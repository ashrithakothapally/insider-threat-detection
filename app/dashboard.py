"""
dashboard.py
Streamlit dashboard for Insider Threat Detection.
Loads the best trained model and lets a user:
  1. Upload a CSV of daily user logs and get predictions
  2. Manually enter a single day's activity and get a risk score
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insider Threat Detection", layout="wide")

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
BEST_MODEL_NAME_PATH = "models/best_model_name.json"

RAW_NUMERIC_COLS = [
    "logon_count", "after_hours_logons", "usb_connections",
    "files_copied_to_usb", "files_accessed", "emails_sent",
    "emails_to_external", "http_requests", "unique_devices_used",
    "session_duration_hrs"
]

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    with open(BEST_MODEL_NAME_PATH) as f:
        best_name = json.load(f)["best_model"]
    return model, scaler, feature_cols, best_name

model, scaler, feature_cols, best_name = load_artifacts()

st.title("🛡️ Insider Threat Detection in Cloud Environments")
st.caption(f"Active model: **{best_name}**")

tab1, tab2 = st.tabs(["📁 Batch Prediction (CSV)", "🧍 Single User Check"])

# ---------- TAB 1: Batch CSV prediction ----------
with tab1:
    st.subheader("Upload daily activity logs (raw format, like insider_threat_logs.csv)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    threshold = st.slider(
        "Malicious probability threshold (lower = catch more threats, more false alarms)",
        0.05, 0.95, 0.5, 0.05, key="batch_threshold"
    )

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded, parse_dates=["date"])
        raw_df = raw_df.sort_values(["user_id", "date"]).reset_index(drop=True)

        # Recreate the same features as preprocess.py
        df = raw_df.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        for col in RAW_NUMERIC_COLS:
            roll_mean = (
                df.groupby("user_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
            )
            df[f"{col}_baseline"] = roll_mean.fillna(df[col].mean())
            df[f"{col}_deviation"] = df[col] - df[f"{col}_baseline"]

        X = df[feature_cols]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = raw_df[["user_id", "date"]].copy()
        results["risk_score"] = probs.round(4)
        results["flagged"] = preds

        st.write(f"**{preds.sum()} flagged entries** out of {len(preds)} "
                 f"(threshold = {threshold})")

        flagged_df = results[results["flagged"] == 1].sort_values(
            "risk_score", ascending=False
        )
        st.dataframe(flagged_df, use_container_width=True)

        st.download_button(
            "Download full results as CSV",
            results.to_csv(index=False),
            file_name="threat_predictions.csv",
            mime="text/csv"
        )

        st.subheader("Risk score distribution")
        fig, ax = plt.subplots()
        ax.hist(probs, bins=50)
        ax.set_xlabel("Predicted malicious probability")
        ax.set_ylabel("Count")
        ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
        ax.legend()
        st.pyplot(fig)

# ---------- TAB 2: Manual single-day check ----------
with tab2:
    st.subheader("Enter a single day's activity for a user")
    st.caption("Baseline/deviation features are approximated as 0 deviation (no history) for a quick manual check.")

    col1, col2 = st.columns(2)
    with col1:
        logon_count = st.number_input("Logon count", 0, 50, 3)
        after_hours_logons = st.number_input("After-hours logons", 0, 20, 0)
        usb_connections = st.number_input("USB connections", 0, 20, 0)
        files_copied_to_usb = st.number_input("Files copied to USB", 0, 500, 1)
        files_accessed = st.number_input("Files accessed", 0, 1000, 15)
    with col2:
        emails_sent = st.number_input("Emails sent", 0, 200, 10)
        emails_to_external = st.number_input("Emails to external domains", 0, 100, 1)
        http_requests = st.number_input("HTTP requests", 0, 1000, 40)
        unique_devices_used = st.number_input("Unique devices used", 1, 10, 1)
        session_duration_hrs = st.number_input("Session duration (hrs)", 0.0, 24.0, 8.0)

    day_of_week = st.selectbox(
        "Day of week", options=list(range(7)),
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )
    is_weekend = 1 if day_of_week >= 5 else 0
    manual_threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="manual_threshold")

    if st.button("Check risk"):
        raw_vals = {
            "logon_count": logon_count, "after_hours_logons": after_hours_logons,
            "usb_connections": usb_connections, "files_copied_to_usb": files_copied_to_usb,
            "files_accessed": files_accessed, "emails_sent": emails_sent,
            "emails_to_external": emails_to_external, "http_requests": http_requests,
            "unique_devices_used": unique_devices_used,
            "session_duration_hrs": session_duration_hrs,
        }
        row = dict(raw_vals)
        for col in RAW_NUMERIC_COLS:
            row[f"{col}_baseline"] = raw_vals[col]   # assume today == baseline (no history)
            row[f"{col}_deviation"] = 0.0
        row["day_of_week"] = day_of_week
        row["is_weekend"] = is_weekend

        X_manual = pd.DataFrame([row])[feature_cols]
        X_manual_scaled = pd.DataFrame(scaler.transform(X_manual), columns=X_manual.columns)

        prob = model.predict_proba(X_manual_scaled)[0, 1]
        is_flagged = prob >= manual_threshold

        if is_flagged:
            st.error(f"⚠️ FLAGGED as potentially malicious — risk score: {prob:.3f}")
        else:
            st.success(f"✅ Normal activity — risk score: {prob:.3f}")

        st.progress(min(int(prob * 100), 100))