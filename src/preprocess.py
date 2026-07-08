"""
preprocess.py
Loads raw daily logs, engineers behavioral features, and outputs
a model-ready feature table.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/insider_threat_logs.csv"
OUTPUT_PATH = "data/features.csv"

NUMERIC_COLS = [
    "logon_count", "after_hours_logons", "usb_connections",
    "files_copied_to_usb", "files_accessed", "emails_sent",
    "emails_to_external", "http_requests", "unique_devices_used",
    "session_duration_hrs"
]

def load_data():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df

def add_calendar_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def add_rolling_baseline_features(df):
    """
    For each user, compute a 7-day rolling mean (excluding current day)
    as their 'personal baseline', then compute how far today's value
    deviates from that baseline. This is the key insider-threat signal:
    behavior that's unusual FOR THAT PERSON, not just unusual overall.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        roll_mean = (
            df.groupby("user_id")[col]
            .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
        )
        df[f"{col}_baseline"] = roll_mean.fillna(df[col].mean())
        df[f"{col}_deviation"] = df[col] - df[f"{col}_baseline"]
    return df

def build_features():
    df = load_data()
    df = add_calendar_features(df)
    df = add_rolling_baseline_features(df)

    feature_cols = (
        NUMERIC_COLS
        + [f"{c}_baseline" for c in NUMERIC_COLS]
        + [f"{c}_deviation" for c in NUMERIC_COLS]
        + ["day_of_week", "is_weekend"]
    )

    final_df = df[["user_id", "date"] + feature_cols + ["is_malicious"]]
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature table saved: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    print(final_df.head())
    return final_df

if __name__ == "__main__":
    build_features()