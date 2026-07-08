"""
generate_data.py
Generates a synthetic insider-threat dataset mimicking the structure
of the CERT Insider Threat dataset. Malicious behavior is injected with
realistic overlap/noise so detection isn't trivially perfect.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_USERS = 500
N_DAYS = 120
MALICIOUS_FRACTION = 0.05
START_DATE = datetime(2024, 1, 1)

def generate_user_ids(n):
    return [f"USER{str(i).zfill(4)}" for i in range(n)]

def generate_dataset():
    user_ids = generate_user_ids(N_USERS)
    malicious_users = set(random.sample(user_ids, int(N_USERS * MALICIOUS_FRACTION)))

    rows = []
    for user in user_ids:
        is_malicious_user = user in malicious_users
        malicious_start_day = random.randint(N_DAYS // 2, N_DAYS - 10) if is_malicious_user else None

        # Each user has their own baseline "noise level" — some normal users
        # are just naturally more active than others (adds overlap)
        user_activity_multiplier = np.random.uniform(0.7, 1.4)

        for day in range(N_DAYS):
            date = START_DATE + timedelta(days=day)
            is_malicious_day = is_malicious_user and malicious_start_day is not None and day >= malicious_start_day

            # On a "malicious" day, behavior only escalates ~70% of the time —
            # real insiders don't act suspiciously every single day.
            if is_malicious_day:
                is_malicious_day = np.random.rand() < 0.7

            # Occasionally a NORMAL user has a genuinely busy/unusual day
            # (deadline crunch, big legit file transfer, business travel, etc.)
            normal_spike = (not is_malicious_day) and (np.random.rand() < 0.03)

            if not is_malicious_day:
                mult = user_activity_multiplier * (1.8 if normal_spike else 1.0)
                logon_count = np.random.poisson(3 * mult)
                after_hours_logons = np.random.poisson(0.3 * mult)
                usb_connections = np.random.poisson(0.4 * mult)
                files_copied_to_usb = np.random.poisson(1.0 * mult)
                files_accessed = np.random.poisson(15 * mult)
                emails_sent = np.random.poisson(10 * mult)
                emails_to_external = np.random.poisson(0.8 * mult)
                http_requests = np.random.poisson(40 * mult)
                unique_devices_used = np.random.choice([1, 1, 2, 2, 3], p=[0.55, 0.25, 0.12, 0.06, 0.02])
                session_duration_hrs = np.clip(np.random.normal(8 * mult, 2), 0, 18)
                label = 0
            else:
                # Malicious days overlap more with normal — softer, noisier escalation
                mult = user_activity_multiplier * np.random.uniform(1.3, 2.2)
                logon_count = np.random.poisson(4 * mult)
                after_hours_logons = np.random.poisson(1.5 * mult)
                usb_connections = np.random.poisson(2.0 * mult)
                files_copied_to_usb = np.random.poisson(8 * mult)
                files_accessed = np.random.poisson(40 * mult)
                emails_sent = np.random.poisson(12 * mult)
                emails_to_external = np.random.poisson(3 * mult)
                http_requests = np.random.poisson(55 * mult)
                unique_devices_used = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
                session_duration_hrs = np.clip(np.random.normal(11 * mult, 2.5), 0, 20)
                label = 1

            rows.append({
                "user_id": user,
                "date": date.strftime("%Y-%m-%d"),
                "logon_count": logon_count,
                "after_hours_logons": after_hours_logons,
                "usb_connections": usb_connections,
                "files_copied_to_usb": files_copied_to_usb,
                "files_accessed": files_accessed,
                "emails_sent": emails_sent,
                "emails_to_external": emails_to_external,
                "http_requests": http_requests,
                "unique_devices_used": unique_devices_used,
                "session_duration_hrs": round(session_duration_hrs, 2),
                "is_malicious": label
            })

    df = pd.DataFrame(rows)

    # Add small label noise (~1%) — mimics real-world imperfect ground truth
    flip_idx = df.sample(frac=0.01, random_state=42).index
    df.loc[flip_idx, "is_malicious"] = 1 - df.loc[flip_idx, "is_malicious"]

    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/insider_threat_logs.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Malicious rows: {df['is_malicious'].sum()} ({df['is_malicious'].mean()*100:.2f}%)")
    print(df.head())