"""
train_models.py
Trains and compares 5 ML models for insider threat detection.
Handles class imbalance with SMOTE. Saves the best model + a results table.
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_PATH = "data/features.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

DROP_COLS = ["user_id", "date", "is_malicious"]

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS)
    y = df["is_malicious"]
    return X, y

def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=42, verbose=0
        ),
    }

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    cm = confusion_matrix(y_test, preds)
    print(f"\n===== {name} =====")
    print(metrics)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test, preds, digits=4))
    return metrics

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (helps AdaBoost/RF a bit, harmless for tree boosters)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Handle severe class imbalance with SMOTE (train set only!)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {X_train_res.shape[0]} rows, "
          f"malicious ratio = {y_train_res.mean():.3f}")

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model
        metrics = evaluate(model, X_test_scaled, y_test, name)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print("\n===== FINAL COMPARISON (sorted by F1) =====")
    print(results_df.to_string(index=False))

    results_df.to_csv(f"{OUTPUTS_DIR}/model_comparison.csv", index=False)

    # Save the best model based on F1 score
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} -> saving to models/best_model.pkl")

    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_columns.pkl")

    with open(f"{MODELS_DIR}/best_model_name.json", "w") as f:
        json.dump({"best_model": best_name}, f)

if __name__ == "__main__":
    main()