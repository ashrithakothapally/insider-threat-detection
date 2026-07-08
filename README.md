# Insider Threat Detection in Cloud Environments using Machine Learning

A machine learning pipeline that detects insider threats (malicious or negligent
user behavior) from daily activity logs — logons, USB usage, file access,
email activity, and web/http traffic — inspired by the CERT Insider Threat
dataset structure.

## 🎯 Problem Statement
Insider threats are among the hardest security risks to detect because the
actor is an authorized user. This project builds a behavior-based anomaly
detection system that flags unusual activity patterns per user, using
supervised ML on engineered behavioral features.

## 🗂️ Project Structure
insider-threat-detection/
├── data/              # raw + processed datasets
├── notebooks/         # EDA, model comparison, SHAP analysis
├── src/               # data generation, preprocessing, training scripts
├── models/            # saved trained model, scaler, feature list
├── app/               # Streamlit dashboard
├── outputs/           # plots, model comparison results
├── requirements.txt
└── README.md
## ⚙️ Tech Stack
- **Language:** Python 3
- **ML Libraries:** scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **App:** Streamlit
- **Dataset:** Synthetic logs structured like the CERT Insider Threat Dataset
  (r4.2/r6.2), generated to simulate realistic behavioral patterns.

## 🧠 Approach
1. Generate/ingest per-user daily activity logs
2. Engineer features: raw counts + 7-day rolling personal baseline + deviation
3. Handle severe class imbalance (~1.5% malicious) using SMOTE
4. Train and compare 5 models: Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost
5. Evaluate using Precision, Recall, F1, ROC-AUC (not accuracy, due to imbalance)
6. Explain predictions using SHAP
7. Serve real-time predictions via a Streamlit dashboard

## 📊 Results
Best model: 

| Model         | Precision | Recall | F1-score | ROC-AUC |
|---------------|----------:|-------:|---------:|--------:|
| CatBoost      | 0.9835    | 0.4897 | 0.6538   | 0.7383  |
| LightGBM      | 0.9833    | 0.4856 | 0.6501   | 0.7380  |
| XGBoost       | 0.9833    | 0.4856 | 0.6501   | 0.7310  |
| RandomForest  | 0.9154    | 0.4897 | 0.6381   | 0.7488  |
| AdaBoost      | 0.5284    | 0.4979 | 0.5127   | 0.7284  |


## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/insider-threat-detection.git
cd insider-threat-detection

# 2. Set up environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Generate data & train
python src\generate_data.py
python src\preprocess.py
python src\train_models.py

# 4. Launch dashboard
streamlit run app\dashboard.py
```

## 🔮 Future Improvements
- Train/validate on the real CERT dataset
- Add network/API-level traffic features (not just user logs)
- Deploy as a cloud-native service (AWS Lambda / Azure Function) with
  real-time log ingestion
- Add sequence models (LSTM/Transformer) to capture temporal attack patterns