# Formula 1 Race Prediction Project

This repository contains the collaborative machine learning pipeline for Formula 1 race predictions. The project is divided into multiple stages based on our project roadmap. 

## 📌 Project Roadmap & Current Status

![Roadmap Outline](docs/WhatsApp%20Image%202026-06-06%20at%2011.41.21.jpeg)

- **Aşama 1 & 2: Veri Birleştirme, Temizlik ve Özellik Mühendisliği (Data Merging, Preprocessing, Feature Engineering)**
  - *Status:* Pending (To be added by Teammate 1)
- **Aşama 3: Model Seçimi - Hedef 1: Sınıflandırma (Classification for Podium Prediction - Top 3)**
  - *Status:* **✅ Completed by Ahsen** (XGBoost, LightGBM)
  - *Metrics:* F1, ROC-AUC, Precision@3
- **Aşama 3: Model Seçimi - Hedef 2: Regresyon (Regression for Fastest Lap Prediction)**
  - *Status:* Pending (To be added by Teammate 2)
  - *Metrics:* MAE, RMSE, R² (Random Forest, LightGBM)
- **Aşama 4: Değerlendirme ve Sunum (Evaluation Metrics & Reporting)**
  - *Status:* Ongoing

---

## 📁 Repository Structure

```text
.
├── data/                        # Processed data, predictions, and metrics
│   ├── df_train_ready.parquet
│   └── ...
├── src/                         # Source code
│   ├── features/                # Data cleaning and feature engineering scripts
│   ├── models/                  # ML models (Classification, Regression)
│   │   └── classification_podium.py
│   └── scraper/                 # (Optional) Scrapers for raw data
├── docs/                        # Reports, guides, and visual assets
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🤖 Instructions for AI Agents & Teammates

Welcome, collaborating Agents! To ensure a smooth integration of the codebase without conflicts, please follow these guidelines:

### For the Agent Handling Stage 1 & 2 (Data Pipeline)
Your task is to merge, clean, and engineer features from the raw data.
1. Place your data preparation and feature engineering scripts inside the `src/features/` directory (e.g., `src/features/data_preprocessing.py`, `src/features/feature_engineering.py`).
2. Save your final output datasets (like the prepared `df_train_ready.parquet` or intermediate CSVs) in the `data/` directory.
3. Ensure that your output schema is compatible with `src/models/classification_podium.py`.

### For the Agent Handling Stage 3 (Regression Model)
Your task is to predict the fastest lap times.
1. Create your regression scripts inside the `src/models/` directory (e.g., `src/models/regression_fastest_lap.py`).
2. Load the cleaned datasets from the `data/` directory.
3. Save your evaluation metrics and predictions back into the `data/` directory (e.g., `regression_predictions.csv`, `regression_metrics.json`).
4. **DO NOT** modify `classification_podium.py` as it is already complete.

### For Stage 4 (Reporting)
All SHAP plots, feature importance visualizations, and final text reports should be placed in the `docs/` directory.

---

## 🛠️ Installation & Setup

1. Clone the repository / Open the folder
2. Create and activate a Virtual Environment
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
