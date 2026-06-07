🏎️ F1 Predictive Modeling & Data Analysis
A comprehensive Python-based machine learning pipeline designed to predict Formula 1 race outcomes. This collaborative project uses historical race data to engineer complex features, predict the Top 3 podium finishes (Classification), and forecast the fastest lap times (Regression).

🚀 Features
- **Data Merging & Cleaning**: Robust data leakage prevention, handling missing values, and merging multiple Formula 1 relational datasets.
- **Advanced Feature Engineering**: Extracts dynamic contextual data such as rolling form averages, win streaks, circuit dominance, qualifying deltas, and pit stop strategies.
- **Podium Prediction (Classification)**: Utilizes XGBoost and LightGBM models to predict whether a driver will finish in the Top 3. Evaluated using F1, ROC-AUC, and Precision@3.
- **Fastest Lap Prediction (Regression)**: Employs Random Forest and LightGBM regression models to predict the fastest lap times. Evaluated using MAE, RMSE, and R².
- **Interpretability**: Integrates feature importance metrics and SHAP analysis for transparent ML predictions.

📁 Project Structure
.
├── data/
│   ├── df_master.parquet            # Merged and cleaned master dataset
│   ├── df_model_ready.parquet       # Final dataset with engineered features
│   ├── df_train_ready.parquet       # Scaled data ready for classification
│   └── ...                          # Metrics (JSON), predictions (CSV), and models (Joblib)
├── docs/
│   ├── aşama1_rapor.txt             # Stage 1: Data preparation reports
│   ├── aşama2_rapor.txt             # Stage 2: Feature engineering reports
│   ├── classification_guide_tr.md   # Documentation for the classification model
│   ├── regression_report.md         # Comprehensive regression analysis
│   └── *.png                        # Model comparison and evaluation plots
├── src/
│   ├── features/
│   │   ├── feature_engineering.ipynb # Notebook for exploratory feature analysis
│   │   └── preprocessing.py         # Data cleaning and manipulation scripts
│   └── models/
│       ├── classification_podium.py # ML Classification models for Podium Prediction
│       └── regression_fastest_lap.py# ML Regression models for Fastest Lap Prediction
├── .gitignore                       # Ignored files and folders
├── requirements.txt                 # Python project dependencies
└── README.md                        # Project documentation

🛠️ Installation & Setup
Clone the repository / Open the folder

Create and Activate a Virtual Environment

```bash
python -m venv .venv
# For Windows
.\.venv\Scripts\activate
# For Mac/Linux
source .venv/bin/activate
```

Install Dependencies
```bash
pip install -r requirements.txt
```

💻 Usage
You can run the individual ML pipelines straight from the terminal. Make sure you are in the project root directory.

To run the data preprocessing pipeline (Stage 1 & 2):
```bash
python src/features/preprocessing.py
```

To train and evaluate the classification models (Podium Prediction):
```bash
python src/models/classification_podium.py
```

To train and evaluate the regression models (Fastest Lap Prediction):
```bash
python src/models/regression_fastest_lap.py
```

🧠 How the Algorithm Works
The project follows a 4-Stage Roadmap:
1. **Data Preprocessing**: Filters pre/post race data points to prevent data leakage and applies rigorous NaN management.
2. **Feature Engineering**: Calculates driver forms, momentum scores, track history, and grid position strategies.
3. **Model Selection**: Dedicated scripts train specialized models (LightGBM/XGBoost/Random Forest) for separate predictive targets.
4. **Evaluation**: Outputs metrics and detailed visuals (saved in `docs/` and `data/`) to compare model efficacy and discover the most important features driving F1 race outcomes.
