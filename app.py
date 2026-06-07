import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="F1 ML Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 ML Predictive Dashboard")
st.markdown("Welcome to the **Applied Machine Learning Project** Demo. This dashboard uses our trained XGBoost models to predict classification race outcomes in real-time.")

st.sidebar.header("🔧 Configuration")

@st.cache_resource
def load_models():
    try:
        clf_model = joblib.load('data/classification_model.joblib')
        clf_transformer = joblib.load('data/classification_transformer.joblib')
        reg_payload = joblib.load('data/regression_fastest_lap_model.joblib')
        return clf_model, clf_transformer, reg_payload
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data():
    try:
        return pd.read_parquet('data/df_train_ready.parquet')
    except:
        return None

clf_model, clf_transformer, reg_payload = load_models()
df = load_data()

if clf_model is None or reg_payload is None or df is None:
    st.error("⚠️ Models or Dataset not found. Please wait for the training scripts to finish saving `.joblib` files.")
    st.stop()
    
def format_ms(ms):
    if pd.isna(ms) or ms <= 0: return "N/A"
    secs = ms / 1000.0
    mins = int(secs // 60)
    rem_secs = secs % 60
    return f"{mins}:{rem_secs:06.3f}"

# Filter for Test Set (Recent Races)
df_test = df[df['year'] >= 2022].copy()

# Selectors
races = df_test[['raceId', 'year', 'round', 'circuit_name']].drop_duplicates()
races['race_label'] = races['year'].astype(str) + " - " + races['circuit_name'].astype(str)
selected_race_label = st.sidebar.selectbox("Select Race", races['race_label'].tolist())

race_id = races[races['race_label'] == selected_race_label]['raceId'].iloc[0]
race_data = df_test[df_test['raceId'] == race_id].copy()

selected_driver = st.sidebar.selectbox("Select Driver", race_data['driver_name'].tolist())

driver_row = race_data[race_data['driver_name'] == selected_driver].copy()

st.subheader(f"🏁 Driver Profile: {selected_driver} ({selected_race_label})")

col1, col2, col3 = st.columns(3)
col1.metric("Constructor", driver_row['constructor_name'].iloc[0])
col2.metric("Grid Position", str(driver_row['grid'].iloc[0]))
col3.metric("Qualifying Position", str(driver_row['quali_position'].iloc[0]))

st.markdown("---")

if st.button("🔮 Predict Podium Finish (Classification)", use_container_width=True):
    with st.spinner("Processing engineered features..."):
        # We need the feature columns. LEAKAGE_COLUMNS must be dropped.
        LEAKAGE_COLUMNS = [
            "is_podium", "personal_best_lap_ms", "pit_stop_count", 
            "avg_pit_dur_ms", "gap_to_fastest_ms", "raceId", "date"
        ]
        feature_columns = [col for col in df_test.columns if col not in LEAKAGE_COLUMNS]
        
        x_input = driver_row[feature_columns].copy()
        
        FORCED_CATEGORICAL_COLUMNS = ["driverId", "constructorId", "circuitId"]
        categorical_columns = list(df_test[feature_columns].select_dtypes(include=["object"]).columns)
        for column in FORCED_CATEGORICAL_COLUMNS:
            if column in feature_columns and column not in categorical_columns:
                categorical_columns.append(column)
                
        for col in categorical_columns:
            x_input[col] = x_input[col].astype(str)
            
        x_input_transformed = clf_transformer.transform(x_input)
        
        proba = clf_model.predict_proba(x_input_transformed)[0][1]
        
        is_podium_pred = proba >= 0.58 # 0.58 was the selected threshold from metrics
        
        st.subheader("📊 Prediction Results")
        if is_podium_pred:
            st.success(f"**YES!** The XGBoost model predicts {selected_driver} WILL finish on the podium! 🍾")
        else:
            st.error(f"**NO.** The XGBoost model predicts {selected_driver} will NOT finish on the podium.")
            
        st.progress(float(proba))
        st.write(f"**Podium Probability:** {proba*100:.1f}%")
        
        actual = driver_row['is_podium'].iloc[0] == 1
        st.info(f"**Actual Result in Real Life:** {'Podium 🍾' if actual else 'No Podium'}")

if st.button("⏱️ Predict Fastest Lap (Regression)", use_container_width=True):
    with st.spinner("Calculating regression model..."):
        reg_pipeline = reg_payload['pipeline']
        reg_features = reg_payload['feature_columns']
        offset_col = reg_payload['offset_column']
        
        x_input_reg = driver_row[reg_features].copy()
        
        # Predict residual and add offset
        pred_residual = reg_pipeline.predict(x_input_reg)[0]
        offset_value = driver_row[offset_col].iloc[0]
        predicted_ms = pred_residual + offset_value
        
        st.subheader("⏱️ Regression Results")
        st.success(f"**Predicted Fastest Lap:** {format_ms(predicted_ms)}")
        
        actual_ms = driver_row['personal_best_lap_ms'].iloc[0]
        st.info(f"**Actual Fastest Lap:** {format_ms(actual_ms)}")
        
        diff = abs(actual_ms - predicted_ms)
        st.write(f"*Model Prediction Error (Absolute):* {format_ms(diff)}")
