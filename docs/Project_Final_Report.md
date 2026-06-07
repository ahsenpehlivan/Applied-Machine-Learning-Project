**Formula 1 Race Predictive Modeling & Data Analysis**  
**Applied Machine Learning - Project Final Report**  
**1. Team Members & Roles**  
- **Yiğit Enes Görgülü** (Email: *[Add Email]*)  
 *Role:* Stage 1 - Data Merging & Preprocessing  
- **Görkem Özden** (Email: *[Add Email]*)  
 *Role:* Stage 2 - Feature Engineering  
- **Ahsen Pehlivan** (Email: *[Add Email]*)  
 *Role:* Stage 3 & 4 - Podium Prediction (Classification Modeling & Evaluation)  
- **Tayfun Özgür** (Email: *[Add Email]*)  
 *Role:* Stage 3 & 4 - Fastest Lap Prediction (Regression Modeling & Evaluation)  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSeYxKS/kJkED6bwYAVvImwJtszMVu0BAPAXx1rd1fn1BACA164HHDwF+DpPyKwAAAAASUVORK5CYII=)  
**2. Goal / Motivation**  
Formula 1 is a highly data-driven sport where fractions of a second and strategic decisions determine the outcome of a race. The primary motivation of this project is to leverage historical F1 data to understand the underlying patterns that contribute to a driver's success.  
Our dual goal is to:  
1. Accurately predict whether a driver will finish on the podium (Top 3) in a given race.  
2. Forecast the personal fastest lap time of a driver during the race, given their qualifying performance and track conditions.  
   
   
By solving these problems, we aim to demonstrate how machine learning can be used in sports analytics to aid teams in strategy formulation and enhance the viewing experience for fans.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJWEPcbJpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaIkEMIPgIvAAAAAASUVORK5CYII=)  
**3. Description of the Learning Problem / Scheme**  
This project tackles two distinct machine learning paradigms simultaneously:  
1. **Classification Scheme (Podium Prediction):**  
  - **Problem:** Binary classification. Predict if is_podium == True (1) or False (0).  
  - **Algorithms Used:** XGBoost, LightGBM.  
  - **Evaluation Metrics:** F1-Score, ROC-AUC, Precision@3.  
2. **Regression Scheme (Fastest Lap Prediction):**  
  - **Problem:** Continuous regression. Predict personal_best_lap_ms.  
  - **Strategy:** Instead of predicting the raw time directly, the model predicts the delta (difference) between the qualifying time and the race lap time, which provides much more stable results.  
  - **Algorithms Used:** Random Forest, LightGBM.  
  - **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSPBCj7fFRYQwYwEZiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AMTJBeJDClAyAAAAAElFTkSuQmCC)  
**4. List of Datasets**  
Our master dataset is constructed by merging several relational tables. The primary source for this historical data is the [Ergast Developer API / Kaggle F1 Dataset.](http://ergast.com/mrd/ "http://ergast.com/mrd/")  
- **Races & Circuits Data:** Contains track locations, dates, and historical circuit characteristics.  
- **Drivers & Constructors Data:** Historical information regarding the drivers and the teams (constructors).  
- **Results Data:** Finishing positions, points, grid starts, and fastest lap times.  
- **Qualifying & Pit Stop Data:** Q1/Q2/Q3 times, pit stop durations, and race strategy metrics.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZIgBKm1gjSMNCwYYCIkd9OP3zJzRMQMAAB+sfqJeroBAMCN2pTWBSSZVtjzAAAAAElFTkSuQmCC)  
**5. Methodology & Implementation**  
**Stage 1: Data Merging & Preprocessing (Yiğit Enes Görgülü)**  
The initial stage focused on integrating the disparate relational datasets using main join keys such as raceId, driverId, and constructorId.  
- **Data Leakage Prevention:** Strict filtering was applied to ensure that post-race features (which would not be available at prediction time, e.g., total pit stops) were not used to predict outcomes.  
- **Missing Value Management:** Handled NaNs dynamically, ensuring the integrity of lap times and position data.  
**Stage 2: Feature Engineering (Görkem Özden)**  
To give the models contextual understanding, we engineered advanced dynamic features:  
- **Form & Momentum:** Rolling averages of previous driver positions and momentum scores.  
- **Circuit Dominance:** Historical track success rates for specific drivers and constructors.  
- **Qualifying Features:**Q3 delta and gap to pole to measure single-lap pace.  
- **Series & Context:** Current win streaks, podium rates, and championship point gaps leading up to the race (Race n/N).  
**Stage 3 & 4: Classification Modeling & Evaluation (Ahsen Pehlivan)**  
To predict podium finishes, LightGBM and XGBoost were trained on the engineered dataset.  
- The models successfully learned to distinguish podium contenders.  
- **Results:** The final LightGBM classification model achieved an impressive  **ROC-AUC of 0.916** and an  **F1-Score of 0.606**. The Precision@3 metric (0.598) indicates high reliability when the model specifically targets the top 3 drivers.  
**Stage 3 & 4: Regression Modeling & Evaluation (Tayfun Özgür)**  
To predict the fastest lap times, Random Forest and LightGBM models were employed.  
- The models utilized a baseline offset strategy (predicting the difference from the qualifying lap).  
- **Results:** LightGBM outperformed Random Forest, achieving a test  **R² of 0.793** with a  **Mean Absolute Error (MAE) of only 2.6 seconds (2601 ms)**, significantly outperforming naive global and circuit median baselines.  
   ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAADUlEQVR4nGP4//8/AwAI/AL+p5qgoAAAAABJRU5ErkJggg==)  
   ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAADUlEQVR4nGP4//8/AwAI/AL+p5qgoAAAAABJRU5ErkJggg==)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsad4FCtY9ecwnkms4E2ELcGWmTmrKwAA/uLeqrU6vp4AAPDa/gDzUgM9+S8z3AAAAABJRU5ErkJggg==)  
**6. Conclusion**  
The Formula 1 Predictive Modeling project successfully integrates data engineering, feature extraction, and dual machine learning schemes. By preventing data leakage and focusing on robust feature engineering (driver form, circuit dominance), our models demonstrate high predictive power. The Classification model accurately identifies podium finishers, while the Regression model effectively forecasts lap times within a highly competitive ~2.6-second margin of error.  
*(For the presentation and demo, the code is fully executable via the provided * *src/* * modules, and visual SHAP / Feature Importance graphs are available in the * *docs/* * directory.)*  
