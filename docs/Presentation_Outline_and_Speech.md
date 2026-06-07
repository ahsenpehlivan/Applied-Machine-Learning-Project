# 🎤 F1 Predictive Modeling: Presentation Outline & Speech Scripts

This document provides a slide-by-slide outline for your project presentation, along with suggested English speech scripts tailored to each team member's specific contribution.

---

## 📊 Presentation Outline (Approx. 10-12 Minutes)

| Slide | Title | Speaker | Content |
| :--- | :--- | :--- | :--- |
| **1** | Title Slide | **Yiğit Enes** | Project Title, Team Members, Course Name |
| **2** | Motivation & Goal | **Yiğit Enes** | Why F1? The dual learning problem (Classification & Regression) |
| **3** | Datasets & Preprocessing | **Yiğit Enes** | Data sources (Ergast API), handling missing values, preventing data leakage |
| **4** | Feature Engineering | **Görkem** | The need for context. Form, momentum, circuit dominance |
| **5** | Advanced Features | **Görkem** | Qualifying deltas, win streaks, and preparing the final `df_model_ready` |
| **6** | Classification Problem | **Ahsen** | Predicting the Podium (Top 3). Target definition, class imbalance |
| **7** | Classification Results | **Ahsen** | XGBoost vs LightGBM. ROC-AUC (0.916), F1-Score, Precision@3 |
| **8** | Regression Problem | **Tayfun** | Predicting Fastest Lap. The offset strategy (Target = Race Lap - Quali Lap) |
| **9** | Regression Results | **Tayfun** | Random Forest vs LightGBM. MAE (2.6s), R² (0.793), showing the plots |
| **10**| Live Demo / Code | **Ahsen / Tayfun**| Briefly showing the terminal execution or the Streamlit dashboard |
| **11**| Conclusion & Future Work | **Yiğit Enes** | Wrap up, potential improvements (weather data, tire degradation) |
| **12**| Q&A | **All Team** | Open floor for professor and class questions |

---

## 🗣️ Speech Scripts

### 1. Yiğit Enes Görgülü (Intro & Stage 1)
**Slide 1: Title**
> "Hello everyone, welcome to our Applied Machine Learning project presentation. Our team consists of Görkem, Ahsen, Tayfun, and myself, Yiğit Enes. Today, we are excited to present our 'Formula 1 Race Predictive Modeling' project."

**Slide 2: Motivation**
> "Formula 1 is a sport where fractions of a second matter. Our motivation was to see if we could use historical data to predict race outcomes. We tackled two distinct problems: First, a Classification problem to predict if a driver will finish on the podium. Second, a Regression problem to predict a driver's fastest lap time during the race."

**Slide 3: Datasets & Preprocessing**
> "For my part in Stage 1, I focused on gathering and cleaning the data. We merged multiple relational datasets, including drivers, constructors, circuits, and historical results. The biggest challenge was *data leakage*. I had to strictly filter out post-race features—like total pit stops—because those wouldn't be known before a race starts. I also handled missing values dynamically to ensure our dataset was solid for the next stages. Now, I'll pass it to Görkem to talk about Feature Engineering."

---

### 2. Görkem Özden (Stage 2 - Feature Engineering)
**Slide 4: Feature Engineering**
> "Thank you, Yiğit. A raw dataset isn't enough to predict F1 races because driver performance is highly contextual. For Stage 2, my task was to engineer dynamic features that capture the current 'form' of a driver and their team."

**Slide 5: Advanced Features**
> "I created rolling averages of previous race positions, calculated momentum scores, and measured circuit dominance—which tells us if a driver historically performs well on a specific track. I also integrated qualifying deltas, meaning the time gap between a driver and the pole position. By extracting these win streaks and podium rates, I transformed the raw data into our `df_model_ready` dataset, which Ahsen and Tayfun then used to train the models. Over to Ahsen for Classification."

---

### 3. Ahsen Pehlivan (Stage 3 & 4 - Classification)
**Slide 6: Classification Problem**
> "Thanks, Görkem. For the third stage, my goal was to solve the Podium Prediction problem. This is a binary classification task where the model predicts whether a driver will finish in the Top 3. The challenge here is class imbalance, since only 3 out of 20 drivers make the podium."

**Slide 7: Classification Results**
> "I trained both XGBoost and LightGBM models on Görkem's engineered features. LightGBM performed exceptionally well. We achieved an ROC-AUC score of 0.916, which shows the model has a very high capability of distinguishing podium contenders from the rest of the pack. Our Precision at Top 3 was nearly 60%, meaning when our model strongly predicts a driver will be on the podium, it is usually right. Now, Tayfun will discuss the regression problem."

---

### 4. Tayfun Özgür (Stage 3 & 4 - Regression)
**Slide 8: Regression Problem**
> "Thank you, Ahsen. While Ahsen predicted the final positions, my task was to predict the exact *Fastest Lap Time* of each driver using regression models. Predicting raw lap times across different tracks is difficult because a lap in Monaco is much shorter than a lap in Spa. So, I used an *offset strategy*. Instead of predicting the raw time, the model predicts the difference between the driver's qualifying time and their race time."

**Slide 9: Regression Results**
> "I evaluated Random Forest and LightGBM. LightGBM was again the winner. As you can see in the comparison graphs on the slide, our model achieved an R-squared of 0.793. More impressively, our Mean Absolute Error (MAE) was just 2.6 seconds. This means, on average, we can predict a driver's fastest lap in a race with an error margin of only 2.6 seconds. Back to Yiğit to wrap up."

---

### Conclusion (Yiğit Enes)
**Slide 11: Conclusion**
> "To conclude, our pipeline successfully processed raw F1 data, extracted powerful contextual features, and trained highly accurate models for both classification and regression. In the future, we could improve this by integrating real-time weather data and tire degradation metrics. Thank you for listening, we are now open to your questions!"
