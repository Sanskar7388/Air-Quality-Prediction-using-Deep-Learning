

# **Air Quality Prediction Using GRU + Attention**

## **1. Problem Statement**

Delhi faces persistently high **PM2.5 pollution levels**, often exceeding WHO safety limits throughout the year. These pollution spikes severely impact public health, disrupt daily life, and complicate government decision-making.

To address this challenge, the objective of this project is to build a **deep learning model capable of predicting the next hour’s PM2.5 concentration** using historical air-quality and meteorological data.
Accurate short-term forecasts can help individuals plan their outdoor activities and assist authorities in deploying timely interventions.

---

## **2. Dataset Description**

### **Source**

Hourly air-quality dataset from **Mendeley Data**, recorded by the **Central Pollution Control Board (CPCB)** across six monitoring stations in Delhi.

### **Dataset Details**

* **Duration:** June 2018 – October 2019
* **Frequency:** Hourly
* **Stations:** 6 locations across Delhi

### **Features Used**

**Pollutants:**
PM2.5, PM10, NO₂, NOx, SO₂, CO, O₃

**Weather Variables:**
Temperature, Humidity, Pressure, Wind Speed

### **Preprocessing Steps**

* Removed non-informative and redundant columns
* Applied **Min–Max scaling** (fitted only on training data to avoid leakage)
* Created **48-hour sliding windows** to predict the next hour
* Used a **chronological split**:

  * 80% Training
  * 20% Testing

---

## **3. Literature Review**

A brief review of recent studies helped shape model selection and methodology.

### **Study 1: Guo et al. (2025)**

Proposed a **CNN–LSTM hybrid** to capture spatial + temporal features, achieving strong accuracy.
DOI: 10.70711/aitr.v2i10.7147

### **Study 2: Gayathri et al. (2024)**

Compared **LSTM, Bi-LSTM, and CNN-BiLSTM**, with CNN-BiLSTM outperforming others in prediction error.
Source: IJISAE

### **Study 3: Bawane et al. (2025)**

Evaluated **Random Forest, XGBoost, LSTM, and GRU**, concluding that GRU achieved the highest accuracy with **R² ≈ 0.95**.
Source: J Neonatal Surg

### **Summary of Literature Findings**

Across the studies:

* Deep learning models consistently outperform classical ML models
* **GRU, LSTM, and hybrid CNN-LSTM approaches** show the best forecasting performance
* GRU is often preferred due to **faster training** and **lower computational load**

---

## **4. Proposed Model: GRU + Attention**

### **Why GRU?**

* Captures long-term temporal patterns effectively
* Computationally lighter than LSTM
* Well-suited for real-time forecasting scenarios

### **Model Pipeline**

* **Input:** Past 48 hours of pollutant + weather data
* **GRU Layer 1:** 128 units (custom implementation)
* **GRU Layer 2:** 64 units
* **Attention Layer:** Highlights the most influential past timesteps
* **Slice Layer:** Selects the final timestep’s processed representation
* **Dense Layer:** 64 units (ReLU)
* **Output Layer:** Predicts PM2.5 for next hour

### **Training Configuration**

* **Train/Test Split:** Chronological 80/20
* **Window Size:** 48 hours → 1 prediction
* **Loss Function:** **Mean Squared Error (MSE)**

  * Chosen because it heavily penalizes large errors, improving sensitivity to high pollution spikes
* **Optimizer:** Adam
* **Epochs:** 30
* **Batch Size:** 4 (train), 2 (validation)

### **Model Performance**

| Model               | R² Score   | MAE        |
| ------------------- | ---------- | ---------- |
| **GRU + Attention** | **0.8523** | **0.0203** |
| Bidirectional LSTM  | 0.8116     | 0.0209     |
| Transformer Encoder | 0.7492     | 0.0330     |
| CNN + LSTM          | 0.9151     | 17.1613    |
| GRU                 | 0.9101     | 17.33      |

The GRU + Attention model demonstrated a strong balance between accuracy and stability, particularly during abrupt changes in PM2.5 concentrations.

---

## **5. Key Learnings & Takeaways**

* **GRU models** are highly effective for time-series forecasting involving long temporal dependencies.
* **Attention mechanisms** improve robustness by identifying critical past moments influencing PM2.5 levels.
* **Chronological splitting** is essential to prevent future data leakage.
* Proper preprocessing—especially **scaling** and **time-window creation**—significantly affects model performance.
* While Bi-LSTM and Transformer models performed reasonably well, **GRU + Attention** offered the best trade-off between **accuracy, computational cost, and interpretability**.


Video Submission :- https://drive.google.com/file/d/1W6H4b332IznDeO400LqWD3TfiuCX5xKb/view?usp=sharing

Authors: Aditya Masutey, Mukund Jha, Sanskar Sengar, Manish bist 
