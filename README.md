**1. Problem Statement**

Delhi experiences extremely high PM2.5 levels throughout the year, often crossing safe limits set by WHO. These pollution spikes affect public health, daily routines, and government planning. Being able to predict PM2.5 levels even one hour in advance can help people prepare, reduce exposure, and support authorities in taking timely actions.

The goal of this project was to build a deep learning model that can forecast the next hour’s PM2.5 value by learning patterns from historical pollution and meteorological data.

---

**2. Dataset**

We used an air-quality dataset published on Mendeley Data, collected by the Central Pollution Control Board (CPCB) from six monitoring stations in Delhi.

Details

Duration: June 2018 to October 2019

Frequency: Hourly data

Stations: 6 locations across Delhi

**Features Used**

Pollutants: PM2.5, PM10, NO₂, NOx, SO₂, CO, O₃

Weather: Temperature, Humidity, Pressure, Wind Speed

**Preprocessing**

Removed non-informative columns
Applied Min-Max scaling
Converted data into 48-hour sequences to predict the next hour
Used a chronological 80% train / 20% test split

---

**3. Literature Review**

The literature review was intentionally kept concise to focus on the models used in previous work and what results they achieved, so we could understand which architectures make sense for our project.

**Paper 1** 

Guo, Zicheng, Shuqi Wu, and Meixing Zhu. “Air Quality PM2.5 Index Prediction Model Based on CNN - LSTM.” Artificial Intelligence Technology Research. 2, no. 10 (July 13, 2025). [https://doi.org/10.70711/aitr.v2i10.7147](https://doi.org/10.70711/aitr.v2i10.7147) .

Model used: CNN-LSTM hybrid
Result: Predicted PM2.5 accurately and captured both spatial and temporal patterns

**Paper 2**

M., Gayathri, Kavitha V., and Anand Jeyaraj. 2024. “Forecasting Air Quality With Deep Learning.” International Journal of Intelligent Systems and Applications in Engineering, May, 01–11. [https://www.ijisae.org](https://www.ijisae.org) .

Model used: LSTM, Bi-LSTM, and CNN-BiLSTM
Result: CNN-BiLSTM achieved the lowest errors among all tested models

**Paper 3**

Bawane, Sheetal, Priyanka Chaudhary, Sanmati Kumar Jain, and Jitendra Singh Dodiya. 2025. “Forecasting of Air Quality Index Using Machine Learning and Deep Learning Models.” Journal of Neonatal Surgery 14 (18s): 1147–55. [https://www.jneonatalsurg.com](https://www.jneonatalsurg.com) .

Model used: Random Forest, XGBoost, LSTM, GRU
Result: GRU performed the best with R² = 0.952

Across all three papers, deep learning—especially GRU, LSTM, and hybrid models—consistently outperformed traditional machine-learning approaches.

---

**4. Our Main Model: GRU + Attention**

For our final model, we designed a Custom GRU + Attention architecture to forecast PM2.5 values using the previous 48 hours of multivariate time-series data.

**Why GRU?**

GRUs learn long-term temporal patterns effectively
They are lighter and faster than LSTMs
Well suited for real-time forecasting tasks

**Model Architecture**

Input Sequence: Last 48 hours of pollution + weather data
Custom GRU Layer 1: 128 units
Custom GRU Layer 2: 64 units
Attention Layer: Helps the model focus on the most relevant past timestamps
Slice Layer: Selects the final processed timestep
Dense Layer: 64 units (ReLU)
Output Layer: Predicts PM2.5 for the next hour

**Training Setup**

Train/Test split: 80/20 in chronological order
Min-Max scaling applied only on training data
Sliding window of 48 steps → 1 prediction
Loss: Mean Squared Error (MSE)
Optimizer: Adam
Epochs: 30
Batch size: 4 (train), 2 (validation)

**Results**

| Model                 | R² Score | MAE     |
| --------------------- | -------- | ------- |
| GRU + Attention Layer | 0.8523   | 0.0203  |
| Bidirectional LSTM    | 0.8116   | 0.0209  |
| Transformer Encoder   | 0.7492   | 0.0330  |
| CNN + LSTM            | 0.9151   | 17.1613 |
| GRU                   | 0.9101   | 17.33   |

**Model Interpretation**

The GRU + Attention model performed reliably with low MAE and a strong R² score.
The attention mechanism helped the model identify important past hours, making the predictions more stable during sudden changes in PM2.5.

---

**5. Final Learning & Takeaways From Our Work**

GRU-based models work extremely well for time-series forecasting.
The attention layer made the model more sensitive to sudden pollution changes.
Using a chronological split was important to avoid data leakage.
Proper scaling and sequence-window creation had a big impact on training stability.
Models like Transformer and Bi-LSTM performed reasonably well, but GRU + Attention balanced accuracy, speed, and interpretability the best.

This project helped us understand how temporal models work and how deep learning can be applied to real environmental problems.

---

Video Submission :- https://drive.google.com/file/d/1W6H4b332IznDeO400LqWD3TfiuCX5xKb/view?usp=sharing

Authors: Aditya Masutey, Mukund Jha, Sanskar Sengar, Manish Bisht
