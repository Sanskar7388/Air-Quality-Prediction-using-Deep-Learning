🌿 Air Quality Prediction using Deep Learning

This project focuses on **predicting Air Quality Index (AQI)** using **deep learning models**. By analyzing key environmental parameters such as temperature, humidity, CO, NO₂, and PM2.5 levels, the model learns to forecast air quality trends — helping detect pollution risks early and promote healthier living environments.

---

## 🚀 Features

- Predicts **Air Quality Index (AQI)** using sensor/environmental data  
- Implements **deep learning algorithms** (ANN, CNN, or LSTM depending on data type)  
- Supports **data preprocessing, normalization, and visualization**  
- Generates **pollution level classification** (Good, Moderate, Unhealthy, etc.)  
- Can be integrated with **IoT-based air monitoring systems**

---

## 🧠 Project Overview

The increasing rate of air pollution worldwide makes AQI forecasting critical for public health and smart city applications.  
This project uses deep learning techniques to:
1. Learn from historical air quality datasets  
2. Predict future AQI values  
3. Identify pollution severity categories  

---

## 🗂️ Dataset

You can use publicly available datasets such as:
- [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- [OpenAQ API](https://openaq.org)
- [Kaggle - Air Quality Data in India or Global Cities](https://www.kaggle.com/datasets)
- https://data.mendeley.com/datasets/bzhzr9b64v/1

**Typical features include:**
- PM2.5, PM10  
- CO, NO₂, SO₂, O₃  
- Temperature, Humidity, Wind Speed  
- Date & Time  

---

## ⚙️ Tech Stack

- **Language:** Python 🐍  
- **Libraries:** TensorFlow / Keras, Pandas, NumPy, Matplotlib, Scikit-learn  
- **Environment:** Jupyter Notebook or VS Code  

---

# PM2.5 Forecasting Project — README

**What this README contains**

* A concise summary of every model you trained
* The important plots to include in your presentation and their recommended filenames
* Code snippets to save each plot at high quality
* Run / reproducibility instructions and environment dependencies
* Suggested slide order and figure captions

---

# Project overview

This project predicts **PM2.5** using an integrated air-quality dataset (`Integrated_AQI_Data.csv`) with features such as `PM10, AT (air temperature), BP, SR, RH, WS, WD, NO, NO2, SO2, Ozone, CO, Benzene, NH3, NOx`, plus `year, month, day, hour`.

You trained multiple models (deep learning and classical/hybrid). This README lists the key evaluation plots to generate and attach to your report or presentation, plus instructions to save high-quality images for slides.

---

# Models included

1. **GRU** (baseline)
2. **GRU + Attention**
3. **Bidirectional LSTM (BiLSTM)**
4. **CNN + LSTM**
5. **Transformer Encoder**
6. **CNN + GRU (hybrid)**
7. **Temporal Convolutional Network (TCN)**
8. **CustomGRU + CustomAttention** (your custom implementation)

> Recommendation: use the same train/validation/test splits and the same scaling for all models to ensure fair comparison.

---

# Essential plots (filenames & purpose)

Below are the **must-have** plots for every model and a few project-level figures. Save each plot to `figures/` with the filename suggested.

## Per-model plots (for every trained model)

1. **Training vs Validation Loss (spike graph)**

   * Filename: `figures/{model_name}_train_val_loss.png`
   * Purpose: show learning dynamics and overfitting/underfitting
   * Save tip: `plt.savefig('figures/{}_train_val_loss.png'.format(model_name), dpi=300, bbox_inches='tight')`

2. **Actual vs Predicted (Scatter + 45° reference)**

   * Filename: `figures/{model_name}_actual_vs_pred_scatter.png`
   * Purpose: visual fit; points along diagonal indicate good fit

3. **Time-series: Actual vs Predicted (full test set)**

   * Filename: `figures/{model_name}_timeseries_full.png`
   * Purpose: show how model captures trends and spikes over time

4. **Time-series: Zoomed (first 200–300 points)**

   * Filename: `figures/{model_name}_timeseries_zoom.png`
   * Purpose: clearer view for presentation slides

5. **Residual Plot (Residuals vs Predicted)**

   * Filename: `figures/{model_name}_residuals.png`
   * Purpose: detect heteroscedasticity or systematic bias

6. **Error Distribution (Histogram of residuals)**

   * Filename: `figures/{model_name}_error_dist.png`
   * Purpose: show whether errors are centered around zero and their spread

## Project-level / comparative plots

1. **Complexity vs Performance**

   * Filenames: `figures/complexity_vs_r2.png`, `figures/complexity_vs_mae.png`
   * Purpose: visualize how model complexity relates to performance (R²/MAE)

2. **Model comparison table / bar chart**

   * Filename: `figures/model_comparison_metrics.png`
   * Purpose: side-by-side R², MAE, RMSE for all models

3. **Correlation Heatmap (annotated)**

   * Filename: `figures/correlation_heatmap.png`
   * Purpose: show feature correlations with PM2.5 and among themselves

4. **Feature importance / SHAP summary (for XGBoost or deep models)**

   * Filename: `figures/shap_summary.png`
   * Purpose: explain which features drive predictions

5. **Attention heatmap (for attention-enabled models)**

   * Filename: `figures/{model_name}_attention_heatmap.png`
   * Purpose: visualize temporal attention weights (how model attends to time steps)

6. **Optional: Confusion matrix on binned categories**

   * Filename: `figures/confusion_binned.png`
   * Purpose: if you convert PM2.5 into categories (Good/Moderate/Severe), show classification confusion

---

# Code snippets — how to save plots (use after each plotting cell)

Use these snippets to save the figures you already plot in your notebook or script.

```python
# generic: save current figure with high resolution
plt.savefig('figures/<filename>.png', dpi=300, bbox_inches='tight')
```

Examples (replace `model_name`):

```python
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Train vs Val Loss')
plt.savefig(f'figures/{model_name}_train_val_loss.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.savefig(f'figures/{model_name}_actual_vs_pred_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

# Suggested file structure

```
project_root/
├─ data/
│  └─ Integrated_AQI_Data.csv
├─ notebooks/
│  └─ training_and_plots.ipynb
├─ scripts/
│  ├─ train_gru.py
│  ├─ train_transformer.py
│  └─ make_plots.py
├─ models/
│  └─ saved_model_{model_name}.h5
├─ figures/
│  └─ (all .png files generated here)
└─ README.md
```

---

# Quick reproduce steps

1. Create `figures/` directory next to your notebook or script: `mkdir figures`
2. Train each model (or load saved weights) using the same preprocessing pipeline.
3. After each model training, run the plotting cells to generate and save the figures listed above.

---

# Environment & dependencies

Minimum recommended environment (try to match your training environment):

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow==2.12.0 keras-tcn keras shap xgboost
```

Notes:

* If you use SHAP or `keras-tcn` install them separately; `keras-tcn` is optional if you use the pure-TensorFlow TCN implementation.
* Use the same TensorFlow major version you used for training to avoid saved-model incompatibilities.

---

# Suggested captions and slide order

1. **Dataset overview** — (table snapshot + correlation heatmap) `figures/correlation_heatmap.png`.
2. **Feature engineering** — describe lag creation and scaling.
3. **Model list & architecture** — small diagram for each model.
4. **Training dynamics** — (one slide per model or grouped) `*_train_val_loss.png`.
5. **Prediction quality** — `*_actual_vs_pred_scatter.png` + `*_timeseries_zoom.png`.
6. **Residual analysis** — `*_residuals.png` and `*_error_dist.png`.
7. **Model comparison** — `model_comparison_metrics.png` + `complexity_vs_r2.png`.
8. **Explainability** — `shap_summary.png` and `attention_heatmap.png`.
9. **Conclusion & future work** — short bullets.

---

# Helpful tips & best practices

* **Use the same test set** for final evaluation across all models — this ensures fairness.
* **Save random seeds** for reproducibility: `np.random.seed(...)`, `tf.random.set_seed(...)`.
* **Use sample windows** (first 200–300 points) for slide visuals so plots are legible.
* **Add 45° reference line** to actual vs predicted scatter plots to make fit obvious.
* **Check residuals** for autocorrelation—if present, consider adding lag features or ensembling.

---

# If you want, I can:

* generate a `make_plots.py` script that automatically loads saved model outputs and writes every figure to `figures/` (I can create this file for you),
* produce a slide-ready PDF layout with the key figures placed in order,
* or add pre-written slide text for each figure.

Tell me which of these you want next.

