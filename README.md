# Timeseries-ds-pipeline

# 🧠 timeseries-ds-pipeline

> **Feature Selection · Anomaly Detection · Statistical Forecasting**  
> A structured, end-to-end data science portfolio project — from raw features to production-ready forecasts.

---

## 📌 Overview

This repository contains a single, cohesive Jupyter notebook that walks through three fundamental data science workflows on real-world datasets. Each module stands alone as a learning reference, but together they form a complete ML pipeline: clean your features, detect anomalies in your signal, then forecast the future.

| Module | Problem | Dataset | Techniques |
|--------|---------|---------|------------|
| 🔬 **1 — Feature Selection** | Which features actually matter? | UCI Wine (scikit-learn) | Variance Threshold, SelectKBest (MI), RFE, Boruta |
| 🚨 **2 — Anomaly Detection** | Detect unusual spikes in cloud metrics | AWS EC2 CPU (Numenta NAB) | MAD Robust Z-score, Isolation Forest, Local Outlier Factor |
| 📈 **3 — Forecasting** | Predict bakery sales 7 days ahead | French Bakery daily sales | ARIMA, SARIMA, ETS, Theta, CES, MSTL+AutoARIMA, Exogenous Features |

---

## 📂 Repository Structure

```
timeseries-ds-pipeline/
│
├── ds_portfolio_combined.ipynb   ← Main notebook (all 3 modules)
│
├── data/
│   ├── ec2_cpu_utilization_24ae8d.csv      ← AWS EC2 CPU (Numenta NAB)
│   └── daily_sales_french_bakery.csv← French Bakery sales
│
├── requirements.txt
└── README.md
```

---

## 🔬 Module 1 — Feature Selection

**Dataset:** UCI Wine Recognition (178 samples, 13 features, 3 classes)  
**Model:** GradientBoostingClassifier evaluated with weighted F1-score

Four methods are benchmarked and compared:

| Method | Type | Features Selected | F1-Score |
|--------|------|:-----------------:|:--------:|
| All features (baseline) | — | 13 | *your result* |
| Variance Threshold | Filter | 11 | *your result* |
| SelectKBest (Mutual Info) | Filter | 3 | *your result* |
| RFE | Wrapper | 3 | *your result* |
| Boruta | All-relevant | ~9 | *your result* |

**Key visual:** Scatter plot of F1 vs. number of features — shows the sweet spot where fewer features match or beat the full model.

---

## 🚨 Module 2 — Anomaly Detection in Time Series

**Dataset:** AWS EC2 CPU utilisation — 4,032 data points at 5-min intervals  
**Ground truth:** 2 labelled anomaly timestamps from the [Numenta NAB benchmark](https://github.com/numenta/NAB)

Three methods compared using Precision / Recall / F1:

| Method | Family | Key Idea |
|--------|--------|----------|
| MAD Robust Z-score | Statistical | Median-based Z-score, resistant to outlier skew |
| Isolation Forest | ML — Tree | Anomalies are isolated faster in random trees |
| Local Outlier Factor | ML — Density | Anomalies have lower local density than neighbours |

**Key visual:** Side-by-side confusion matrices + grouped Precision/Recall/F1 bar chart.

---

## 📈 Module 3 — Statistical Time Series Forecasting

**Dataset:** French Bakery daily sales — BAGUETTE and CROISSANT  
**Horizon:** 7 days | **Evaluation:** 8-fold rolling cross-validation

### Model Ladder

```
Naive baselines ──→ ARIMA/SARIMA ──→ ETS / Theta / CES ──→ MSTL+ARIMA ──→ + Exogenous features
```

| Model | Family | Seasonal | Exogenous |
|-------|--------|:--------:|:---------:|
| ARIMA | ARIMA | ❌ | ✅ |
| SARIMA | ARIMA | ✅ | ✅ |
| AutoETS | Exponential Smoothing | ✅ | ❌ |
| AutoTheta | Theta | ✅ | ❌ |
| AutoCES | Complex ES | ✅ | ❌ |
| MSTL + AutoARIMA | Decomposition | ✅ | ❌ |
| SARIMA + Price | ARIMA + Exog | ✅ | ✅ |
| SARIMA + Time Features | ARIMA + Exog | ✅ | ✅ |

**Key visuals:**
- 🥇🥈🥉 Medal-coloured leaderboard across all 8 model variants
- Improvement-over-baseline chart (green = better, red = worse)
- 3×3 metric grid (MAE, RMSE, MAPE, sMAPE, MASE, CRPS) with gold-border winner per metric
- 80% prediction intervals via cross-validated AutoARIMA

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Sargam-max/timeseries-ds-pipeline.git
cd timeseries-ds-pipeline
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the datasets

**EC2 CPU data (Module 2):**
- Download `ec2_cpu_utilization_24ae8d.csv` from the [Numenta NAB repo](https://github.com/numenta/NAB/blob/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv)
- Save as `ec2_cpu_utilization.csv`

**French Bakery data (Module 3):**
- Download from [Marco Peixeiro's data](https://github.com/marcopeix)
- Save as `daily_sales_french_bakery.csv`

### 5. Launch Jupyter
```bash
jupyter notebook ds_portfolio_combined_v2.ipynb
```

Run all cells top-to-bottom with **Cell → Run All**.

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
statsforecast
utilsforecast
boruta-py
jupyter
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Results Summary

| Module | Best Method | Score |
|--------|-------------|-------|
| Feature Selection | RFE (3 features) | F1 = *1.0* |
| Anomaly Detection | Isolation Forest | F1 = *0.0* |
| Forecasting | *Best model* | MAE = *19.281* |

---

## 📚 References

1. Guyon & Elisseeff (2003). *An Introduction to Variable and Feature Selection*. JMLR.
2. Kursa & Rudnicki (2010). *Feature Selection with the Boruta Package*. JSS.
3. Chandola et al. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys.
4. Liu et al. (2008). *Isolation Forest*. IEEE ICDM.
5. Breunig et al. (2000). *LOF: Identifying Density-Based Local Outliers*. SIGMOD.
6. Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

---

## 🙋 Author

**[Sargam Tripathi]**  
[LinkedIn](https://in.linkedin.com/in/sargam-tripathi-304089318) · [GitHub](https://github.com/Sargam-max)

---

*Built as a Data Science internship portfolio project.*
