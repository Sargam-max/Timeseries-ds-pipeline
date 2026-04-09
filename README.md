# 🧠Timeseries-ds-pipeline

> **Feature Selection · Anomaly Detection · Statistical Forecasting**  
> An end-to-end data science portfolio project — from raw features to production-ready forecasts.

**Author:** Sargam Tripathi · **Role Applied For:** Data Science Intern  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sargam%20Tripathi-blue?logo=linkedin)](https://in.linkedin.com/in/sargam-tripathi-304089318)
[![GitHub](https://img.shields.io/badge/GitHub-Sargam--max-black?logo=github)](https://github.com/Sargam-max)

---

## 📌 Overview

This repository contains a single cohesive Jupyter notebook integrating three fundamental data science workflows on real-world datasets. Each module stands alone as a learning reference — but together they form a complete ML pipeline:

```
Raw data  ──▶  Feature Selection  ──▶  Anomaly Detection  ──▶  Forecasting
              (remove noise)            (clean the signal)      (predict the future)
```

| Module | Problem | Dataset | Key Techniques |
|--------|---------|---------|----------------|
| 🔬 **1 — Feature Selection** | Which features actually matter? | UCI Wine (178 samples, 13 features) | Variance Threshold, SelectKBest (MI), RFE, Boruta |
| 🚨 **2 — Anomaly Detection** | Detect unusual spikes in cloud metrics | AWS EC2 CPU — Numenta NAB (4,032 pts) | MAD Robust Z-score, Isolation Forest, Local Outlier Factor |
| 📈 **3 — Forecasting** | Predict bakery sales 7 days ahead | French Bakery daily sales | Naive baselines, ARIMA, SARIMA, Exogenous features, Prediction intervals |

---

## 📂 Repository Structure

```
timeseries-ds-pipeline/
│
├── ds_portfolio_combined.ipynb        ← Main notebook (all 3 modules, 82 cells)
│
├── data/
│   ├── ec2_cpu_utilization.csv        ← AWS EC2 CPU utilisation (Numenta NAB)
│   └── daily_sales_french_bakery.csv  ← French Bakery daily sales + unit prices
│
├── requirements.txt
└── README.md
```

---

## 🔬 Module 1 — Feature Selection

**Dataset:** UCI Wine Recognition — loaded directly via `sklearn.datasets.load_wine()`  
**Task:** 3-class wine cultivar classification  
**Classifier:** `GradientBoostingClassifier(max_depth=5)` · Metric: weighted F1-score  
**Split:** Stratified 70/30 train/test (`random_state=42`)

### Methods Compared

| Method | Type | Features Selected | Key Idea |
|--------|------|:-----------------:|----------|
| All features (baseline) | — | 13 | Benchmark to beat |
| Variance Threshold | Filter | 11 | Drop `ash` & `magnesium` — near-zero variance after MinMax scaling |
| SelectKBest — Mutual Info | Filter | 3 | Score features by non-linear dependence with target; sweep k=1…13 |
| RFE | Wrapper | 3 | Recursively remove weakest feature; model-aware; sweep k=1…13 |
| Boruta | All-relevant | ~9 | Compare each feature against random shadow features using Random Forest |

### Key Visuals
- Variance bar chart with threshold line — identifies low-information features at a glance  
- F1 vs. k sweep plots for SelectKBest and RFE — pinpoints the optimal feature count  
- Side-by-side scatter: F1-score vs. number of features (efficiency plot)  
- Summary table: method · N features · F1 · Δ vs baseline · speed · model-awareness  

---

## 🚨 Module 2 — Anomaly Detection in Time Series

**Dataset:** AWS EC2 CPU utilisation (`ec2_cpu_utilization_24ae8d`) from the [Numenta NAB benchmark](https://github.com/numenta/NAB)  
- **4,032 data points** recorded every 5 minutes starting 2014-02-14  
- **2 labelled ground-truth anomalies:** `2014-02-26 22:05` and `2014-02-27 17:15`  
- Licence: AGPL-3.0

**Split:** Temporal — train on `[:3550]`, test on `[3550:]` (both anomalies fall in the test set)  
**Contamination:** `1 / len(train)` for ML models

### Methods Compared

| Method | Family | Core Formula / Algorithm | Threshold |
|--------|--------|--------------------------|:---------:|
| MAD Robust Z-score | Statistical | `Z = 0.6745·(x − median) / MAD` | \|Z\| > 3.5 |
| Isolation Forest | ML — Tree-based | Anomalies isolated in fewer random tree splits | contamination |
| Local Outlier Factor | ML — Density-based | Points with lower local density than k-NN are anomalous | contamination |

### Key Visuals
- Raw time series with true anomaly positions marked (×)  
- KDE of CPU value distribution with median line  
- Individual confusion matrices per method (Blues / Oranges / Greens)  
- 3-panel side-by-side confusion matrix grid  
- Grouped bar chart: Precision / Recall / F1 for all three methods  
- TP / FP / FN breakdown chart  

---

## 📈 Module 3 — Statistical Time Series Forecasting

**Dataset:** French Bakery daily sales — products `BAGUETTE` and `CROISSANT`  
- Unit price included as exogenous variable  
- Strong **weekly seasonality** (higher weekend sales)  
- Filtered to series with ≥ 28 observations  

**Horizon:** 7 days | **Evaluation:** 8-fold rolling-origin cross-validation  
(`n_windows=8`, `step_size=7`, `refit=True`)

---

### 3.2 — Baseline Models

| Model | Logic |
|-------|-------|
| Naive | Last observed value |
| HistoricAverage | Mean of all history |
| WindowAverage(7) | Mean of last 7 days |
| **SeasonalNaive(7)** | Same weekday from last week — **best baseline** |

---

### 3.3 — ARIMA and SARIMA

| Model | Config | Captures |
|-------|--------|----------|
| ARIMA | `AutoARIMA(seasonal=False)` | Trend + autocorrelation, no seasonality |
| SARIMA | `AutoARIMA(season_length=7)` | Trend + autocorrelation + weekly seasonality |

`AutoARIMA` auto-selects the best (p,d,q) order by minimising AIC — no manual grid search needed.  
Both models evaluated via 8-fold CV and benchmarked against SeasonalNaive.

---

### 3.4 — Forecasting with Exogenous Features

| Variant | Extra Signal | How |
|---------|-------------|-----|
| `SARIMA_exog` | Unit price of each product | Pass `X_df=futr_exog_df` to `sf.predict()` |
| `SARIMA_time_exog` | Fourier terms (k=2, s=7) + day/week/month calendar | `utilsforecast.feature_engineering.pipeline` |

---

### 3.5 — Prediction Intervals

AutoARIMA with **80% prediction intervals** — shows the uncertainty band around every point forecast.  
Evaluated using both a held-out test split and 8-fold cross-validation.

---

### 3.6 — Full Evaluation Suite

Final face-off: **SARIMA_exog vs SeasonalNaive** across 7 metrics.

| Metric | What It Penalises |
|--------|------------------|
| MAE | All errors equally |
| MSE | Large errors more (squared) |
| RMSE | Large errors — same units as target |
| MAPE | Percentage error (scale-free) |
| sMAPE | Symmetric MAPE — handles near-zero values better |
| MASE (s=7) | Normalised vs seasonal naive |
| scaled CRPS | Full probabilistic forecast quality |

Visualised as a **3×3 metric grid** with a gold border on the winner per metric.

---

### 📊 Module 3 Leaderboard (Cross-Validated MAE, 8 folds)

| Rank | Model | MAE |
|:----:|-------|:---:|
| 🥇 | SARIMA + Price Exog | **19.210** |
| 🥈 | SARIMA | 19.281 |
| 🥉 | SARIMA + Time Features | 19.533 |
| 4 | Seasonal Naive | 21.118 |
| 5 | ARIMA | 21.229 |

---

## 📊 Results Summary

| Module | Best Method | Score |
|--------|-------------|-------|
| 🔬 Feature Selection | RFE — 3 features | F1 = **1.000** |
| 🚨 Anomaly Detection | Isolation Forest | F1 = **1.000** (test set) |
| 📈 Forecasting | SARIMA + Price Exog | MAE = **19.210** |

---

## 🛠️ Setup

### 1 — Clone the repository
```bash
git clone https://github.com/Sargam-max/timeseries-ds-pipeline.git
cd timeseries-ds-pipeline
```

### 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Download the datasets

Create a `data/` folder in the project root, then download:

**Module 2 — EC2 CPU data (Numenta NAB, AGPL-3.0):**
```
https://github.com/numenta/NAB/blob/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv
→ Save as: data/ec2_cpu_utilization.csv
```

**Module 3 — French Bakery sales:**
```
https://github.com/marcopeix
→ Save as: data/daily_sales_french_bakery.csv
```

### 5 — Launch the notebook
```bash
jupyter notebook ds_portfolio_combined.ipynb
```

Run all cells top-to-bottom with **Cell → Run All**.

> ⚠️ `boruta-py` is required for Module 1 Section 1.6 — it is included in `requirements.txt`.

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
statsforecast>=1.7.0
utilsforecast>=0.1.0
boruta-py>=0.3.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 📚 References

1. Guyon & Elisseeff (2003). *An Introduction to Variable and Feature Selection*. JMLR.
2. Kursa & Rudnicki (2010). *Feature Selection with the Boruta Package*. JSS.
3. Chandola, Banerjee & Kumar (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys.
4. Liu, Ting & Zhou (2008). *Isolation Forest*. IEEE ICDM.
5. Breunig, Kriegel, Ng & Sander (2000). *LOF: Identifying Density-Based Local Outliers*. SIGMOD.
6. Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.


---

## 🙋 Author

**Sargam Tripathi**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://in.linkedin.com/in/sargam-tripathi-304089318)
[![GitHub](https://img.shields.io/badge/GitHub-Sargam--max-black?logo=github)](https://github.com/Sargam-max)

---

*Built as a Data Science internship portfolio project*
