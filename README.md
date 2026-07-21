#  StockSense – Demand Forecasting for Inventory

##  Overview

StockSense is a machine learning demand forecasting system developed to help distribution businesses optimize inventory management. The system predicts future daily demand for each Store–Item (SKU) combination, enabling purchasing teams to make better replenishment decisions.

By forecasting demand accurately, the business can reduce:

- **Stockouts**, which lead to lost sales and dissatisfied customers.
- **Overstock**, which increases storage costs and ties up inventory.

The project reconstructs a complete machine learning pipeline, from problem framing and feature engineering to model evaluation and deployment.

---

#  Business Problem

A distribution company manages inventory across multiple stores and hundreds of products.

Without accurate demand forecasts:

- Popular products may run out of stock, causing lost revenue.
- Slow-moving products may be over-ordered, increasing inventory costs.

The objective of this project is to forecast future demand for every Store × Item combination so purchasing teams can make data-driven ordering decisions.

---

#  Dataset

**Dataset:** Store Item Demand Forecasting Challenge (Kaggle)

The dataset contains:

- 5 years of daily sales data
- 10 stores
- 50 products (SKUs)
- 500 Store × Item time series
- 913,000 total records

Each record contains:

| Feature | Description |
|----------|-------------|
| date | Date of sale |
| store | Store identifier |
| item | Product identifier |
| sales | Number of units sold (Target) |

---

#  Problem Framing

This project is formulated as a **time-series regression** problem.

### Target

Predict the daily sales quantity (`sales`).

### Forecast Granularity

- Store × Item × Day

### Forecast Horizon

Daily demand forecasting using historical observations.

### Modeling Strategy

A **single global forecasting model** was trained across all 500 Store × Item time series instead of training one model per series.

This approach was selected because it:

- Learns shared demand patterns across stores and products.
- Scales efficiently to many time series.
- Generalizes better for products with limited historical data.

---

#  Data Assumptions

The following assumptions were made during model development:

- Historical sales are representative of future demand patterns.
- The dataset contains complete daily sales records with no missing values.
- Store and item identifiers remain consistent throughout the dataset.
- Holiday, promotion, and weather information are unavailable, so forecasts rely on historical sales and calendar-based features only.
- Lag and rolling statistics are computed using only past observations to prevent data leakage.
- The model predicts demand one day at a time for each Store × Item combination.

---

#  Feature Engineering

Several features were engineered to improve forecasting performance.

## Calendar Features

- Year
- Month
- Day
- Day of Week
- Week of Year
- Quarter
- Weekend Indicator

These features help the model learn seasonal and weekly demand patterns.

## Lag Features

Historical demand was incorporated using:

- Lag 1 (previous day)
- Lag 7 (same day last week)
- Lag 30 (same day last month)

Lag features allow the model to use recent demand history when making predictions.

## Rolling Statistics

Demand trends were captured using:

- 7-Day Rolling Mean
- 30-Day Rolling Mean

Rolling averages reduce daily fluctuations and help the model capture recent demand trends.

---

#  Model

The forecasting model was built using **XGBoost Regressor**.

### Model Parameters

- n_estimators = 300
- learning_rate = 0.05
- max_depth = 8
- subsample = 0.8
- colsample_bytree = 0.8
- objective = reg:squarederror

XGBoost was selected because it performs exceptionally well on structured tabular data with engineered features and efficiently models nonlinear relationships.

---

#  Model Evaluation

Instead of using a random train-test split, the model was evaluated using **rolling-origin backtesting**, which better reflects real-world forecasting.

## Backtesting Strategy

| Training Period | Validation Period |
|-----------------|-------------------|
| 2013–2014 | 2015 |
| 2013–2015 | 2016 |
| 2013–2016 | 2017 |

This approach evaluates the model on multiple future periods rather than a single validation split.

---

## Evaluation Metrics

The following metrics were used:

### MAE (Mean Absolute Error)

Measures the average prediction error in units sold.

It is easy to interpret from a business perspective because it tells us approximately how many units the prediction differs from the actual demand.

### RMSE (Root Mean Squared Error)

Measures prediction error while giving larger penalties to larger mistakes.

This is useful because large forecasting errors can significantly impact inventory decisions.

### MAPE (Mean Absolute Percentage Error)

Measures prediction error as a percentage of actual demand.

It allows performance to be compared across products with different sales volumes.

### Bias

Measures whether the model consistently overestimates or underestimates demand.

A bias close to zero is desirable because:

- Positive bias may cause overstock.
- Negative bias may increase stockouts.

---

## Average Backtesting Results

| Metric | Value |
|---------|------:|
| MAE | 5.95 |
| RMSE | 7.74 |
| MAPE | 12.94% |
| Bias | -0.076 |

The low bias indicates that the model remains balanced and does not consistently over-forecast or under-forecast demand.

---

# Feature Importance

Feature importance analysis showed that historical demand features were significantly more informative than static identifiers such as store or item.

The most influential features were:

- 7-Day Rolling Mean
- Lag 7
- 30-Day Rolling Mean
- Weekend Indicator
- Day of Week

This indicates that recent demand history is the strongest predictor of future sales.

---

#  Model Serving

The trained model is exported as:

- `model.pkl`
- `feature_columns.pkl`
- `metrics.pkl`

These files are loaded by a FastAPI application that serves predictions.

### API Input

```json
{
  "store": 1,
  "item": 10,
  "forecast_date": "2018-01-05"
}
```

The API generates the required engineered features before passing them to the trained model.

### API Response

```json
{
  "predicted_sales": 43.18
}
```

**API Development**
- FastAPI implementation
- Model serving
- Prediction endpoint
- Deployment

---

#  Monitoring Plan

After deployment, the forecasting system should be continuously monitored to ensure prediction quality.

The following metrics should be tracked over time:

- MAE
- RMSE
- MAPE
- Bias

The system should also monitor:

- Changes in sales patterns that may indicate data drift.
- API usage and prediction volume.
- Forecast performance across different stores and products.

If forecasting performance degrades or significant drift is detected, the model should be retrained using the latest available sales data.

---

#  Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Joblib
- Matplotlib
- Kaggle Notebooks
- FastAPI

---

## Team Members

- Fatimah Alwarsh
- Raed Almozel

---

#  Repository Structure

```
StockSense/
│
├── notebooks/
│   └── Demand forecasting for inventory.ipynb
│
├── model/
│   ├── model.pkl
│   ├── feature_columns.pkl
│   └── metrics.pkl
│
├── api/
│   ├── main.py
│   ├── predictor.py
│   └── requirements.txt
│
└── README.md



---



