# Time Series Analytics on Cryptocurrency Prices (Bitcoin)

## Project Overview
This project focuses on **time series analysis and forecasting of Bitcoin (BTC-USD) prices** using real-world historical data.  
The goal is to understand price trends, volatility, and market behavior, and to build reliable forecasting models using both **statistical** and **advanced time series techniques**.

The project was completed as part of a **Data Analytics Internship**, following an end-to-end workflow used by professional data analysts.

---

## Objectives
- Collect real-world cryptocurrency price data
- Clean and preprocess messy time series data
- Perform exploratory data analysis (EDA)
- Build a baseline forecasting model using **ARIMA**
- Build an advanced forecasting model using **Prophet**
- Evaluate and compare models using quantitative metrics
- Select the best model based on evidence

---

## Key Concepts Applied
- Time Series Analysis  
- Stationarity & Differencing  
- Exploratory Data Analysis (EDA)  
- ARIMA Modeling  
- Prophet Forecasting  
- Model Evaluation (RMSE, MAE)  
- Forecast Interpretation  

---

## Workflow Summary

### Data Collection
- Historical Bitcoin price data (`BTC-USD`) collected using Yahoo Finance  
- Date range: **2019 – 2026**  
- Raw data preserved for reproducibility  

### Data Cleaning & Preprocessing
- Removed invalid index rows from the raw dataset  
- Converted timestamps into a proper `DatetimeIndex`  
- Ensured numeric data types for price and volume  
- Selected relevant features (`Close`, `Volume`)  
- Created a clean, model-ready dataset  

### Exploratory Data Analysis (EDA)
- Analyzed long-term price trends  
- Studied trading volume behavior  
- Used rolling averages to smooth volatility  
- Analyzed daily returns to understand risk and distribution  
- Identified high volatility and non-normal return patterns  

### Time Series Modeling
- **ARIMA(1,1,1)** used as a baseline statistical model  
- Stationarity validated using the ADF test  
- Differencing applied to stabilize the series  

### Advanced Forecasting
- **Prophet** model implemented to handle trend changes and seasonality  
- Forecasts generated with confidence intervals  
- Prophet results compared against ARIMA  

---

## Model Evaluation & Comparison

| Model   | RMSE   | MAE   | Interpretation |
|--------|--------|-------|----------------|
| ARIMA  | ~39,621 | ~35,678 | Underfits volatile crypto data |
| Prophet | ~14,636 | ~12,289 | Captures trends and volatility better |

### Final Conclusion
> Prophet outperformed ARIMA by reducing forecasting error by **more than 60%**, making it a better choice for highly volatile cryptocurrency time series data.

---

## Key Insights
- Bitcoin prices are **non-stationary and highly volatile**
- Linear models like ARIMA struggle with nonlinear market behavior
- Prophet adapts better to trend changes and seasonality
- Model selection should always be **data-driven and metric-based**

---

## Tools & Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Statsmodels  
- Prophet  
- Scikit-learn  
- Jupyter Notebook  

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
