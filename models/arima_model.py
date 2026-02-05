"""ARIMA model for time series forecasting."""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """ARIMA model for cryptocurrency price forecasting."""
    
    def __init__(self, order: tuple = (5, 1, 0)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.model_fit = None
        self.history = None
    
    def check_stationarity(self, series: pd.Series) -> dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        return {
            "test_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05
        }
    
    def find_optimal_order(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> tuple:
        """
        Find optimal ARIMA order using AIC.
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values for order parameters
            
        Returns:
            Optimal order tuple (p, d, q)
        """
        best_aic = float("inf")
        best_order = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        self.order = best_order
        return best_order
    
    def fit(self, series: pd.Series) -> None:
        """
        Fit the ARIMA model.
        
        Args:
            series: Time series data for training
        """
        self.history = series.values.tolist()
        self.model = ARIMA(series, order=self.order)
        self.model_fit = self.model.fit()
    
    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_fit.get_forecast(steps=steps)
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        result = pd.DataFrame({
            "Prediction": predictions.values,
            "Lower_CI": conf_int.iloc[:, 0].values,
            "Upper_CI": conf_int.iloc[:, 1].values
        })
        
        return result
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict:
        """
        Evaluate model performance.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        }
    
    def walk_forward_validation(self, series: pd.Series, train_size: float = 0.8) -> dict:
        """
        Perform walk-forward validation.
        
        Args:
            series: Complete time series
            train_size: Proportion of data for training
            
        Returns:
            Dictionary with predictions and metrics
        """
        n = len(series)
        train_end = int(n * train_size)
        
        train = series[:train_end].values.tolist()
        test = series[train_end:].values
        predictions = []
        
        for i in range(len(test)):
            model = ARIMA(train, order=self.order)
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
            predictions.append(pred)
            train.append(test[i])
        
        predictions = np.array(predictions)
        metrics = self.evaluate(test, predictions)
        
        return {
            "predictions": predictions,
            "actual": test,
            "metrics": metrics
        }
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if self.model_fit is None:
            return "Model not fitted."
        return str(self.model_fit.summary())


if __name__ == "__main__":
    from data_collector import fetch_sample_data
    
    df = fetch_sample_data()
    
    forecaster = ARIMAForecaster(order=(2, 1, 2))
    forecaster.fit(df["Price"])
    
    predictions = forecaster.predict(steps=7)
    print("7-Day Forecast:")
    print(predictions)
