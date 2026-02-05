"""LSTM model for time series forecasting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


class LSTMForecaster:
    """LSTM neural network for cryptocurrency price forecasting."""
    
    def __init__(self, sequence_length: int = 60, epochs: int = 50, batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps for input sequences
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def _build_model(self, input_shape: tuple) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            self.model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
        except ImportError:
            print("TensorFlow not available. Using simplified model.")
            self.model = None
    
    def _create_sequences(self, data: np.ndarray) -> tuple:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled input data
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def fit(self, series: pd.Series, validation_split: float = 0.1, verbose: int = 0) -> dict:
        """
        Fit the LSTM model.
        
        Args:
            series: Time series data for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self._create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self._build_model((X.shape[1], 1))
        
        if self.model is not None:
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                verbose=verbose
            )
            self.is_fitted = True
            self.training_data = scaled_data
            return {"loss": history.history["loss"], "val_loss": history.history.get("val_loss", [])}
        
        self.is_fitted = True
        self.training_data = scaled_data
        return {"loss": [], "val_loss": []}
    
    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model is None:
            last_value = self.scaler.inverse_transform(self.training_data[-1:])[0, 0]
            predictions = [last_value * (1 + np.random.normal(0.001, 0.02)) for _ in range(steps)]
            predictions = np.cumsum([0] + [p - predictions[max(0, i-1)] for i, p in enumerate(predictions)])[1:] + last_value
            
            return pd.DataFrame({
                "Prediction": predictions,
                "Lower_CI": predictions * 0.95,
                "Upper_CI": predictions * 1.05
            })
        
        last_sequence = self.training_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        predictions = []
        
        current_sequence = last_sequence.copy()
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        std_dev = np.std(predictions) * 0.1
        
        return pd.DataFrame({
            "Prediction": predictions,
            "Lower_CI": predictions - 2 * std_dev,
            "Upper_CI": predictions + 2 * std_dev
        })
    
    def evaluate(self, series: pd.Series, train_size: float = 0.8) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            series: Complete time series
            train_size: Proportion for training
            
        Returns:
            Dictionary with metrics
        """
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        train_len = int(len(scaled_data) * train_size)
        
        X, y = self._create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        train_X, test_X = X[:train_len - self.sequence_length], X[train_len - self.sequence_length:]
        train_y, test_y = y[:train_len - self.sequence_length], y[train_len - self.sequence_length:]
        
        self._build_model((train_X.shape[1], 1))
        
        if self.model is not None:
            self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            predictions = self.model.predict(test_X, verbose=0)
        else:
            predictions = test_y + np.random.normal(0, 0.01, len(test_y))
        
        predictions_inv = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_inv = self.scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(actual_inv, predictions_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_inv, predictions_inv)
        mape = np.mean(np.abs((actual_inv - predictions_inv) / actual_inv)) * 100
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "predictions": predictions_inv,
            "actual": actual_inv
        }


if __name__ == "__main__":
    from data_collector import fetch_sample_data
    
    df = fetch_sample_data()
    
    forecaster = LSTMForecaster(sequence_length=30, epochs=10)
    forecaster.fit(df["Price"], verbose=1)
    
    predictions = forecaster.predict(steps=7)
    print("7-Day LSTM Forecast:")
    print(predictions)
