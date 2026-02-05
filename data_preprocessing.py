"""Data preprocessing and exploration module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Handles data cleaning, transformation, and feature engineering."""
    
    def __init__(self):
        self.scaler = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing values ('interpolate', 'ffill', 'bfill', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if method == "interpolate":
            df = df.interpolate(method="linear")
        elif method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "drop":
            df = df.dropna()
        
        # Fill any remaining NaN at edges
        df = df.bfill().ffill()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame, price_col: str = "Price") -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the price column
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df["MA_7"] = df[price_col].rolling(window=7).mean()
        df["MA_21"] = df[price_col].rolling(window=21).mean()
        df["MA_50"] = df[price_col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df["EMA_12"] = df[price_col].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        df["RSI"] = self.calculate_rsi(df[price_col])
        
        # Bollinger Bands
        df["BB_Middle"] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        
        # Daily Returns
        df["Daily_Return"] = df[price_col].pct_change()
        
        # Volatility (Rolling Standard Deviation of Returns)
        df["Volatility_7"] = df["Daily_Return"].rolling(window=7).std()
        df["Volatility_30"] = df["Daily_Return"].rolling(window=30).std()
        
        # Price Rate of Change
        df["ROC_7"] = df[price_col].pct_change(periods=7)
        df["ROC_14"] = df[price_col].pct_change(periods=14)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def normalize_data(self, df: pd.DataFrame, columns: list = None, method: str = "minmax") -> tuple:
        """
        Normalize specified columns.
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize (default: all numeric)
            method: Normalization method ('minmax' or 'standard')
            
        Returns:
            Tuple of (normalized DataFrame, scaler)
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        df[columns] = self.scaler.fit_transform(df[columns])
        
        return df, self.scaler
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("No scaler fitted. Call normalize_data first.")
        return self.scaler.inverse_transform(data)
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> tuple:
        """
        Create sequences for LSTM model.
        
        Args:
            data: Input data array
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def calculate_statistics(self, df: pd.DataFrame, price_col: str = "Price") -> dict:
        """
        Calculate descriptive statistics.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the price column
            
        Returns:
            Dictionary of statistics
        """
        prices = df[price_col]
        returns = prices.pct_change().dropna()
        
        stats = {
            "mean": prices.mean(),
            "median": prices.median(),
            "std": prices.std(),
            "min": prices.min(),
            "max": prices.max(),
            "range": prices.max() - prices.min(),
            "mean_return": returns.mean(),
            "volatility": returns.std(),
            "sharpe_ratio": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            "max_drawdown": self.calculate_max_drawdown(prices),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis()
        }
        
        return stats
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown as percentage
        """
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()


if __name__ == "__main__":
    from data_collector import fetch_sample_data
    
    df = fetch_sample_data()
    preprocessor = DataPreprocessor()
    
    df_clean = preprocessor.clean_data(df)
    df_indicators = preprocessor.add_technical_indicators(df_clean)
    stats = preprocessor.calculate_statistics(df_indicators)
    
    print("Sample Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
