"""Data collection module for cryptocurrency price data."""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import COINGECKO_API_URL, SUPPORTED_CRYPTOS


class CryptoDataCollector:
    """Collects cryptocurrency data from multiple API sources."""
    
    def __init__(self):
        self.coingecko_url = COINGECKO_API_URL
    
    def get_historical_data_coingecko(self, crypto_id: str, vs_currency: str = "usd", days: int = 365) -> pd.DataFrame:
        """
        Fetch historical price data from CoinGecko API.
        
        Args:
            crypto_id: Cryptocurrency ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Target currency (default: 'usd')
            days: Number of days of historical data
            
        Returns:
            DataFrame with Date, Price, Market_Cap, Volume columns
        """
        url = f"{self.coingecko_url}/coins/{crypto_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame({
                "Date": [datetime.fromtimestamp(x[0] / 1000) for x in data["prices"]],
                "Price": [x[1] for x in data["prices"]],
                "Market_Cap": [x[1] for x in data["market_caps"]],
                "Volume": [x[1] for x in data["total_volumes"]]
            })
            
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df = df.drop_duplicates(subset=["Date"])
            df.set_index("Date", inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from CoinGecko: {e}")
            return pd.DataFrame()
    
    def get_historical_data_yfinance(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Time period (e.g., '1y', '2y', '5y')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            df.index = df.index.date
            df.index.name = "Date"
            return df
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, crypto_id: str, vs_currency: str = "usd") -> dict:
        """
        Fetch current price and market data.
        
        Args:
            crypto_id: Cryptocurrency ID
            vs_currency: Target currency
            
        Returns:
            Dictionary with current market data
        """
        url = f"{self.coingecko_url}/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": vs_currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "price": data[crypto_id][vs_currency],
                "market_cap": data[crypto_id].get(f"{vs_currency}_market_cap", 0),
                "volume_24h": data[crypto_id].get(f"{vs_currency}_24h_vol", 0),
                "change_24h": data[crypto_id].get(f"{vs_currency}_24h_change", 0)
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current price: {e}")
            return {}
    
    def get_multiple_cryptos(self, crypto_ids: list, vs_currency: str = "usd", days: int = 365) -> dict:
        """
        Fetch historical data for multiple cryptocurrencies.
        
        Args:
            crypto_ids: List of cryptocurrency IDs
            vs_currency: Target currency
            days: Number of days
            
        Returns:
            Dictionary with crypto_id as key and DataFrame as value
        """
        data = {}
        for crypto_id in crypto_ids:
            df = self.get_historical_data_coingecko(crypto_id, vs_currency, days)
            if not df.empty:
                data[crypto_id] = df
        return data


def fetch_sample_data():
    """Generate sample data for testing when API is unavailable."""
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=365, freq="D")
    np.random.seed(42)
    
    base_price = 45000
    returns = np.random.normal(0.001, 0.03, 365)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "Date": dates,
        "Price": prices,
        "Market_Cap": prices * 19000000,
        "Volume": np.random.uniform(20e9, 50e9, 365)
    })
    
    df["Date"] = df["Date"].dt.date
    df.set_index("Date", inplace=True)
    
    return df


if __name__ == "__main__":
    collector = CryptoDataCollector()
    btc_data = collector.get_historical_data_coingecko("bitcoin", days=30)
    print(btc_data.head())
