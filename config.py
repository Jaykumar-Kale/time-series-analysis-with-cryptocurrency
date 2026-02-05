"""Configuration settings for the cryptocurrency analysis project."""

# API Configuration
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
BINANCE_API_URL = "https://api.binance.com/api/v3"

# Supported Cryptocurrencies
SUPPORTED_CRYPTOS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "litecoin": "LTC",
    "avalanche-2": "AVAX"
}

# Default Settings
DEFAULT_CRYPTO = "bitcoin"
DEFAULT_CURRENCY = "usd"
DEFAULT_DAYS = 365

# Model Parameters
ARIMA_ORDER = (5, 1, 0)
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_SEQUENCE_LENGTH = 60
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Visualization Settings
CHART_THEME = "plotly_dark"
PRIMARY_COLOR = "#00D4AA"
SECONDARY_COLOR = "#FF6B6B"
