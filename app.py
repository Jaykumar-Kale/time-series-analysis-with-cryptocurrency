"""Streamlit GUI for Cryptocurrency Time Series Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import SUPPORTED_CRYPTOS, DEFAULT_CRYPTO, DEFAULT_DAYS
from data_collector import CryptoDataCollector, fetch_sample_data
from data_preprocessing import DataPreprocessor
from models.arima_model import ARIMAForecaster
from models.lstm_model import LSTMForecaster
from models.prophet_model import ProphetForecaster
from sentiment_analysis import SentimentAnalyzer, VolatilityAnalyzer
from visualizations import CryptoVisualizer

st.set_page_config(
    page_title="Crypto Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stMetric {
        background-color: #262626;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_crypto_data(crypto_id: str, days: int):
    """Load cryptocurrency data with caching."""
    collector = CryptoDataCollector()
    df = collector.get_historical_data_coingecko(crypto_id, days=days)
    
    if df.empty:
        st.warning("Could not fetch live data. Using sample data.")
        df = fetch_sample_data()
    
    return df


@st.cache_data(ttl=3600)
def get_current_price_data(crypto_id: str):
    """Get current price with caching."""
    collector = CryptoDataCollector()
    return collector.get_current_price(crypto_id)


def main():
    st.markdown('<h1 class="main-header">📈 Cryptocurrency Time Series Analysis</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        selected_crypto = st.selectbox(
            "Select Cryptocurrency",
            options=list(SUPPORTED_CRYPTOS.keys()),
            format_func=lambda x: f"{SUPPORTED_CRYPTOS[x]} - {x.title()}",
            index=0
        )
        
        days = st.slider("Historical Data (Days)", 30, 365, DEFAULT_DAYS)
        
        st.divider()
        
        st.subheader("📊 Analysis Options")
        show_indicators = st.checkbox("Show Technical Indicators", value=True)
        show_forecast = st.checkbox("Show Price Forecast", value=True)
        show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
        show_volatility = st.checkbox("Show Volatility Analysis", value=True)
        
        if show_forecast:
            st.subheader("🔮 Forecast Settings")
            forecast_model = st.selectbox(
                "Forecasting Model",
                ["ARIMA", "LSTM", "Prophet"]
            )
            forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)
    
    with st.spinner("Loading cryptocurrency data..."):
        df = load_crypto_data(selected_crypto, days)
        current_price = get_current_price_data(selected_crypto)
    
    preprocessor = DataPreprocessor()
    visualizer = CryptoVisualizer()
    
    df_clean = preprocessor.clean_data(df)
    df_indicators = preprocessor.add_technical_indicators(df_clean)
    stats = preprocessor.calculate_statistics(df_indicators)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if current_price:
            st.metric(
                "Current Price",
                f"${current_price['price']:,.2f}",
                f"{current_price['change_24h']:.2f}%"
            )
        else:
            st.metric("Latest Price", f"${df_indicators['Price'].iloc[-1]:,.2f}")
    
    with col2:
        st.metric("24h Volume", f"${current_price.get('volume_24h', 0)/1e9:.2f}B" if current_price else "N/A")
    
    with col3:
        st.metric("Volatility (30d)", f"{stats['volatility']*100:.2f}%")
    
    with col4:
        st.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")
    
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Chart", "📊 Technical Analysis", "🔮 Forecasting", "💭 Sentiment", "📉 Volatility"
    ])
    
    with tab1:
        st.subheader(f"{SUPPORTED_CRYPTOS[selected_crypto]} Price History")
        
        if show_indicators:
            fig = visualizer.plot_with_indicators(df_indicators, ["MA_7", "MA_21", "MA_50"])
        else:
            fig = visualizer.plot_price_history(df_indicators, f"{selected_crypto.title()} Price")
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Mean Price", "Median Price", "Std Dev", "Min", "Max", "Sharpe Ratio"],
                "Value": [
                    f"${stats['mean']:,.2f}",
                    f"${stats['median']:,.2f}",
                    f"${stats['std']:,.2f}",
                    f"${stats['min']:,.2f}",
                    f"${stats['max']:,.2f}",
                    f"{stats['sharpe_ratio']:.4f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            fig_dist = visualizer.plot_returns_distribution(df_indicators)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### RSI (Relative Strength Index)")
            fig_rsi = visualizer.plot_rsi(df_indicators)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            current_rsi = df_indicators["RSI"].iloc[-1]
            if current_rsi > 70:
                st.warning(f"⚠️ RSI at {current_rsi:.1f} - Potentially overbought")
            elif current_rsi < 30:
                st.success(f"✅ RSI at {current_rsi:.1f} - Potentially oversold")
            else:
                st.info(f"ℹ️ RSI at {current_rsi:.1f} - Neutral territory")
        
        with col2:
            st.markdown("### Bollinger Bands")
            fig_bb = visualizer.plot_bollinger_bands(df_indicators)
            st.plotly_chart(fig_bb, use_container_width=True)
        
        st.markdown("### MACD")
        fig_macd = visualizer.plot_macd(df_indicators)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tab3:
        if show_forecast:
            st.subheader(f"🔮 {forecast_model} Price Forecast")
            
            with st.spinner(f"Training {forecast_model} model..."):
                try:
                    if forecast_model == "ARIMA":
                        model = ARIMAForecaster(order=(5, 1, 2))
                        model.fit(df_indicators["Price"])
                        predictions = model.predict(steps=forecast_days)
                        
                    elif forecast_model == "LSTM":
                        model = LSTMForecaster(sequence_length=30, epochs=20)
                        model.fit(df_indicators["Price"], verbose=0)
                        predictions = model.predict(steps=forecast_days)
                        
                    else:
                        model = ProphetForecaster()
                        model.fit(df_indicators["Price"])
                        predictions = model.predict(steps=forecast_days)
                    
                    fig_forecast = visualizer.plot_forecast(
                        df_indicators, predictions,
                        f"{selected_crypto.title()} {forecast_days}-Day Forecast ({forecast_model})"
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    current_price_val = df_indicators["Price"].iloc[-1]
                    predicted_price = predictions["Prediction"].iloc[-1]
                    price_change = ((predicted_price - current_price_val) / current_price_val) * 100
                    
                    with col1:
                        st.metric("Current Price", f"${current_price_val:,.2f}")
                    with col2:
                        st.metric(f"Predicted ({forecast_days}d)", f"${predicted_price:,.2f}")
                    with col3:
                        st.metric("Expected Change", f"{price_change:.2f}%",
                                 delta=f"{price_change:.2f}%")
                    
                    with st.expander("📋 Forecast Details"):
                        st.dataframe(predictions, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
        else:
            st.info("Enable 'Show Price Forecast' in the sidebar to view predictions.")
    
    with tab4:
        if show_sentiment:
            st.subheader("💭 Sentiment Analysis")
            
            sentiment_analyzer = SentimentAnalyzer()
            
            with st.spinner("Analyzing market sentiment..."):
                report = sentiment_analyzer.generate_sentiment_report(selected_crypto)
                fear_greed = report["fear_greed_index"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Fear & Greed Index")
                fig_fg = visualizer.plot_fear_greed_index(
                    fear_greed["value"],
                    fear_greed["classification"]
                )
                st.plotly_chart(fig_fg, use_container_width=True)
            
            with col2:
                st.markdown("### Overall Sentiment")
                fig_sentiment = visualizer.plot_sentiment_gauge(
                    report["combined_score"],
                    "Market Sentiment"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            st.markdown(f"### 📝 Recommendation")
            st.info(report["recommendation"])
            
            st.markdown("### 📰 Recent News Sentiment")
            news_df = pd.DataFrame(report["news_details"][:10])
            if not news_df.empty:
                news_df["sentiment_color"] = news_df["sentiment"].map({
                    "Positive": "🟢",
                    "Negative": "🔴",
                    "Neutral": "🟡"
                })
                st.dataframe(
                    news_df[["sentiment_color", "title", "source", "polarity"]],
                    column_config={
                        "sentiment_color": "📊",
                        "title": "Headline",
                        "source": "Source",
                        "polarity": st.column_config.ProgressColumn("Polarity", min_value=-1, max_value=1)
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("Enable 'Show Sentiment Analysis' in the sidebar to view sentiment data.")
    
    with tab5:
        if show_volatility:
            st.subheader("📉 Volatility Analysis")
            
            volatility_analyzer = VolatilityAnalyzer()
            vol_metrics = volatility_analyzer.calculate_volatility_metrics(df_indicators)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Daily Volatility", f"{vol_metrics['daily_volatility']*100:.2f}%")
            with col2:
                st.metric("Annualized Vol", f"{vol_metrics['annualized_volatility']*100:.2f}%")
            with col3:
                st.metric("7-Day Vol", f"{vol_metrics['current_vol_7d']*100:.2f}%")
            with col4:
                st.metric("Vol Trend", vol_metrics['volatility_trend'])
            
            fig_vol = visualizer.plot_volatility(df_indicators)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            df_regimes = volatility_analyzer.detect_volatility_regimes(df_indicators)
            
            st.markdown("### Volatility Regimes")
            regime_counts = df_regimes["regime"].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Vol Days", regime_counts.get("High", 0))
            with col2:
                st.metric("Normal Vol Days", regime_counts.get("Normal", 0))
            with col3:
                st.metric("Low Vol Days", regime_counts.get("Low", 0))
        else:
            st.info("Enable 'Show Volatility Analysis' in the sidebar to view volatility data.")
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>📈 Cryptocurrency Time Series Analysis Dashboard</p>
        <p style='font-size: 0.8rem;'>Data provided by CoinGecko API | For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
