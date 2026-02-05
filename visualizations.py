"""Visualization module for cryptocurrency analysis."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import CHART_THEME, PRIMARY_COLOR, SECONDARY_COLOR


class CryptoVisualizer:
    """Creates interactive visualizations for crypto analysis."""
    
    def __init__(self):
        self.theme = CHART_THEME
        self.primary_color = PRIMARY_COLOR
        self.secondary_color = SECONDARY_COLOR
    
    def plot_price_history(self, df: pd.DataFrame, title: str = "Price History") -> go.Figure:
        """
        Create interactive price history chart.
        
        Args:
            df: DataFrame with Date index and Price column
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Price"],
            mode="lines",
            name="Price",
            line=dict(color=self.primary_color, width=2),
            hovertemplate="Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=self.theme,
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
    
    def plot_candlestick(self, df: pd.DataFrame, title: str = "OHLC Chart") -> go.Figure:
        """
        Create candlestick chart for OHLC data.
        
        Args:
            df: DataFrame with Open, High, Low, Close columns
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=self.primary_color,
            decreasing_line_color=self.secondary_color
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=self.theme,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_with_indicators(self, df: pd.DataFrame, indicators: list = None) -> go.Figure:
        """
        Plot price with technical indicators.
        
        Args:
            df: DataFrame with price and indicator columns
            indicators: List of indicator columns to plot
            
        Returns:
            Plotly Figure
        """
        if indicators is None:
            indicators = ["MA_7", "MA_21", "MA_50"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Price"],
            mode="lines",
            name="Price",
            line=dict(color=self.primary_color, width=2)
        ))
        
        colors = ["#FFD700", "#FF6347", "#4169E1", "#32CD32", "#9370DB"]
        for i, indicator in enumerate(indicators):
            if indicator in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode="lines",
                    name=indicator,
                    line=dict(color=colors[i % len(colors)], width=1, dash="dash")
                ))
        
        fig.update_layout(
            title="Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=self.theme,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def plot_bollinger_bands(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot Bollinger Bands.
        
        Args:
            df: DataFrame with Price and Bollinger Band columns
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            mode="lines", name="Upper Band",
            line=dict(color="rgba(255,255,255,0.3)")
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            mode="lines", name="Lower Band",
            line=dict(color="rgba(255,255,255,0.3)"),
            fill="tonexty",
            fillcolor="rgba(100,100,100,0.2)"
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Middle"],
            mode="lines", name="Middle Band",
            line=dict(color="#FFD700", width=1, dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Price"],
            mode="lines", name="Price",
            line=dict(color=self.primary_color, width=2)
        ))
        
        fig.update_layout(
            title="Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=self.theme,
            hovermode="x unified"
        )
        
        return fig
    
    def plot_rsi(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot RSI indicator.
        
        Args:
            df: DataFrame with RSI column
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            mode="lines", name="RSI",
            line=dict(color=self.primary_color, width=2)
        ))
        
        fig.add_hline(y=70, line_dash="dash", line_color=self.secondary_color, 
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#32CD32",
                      annotation_text="Oversold (30)")
        
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,0,0.1)", line_width=0)
        
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            template=self.theme
        )
        
        return fig
    
    def plot_macd(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot MACD indicator.
        
        Args:
            df: DataFrame with MACD columns
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, row_heights=[0.6, 0.4])
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Price"],
            mode="lines", name="Price",
            line=dict(color=self.primary_color, width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            mode="lines", name="MACD",
            line=dict(color="#00BFFF", width=1.5)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            mode="lines", name="Signal",
            line=dict(color="#FF6347", width=1.5)
        ), row=2, col=1)
        
        histogram = df["MACD"] - df["MACD_Signal"]
        colors = [self.primary_color if v >= 0 else self.secondary_color for v in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram,
            name="Histogram",
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(
            title="MACD Indicator",
            template=self.theme,
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
    
    def plot_forecast(self, historical: pd.DataFrame, forecast: pd.DataFrame,
                     title: str = "Price Forecast") -> go.Figure:
        """
        Plot historical data with forecast.
        
        Args:
            historical: DataFrame with historical prices
            forecast: DataFrame with predictions and confidence intervals
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical.index, y=historical["Price"],
            mode="lines", name="Historical",
            line=dict(color=self.primary_color, width=2)
        ))
        
        last_date = pd.to_datetime(historical.index[-1])
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=len(forecast))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast["Prediction"],
            mode="lines", name="Forecast",
            line=dict(color="#FFD700", width=2, dash="dash")
        ))
        
        if "Upper_CI" in forecast.columns and "Lower_CI" in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast["Upper_CI"],
                mode="lines", name="Upper CI",
                line=dict(color="rgba(255,215,0,0.3)")
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast["Lower_CI"],
                mode="lines", name="Lower CI",
                line=dict(color="rgba(255,215,0,0.3)"),
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.1)"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=self.theme,
            hovermode="x unified"
        )
        
        return fig
    
    def plot_volatility(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot volatility analysis.
        
        Args:
            df: DataFrame with volatility columns
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, row_heights=[0.6, 0.4])
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Price"],
            mode="lines", name="Price",
            line=dict(color=self.primary_color, width=2)
        ), row=1, col=1)
        
        if "Volatility_7" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Volatility_7"] * 100,
                mode="lines", name="7-Day Volatility",
                line=dict(color="#FFD700", width=1.5)
            ), row=2, col=1)
        
        if "Volatility_30" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Volatility_30"] * 100,
                mode="lines", name="30-Day Volatility",
                line=dict(color="#FF6347", width=1.5)
            ), row=2, col=1)
        
        fig.update_layout(
            title="Price and Volatility Analysis",
            template=self.theme,
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        
        return fig
    
    def plot_sentiment_gauge(self, value: float, title: str = "Sentiment Score") -> go.Figure:
        """
        Create sentiment gauge chart.
        
        Args:
            value: Sentiment value (-1 to 1)
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        normalized = (value + 1) / 2 * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=normalized,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=self.primary_color),
                steps=[
                    dict(range=[0, 33], color="#FF6B6B"),
                    dict(range=[33, 66], color="#FFD93D"),
                    dict(range=[66, 100], color="#6BCB77")
                ],
                threshold=dict(
                    line=dict(color="white", width=4),
                    thickness=0.75,
                    value=normalized
                )
            )
        ))
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    def plot_fear_greed_index(self, value: int, classification: str) -> go.Figure:
        """
        Create Fear & Greed Index visualization.
        
        Args:
            value: Index value (0-100)
            classification: Classification text
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=f"Fear & Greed Index<br><span style='font-size:0.8em'>{classification}</span>"),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="white"),
                steps=[
                    dict(range=[0, 25], color="#FF4444"),
                    dict(range=[25, 45], color="#FF8C00"),
                    dict(range=[45, 55], color="#FFD700"),
                    dict(range=[55, 75], color="#90EE90"),
                    dict(range=[75, 100], color="#00AA00")
                ]
            )
        ))
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame with multiple crypto prices
            
        Returns:
            Plotly Figure
        """
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12)
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            template=self.theme
        )
        
        return fig
    
    def plot_returns_distribution(self, df: pd.DataFrame, price_col: str = "Price") -> go.Figure:
        """
        Plot distribution of returns.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            Plotly Figure
        """
        returns = df[price_col].pct_change().dropna() * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns Distribution",
            marker_color=self.primary_color
        ))
        
        fig.add_vline(x=returns.mean(), line_dash="dash", line_color="#FFD700",
                      annotation_text=f"Mean: {returns.mean():.2f}%")
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template=self.theme
        )
        
        return fig


if __name__ == "__main__":
    from data_collector import fetch_sample_data
    from data_preprocessing import DataPreprocessor
    
    df = fetch_sample_data()
    preprocessor = DataPreprocessor()
    df = preprocessor.add_technical_indicators(df)
    
    visualizer = CryptoVisualizer()
    fig = visualizer.plot_with_indicators(df, ["MA_7", "MA_21"])
    fig.show()
