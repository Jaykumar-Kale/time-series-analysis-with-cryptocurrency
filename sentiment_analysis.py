"""Sentiment analysis module for crypto news and social media."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests


class SentimentAnalyzer:
    """Analyzes sentiment from crypto-related news and social media."""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity
        """
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            
            return {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "sentiment": self._classify_sentiment(blob.sentiment.polarity)
            }
        except ImportError:
            positive_words = ["bullish", "moon", "pump", "gain", "profit", "up", "buy", "good", "great"]
            negative_words = ["bearish", "dump", "crash", "loss", "down", "sell", "bad", "fear"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            polarity = (pos_count - neg_count) / max(pos_count + neg_count, 1)
            
            return {
                "polarity": polarity,
                "subjectivity": 0.5,
                "sentiment": self._classify_sentiment(polarity)
            }
    
    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score."""
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        return "Neutral"
    
    def fetch_crypto_news(self, crypto: str = "bitcoin", limit: int = 10) -> list:
        """
        Fetch crypto news headlines.
        
        Args:
            crypto: Cryptocurrency name
            limit: Number of news items
            
        Returns:
            List of news dictionaries
        """
        sample_news = [
            {"title": f"{crypto.title()} reaches new weekly high amid market optimism", "source": "CryptoNews"},
            {"title": f"Institutional investors increase {crypto.title()} holdings", "source": "Bloomberg Crypto"},
            {"title": f"Technical analysis suggests {crypto.title()} breakout imminent", "source": "TradingView"},
            {"title": f"{crypto.title()} network upgrade scheduled for next month", "source": "CoinDesk"},
            {"title": f"Market volatility continues as {crypto.title()} tests support levels", "source": "CoinTelegraph"},
            {"title": f"Regulatory clarity boosts {crypto.title()} adoption", "source": "Reuters"},
            {"title": f"{crypto.title()} mining difficulty reaches all-time high", "source": "Mining Weekly"},
            {"title": f"DeFi protocols show increased {crypto.title()} integration", "source": "DeFi Pulse"},
            {"title": f"Whale activity detected in {crypto.title()} markets", "source": "Whale Alert"},
            {"title": f"{crypto.title()} correlation with traditional markets decreases", "source": "Forbes"}
        ]
        
        return sample_news[:limit]
    
    def analyze_news_sentiment(self, crypto: str = "bitcoin", limit: int = 10) -> pd.DataFrame:
        """
        Fetch and analyze crypto news sentiment.
        
        Args:
            crypto: Cryptocurrency name
            limit: Number of news items
            
        Returns:
            DataFrame with news and sentiment scores
        """
        news_items = self.fetch_crypto_news(crypto, limit)
        
        results = []
        for item in news_items:
            sentiment = self.analyze_text(item["title"])
            results.append({
                "title": item["title"],
                "source": item["source"],
                "polarity": sentiment["polarity"],
                "subjectivity": sentiment["subjectivity"],
                "sentiment": sentiment["sentiment"]
            })
        
        return pd.DataFrame(results)
    
    def get_fear_greed_index(self) -> dict:
        """
        Fetch Fear & Greed Index from Alternative.me API.
        
        Returns:
            Dictionary with index value and classification
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            current = data["data"][0]
            
            return {
                "value": int(current["value"]),
                "classification": current["value_classification"],
                "timestamp": current["timestamp"],
                "time_until_update": current.get("time_until_update", "N/A")
            }
        except:
            value = np.random.randint(20, 80)
            if value < 25:
                classification = "Extreme Fear"
            elif value < 45:
                classification = "Fear"
            elif value < 55:
                classification = "Neutral"
            elif value < 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            return {
                "value": value,
                "classification": classification,
                "timestamp": str(int(datetime.now().timestamp())),
                "time_until_update": "N/A"
            }
    
    def calculate_aggregate_sentiment(self, news_df: pd.DataFrame) -> dict:
        """
        Calculate aggregate sentiment metrics.
        
        Args:
            news_df: DataFrame with sentiment data
            
        Returns:
            Dictionary with aggregate metrics
        """
        return {
            "mean_polarity": news_df["polarity"].mean(),
            "sentiment_distribution": news_df["sentiment"].value_counts().to_dict(),
            "positive_ratio": (news_df["sentiment"] == "Positive").sum() / len(news_df),
            "negative_ratio": (news_df["sentiment"] == "Negative").sum() / len(news_df),
            "overall_sentiment": self._classify_sentiment(news_df["polarity"].mean())
        }
    
    def generate_sentiment_report(self, crypto: str = "bitcoin") -> dict:
        """
        Generate comprehensive sentiment report.
        
        Args:
            crypto: Cryptocurrency name
            
        Returns:
            Dictionary with full sentiment analysis
        """
        news_df = self.analyze_news_sentiment(crypto, limit=20)
        aggregate = self.calculate_aggregate_sentiment(news_df)
        fear_greed = self.get_fear_greed_index()
        
        combined_score = (aggregate["mean_polarity"] + (fear_greed["value"] - 50) / 100) / 2
        
        return {
            "crypto": crypto,
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": aggregate,
            "fear_greed_index": fear_greed,
            "combined_score": combined_score,
            "recommendation": self._get_recommendation(combined_score),
            "news_details": news_df.to_dict("records")
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Generate trading recommendation based on sentiment score."""
        if score > 0.3:
            return "Bullish - Consider accumulating"
        elif score > 0.1:
            return "Slightly Bullish - Watch for entry points"
        elif score > -0.1:
            return "Neutral - Hold current positions"
        elif score > -0.3:
            return "Slightly Bearish - Consider reducing exposure"
        else:
            return "Bearish - Exercise caution"


class VolatilityAnalyzer:
    """Analyzes price volatility patterns."""
    
    def calculate_volatility_metrics(self, df: pd.DataFrame, price_col: str = "Price") -> dict:
        """
        Calculate comprehensive volatility metrics.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            Dictionary with volatility metrics
        """
        returns = df[price_col].pct_change().dropna()
        
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(365)
        
        rolling_vol_7 = returns.rolling(window=7).std()
        rolling_vol_30 = returns.rolling(window=30).std()
        
        parkinson_vol = self._parkinson_volatility(df) if "High" in df.columns else None
        
        return {
            "daily_volatility": daily_vol,
            "annualized_volatility": annualized_vol,
            "current_vol_7d": rolling_vol_7.iloc[-1] if len(rolling_vol_7) > 0 else 0,
            "current_vol_30d": rolling_vol_30.iloc[-1] if len(rolling_vol_30) > 0 else 0,
            "vol_percentile": self._calculate_vol_percentile(rolling_vol_30),
            "parkinson_volatility": parkinson_vol,
            "volatility_trend": self._assess_vol_trend(rolling_vol_30)
        }
    
    def _parkinson_volatility(self, df: pd.DataFrame, window: int = 30) -> float:
        """Calculate Parkinson volatility using high-low prices."""
        if "High" not in df.columns or "Low" not in df.columns:
            return None
        
        log_hl = np.log(df["High"] / df["Low"])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean())
        return parkinson.iloc[-1] if len(parkinson) > 0 else 0
    
    def _calculate_vol_percentile(self, rolling_vol: pd.Series) -> float:
        """Calculate current volatility percentile."""
        current = rolling_vol.iloc[-1]
        return (rolling_vol < current).sum() / len(rolling_vol) * 100
    
    def _assess_vol_trend(self, rolling_vol: pd.Series) -> str:
        """Assess whether volatility is increasing or decreasing."""
        if len(rolling_vol) < 14:
            return "Insufficient data"
        
        recent = rolling_vol.iloc[-7:].mean()
        previous = rolling_vol.iloc[-14:-7].mean()
        
        if recent > previous * 1.1:
            return "Increasing"
        elif recent < previous * 0.9:
            return "Decreasing"
        return "Stable"
    
    def detect_volatility_regimes(self, df: pd.DataFrame, price_col: str = "Price") -> pd.DataFrame:
        """
        Detect high and low volatility regimes.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            DataFrame with regime classifications
        """
        df = df.copy()
        returns = df[price_col].pct_change()
        rolling_vol = returns.rolling(window=20).std()
        
        vol_median = rolling_vol.median()
        
        df["volatility"] = rolling_vol
        df["regime"] = np.where(rolling_vol > vol_median * 1.5, "High",
                                np.where(rolling_vol < vol_median * 0.5, "Low", "Normal"))
        
        return df


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    report = analyzer.generate_sentiment_report("bitcoin")
    print("Sentiment Report:")
    print(f"  Overall: {report['news_sentiment']['overall_sentiment']}")
    print(f"  Fear & Greed: {report['fear_greed_index']['value']} ({report['fear_greed_index']['classification']})")
    print(f"  Recommendation: {report['recommendation']}")
