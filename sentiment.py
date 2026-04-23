"""
sentiment.py
Performs sentiment analysis using VADER (with crypto lexicon) and TextBlob.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

# --- Custom crypto-aware lexicon for VADER ---
CRYPTO_LEXICON = {
    # Positive signals
    "bullish": 3.0,
    "bull": 2.5,
    "rally": 2.5,
    "surge": 2.5,
    "soar": 2.5,
    "moon": 2.0,
    "mooning": 2.5,
    "breakout": 2.0,
    "halving": 1.5,
    "adoption": 2.0,
    "accumulate": 1.5,
    "hodl": 1.5,
    "all-time high": 3.0,
    "ath": 2.5,
    "etf": 1.5,
    "approval": 2.0,
    "institutional": 1.5,
    "mainstream": 1.5,
    # Negative signals
    "bearish": -3.0,
    "bear": -2.5,
    "crash": -3.0,
    "dump": -2.5,
    "dumping": -2.5,
    "sell-off": -2.5,
    "selloff": -2.5,
    "crackdown": -2.5,
    "ban": -2.5,
    "hack": -2.5,
    "hacked": -3.0,
    "scam": -3.0,
    "fraud": -3.0,
    "fear": -2.0,
    "panic": -2.5,
    "whale": -1.0,
    "manipulation": -2.0,
    "liquidation": -2.0,
    "plunge": -2.5,
    "plunges": -2.5,
    "drop": -1.5,
    "drops": -1.5,
}

vader = SentimentIntensityAnalyzer()
vader.lexicon.update(CRYPTO_LEXICON)


def get_vader_sentiment(text):
    """Returns compound sentiment score (-1 to 1) using VADER + crypto lexicon."""
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return score, "Positive"
    elif score <= -0.05:
        return score, "Negative"
    else:
        return score, "Neutral"


def get_textblob_sentiment(text):
    """Returns polarity (-1 to 1) and subjectivity (0 to 1) using TextBlob."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


def analyze_dataframe(df, text_column="text"):
    """Adds sentiment columns to a DataFrame."""
    df[["vader_score", "vader_label"]] = df[text_column].apply(
        lambda x: pd.Series(get_vader_sentiment(x))
    )
    df[["tb_polarity", "tb_subjectivity"]] = df[text_column].apply(
        lambda x: pd.Series(get_textblob_sentiment(x))
    )
    return df


if __name__ == "__main__":
    test_texts = [
        "Bitcoin hits new all-time high as institutional investors pour in",
        "Crypto market crashes as regulators announce crackdown",
        "Bitcoin adoption grows in emerging markets",
        "Whale wallets dump large BTC holdings, market fears sell-off",
        "Bitcoin ETF approval sparks massive rally",
    ]
    print("Testing crypto-enhanced VADER:\n")
    for text in test_texts:
        score, label = get_vader_sentiment(text)
        print(f"  [{label:8s}] ({score:+.2f}) → {text}")
