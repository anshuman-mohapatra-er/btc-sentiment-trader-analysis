"""
fetch_data.py
Fetches BTC price data from CoinGecko (free, no API key needed)
and sample headlines for sentiment analysis.
"""

import requests
import pandas as pd


def fetch_btc_price(days=30):
    """Fetch BTC price history from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.drop(columns=["timestamp"])
        df["price_change"] = df["price"].pct_change() * 100  # daily % change
        df["movement"] = df["price_change"].apply(label_price_movement)
        print(f"    Fetched {len(df)} days of BTC price data from CoinGecko")
        return df
    except Exception as e:
        print(f"    [Warning] Could not fetch live price data: {e}")
        print("    Using offline sample price data instead.")
        return get_sample_prices()


def get_sample_prices():
    """Offline fallback BTC price data."""
    data = {
        "date": ["2024-01-01","2024-01-02","2024-01-03",
                 "2024-01-04","2024-01-05","2024-01-06","2024-01-07"],
        "price": [42000, 40500, 43200, 41800, 45000, 45300, 43100],
    }
    df = pd.DataFrame(data)
    df["price_change"] = df["price"].pct_change() * 100
    df["movement"] = df["price_change"].apply(label_price_movement)
    return df


def label_price_movement(pct_change):
    """Label daily price change as Up/Down/Flat."""
    if pd.isna(pct_change):
        return "Flat"
    if pct_change > 1.0:
        return "Up"
    elif pct_change < -1.0:
        return "Down"
    else:
        return "Flat"


def get_sample_headlines():
    """Sample BTC headlines for testing (replace with real API data)."""
    headlines = [
        {"date": "2024-01-01", "text": "Bitcoin hits new all-time high as institutional investors pour in"},
        {"date": "2024-01-02", "text": "Crypto market crashes as regulators announce crackdown"},
        {"date": "2024-01-03", "text": "Bitcoin adoption grows in emerging markets"},
        {"date": "2024-01-04", "text": "Whale wallets dump large BTC holdings, market fears sell-off"},
        {"date": "2024-01-05", "text": "Bitcoin ETF approval sparks massive rally"},
        {"date": "2024-01-06", "text": "BTC price stable as market consolidates gains"},
        {"date": "2024-01-07", "text": "Hackers target crypto exchange, Bitcoin drops sharply"},
    ]
    return pd.DataFrame(headlines)


if __name__ == "__main__":
    print("Fetching BTC price data...")
    df = fetch_btc_price(days=7)
    print(df.to_string(index=False))
