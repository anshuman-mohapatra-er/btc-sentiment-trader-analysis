"""
main.py
Entry point for the BTC Sentiment Analysis project.
Runs sentiment analysis, fetches BTC price, merges data, and saves outputs.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for WSL/servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fetch_data import get_sample_headlines, fetch_btc_price
from sentiment import analyze_dataframe


def plot_results(sentiment_df, price_df):
    """Generate and save a combined sentiment + price chart."""
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("BTC Sentiment Analysis Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Chart 1: Sentiment Distribution (bar) ---
    ax1 = fig.add_subplot(gs[0, 0])
    counts = sentiment_df["vader_label"].value_counts()
    colors = {"Positive": "#4caf50", "Neutral": "#ff9800", "Negative": "#f44336"}
    bar_colors = [colors.get(x, "gray") for x in counts.index]
    counts.plot(kind="bar", ax=ax1, color=bar_colors, edgecolor="white")
    ax1.set_title("Sentiment Distribution")
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=0)

    # --- Chart 2: VADER Score per Headline ---
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors2 = [
        "#4caf50" if s > 0.05 else "#f44336" if s < -0.05 else "#ff9800"
        for s in sentiment_df["vader_score"]
    ]
    ax2.barh(range(len(sentiment_df)), sentiment_df["vader_score"], color=bar_colors2)
    ax2.set_yticks(range(len(sentiment_df)))
    ax2.set_yticklabels(
        [d.strftime("%b %d") if hasattr(d, "strftime") else str(d)
         for d in pd.to_datetime(sentiment_df["date"])],
        fontsize=8,
    )
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("VADER Score by Date")
    ax2.set_xlabel("Sentiment Score")

    # --- Chart 3: BTC Price (if available) ---
    ax3 = fig.add_subplot(gs[1, :])
    price_clean = price_df.dropna(subset=["price_change"])
    ax3.plot(
        pd.to_datetime(price_df["date"]),
        price_df["price"],
        color="#f7931a",
        linewidth=2,
        marker="o",
        markersize=5,
        label="BTC Price (USD)",
    )
    ax3.set_title("BTC Price Over Period")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price (USD)")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()

    plt.savefig("data/btc_sentiment_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Chart saved → data/btc_sentiment_dashboard.png")


def main():
    print("=" * 55)
    print("   BTC Sentiment Analysis")
    print("=" * 55)

    # Step 1: Headlines
    print("\n[1] Loading headlines...")
    df = get_sample_headlines()
    print(f"    Loaded {len(df)} headlines")

    # Step 2: Sentiment
    print("\n[2] Running sentiment analysis (crypto-enhanced VADER)...")
    df = analyze_dataframe(df, text_column="text")
    print(df[["date", "vader_label", "vader_score", "text"]].to_string(index=False))

    # Step 3: BTC Price
    print("\n[3] Fetching BTC price data...")
    price_df = fetch_btc_price(days=7)

    # Step 4: Merge sentiment + price
    print("\n[4] Merging sentiment with price data...")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    merged = pd.merge(df, price_df, on="date", how="inner")
    if not merged.empty:
        print(merged[["date", "vader_label", "vader_score", "price", "movement"]].to_string(index=False))
    else:
        print("    (Dates don't overlap — using separate outputs)")

    # Step 5: Save outputs
    print("\n[5] Saving outputs...")
    df.to_csv("data/sentiment_results.csv", index=False)
    price_df.to_csv("data/btc_price.csv", index=False)
    if not merged.empty:
        merged.to_csv("data/merged_sentiment_price.csv", index=False)
    plot_results(df, price_df)

    print("\n✓ Done! Files saved in data/")
    print("  → data/sentiment_results.csv")
    print("  → data/btc_price.csv")
    print("  → data/btc_sentiment_dashboard.png")


if __name__ == "__main__":
    main()
