# 🪙 BTC Sentiment Trader Analysis

A Python-based project that analyzes Bitcoin news sentiment and correlates it with price movements using NLP and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Features

- **Crypto-enhanced VADER sentiment analysis** — custom lexicon with 30+ Bitcoin-specific terms
- **TextBlob dual scoring** — polarity + subjectivity for each headline
- **Live BTC price data** — fetched from CoinGecko API (no API key needed)
- **Sentiment + Price merge** — correlate news sentiment with daily price movement
- **ML Price Predictor** — classify BTC as Up/Down/Flat based on sentiment score
- **Dashboard chart** — auto-generated visual report saved to `data/`

---

## 📁 Project Structure

```
btc_sentiment_project/
├── main.py              # Entry point — runs full pipeline
├── sentiment.py         # VADER + TextBlob sentiment analysis
├── fetch_data.py        # BTC price from CoinGecko + sample headlines
├── model.py             # ML model — predicts price movement from sentiment
├── utils/
│   └── helpers.py       # Text cleaning utilities
├── notebooks/           # Jupyter notebooks for EDA
├── requirements.txt     # All dependencies
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/anshuman-mohapatra-er/btc-sentiment-trader-analysis.git
cd btc-sentiment-trader-analysis

# 2. Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download TextBlob corpora
python3 -m textblob.download_corpora lite
```

---

## 🚀 Run

```bash
python3 main.py
```

---

## 📊 Sample Output

| Date       | Sentiment | VADER Score | BTC Movement |
|------------|-----------|-------------|--------------|
| 2024-01-01 | Positive  | +0.36       | Up           |
| 2024-01-02 | Negative  | -0.54       | Down         |
| 2024-01-04 | Negative  | -0.90       | Down         |
| 2024-01-05 | Positive  | +0.87       | Up           |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| VADER Sentiment | Crypto-enhanced NLP scoring |
| TextBlob | Polarity + subjectivity analysis |
| scikit-learn | ML price direction classifier |
| pandas + numpy | Data processing |
| matplotlib | Charts and visualizations |
| CoinGecko API | Free live BTC price data |

---

## 📈 Roadmap

- [x] Sentiment analysis with crypto VADER lexicon
- [x] Live BTC price fetching
- [x] Sentiment + price merge
- [x] Dashboard chart generation
- [x] ML model for price direction prediction

---

## 👤 Author

**Anshuman Mohapatra**
- GitHub: [@anshuman-mohapatra-er](https://github.com/anshuman-mohapatra-er)

---

## 📄 License

MIT License — free to use and modify.
