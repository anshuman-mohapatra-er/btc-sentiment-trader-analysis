"""
model.py
Trains a simple ML classifier to predict BTC price direction
(Up / Down / Flat) based on sentiment scores.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os


def build_training_data():
    """
    Build a synthetic training dataset combining sentiment scores
    with price movement labels. Replace with real merged data when available.
    """
    np.random.seed(42)
    n = 200

    # Simulate realistic sentiment-price relationships
    data = []
    for _ in range(n):
        vader = np.random.uniform(-1, 1)
        tb_polarity = vader * 0.6 + np.random.uniform(-0.2, 0.2)
        tb_subjectivity = np.random.uniform(0, 1)

        # Sentiment influences movement with some noise
        score = vader * 0.7 + tb_polarity * 0.3 + np.random.uniform(-0.3, 0.3)
        if score > 0.15:
            movement = "Up"
        elif score < -0.15:
            movement = "Down"
        else:
            movement = "Flat"

        data.append({
            "vader_score": round(vader, 4),
            "tb_polarity": round(tb_polarity, 4),
            "tb_subjectivity": round(tb_subjectivity, 4),
            "movement": movement,
        })

    return pd.DataFrame(data)


def load_real_data():
    """Load merged sentiment+price data if available."""
    path = "data/merged_sentiment_price.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        required = ["vader_score", "tb_polarity", "tb_subjectivity", "movement"]
        if all(col in df.columns for col in required):
            print(f"    Loaded real merged data: {len(df)} rows")
            return df
    return None


def train_model(df):
    """Train and evaluate the sentiment-based price predictor."""
    features = ["vader_score", "tb_polarity", "tb_subjectivity"]
    X = df[features]
    y = df["movement"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Try both models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }

    best_model = None
    best_score = 0
    best_name = ""

    print("\n    Model Evaluation:")
    print("    " + "-" * 40)

    for name, clf in models.items():
        cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"    {name}")
        print(f"      CV Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        print(f"      Test Accuracy: {acc:.2%}")

        if acc > best_score:
            best_score = acc
            best_model = clf
            best_name = name

    print(f"\n    ✓ Best model: {best_name} ({best_score:.2%} accuracy)")

    # Full report for best model
    y_pred_best = best_model.predict(X_test)
    print("\n    Classification Report:")
    print(classification_report(
        y_test, y_pred_best,
        target_names=le.classes_,
        zero_division=0
    ))

    return best_model, le, features


def predict(model, label_encoder, features, vader_score, tb_polarity, tb_subjectivity):
    """Predict BTC movement for a single input."""
    X = pd.DataFrame([[vader_score, tb_polarity, tb_subjectivity]], columns=features)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = label_encoder.inverse_transform([pred])[0]
    confidence = max(proba) * 100
    return label, confidence


def save_model(model, label_encoder, features):
    """Save trained model to models/ directory."""
    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump({"model": model, "label_encoder": label_encoder, "features": features}, f)
    print("    Model saved → models/sentiment_model.pkl")


def load_saved_model():
    """Load previously trained model."""
    path = "models/sentiment_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["label_encoder"], data["features"]
    return None, None, None


if __name__ == "__main__":
    print("=" * 55)
    print("   BTC Price Direction Predictor (ML)")
    print("=" * 55)

    # Load real data if available, else use synthetic
    print("\n[1] Loading training data...")
    df = load_real_data()
    if df is None:
        print("    No real data found — using synthetic training data")
        df = build_training_data()
        print(f"    Generated {len(df)} synthetic samples")

    # Train
    print("\n[2] Training models...")
    model, le, features = train_model(df)

    # Save
    print("\n[3] Saving model...")
    save_model(model, le, features)

    # Demo predictions
    print("\n[4] Sample Predictions:")
    print("    " + "-" * 50)
    test_cases = [
        (0.87, 0.60, 0.4, "ETF approval sparks massive rally"),
        (-0.90, -0.70, 0.6, "Whale wallets dump, market fears sell-off"),
        (0.00, 0.05, 0.3, "BTC price stable, market consolidates"),
        (0.36, 0.30, 0.5, "Bitcoin hits all-time high"),
        (-0.54, -0.40, 0.7, "Crypto market crashes on regulation news"),
    ]

    for vader, tb_pol, tb_sub, headline in test_cases:
        label, conf = predict(model, le, features, vader, tb_pol, tb_sub)
        icon = "📈" if label == "Up" else "📉" if label == "Down" else "➡️"
        print(f"    {icon} [{label:4s}] ({conf:.0f}%) → {headline}")
