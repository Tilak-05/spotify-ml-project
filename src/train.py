# src/train.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.lastfm_client import LastFMClient
from src.preprocess import Preprocessor
from src.config import LASTFM_FEATURES

# ----------------------------
# CONFIG
# ----------------------------
POPULARITY_THRESHOLD = 10_000_000
MODEL_PATH = "models/popularity_model.joblib"
SCALER_PATH = "models/scaler.joblib"


def create_label(playcount: int) -> int:
    """
    Define popularity label based on playcount
    """
    return 1 if playcount >= POPULARITY_THRESHOLD else 0


def collect_training_data():
    """
    Collect sample data using Last.fm API
    (for demo / mini-project purpose)
    """
    client = LastFMClient()

    # Sample songs (can expand later)
    samples = [
        ("Blinding Lights", "The Weeknd"),
        ("Shape of You", "Ed Sheeran"),
        ("Believer", "Imagine Dragons"),
        ("Let Me Love You", "DJ Snake"),
        ("Cheap Thrills", "Sia"),
    ]

    rows = []

    for song, artist in samples:
        data = client.get_track_info(song, artist)
        if data:
            data["label"] = create_label(data["playcount"])
            rows.append(data)

    return pd.DataFrame(rows)


def train_model():
    print("📥 Collecting training data...")
    df = collect_training_data()

    print("\nTraining data:")
    print(df[LASTFM_FEATURES + ["label"]])

    X = df[LASTFM_FEATURES]
    y = df["label"]

    # Preprocessing
    preprocessor = Preprocessor()
    X_scaled = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model trained successfully")
    print(f"📊 Accuracy: {accuracy:.2f}")

    # Save model & scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor.scaler, SCALER_PATH)

    print("\n💾 Model and scaler saved")


if __name__ == "__main__":
    train_model()
