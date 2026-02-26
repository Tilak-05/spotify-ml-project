# src/train.py

import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Paths
DATA_PATH = "data/spotify_tracks.csv"
MODEL_PATH = "models/popularity_model.joblib"
SCALER_PATH = "models/scaler.joblib"
METRICS_PATH = "models/metrics.json"


# Features used for ML
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


# Popularity threshold
POPULARITY_THRESHOLD = 60


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset size: {len(df)} samples")

    return df


def prepare_data(df):

    print("Preparing data...")

    X = df[FEATURE_COLUMNS]

    # create label
    y = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_model():

    df = load_dataset()

    X, y, scaler = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training model...")

    model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
       zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nModel Performance:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Save metrics
    metrics = {

        "model": "RandomForestClassifier",

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "training_samples": int(len(df)),

        "features": FEATURE_COLUMNS,

        "timestamp": datetime.now().isoformat()
    }

    with open(METRICS_PATH, "w") as f:

        json.dump(metrics, f, indent=4)

    print("\nModel, scaler, and metrics saved successfully.")


if __name__ == "__main__":
    train_model()