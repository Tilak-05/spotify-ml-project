# src/predict.py

import joblib
import numpy as np

from src.lastfm_client import LastFMClient
from src.preprocess import Preprocessor
from src.config import LASTFM_FEATURES

MODEL_PATH = "models/popularity_model.joblib"
SCALER_PATH = "models/scaler.joblib"


class PopularityPredictor:
    """
    Handles real-time popularity prediction
    """

    def __init__(self):
        # Load trained model and scaler
        self.model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        self.preprocessor = Preprocessor(scaler=scaler)
        self.client = LastFMClient()

    def predict_song(self, song_name: str, artist_name: str) -> dict:
        """
        Predict popularity for a given song
        """
        # Fetch real-time data
        data = self.client.get_track_info(song_name, artist_name)
        if not data:
            return {"error": "Song not found"}

        # Preprocess
        X = self.preprocessor.preprocess_for_prediction(data)

        # Prediction
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])

        return {
            "song": data["track_name"],
            "artist": data["artist_name"],
            "prediction": "Popular" if prediction == 1 else "Not Popular",
            "confidence": round(probability * 100, 2),
            "features_used": {
                feature: data.get(feature, 0)
                for feature in LASTFM_FEATURES
            }
        }
