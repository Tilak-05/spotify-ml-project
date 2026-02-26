import joblib
import pandas as pd
from rapidfuzz import fuzz
from src.spotify_client import SpotifyClient

MODEL_PATH = "models/popularity_model.joblib"
SCALER_PATH = "models/scaler.joblib"
DATASET_PATH = "data/spotify_tracks.csv"

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


class PopularityPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.client = SpotifyClient()

        # load dataset
        self.dataset = pd.read_csv(DATASET_PATH)

    from rapidfuzz import fuzz

    def find_song_features(self, song_name, artist_name):

        best_score = 0
        best_row = None

        for _, row in self.dataset.iterrows():

            dataset_song = str(row["track_name"]).lower()
            dataset_artist = str(row["artists"]).lower()

            song_score = fuzz.partial_ratio(song_name.lower(), dataset_song)
            artist_score = fuzz.partial_ratio(artist_name.lower(), dataset_artist)

            total_score = (song_score + artist_score) / 2

            if total_score > best_score:
                best_score = total_score
                best_row = row

        if best_score < 70:
            return None

        return best_row[FEATURE_COLUMNS]

    def predict_song(self, song_name, artist_name):

        # verify song exists on spotify
        track_info = self.client.search_track_full(song_name, artist_name)

        if not track_info:
            return {"error": "Song not found on Spotify"}

        features = self.find_song_features(track_info["name"], track_info["artist"])

        if features is None:
            return {"error": "Song not found in dataset"}

        df = pd.DataFrame([features])

        X_scaled = self.scaler.transform(df)

        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0][1]

        return {
        "song": track_info["name"],
        "artist": track_info["artist"],
        "image": track_info["image"],   # NEW LINE (album cover)
        "prediction": "Popular" if pred == 1 else "Not Popular",
        "confidence": round(prob * 100, 2),
        "features_used": features.to_dict()
    }