# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import LASTFM_FEATURES


class Preprocessor:
    """
    Preprocess Last.fm data for ML models
    """

    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()

    def to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """
        Convert Last.fm dictionary to DataFrame
        """
        return pd.DataFrame([raw_data])

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only Last.fm numeric features
        """
        return df[LASTFM_FEATURES]

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values safely
        """
        return df.fillna(0)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Used during training
        """
        return self.scaler.fit_transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Used during prediction
        """
        return self.scaler.transform(df)

    def preprocess_for_prediction(self, raw_data: dict) -> np.ndarray:
        """
        Complete preprocessing pipeline for prediction
        """
        df = self.to_dataframe(raw_data)
        df = self.select_features(df)
        df = self.handle_missing(df)
        return self.transform(df)
