# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# Features we will use for ML
LASTFM_FEATURES = [
    "listeners",
    "playcount",
    "artist_listeners",
    "artist_playcount"
]
