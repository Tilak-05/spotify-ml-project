# src/lastfm_client.py

import requests
from src.config import LASTFM_API_KEY


class LastFMClient:
    """
    Handles communication with Last.fm API
    """

    BASE_URL = "http://ws.audioscrobbler.com/2.0/"

    def __init__(self):
        if not LASTFM_API_KEY:
            raise ValueError("Last.fm API key not found in .env file")

    def get_track_info(self, track_name: str, artist_name: str):
        params = {
            "method": "track.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist": artist_name,
            "track": track_name,
            "format": "json"
        }

        response = requests.get(self.BASE_URL, params=params)
        data = response.json()

        if "track" not in data:
            return None

        track = data["track"]
        artist = track["artist"]

        return {
            "track_name": track.get("name"),
            "artist_name": artist.get("name"),
            "listeners": int(track.get("listeners", 0)),
            "playcount": int(track.get("playcount", 0)),
            "artist_listeners": int(artist.get("stats", {}).get("listeners", 0)),
            "artist_playcount": int(artist.get("stats", {}).get("playcount", 0)),
        }
