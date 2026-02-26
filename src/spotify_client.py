import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

class SpotifyClient:

    def __init__(self):

        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )

        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def search_track(self, song_name, artist_name):

        query = f"track:{song_name} artist:{artist_name}"

        results = self.sp.search(
            q=query,
            type="track",
            limit=1
        )

    # Check if results exist
        if "tracks" not in results or len(results["tracks"]["items"]) == 0:
            return None

        track = results["tracks"]["items"][0]

    # Safe extraction
        name = track.get("name", "")
        artist = track.get("artists", [{}])[0].get("name", "")
        popularity = track.get("popularity", 0)

        return {
        "name": os.name,
        "artist": artist,
        "popularity": popularity
        }

    def search_track_full(self, song_name, artist_name):

        query = f"track:{song_name} artist:{artist_name}"

        results = self.sp.search(q=query, type="track", limit=1)

        if not results["tracks"]["items"]:
            return None

        track = results["tracks"]["items"][0]

        return {
        "name": track.get("name"),
        "artist": track.get("artists")[0].get("name"),
        "image": track.get("album").get("images")[0].get("url")
        }