from src.lastfm_client import LastFMClient
from src.preprocess import Preprocessor

client = LastFMClient()
pre = Preprocessor()

data = client.get_track_info("Blinding Lights", "The Weeknd")

df = pre.to_dataframe(data)
df = pre.select_features(df)
df = pre.handle_missing(df)

print(df)
