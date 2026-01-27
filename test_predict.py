from src.predict import PopularityPredictor

predictor = PopularityPredictor()

result = predictor.predict_song("Blinding Lights", "The Weeknd")
print(result)
