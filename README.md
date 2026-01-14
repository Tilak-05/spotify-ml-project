# Spotify ML Project

This project analyzes Spotify music data using Machine Learning.

🎵 Real-Time Music Popularity Prediction System

A real-time machine learning project that predicts music popularity using live data from the Last.fm API, instead of static datasets.
The system is designed with an industry-standard pipeline and is deployable using Streamlit.

📌 Project Overview

Thousands of songs are released daily, making it difficult to identify which tracks are popular or trending.
This project solves the problem by fetching real-time music popularity metrics and preparing them for machine learning-based analysis.

Instead of relying on outdated CSV files, the system uses live API data, making it suitable for real-world deployment.

🚀 Features Implemented (Current Stage)

✔ Real-time data ingestion using Last.fm API
✔ Secure API key management using .env
✔ Robust preprocessing pipeline for ML readiness
✔ Clean project structure with modular design
✔ Professional Git workflow

🏗️ Project Structure
spotify-ml-project/
│
├── src/
│   ├── config.py            # Environment config & feature definitions
│   ├── lastfm_client.py     # Last.fm real-time API integration
│   └── preprocess.py        # Data preprocessing pipeline
│
├── test_preprocess.py       # Preprocessing test script
│
├── .env                     # API keys (ignored in Git)
├── requirements.txt
├── README.md

🔑 API Used
Last.fm API

Provides real-time popularity metrics

Easy authentication (API key only)

Used for:

Track listeners

Track playcount

Artist popularity indicators

Spotify Web API was initially planned, but due to temporary developer access restrictions, Last.fm was used as a reliable alternative.

🔐 Environment Setup
1️⃣ Create .env file (project root)
LASTFM_API_KEY=your_lastfm_api_key_here


⚠️ Do not commit this file to GitHub.

2️⃣ Install dependencies
pip install -r requirements.txt

🧪 Testing Real-Time Data

Run the test script:

python test_preprocess.py

Sample Output
{
 'track_name': 'Blinding Lights',
 'artist_name': 'The Weeknd',
 'listeners': 2275434,
 'playcount': 37246021,
 'artist_listeners': 0,
 'artist_playcount': 0
}


This confirms successful real-time data fetching and preprocessing.

🧠 Preprocessing Pipeline

The preprocessing module performs:

Conversion of API response to DataFrame

Selection of relevant numeric features

Handling missing or incomplete values

Preparation of ML-ready inputs

This design supports both training and real-time prediction workflows.

📊 Technologies Used

Python 3.x

Last.fm Web API

Pandas & NumPy

Scikit-learn

Git & GitHub

Streamlit (planned)