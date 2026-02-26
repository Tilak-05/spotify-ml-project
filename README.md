
=======
#  AI Music Popularity Predictor

# Overview

## Project Overview

This project implements a complete end-to-end machine learning pipeline that predicts whether a song is popular or not using real-time music popularity metrics. Unlike traditional ML projects that rely on static datasets, this system fetches live data from the Last.fm API, preprocesses it, applies a trained classification model, and provides predictions through an interactive Streamlit web application.

The system demonstrates practical machine learning engineering skills including real-time data ingestion, preprocessing, model training, inference pipeline design, and deployment-ready interface development.

---

## Problem Statement

Music popularity prediction is important for streaming platforms, recommendation systems, and analytics. However, most academic ML projects rely on static datasets, which do not reflect real-time popularity.

This project solves that limitation by:

* Fetching real-time popularity metrics from an external API
* Processing the data into machine learning features
* Training a classification model
* Providing live predictions through a web interface

---

## System Architecture

Pipeline flow:

User Input (Song, Artist)
→ Last.fm API (Real-Time Data)
→ Preprocessing Pipeline
→ Trained Machine Learning Model
→ Prediction Output
→ Streamlit Web Interface

---

## Features Implemented

### Real-Time Data Ingestion

* Integrated Last.fm API using secure API authentication
* Fetches live popularity metrics including:

  * Track listeners
  * Track play count
  * Artist popularity indicators

### Data Preprocessing Pipeline

* Feature selection and transformation
* Missing value handling
* Feature scaling using StandardScaler
* Consistent preprocessing for both training and inference

### Machine Learning Model

* Model: Logistic Regression
* Task: Binary classification (Popular vs Not Popular)
* Popularity label defined based on play count threshold
* Model and scaler persisted using joblib for reuse

### Prediction Pipeline

* Loads trained model and scaler
* Fetches real-time data
* Applies preprocessing
* Generates prediction and confidence score

### Interactive Web Interface

* Built using Streamlit
* Allows user input of song and artist
* Displays:

  * Prediction result
  * Confidence score
  * Feature values used

### Software Engineering Practices

* Modular project structure
* Environment-based configuration
* Version control using Git and GitHub
* Separation of training and inference pipelines
=======

The system includes:

* Machine Learning model trained on **114,000+ Spotify tracks**
* Real-time prediction pipeline
* Modern **Spotify-style Streamlit web interface**
* Confidence score visualization using animated gauge meter
* Production-ready modular architecture

---

# Application Demo

## Main Interface

Features shown:

* Spotify-style UI
* Album artwork display
* Confidence gauge meter
* Popularity classification
* Model performance metrics

---

# Machine Learning Details

## Model Information

| Property     | Value                    |
| ------------ | ------------------------ |
| Model        | Random Forest Classifier |
| Dataset Size | 114,000 songs            |
| Task         | Binary Classification    |
| Framework    | Scikit-Learn             |

---

## Features Used

```
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
```

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 85.39% |
| Precision | 45.36% |
| Recall    | 60.56% |
| F1 Score  | 51.87% |

---

# System Architecture

```
User selects song from UI
        ↓
Dataset lookup and feature extraction
        ↓
Feature scaling (StandardScaler)
        ↓
Random Forest model prediction
        ↓
Confidence score calculation
        ↓
Interactive Streamlit visualization
```

---

# Features

## Machine Learning Pipeline

* Feature preprocessing
* Model training and persistence
* Prediction pipeline
* Confidence score generation

## Frontend Interface

* Spotify-style animated UI
* Album cover display with glow effect
* Animated confidence gauge
* Color-coded popularity indicator
* Compact professional layout

## Engineering Features

* Modular architecture
* Production-ready code
* Model persistence with joblib
* Deployment-ready Streamlit app

---

#  Project Structure

```
spotify-ml-project/
│
├── app.py
│
├── src/
│   ├── config.py
│   ├── lastfm_client.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
=======
├── data/
│   └── spotify_tracks.csv
│
├── models/
│   ├── popularity_model.joblib
│   ├── scaler.joblib
│   └── metrics.json
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── spotify_client.py
│
├── test_preprocess.py
├── test_predict.py
├── requirements.txt
├── README.md
└── .env
=======
├── requirements.txt
└── README.md
```

---

## Technologies Used

Programming Language:

* Python

Machine Learning:

* scikit-learn
* Logistic Regression
* StandardScaler

Data Processing:

* Pandas
* NumPy

API Integration:

* Last.fm API
* Requests

Web Application:

* Streamlit

Model Persistence:

* Joblib

Version Control:

* Git
* GitHub

---

## Installation and Setup

### Clone the repository

```
git clone https://github.com/Tilak-05/spotify-ml-project.git
cd spotify-ml-project
```

### Create virtual environment

=======
# Installation

## Clone Repository

```
git clone https://github.com/Tilak-05/spotify-ml-project.git
cd spotify-ml-project
```

---

## Create Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Configure API key

Create `.env` file:

```
LASTFM_API_KEY=your_api_key_here
```

---

## Running the Application

Launch the Streamlit interface:

```
streamlit run app.py
```

The application will open in the browser and allow real-time predictions.

---

## Model Training

To retrain the model:

```
python -m src.train
```

This will:

* Collect training samples
* Train the classification model
* Save model and scaler

---

## Testing

Run validation tests:

```
python test_preprocess.py
python test_predict.py
=======
# Run Application

```
streamlit run app.py
```

Open browser:

```
http://localhost:8501

```

These tests verify:

* API integration
* Preprocessing correctness
* Prediction pipeline functionality

---

## Example Prediction Output

```
Song: Blinding Lights
Artist: The Weeknd

Prediction: Popular
Confidence: 92.34%

Features:
listeners: 2275434
playcount: 37246021
```

---

## Engineering Highlights (Resume-Relevant)

This project demonstrates:

* Real-time ML pipeline development
* API-based feature engineering
* Training and inference separation
* Model persistence and reuse
* Production-style project structure
* Interactive ML application deployment
=======
# Train Model

```
python -m src.train
```

This generates:

```
models/popularity_model.joblib
models/scaler.joblib
models/metrics.json
```

---

# Example Prediction

The system successfully performs real-time popularity prediction using live data and is fully functional for demonstration and deployment.

---

## Future Improvements

* Collect larger training dataset
* Use advanced models such as Random Forest or XGBoost
* Add model evaluation metrics dashboard
* Deploy on Streamlit Cloud
* Add automated retraining pipeline

---


=======
Input:

```
Song: Chammak Challo
Artist: Vishal-Shekhar
```

Output:

```
Prediction: Medium Popularity
Confidence: 66.0%
```

---

# Technologies Used

## Machine Learning

* Python
* Scikit-Learn
* Random Forest

## Data Processing

* Pandas
* NumPy

## Frontend

* Streamlit
* Plotly
* Glass UI CSS

## Utilities

* Joblib
* RapidFuzz

---

# Engineering Highlights

This project demonstrates:

* End-to-end ML system design
* Training on large real-world dataset
* Model deployment via web interface
* Production-level code organization
* Professional UI/UX design

---

# Deployment Ready

Can be deployed on:

* Streamlit Cloud
* Render
* Railway
* Docker

---

