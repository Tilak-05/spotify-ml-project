
---

# Real-Time Music Popularity Prediction System

## Overview

This project implements an end-to-end machine learning system that predicts whether a song is **popular or not** using **real-time music popularity data**.

Instead of relying on static datasets, the system fetches **live data from the Last.fm API**, processes it through a preprocessing pipeline, applies a trained machine learning model, and presents results through a **Streamlit-based web interface**.

The project demonstrates practical machine learning engineering concepts, including **data ingestion, preprocessing, model training, inference, and deployment readiness**.

---

## Key Features Implemented

* Real-time data ingestion using Last.fm API
* Secure API key management using environment variables
* Robust preprocessing pipeline for machine learning
* Binary popularity classification model
* Model persistence using joblib
* Streamlit-based interactive web application
* Clean and modular project structure
* Version-controlled development using Git and GitHub

---

## Project Structure

```
spotify-ml-project/
│
├── app.py                    # Streamlit web application
│
├── src/
│   ├── config.py             # Environment configuration and feature definitions
│   ├── lastfm_client.py      # Last.fm API integration
│   ├── preprocess.py         # Data preprocessing pipeline
│   ├── train.py              # Machine learning model training
│   └── predict.py            # Real-time prediction logic
│
├── models/
│   ├── popularity_model.joblib
│   └── scaler.joblib
│
├── test_preprocess.py        # Preprocessing test script
├── test_predict.py           # Prediction test script
├── requirements.txt
├── README.md
└── .env                      # Environment variables (ignored in Git)
```

---

## Data Source

### Last.fm API

The system uses the **Last.fm API** to fetch real-time popularity metrics, including:

* Track listener count
* Track play count
* Artist popularity indicators

Spotify Web API was initially planned, but due to temporary developer access restrictions, Last.fm was selected as a reliable alternative.

---

## Environment Setup

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\Activate.ps1  # Windows
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
LASTFM_API_KEY=your_lastfm_api_key_here
```

> Note: This file should not be committed to GitHub.

---

## Preprocessing Pipeline

The preprocessing module performs the following steps:

* Converts raw API responses into structured DataFrame format
* Selects relevant numeric features
* Handles missing or incomplete values safely
* Scales features for consistent model input

The same preprocessing logic is reused during both **training and prediction** to maintain consistency.

---

## Machine Learning Model

* **Model Type:** Logistic Regression
* **Task:** Binary Classification (Popular / Not Popular)

### Popularity Definition

A song is labeled as **Popular** if its play count exceeds a predefined threshold.

The trained model is persisted using `joblib` and reused during real-time inference.

---

## Running the Streamlit Application

To launch the web application locally:

```bash
streamlit run app.py
```

The application allows users to:

* Enter a song name and artist name
* View popularity prediction
* View confidence score
* Inspect feature values used for prediction

---

## Testing

Basic test scripts are included to validate:

* Real-time data fetching
* Preprocessing correctness
* Prediction pipeline integrity

Run the tests using:

```bash
python test_preprocess.py
python test_predict.py
```

---

## Version Control

All development has been tracked using **Git** and pushed to **GitHub** with clean, descriptive commit messages.

Sensitive files such as `.env`, virtual environments, and cache directories are excluded using `.gitignore`.

---

## Current Status

All planned objectives up to this phase have been successfully completed.

The project currently provides a **complete, deployable, real-time machine learning application**.

---
