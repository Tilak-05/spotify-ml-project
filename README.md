
# AI Music Popularity Predictor

# Overview

## Project Overview

This project implements a complete end-to-end machine learning pipeline that predicts whether a song is popular or not using real-time music popularity metrics. The system uses a trained classification model on a Spotify tracks dataset, preprocesses it, and provides predictions through an interactive Streamlit web application.

The system demonstrates practical machine learning engineering skills including data ingestion, preprocessing, model training, inference pipeline design, and deployment-ready interface development.

---

## Problem Statement

Music popularity prediction is important for streaming platforms, recommendation systems, and analytics. However, most academic ML projects rely on static datasets, which do not reflect real-world prediction systems.

This project solves that limitation by:

* Processing music data into machine learning features
* Training a classification model
* Providing live predictions through a web interface
* Deploying a production-ready machine learning application

---

## System Architecture

Pipeline flow:

User selects song from UI  
→ Dataset lookup and feature extraction  
→ Preprocessing Pipeline  
→ Trained Machine Learning Model  
→ Prediction Output  
→ Streamlit Web Interface  

---

## Features Implemented

### Machine Learning Pipeline

* Feature preprocessing
* Model training and persistence
* Prediction pipeline
* Confidence score generation

### Data Preprocessing Pipeline

* Feature selection and transformation
* Missing value handling
* Feature scaling using StandardScaler
* Consistent preprocessing for both training and inference

### Machine Learning Model

* Model: Random Forest Classifier
* Task: Binary classification (Popular vs Not Popular)
* Model persisted using joblib for reuse

### Interactive Web Interface

* Built using Streamlit
* Allows user input of song and artist
* Displays:

  * Prediction result
  * Confidence score
  * Model performance metrics

### Software Engineering Practices

* Modular project structure
* Version control using Git and GitHub
* Separation of training and inference pipelines

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

# Project Structure

```

spotify-ml-project/
│
├── app.py
│
├── data/
│   └── spotify_tracks.csv
│
├── models/
│   └── metrics.json
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── spotify_client.py
│
├── requirements.txt
└── README.md

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

# Installation

## Clone Repository

```

git clone [https://github.com/Tilak-05/spotify-ml-project.git](https://github.com/Tilak-05/spotify-ml-project.git)
cd spotify-ml-project

```

---

## Create Virtual Environment

```

python -m venv .venv
.venv\Scripts\activate

```

---

## Install Dependencies

```

pip install -r requirements.txt

```

---

# Run Application

```

streamlit run app.py

```

Open browser:

```

[http://localhost:8501](http://localhost:8501)

```

---

# Train Model

```

python -m src.train

```

---

# Example Prediction

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

# Future Improvements

* Use advanced models such as XGBoost
* Add automated retraining pipeline
* Deploy on cloud infrastructure
```

---

This version is now:

* Clean
* Professional
* Conflict-free
* Resume-ready
* GitHub-ready
