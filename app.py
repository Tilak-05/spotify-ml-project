import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from src.predict import PopularityPredictor


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Music Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)


# =========================
# SPOTIFY ANIMATED UI CSS
# =========================

st.markdown("""
<style>

/* Animated background */
.stApp {
    background: linear-gradient(-45deg, #020617, #0f172a, #020617, #1DB95420);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* Center layout */
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

/* Title */
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1DB954;
}

/* Album glow */
.album-glow img {
    border-radius:15px;
    box-shadow: 0 0 30px rgba(29,185,84,0.6);
    transition: 0.3s;
}

.album-glow img:hover {
    box-shadow: 0 0 60px rgba(29,185,84,1);
}

/* Confidence badges */
.high {
    background:#1DB954;
    padding:10px;
    border-radius:8px;
    text-align:center;
    font-weight:bold;
}

.medium {
    background:#f59e0b;
    padding:10px;
    border-radius:8px;
    text-align:center;
    font-weight:bold;
}

.low {
    background:#ef4444;
    padding:10px;
    border-radius:8px;
    text-align:center;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================

dataset = pd.read_csv("data/spotify_tracks.csv")

# Support both column formats
if "artists" in dataset.columns:
    dataset["label"] = dataset["track_name"] + " — " + dataset["artists"]
else:
    dataset["label"] = dataset["track_name"] + " — " + dataset["artist_name"]


# =========================
# LOAD METRICS
# =========================

with open("models/metrics.json") as f:
    metrics = json.load(f)


# =========================
# HEADER
# =========================

st.markdown('<div class="title">AI Music Popularity Predictor</div>', unsafe_allow_html=True)

st.write("")


# =========================
# SEARCH CARD
# =========================

st.markdown('<div class="glass">', unsafe_allow_html=True)

selected = st.selectbox(
    "Select Song",
    dataset["label"].unique()
)

predict = st.button("Predict Popularity", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# RESULT CARD
# =========================

if predict:

    song = selected.split(" — ")[0]
    artist = selected.split(" — ")[1]

    predictor = PopularityPredictor()
    result = predictor.predict_song(song, artist)

    st.write("")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Layout: album left, gauge right
    col1, col2 = st.columns([1,1])

    with col1:

        if result.get("image"):

            st.markdown('<div class="album-glow">', unsafe_allow_html=True)
            st.image(result["image"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"### {result['song']}")
        st.write(result["artist"])


        # Confidence badge
        confidence = result.get("confidence", 0)

        if confidence >= 70:
            badge_class = "high"
            badge_text = "HIGH POPULARITY"

        elif confidence >= 40:
            badge_class = "medium"
            badge_text = "MEDIUM POPULARITY"

        else:
            badge_class = "low"
            badge_text = "LOW POPULARITY"


        st.markdown(
            f'<div class="{badge_class}">{badge_text} ({confidence:.1f}%)</div>',
            unsafe_allow_html=True
        )


    with col2:

        # Gauge color logic
        if confidence >= 70:
            gauge_color = "#1DB954"

        elif confidence >= 40:
            gauge_color = "#f59e0b"

        else:
            gauge_color = "#ef4444"


        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text":"Confidence"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gauge_color}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# METRICS CARD
# =========================

st.write("")

st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
col2.metric("Precision", f"{metrics['precision']*100:.1f}%")
col3.metric("Recall", f"{metrics['recall']*100:.1f}%")

f1_value = metrics.get("f1_score", metrics.get("f1", 0))

col4.metric("F1 Score", f"{f1_value*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)