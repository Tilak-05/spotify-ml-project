# app.py

import streamlit as st
import pandas as pd
from src.predict import PopularityPredictor

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Music Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🎵 Music Popularity Predictor</h1>
    <p style='text-align: center;'>
    Predict song popularity using <b>real-time Last.fm data</b> and 
    <b>Machine Learning</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Input Section
# -------------------------
st.subheader("🎧 Enter Song Details")

col1, col2 = st.columns(2)

with col1:
    song_name = st.text_input(
        "Song Name",
        placeholder="e.g. Blinding Lights"
    )

with col2:
    artist_name = st.text_input(
        "Artist Name",
        placeholder="e.g. The Weeknd"
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🚀 Predict Popularity", use_container_width=True)

# -------------------------
# Prediction Section
# -------------------------
if predict_button:
    if not song_name or not artist_name:
        st.warning("⚠️ Please enter both song name and artist name.")
    else:
        with st.spinner("🔍 Fetching real-time data & predicting..."):
            predictor = PopularityPredictor()
            result = predictor.predict_song(song_name, artist_name)

        if "error" in result:
            st.error(result["error"])
        else:
            st.divider()
            st.subheader("📊 Prediction Result")

            # Result summary
            colA, colB = st.columns(2)

            with colA:
                if result["prediction"] == "Popular":
                    st.success("🔥 **POPULAR SONG**")
                else:
                    st.info("❄️ **NOT POPULAR**")

            with colB:
                st.metric(
                    label="Confidence Level",
                    value=f"{result['confidence']} %"
                )

            # -------------------------
            # Feature Table
            # -------------------------
            st.subheader("📈 Popularity Metrics Used")

            feature_df = pd.DataFrame(
                list(result["features_used"].items()),
                columns=["Feature", "Value"]
            )

            st.table(feature_df)

            st.caption(
                "These real-time metrics were used by the machine learning model "
                "to predict song popularity."
            )

st.divider()

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <p style='text-align: center; font-size: 14px;'>
    Built with ❤️ using <b>Last.fm API</b>, <b>Machine Learning</b>, and <b>Streamlit</b><br>
    Academic Project | Deployable Web App
    </p>
    """,
    unsafe_allow_html=True
)
