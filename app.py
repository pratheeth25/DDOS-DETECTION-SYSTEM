import streamlit as st
import pandas as pd

import os
import subprocess
import sys


# ----------------------------------------
# Auto setup: preprocessing + training
# ----------------------------------------

def setup_models():

    # If main model does not exist → train everything
    if not os.path.exists("saved_models/xgb.pkl"):

        st.info("⚙️ First-time setup: Training models... Please wait.")

        # Create folder
        os.makedirs("saved_models", exist_ok=True)

        # --------------------
        # Run preprocessing
        # --------------------
        subprocess.run([sys.executable, "preprocessing/clean_data.py"], check=True)
        subprocess.run([sys.executable, "preprocessing/feature_engineer.py"], check=True)
        subprocess.run([sys.executable, "preprocessing/scaler.py"], check=True)

        # --------------------
        # Run training
        # --------------------
        subprocess.run([sys.executable, "-m", "training.train_xgboost"], check=True)
        subprocess.run([sys.executable, "-m", "training.train_iforest"], check=True)
        subprocess.run([sys.executable, "-m", "training.train_autoencoder"], check=True)

        st.success("✅ Model training completed!")


# ----------------------------------------
# IMPORTANT: Train before importing models
# ----------------------------------------

setup_models()

# Import AFTER setup
from inference.predictor import predict_ddos


# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(
    page_title="DDoS Detector",
    layout="centered"
)

st.title("🚨 DDoS Detection System")

st.write("Upload a network traffic CSV file for analysis")

file = st.file_uploader("Upload CSV File", type="csv")


if file is not None:

    try:
        # Load file
        df = pd.read_csv(file)

        st.subheader("📄 Input Data")
        st.dataframe(df)

        # Remove label if present
        X = df.drop("label", axis=1, errors="ignore")

        # Predict
        with st.spinner("🔍 Analyzing traffic..."):
            result = predict_ddos(X)

        # Show results
        df["Prediction"] = result

        st.subheader("📊 Detection Results")
        st.dataframe(df)

        st.success("✅ Detection Complete!")

    except Exception as e:

        st.error("❌ Error during processing")
        st.exception(e)


else:
    st.info("⬆️ Please upload a CSV file to start detection.")
