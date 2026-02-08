import streamlit as st
import pandas as pd
from inference.predictor import predict_ddos

import os
import subprocess
import sys


def setup_models():

    if not os.path.exists("saved_models/xgb.pkl"):

        st.info("⚙️ Training models for first-time setup...")

        os.makedirs("saved_models", exist_ok=True)

        # Preprocessing
        subprocess.run([sys.executable, "preprocessing/clean_data.py"])
        subprocess.run([sys.executable, "preprocessing/feature_engineer.py"])
        subprocess.run([sys.executable, "preprocessing/scaler.py"])

        # Training
        subprocess.run([sys.executable, "-m", "training.train_xgboost"])
        subprocess.run([sys.executable, "-m", "training.train_iforest"])
        subprocess.run([sys.executable, "-m", "training.train_autoencoder"])

        st.success("✅ Model training complete!")


# Run setup
setup_models()


# UI
st.set_page_config(page_title="DDoS Detector")

st.title("🚨 DDoS Detection System")

st.write("Upload network traffic CSV file for analysis")

file = st.file_uploader("Upload CSV", type="csv")


if file:

    df = pd.read_csv(file)

    st.subheader("📄 Input Data")
    st.dataframe(df)

    # Remove label if present
    X = df.drop("label", axis=1, errors="ignore")

    # Predict
    result = predict_ddos(X)

    df["Prediction"] = result

    st.subheader("📊 Detection Results")
    st.dataframe(df)

    st.success("Detection Complete ✅")
