import streamlit as st
import pandas as pd

import os
import subprocess
import sys


def setup_models():

    if not os.path.exists("saved_models/xgb.pkl"):

        st.info("⚙️ Training models (first run)...")

        os.makedirs("saved_models", exist_ok=True)

        subprocess.run([sys.executable, "preprocessing/clean_data.py"])
        subprocess.run([sys.executable, "preprocessing/feature_engineer.py"])
        subprocess.run([sys.executable, "preprocessing/scaler.py"])

        subprocess.run([sys.executable, "-m", "training.train_xgboost"])
        subprocess.run([sys.executable, "-m", "training.train_iforest"])
        subprocess.run([sys.executable, "-m", "training.train_autoencoder"])

        st.success("✅ Training completed")


# TRAIN FIRST
setup_models()

# THEN import predictor
from inference.predictor import predict_ddos


st.set_page_config(page_title="DDoS Detector")

st.title("🚨 DDoS Detection System")

st.write("Upload network traffic CSV file")

file = st.file_uploader("Upload CSV", type="csv")


if file:

    df = pd.read_csv(file)

    st.subheader("Input Data")
    st.dataframe(df)

    X = df.drop("label", axis=1, errors="ignore")

    result = predict_ddos(X)

    df["Prediction"] = result

    st.subheader("Results")
    st.dataframe(df)

    st.success("Detection Complete ✅")
