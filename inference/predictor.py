import numpy as np
import joblib
import ipaddress
import os

from inference.load_models import load_all_models


# Global cache (loaded only once)
_models = None
_encoders = None


def load_resources():
    """
    Load models and encoders only after they exist
    """
    global _models, _encoders

    if _models is None:

        # Load ML models
        _models = load_all_models()

    if _encoders is None:

        proto = joblib.load("saved_models/proto.pkl")
        label = joblib.load("saved_models/label_encoder.pkl")

        _encoders = (proto, label)

    return _models, _encoders


def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))


def preprocess_input(df):

    df = df.copy()

    (xgb, iforest, ae, scaler), (proto_enc, _) = load_resources()

    # Convert IP
    df["ip"] = df["ip"].apply(ip_to_int)

    # Encode protocol
    df["protocol"] = proto_enc.transform(df["protocol"])

    return df


def predict_ddos(df):

    # Load everything safely
    (xgb, iforest, ae, scaler), (proto_enc, label_enc) = load_resources()

    # Preprocess
    df = preprocess_input(df)

    # Scale
    X = scaler.transform(df)

    # XGBoost
    xgb_num = xgb.predict(X)
    xgb_pred = label_enc.inverse_transform(
        xgb_num.astype(int)
    )

    # Isolation Forest
    if_pred = iforest.predict(X)

    # Autoencoder
    ae_pred = ae.predict(X)
    error = np.mean((X - ae_pred) ** 2, axis=1)

    final = []

    for i in range(len(X)):

        if xgb_pred[i] == "DDoS" or error[i] > 0.05:
            final.append("DDoS")

        elif if_pred[i] == -1:
            final.append("Bot")

        else:
            final.append("Legit")

    return final
