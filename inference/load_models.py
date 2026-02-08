import joblib
from tensorflow import keras
import os


def load_all_models():

    base = "saved_models"

    xgb = joblib.load(os.path.join(base, "xgb.pkl"))
    iforest = joblib.load(os.path.join(base, "iforest.pkl"))
    ae = keras.models.load_model(os.path.join(base, "ae.keras"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))

    return xgb, iforest, ae, scaler
