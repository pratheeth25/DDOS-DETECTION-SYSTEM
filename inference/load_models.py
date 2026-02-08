import joblib
from tensorflow import keras


# Load ML models
xgb = joblib.load("saved_models/xgb.pkl")
iforest = joblib.load("saved_models/iforest.pkl")

# Load Autoencoder (new format)
ae = keras.models.load_model("saved_models/ae.keras")

# Load scaler
scaler = joblib.load("saved_models/scaler.pkl")
