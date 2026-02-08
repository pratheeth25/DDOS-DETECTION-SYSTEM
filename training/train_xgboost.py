import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

from models.xgboost_model import build


# Load data
df = pd.read_csv("data/processed/final.csv")

X = df.drop("label", axis=1)
y = df["label"]


# Encode labels (Bot, DDoS, Legit -> 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save encoder
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")


# Train model
model = build()
model.fit(X, y_encoded)


# Save model
joblib.dump(model, "saved_models/xgb.pkl")

print("XGBoost trained successfully")
