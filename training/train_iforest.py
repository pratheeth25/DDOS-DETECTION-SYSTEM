import pandas as pd
import joblib
from models.isolation_forest import build


df = pd.read_csv("data/processed/final.csv")

X = df.drop("label", axis=1)

model = build()
model.fit(X)

joblib.dump(model, "saved_models/iforest.pkl")

print("IsolationForest trained")
