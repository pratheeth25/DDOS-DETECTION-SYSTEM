import pandas as pd
from models.autoencoder import build


# Load data
df = pd.read_csv("data/processed/final.csv")

X = df.drop("label", axis=1).values


# Build model
model = build(X.shape[1])


# Train
model.fit(
    X, X,
    epochs=30,
    batch_size=16,
    verbose=1
)


# Save in NEW format (important)
model.save("saved_models/ae.keras")

print("Autoencoder trained and saved")
