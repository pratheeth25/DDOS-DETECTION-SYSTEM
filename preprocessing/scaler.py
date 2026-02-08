import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def scale(inp, out):

    df = pd.read_csv(inp)

    y = df["label"]
    X = df.drop("label", axis=1)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    joblib.dump(sc, "saved_models/scaler.pkl")

    final = pd.DataFrame(Xs, columns=X.columns)
    final["label"] = y

    final.to_csv(out, index=False)

    print("Final data saved")


if __name__ == "__main__":

    scale(
        "data/processed/featured.csv",
        "data/processed/final.csv"
    )
