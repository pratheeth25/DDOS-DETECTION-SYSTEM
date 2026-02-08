import pandas as pd


def clean_data(inp, out):

    df = pd.read_csv(inp)

    df = df.dropna()
    df = df.drop_duplicates()

    df = df[df["requests_per_sec"] >= 0]
    df = df[df["packet_size"] > 0]

    df.to_csv(out, index=False)

    print("Cleaned data saved")


if __name__ == "__main__":

    clean_data(
        "data/raw/traffic_data.csv",
        "data/processed/cleaned.csv"
    )
