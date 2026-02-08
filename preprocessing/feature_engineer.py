import pandas as pd
import joblib
import ipaddress
from sklearn.preprocessing import LabelEncoder


def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))


def engineer(inp, out):

    df = pd.read_csv(inp)

    # Convert IP to number
    df["ip"] = df["ip"].apply(ip_to_int)

    # Encode protocol only
    proto_enc = LabelEncoder()
    df["protocol"] = proto_enc.fit_transform(df["protocol"])

    joblib.dump(proto_enc, "saved_models/proto.pkl")

    df.to_csv(out, index=False)

    print("Features saved")


if __name__ == "__main__":

    engineer(
        "data/processed/cleaned.csv",
        "data/processed/featured.csv"
    )
