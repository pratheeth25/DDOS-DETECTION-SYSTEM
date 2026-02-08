import numpy as np
import joblib
import ipaddress

from inference.load_models import xgb, iforest, ae, scaler


# Load protocol encoder + label encoder
proto_encoder = joblib.load("saved_models/proto.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")


def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))


def preprocess_input(df):

    df = df.copy()

    # Convert IP to number (no crash now)
    df["ip"] = df["ip"].apply(ip_to_int)

    # Encode protocol
    df["protocol"] = proto_encoder.transform(df["protocol"])

    return df


def predict_ddos(df):

    df = preprocess_input(df)

    X = scaler.transform(df)

    xgb_num = xgb.predict(X)
    xgb_pred = label_encoder.inverse_transform(
        xgb_num.astype(int)
    )

    if_pred = iforest.predict(X)

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
