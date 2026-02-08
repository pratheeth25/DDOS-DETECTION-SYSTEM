from sklearn.ensemble import IsolationForest


def build():

    return IsolationForest(
        contamination=0.1,
        random_state=42
    )
