import xgboost as xgb


def build():

    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
