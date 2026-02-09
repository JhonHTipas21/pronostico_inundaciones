# app/services/predict_service.py
import pandas as pd
from app.services.feature_service import build_features

def make_predictions(model, meta, payload_df: pd.DataFrame, horizon: int):
    dfm, feats_num, feats_cat = build_features(payload_df, horizon=horizon)
    feats = meta["features_numeric"] + meta["features_categorical"]

    missing = [f for f in feats if f not in (dfm.columns)]
    if missing:
        raise ValueError(f"Faltan features en payload: {missing}")

    yhat = model.predict(dfm[feats])
    return yhat.tolist()
