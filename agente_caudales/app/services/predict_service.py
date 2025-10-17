import pandas as pd
from app.services.feature_service import build_features

def make_predictions(model, meta, payload_df: pd.DataFrame, horizon: int):
    dfm, feats = build_features(payload_df, horizon=horizon)
    # validar features
    miss = [f for f in meta["features"] if f not in dfm.columns]
    if miss:
        raise ValueError(f"Faltan features en payload: {miss}")
    yhat = model.predict(dfm[meta["features"]].values)
    return yhat.tolist()
