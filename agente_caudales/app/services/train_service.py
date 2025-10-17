from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from app.services.feature_service import build_features


def train_from_df(df: pd.DataFrame, horizon: int, model_type: str, model_dir: str):
    # Construcción de features y target
    dfm, feats = build_features(df, horizon=horizon)
    X, y = dfm[feats].values, dfm["y_target"].values

    # Modelo
    est = Lasso(alpha=0.01, max_iter=10000, random_state=0) if model_type == "lasso" else Ridge(alpha=1.0, random_state=0)
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", est)])

    # Validación temporal (walk-forward)
    tscv = TimeSeriesSplit(n_splits=5)
    r2s, rmses = [], []
    for tr, te in tscv.split(X):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])

        # Métricas robustas (RMSE sin usar el flag 'squared')
        r2s.append(float(r2_score(y[te], yp)))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], yp))))

    # Entrenamiento final con todo el set
    pipe.fit(X, y)

    # Persistencia
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "modelo_regresion.pkl")

    meta = {
        "model_type": model_type,
        "features": feats,
        "r2_cv_mean": float(np.mean(r2s)) if r2s else None,
        "rmse_cv_mean": float(np.mean(rmses)) if rmses else None,
        "n_samples": int(len(dfm)),
        "horizon": int(horizon),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load_model(model_dir: str):
    p = Path(model_dir)
    model = joblib.load(p / "modelo_regresion.pkl")
    meta = json.loads((p / "meta.json").read_text())
    return model, meta
