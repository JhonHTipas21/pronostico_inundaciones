# app/services/train_service.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
import joblib

from app.services.feature_service import build_features


def _make_estimator(model_type: str):
    if model_type == "ridge":
        base = Ridge(random_state=0)
        grid = {"reg__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]}
    elif model_type == "lasso":
        base = Lasso(max_iter=50_000, random_state=0)
        grid = {"reg__alpha": [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}
    else:
        raise ValueError("model_type debe ser 'ridge' o 'lasso'")
    return base, grid


def train_from_df(df: pd.DataFrame, horizon: int, model_type: str, model_dir: str):
    dfm, feats_num, feats_cat = build_features(df, horizon=horizon)
    X = dfm[feats_num + feats_cat].copy()
    y = dfm["y_target"].values

    X_num = X[feats_num].to_numpy(dtype=float)
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("El vector objetivo y_target contiene NaN/Inf.")
    if np.isnan(X_num).any() or np.isinf(X_num).any():
        raise ValueError("Las features numéricas contienen NaN/Inf.")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feats_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feats_cat),
        ],
        remainder="drop"
    )

    base_est, grid = _make_estimator(model_type)

    pipe_core = Pipeline([
        ("pre", pre),
        ("reg", base_est),
    ])

    model = TransformedTargetRegressor(
        regressor=pipe_core,
        func=np.log1p,
        inverse_func=np.expm1
    )

    param_grid = {f"regressor__{k}": v for k, v in grid.items()}

    tscv = TimeSeriesSplit(n_splits=5)
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        refit="rmse",                 # ⬅️ selecciona por RMSE (mejor generalización)
        n_jobs=-1,
        verbose=0,
        error_score="raise"
    )
    gscv.fit(X, y)

    best = gscv.best_estimator_
    best_params = gscv.best_params_

    # Métricas CV con el mejor (refit por RMSE)
    r2s, rmses = [], []
    for tr, te in tscv.split(X):
        best.fit(X.iloc[tr], y[tr])
        yp = best.predict(X.iloc[te])
        r2s.append(float(r2_score(y[te], yp)))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], yp))))
    r2_cv = float(np.mean(r2s))
    rmse_cv = float(np.mean(rmses))

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    best.fit(X, y)
    joblib.dump(best, out / "modelo_regresion.pkl")

    meta = {
        "model_type": model_type,
        "horizon": int(horizon),
        "features_numeric": list(feats_num),
        "features_categorical": list(feats_cat),
        "best_params": best_params,
        "r2_cv_mean": r2_cv,
        "rmse_cv_mean": rmse_cv,
        "n_samples": int(len(dfm)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load_model(model_dir: str):
    p = Path(model_dir)
    model = joblib.load(p / "modelo_regresion.pkl")
    meta = json.loads((p / "meta.json").read_text())
    return model, meta
