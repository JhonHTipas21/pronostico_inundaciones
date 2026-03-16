# train_per_station.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

from app.config import settings
from app.services.feature_service import build_features

MODEL_ROOT = Path(settings.MODEL_DIR) / "per_estacion"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

def make_estimator(model_type="ridge"):
    if model_type == "ridge":
        base = Ridge(random_state=0)
        grid = {"reg__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]}
    else:
        base = Lasso(max_iter=50_000, random_state=0)
        grid = {"reg__alpha": [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}
    return base, grid

def train_one(df_est, horizon, model_type="ridge"):
    dfe, feats_num, feats_cat = build_features(df_est, horizon=horizon)
    X = dfe[feats_num + feats_cat].copy()
    y = dfe["y_target"].values

    pre = ColumnTransformer([
        ("num", StandardScaler(), feats_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feats_cat)
    ])

    base, grid = make_estimator(model_type)
    pipe = Pipeline([("pre", pre), ("reg", base)])
    model = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    param_grid = {f"regressor__{k}": v for k, v in grid.items()}

    tscv = TimeSeriesSplit(n_splits=5)
    gscv = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        refit="rmse",
        n_jobs=-1,
        verbose=0
    )
    gscv.fit(X, y)

    best = gscv.best_estimator_

    r2s, rmses = [], []
    for tr, te in tscv.split(X):
        best.fit(X.iloc[tr], y[tr])
        yp = best.predict(X.iloc[te])
        r2s.append(r2_score(y[te], yp))
        rmses.append(np.sqrt(mean_squared_error(y[te], yp)))

    return best, feats_num, feats_cat, float(np.mean(r2s)), float(np.mean(rmses)), gscv.best_params_

if __name__ == "__main__":
    df = pd.read_csv(settings.DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    for est, sub in df.groupby("estacion"):
        model, fn, fc, r2cv, rmsecv, params = train_one(sub, settings.HORIZON, model_type="ridge")
        est_dir = MODEL_ROOT / est.replace(" ", "_").replace("/", "_")
        est_dir.mkdir(exist_ok=True, parents=True)
        joblib.dump(model, est_dir / "modelo.pkl")
        (est_dir / "meta.json").write_text(json.dumps({
            "estacion": est,
            "features_numeric": fn,
            "features_categorical": fc,
            "r2_cv_mean": r2cv,
            "rmse_cv_mean": rmsecv,
            "best_params": params
        }, indent=2, ensure_ascii=False))
        print(f"[{est}] R2_cv={r2cv:.3f} RMSE_cv={rmsecv:.3f}")
