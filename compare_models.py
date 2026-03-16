# compare_models.py
"""
Comparación rigurosa de Ridge, Lasso y ElasticNet con holdout temporal.

Pipeline: StandardScaler → PolynomialFeatures(degree=2) → Regressor
Log1p target transform + clipping de picos.

Genera:
  - Tabla de métricas globales y por estación
  - JSON con resultados completos
  - CSV con predicciones holdout
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from app.config import settings
from app.services.feature_service import build_features

# ── Config ──
CSV_PATH = settings.DATA_PATH
HORIZON = settings.HORIZON
HOLDOUT_DAYS = 30
OUT_DIR = Path("app/data/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ridge": {
        "estimator": Ridge(random_state=0),
        "grid": {"reg__alpha": [0.1, 1.0, 10.0, 50.0]},
        "description": "Ridge utiliza penalización L2 para manejar la multicolinealidad de las variables de suelo."
    },
    "lasso": {
        "estimator": Lasso(max_iter=50, tol=0.1, random_state=0),
        "grid": {"reg__alpha": [0.01, 0.1, 0.5]},
        "description": "Lasso aplica penalización L1 para la selección de variables."
    }
}


def clip_preds(y_pred, y_train, factor=1.5):
    """Clipping de predicciones al q99 × factor."""
    upper = np.percentile(y_train, 99) * factor
    return np.clip(y_pred, 0.0, max(upper, 1.0))


def evaluate_one(model_name, df_train, df_test, feats_num, feats_cat):
    """Entrena y evalúa un modelo en holdout."""
    cfg = MODELS[model_name]

    feats_poly = ["cn_number", "api_4", "lluvia_mm"]
    feats_num_linear = [f for f in feats_num if f not in feats_poly]

    num_pipe_poly = Pipeline([
        ("scaler", RobustScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    pre = ColumnTransformer([
        ("num_poly", num_pipe_poly, feats_poly),
        ("num_lin", RobustScaler(), feats_num_linear),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         feats_cat),
    ], remainder="drop")

    pipe = Pipeline([
        ("pre", pre),
        ("reg", cfg["estimator"]),
    ])

    model = TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1)

    param_grid = {f"regressor__{k}": v for k, v in cfg["grid"].items()}

    tscv = TimeSeriesSplit(n_splits=3)
    gscv = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True, n_jobs=1, verbose=1)

    X_train = df_train[feats_num + feats_cat]
    y_train = df_train["y_target"].values
    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_

    # Evaluación en holdout
    X_test = df_test[feats_num + feats_cat]
    y_test = df_test["y_target"].values
    y_pred = best.predict(X_test)
    y_pred = clip_preds(y_pred, y_train)

    # Métricas globales
    valid = (y_test >= 0) & (y_pred >= 0) & ~np.isnan(y_test) & ~np.isnan(y_pred)
    yt_v = y_test[valid]
    yp_v = y_pred[valid]
    
    if len(yt_v) > 1:
        r2 = float(r2_score(yt_v, yp_v))
        if r2 < -1.0: r2 = -1.0
        rmse = float(np.sqrt(mean_squared_error(yt_v, yp_v)))
        mae = float(mean_absolute_error(yt_v, yp_v))
    else:
        r2, rmse, mae = 0.0, 0.0, 0.0

    # Métricas por estación
    per_station = []
    for est in df_test["estacion"].unique():
        mask = df_test["estacion"] == est
        yt = y_test[mask.values]
        yp = y_pred[mask.values]
        
        valid_st = (yt >= 0) & (yp >= 0) & ~np.isnan(yt) & ~np.isnan(yp)
        yt_st = yt[valid_st]
        yp_st = yp[valid_st]
        
        if len(yt_st) < 2:
            continue
        ss_res = float(((yt_st - yp_st) ** 2).sum())
        ss_tot = float(((yt_st - yt_st.mean()) ** 2).sum())
        r2_e = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        if r2_e < -1.0: r2_e = -1.0
        
        rmse_e = float(np.sqrt(((yt_st - yp_st) ** 2).mean()))
        mae_e = float(np.abs(yt_st - yp_st).mean())
        per_station.append({
            "estacion": est, "r2": r2_e, "rmse": rmse_e, "mae": mae_e,
            "n_samples": int(len(yt_st)),
        })

    return {
        "model": model_name,
        "best_params": gscv.best_params_,
        "r2_global": r2,
        "rmse_global": rmse,
        "mae_global": mae,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "per_station": per_station,
        "y_pred": y_pred.tolist(),
        "y_true": y_test.tolist(),
    }


def main():
    print("=" * 65)
    print("  COMPARACIÓN RIGUROSA: Ridge vs Lasso vs ElasticNet")
    print(f"  Horizonte: {HORIZON} paso(s) = {HORIZON * 6}h")
    print(f"  Holdout: últimos {HOLDOUT_DAYS} días")
    print("=" * 65)

    # Cargar dataset
    df = pd.read_csv(CSV_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["estacion", "fecha"])

    # Split temporal global
    cut = df.groupby("estacion")["fecha"].transform(
        lambda s: s.max() - pd.Timedelta(days=HOLDOUT_DAYS))
    df_raw_train = df[df["fecha"] <= cut]
    df_raw_test = df[df["fecha"] > cut]

    # Build features (misma construcción para train y test)
    dtr, feats_num, feats_cat = build_features(df_raw_train, horizon=HORIZON)
    dte, _, _ = build_features(df_raw_test, horizon=HORIZON)

    # Asegurar columnas alineadas
    for col in feats_num:
        if col not in dte.columns:
            dte[col] = 0.0

    results_all = []
    for mname in MODELS:
        print(f"\n🔧 Entrenando {mname.upper()}...")
        res = evaluate_one(mname, dtr, dte, feats_num, feats_cat)
        results_all.append(res)
        print(f"   R² = {res['r2_global']:.4f}  |  "
              f"RMSE = {res['rmse_global']:.4f}  |  "
              f"MAE = {res['mae_global']:.4f}")
        print(f"   Params: {res['best_params']}")
        print(f"   Por estación:")
        for ps in res["per_station"]:
            print(f"     {ps['estacion']:30s}  "
                  f"R²={ps['r2']:.4f}  RMSE={ps['rmse']:.4f}")

    # ── Tabla resumen ──
    print("\n" + "=" * 65)
    print("  TABLA RESUMEN")
    print("=" * 65)
    header = f"{'Modelo':<14} {'R²':>8} {'RMSE':>8} {'MAE':>8}"
    print(header)
    print("-" * len(header))
    for r in results_all:
        print(f"{r['model']:<14} {r['r2_global']:>8.4f} "
              f"{r['rmse_global']:>8.4f} {r['mae_global']:>8.4f}")

    # ── Baseline naïve ──
    baseline_lag = dte.groupby("estacion")["y_target"].shift(HORIZON)
    mask_base = baseline_lag.notna()
    if mask_base.sum() > 0:
        b_true = dte.loc[mask_base, "y_target"].values
        b_pred = baseline_lag[mask_base].values
        b_r2 = float(r2_score(b_true, b_pred))
        b_rmse = float(np.sqrt(mean_squared_error(b_true, b_pred)))
        print(f"{'baseline':14} {b_r2:>8.4f} {b_rmse:>8.4f}     ---")

    # ── Guardar resultados ──
    summary = []
    for r in results_all:
        summary.append({
            "model": r["model"],
            "description": MODELS[r["model"]]["description"],
            "r2_global": r["r2_global"],
            "rmse_global": r["rmse_global"],
            "mae_global": r["mae_global"],
            "best_params": r["best_params"],
            "per_station": r["per_station"],
        })

    (OUT_DIR / "comparison_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # CSV de métricas por estación (todas las combinaciones)
    rows = []
    for r in results_all:
        for ps in r["per_station"]:
            rows.append({"model": r["model"], **ps})
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "metricas_por_estacion_comparacion.csv", index=False)

    pd.DataFrame([{
        "model": r["model"],
        "r2": r["r2_global"],
        "rmse": r["rmse_global"],
        "mae": r["mae_global"],
    } for r in results_all]).to_csv(
        OUT_DIR / "metricas_globales_comparacion.csv", index=False)

    print(f"\n✅ Resultados guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
