# eval_holdout.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.services.feature_service import build_features
from app.services.train_service import load_model

# --- CONFIG ---
CSV_PATH = Path(settings.DATA_PATH)
MODEL_DIR = Path(settings.MODEL_DIR)
OUT_DIR = Path("app/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_DAYS = 30            # tramo reciente para evaluar
HORIZON = settings.HORIZON   # debe coincidir con el entrenamiento

# --- 1) Cargar dataset base ---
df = pd.read_csv(CSV_PATH)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values(["estacion", "fecha"])

# --- 2) Split temporal por estación (últimos N días como holdout) ---
cut = df.groupby("estacion")["fecha"].transform(
    lambda s: s.max() - pd.Timedelta(days=HOLDOUT_DAYS)
)
df_train = df[df["fecha"] <= cut]
df_test = df[df["fecha"] > cut]

# --- 3) Features con el mismo pipeline ---
dtr, feats_num, feats_cat = build_features(df_train, horizon=HORIZON)
dte, _, _ = build_features(df_test, horizon=HORIZON)

# --- 4) Cargar modelo y features desde meta ---
model, meta = load_model(MODEL_DIR)
feats = meta["features_numeric"] + meta["features_categorical"]

# Seguridad: chequear columnas requeridas
missing = [c for c in feats if c not in dte.columns]
if missing:
    raise RuntimeError(f"Faltan columnas en dte: {missing}")

# --- 5) Predicción y métricas globales ---
y_true = dte["y_target"].to_numpy()
y_pred = model.predict(dte[feats])

ss_res = float(((y_true - y_pred) ** 2).sum())
ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

# --- 6) Guardar resultados detallados ---
preds_df = (
    pd.DataFrame(
        {
            "fecha": dte["fecha"].to_numpy(),
            "estacion": dte["estacion"].to_numpy(),
            "real": y_true,
            "pred": y_pred,
        }
    )
    .sort_values(["estacion", "fecha"])
)

preds_df.to_csv(OUT_DIR / "holdout_preds.csv", index=False)
(OUT_DIR / "holdout_metrics.json").write_text(
    json.dumps({"r2": r2, "rmse": rmse}, indent=2, ensure_ascii=False)
)

# --- 7) Métricas por estación (diagnóstico) ---
rows = []
for est, sub in preds_df.groupby("estacion"):
    y = sub["real"].to_numpy()
    p = sub["pred"].to_numpy()
    ss_res_e = float(((y - p) ** 2).sum())
    ss_tot_e = float(((y - y.mean()) ** 2).sum())
    r2_e = float(1.0 - ss_res_e / ss_tot_e) if ss_tot_e > 0 else float("nan")
    rmse_e = float(np.sqrt(((y - p) ** 2).mean()))
    rows.append({"estacion": est, "r2": r2_e, "rmse": rmse_e})

pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(
    OUT_DIR / "metricas_por_estacion_holdout.csv", index=False
)

# --- 8) Baseline ingenuo (ŷ = y_{t-h}) para comparar ---
baseline = dte.copy()
baseline["y_lag_h"] = baseline.groupby("estacion")["y_target"].shift(HORIZON)
base = baseline.dropna(subset=["y_lag_h", "y_target"]).copy()

b_ss_res = float(((base["y_target"] - base["y_lag_h"]) ** 2).sum())
b_ss_tot = float(((base["y_target"] - base["y_target"].mean()) ** 2).sum())
b_r2 = float(1.0 - b_ss_res / b_ss_tot) if b_ss_tot > 0 else float("nan")
b_rmse = float(np.sqrt(((base["y_target"] - base["y_lag_h"]) ** 2).mean()))

(OUT_DIR / "baseline_metrics.json").write_text(
    json.dumps(
        {"r2": b_r2, "rmse": b_rmse, "horizon": int(HORIZON)},
        indent=2,
        ensure_ascii=False,
    )
)

print(f"Baseline ingenuo (y_t ≈ y_(t-{HORIZON})) -> R² = {b_r2:.4f}, RMSE = {b_rmse:.4f}")
print(f"✅ Resultados guardados en: {OUT_DIR}")
