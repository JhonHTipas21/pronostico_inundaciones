# eval_per_station_holdout.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from app.config import settings
from app.services.feature_service import build_features

HOLDOUT_DAYS = 30
MODEL_ROOT = Path(settings.MODEL_DIR) / "per_estacion"
OUT_DIR = Path("app/data/processed_per_estacion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(settings.DATA_PATH)
df["fecha"] = pd.to_datetime(df["fecha"])

rows = []
for est, sub in df.groupby("estacion"):
    # split temporal para esa estación
    cut = sub["fecha"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    tr, te = sub[sub["fecha"] <= cut], sub[sub["fecha"] > cut]

    dtr, fnum, fcat = build_features(tr, horizon=settings.HORIZON)
    dte, _, _       = build_features(te, horizon=settings.HORIZON)
    feats = fnum + fcat

    est_dir = MODEL_ROOT / est.replace(" ", "_").replace("/", "_")
    model = joblib.load(est_dir / "modelo.pkl")

    y = dte["y_target"].to_numpy()
    p = model.predict(dte[feats])

    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    rows.append({"estacion": est, "r2": r2, "rmse": rmse})

pd.DataFrame(rows).to_csv(OUT_DIR / "holdout_metricas_por_estacion.csv", index=False)
print("✅ Guardado:", OUT_DIR / "holdout_metricas_por_estacion.csv")
