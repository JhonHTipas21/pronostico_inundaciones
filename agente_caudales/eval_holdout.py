import pandas as pd
import numpy as np
import json
from pathlib import Path
from app.services.train_service import load_model
from app.services.feature_service import build_features
from app.config import settings

# === CONFIGURACIÓN ===
CSV_PATH = Path("app/data/dataset.csv")
MODEL_DIR = Path("app/model")
OUT_DIR = Path("app/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_DAYS = 90  # número de días para el tramo de validación
HORIZON = 3  # horizonte usado en el modelo

# === 1. CARGAR DATASET ===
df = pd.read_csv(CSV_PATH)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values(["estacion", "fecha"])

# === 2. SEPARAR TRAIN Y HOLDOUT ===
cut_date = df.groupby("estacion")["fecha"].transform(lambda s: s.max() - pd.Timedelta(days=HOLDOUT_DAYS))
df_train = df[df["fecha"] <= cut_date]
df_test = df[df["fecha"] > cut_date]

# === 3. CONSTRUIR FEATURES ===
train_df, meta = build_features(df_train, horizon=HORIZON)
test_df, _ = build_features(df_test, horizon=HORIZON)

# === 4. CARGAR MODELO ENTRENADO ===
model, meta = load_model(MODEL_DIR)

# === 5. PREDICCIONES ===
y_true = test_df["y_target"].values
y_pred = model.predict(test_df[meta["features"]].values)

# === 6. MÉTRICAS ===
r2 = float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

# === 7. GUARDAR RESULTADOS ===
pd.DataFrame({
    "fecha": test_df["fecha"],
    "estacion": test_df["estacion"],
    "real": y_true,
    "pred": y_pred
}).to_csv(OUT_DIR / "holdout_preds.csv", index=False)

Path(OUT_DIR / "holdout_metrics.json").write_text(json.dumps({"r2": r2, "rmse": rmse}, indent=2, ensure_ascii=False))
print("✅ Resultados guardados en app/data/processed/")
