import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# === RUTAS ===
PROC_DIR = Path("app/data/processed")
metrics_path = PROC_DIR / "holdout_metrics.json"
preds_path = PROC_DIR / "holdout_preds.csv"

# === CARGAR DATOS ===
metrics = json.loads(metrics_path.read_text())
df = pd.read_csv(preds_path)
df["fecha"] = pd.to_datetime(df["fecha"])

# === 1. GRAFICO DE SERIES (Real vs Predicho) ===
for est in df["estacion"].unique():
    sub = df[df["estacion"] == est].sort_values("fecha")

    plt.figure(figsize=(10,5))
    plt.plot(sub["fecha"], sub["real"], label="Real", linewidth=2)
    plt.plot(sub["fecha"], sub["pred"], label="Predicho", linestyle="--")
    plt.title(f"Caudal - {est}")
    plt.xlabel("Fecha")
    plt.ylabel("Caudal (m³/s)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROC_DIR / f"serie_{est}.png", dpi=150)
    plt.close()

# === 2. DISPERSIÓN (Real vs Predicho) ===
plt.figure(figsize=(6,6))
plt.scatter(df["real"], df["pred"], alpha=0.6)
plt.xlabel("Caudal Real (m³/s)")
plt.ylabel("Caudal Predicho (m³/s)")
plt.title(f"Dispersión Real vs Predicho\nR²={metrics['r2']:.3f} | RMSE={metrics['rmse']:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROC_DIR / "dispersion_holdout.png", dpi=150)
plt.close()

# === 3. TABLA MÉTRICAS POR ESTACIÓN ===
resultados = []
for est in df["estacion"].unique():
    sub = df[df["estacion"] == est]
    y = sub["real"].values
    yhat = sub["pred"].values
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = ((y - yhat) ** 2).mean() ** 0.5
    resultados.append({"estacion": est, "r2": r2, "rmse": rmse})

tabla = pd.DataFrame(resultados).sort_values("r2", ascending=False)
tabla.to_csv(PROC_DIR / "metricas_por_estacion.csv", index=False)
print("✅ Gráficas y tabla generadas correctamente")
