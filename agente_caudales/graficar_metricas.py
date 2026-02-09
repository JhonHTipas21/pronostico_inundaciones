import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta del CSV existente
csv_path = Path("app/data/processed/metricas_por_estacion.csv")
output_path = Path("app/data/processed/metricas_por_estacion.png")

# Leer el archivo CSV
df = pd.read_csv(csv_path)

# Crear figura
fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()

# Eje X
x = range(len(df))
ax1.bar(x, df["r2"], width=0.4, label="R²", color="skyblue")
ax2.plot(x, df["rmse"], marker="o", color="orange", linewidth=2, label="RMSE")

# Etiquetas y formato
ax1.set_xticks(x)
ax1.set_xticklabels(df["estacion"], rotation=30, ha="right")
ax1.set_ylabel("R²")
ax2.set_ylabel("RMSE (m³/s)")
plt.title("Métricas por estación")

# Leyenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig(output_path, dpi=200)
plt.show()

print(f"✅ Imagen guardada en: {output_path}")
