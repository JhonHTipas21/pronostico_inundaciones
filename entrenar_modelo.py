import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

# Simular datos históricos
np.random.seed(42)
fechas = pd.date_range("2023-06-01", periods=720, freq='H')

datos = pd.DataFrame({
    'fecha': fechas,
    'lluvia_mm': np.random.gamma(1.5, 5, 720),
    'temperatura_C': np.random.normal(25, 3, 720),
    'impermeabilidad_pct': np.random.uniform(30, 90, 720)
})

# Calcular caudal
datos['caudal_m3s'] = (
    0.8 * datos['lluvia_mm'].shift(3).fillna(0) +
    0.2 * (100 - datos['impermeabilidad_pct']) +
    0.1 * datos['temperatura_C']
)

# Guardar dataset
datos.to_csv(os.path.join(DATA_DIR, "dataset.csv"), index=False)

# Modelo de entrenamiento
X = datos[['lluvia_mm', 'temperatura_C', 'impermeabilidad_pct']]
y = datos['caudal_m3s']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

modelo = Ridge()
modelo.fit(X_train_scaled, y_train)

# Guardar modelo y escalador
joblib.dump(modelo, os.path.join(MODEL_DIR, "modelo_regresion.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "escalador.pkl"))

print("✅ Modelo y escalador guardados correctamente.")
