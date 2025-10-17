# 🌊 Agente_Caudales — Sistema de predicción de caudales pluviales en Cali sector Sur

---

## 📌 Descripción general

En Santiago de Cali, el manejo de las lluvias intensas representa un desafío crítico. El desbordamiento de los drenajes urbanos por precipitaciones súbitas afecta la movilidad, daña la infraestructura y pone en riesgo la vida de los ciudadanos. La respuesta tradicional, basada en acciones reactivas como limpiar rejillas o bombear agua después de la emergencia, ha demostrado ser costosa, tardía e ineficaz.

Este proyecto propone el diseño de un **prototipo de sistema predictivo**, basado en técnicas de **machine learning** e implementado como un **microservicio FastAPI**, capaz de anticipar los caudales (m³/s) con hasta 6 horas de antelación. Este sistema permitirá transformar datos históricos en conocimiento útil para la gestión preventiva del riesgo por inundaciones urbanas

---

## 🎯 Objetivo general

Diseñar un prototipo de sistema predictivo, basado en modelos de regresión lineal implementados en Python, para estimar los caudales (m³/s) en drenajes pluviales urbanos de la ciudad de Santiago de Cali, utilizando series históricas de lluvia, temperatura y datos topográficos correspondientes al periodo de junio 2023 – junio 2025, con un adelanto de 3 a 6 horas, a fin de respaldar la toma de decisiones en la gestión preventiva de inundaciones.

---

## 📌 Objetivos específicos

1. Compilar datos históricos de precipitación, temperatura, caudales y cobertura de suelo de la ciudad de Cali, provenientes de fuentes como el radar Munchique, estaciones locales y mapas geoespaciales.

2. Modelar un proceso de limpieza, transformación y enriquecimiento de los datos mediante herramientas como Pandas, NumPy y GeoPandas.

3. Estimar el desempeño del modelo utilizando métricas como R² y RMSE, garantizando su precisión y capacidad predictiva.

---

## 📊 Metodología disciplinar

**KDD (Knowledge Discovery in Databases)**  
Se aplica la metodología KDD como enfoque disciplinar, ya que permite transformar grandes volúmenes de datos históricos en información útil y accionable. Dado que este sistema se basa en el análisis de variables climáticas y topográficas, KDD guía cada fase del proyecto: selección, preprocesamiento, transformación, modelado y evaluación, alineándose con los principios del desarrollo de inteligencia artificial aplicada.

---

## 🛠️ Stack tecnológico

| Tecnología        | Función principal                                |
|------------------|--------------------------------------------------|
| **Python 3.10+** | Lenguaje de programación base                    |
| **FastAPI**      | Desarrollo del microservicio RESTful             |
| **Scikit-learn** | Modelos de regresión (lineal, Ridge, Lasso)      |
| **Pandas**       | Limpieza y manipulación de datos tabulares       |
| **NumPy**        | Operaciones numéricas y matriciales              |
| **GeoPandas**    | Análisis geoespacial de cobertura del suelo      |
| **Joblib**       | Serialización de modelos para producción         |
| **Uvicorn**      | Servidor ASGI para ejecutar FastAPI              |

---

## 🧠 ¿Qué es este sistema?

Un microservicio inteligente diseñado para predecir caudales urbanos de forma automatizada, con las siguientes capacidades:

- 📥 Ingesta de datos climáticos y geoespaciales (lluvia, temperatura, impermeabilidad)
- 🧪 Procesamiento, transformación y modelado de datos históricos
- 🤖 Predicción de caudal pluvial a corto plazo (3 a 6 horas)
- 🔁 Reentrenamiento del modelo con nuevos datos en cualquier momento

---
## 🔁 Arquitectura del sistema

          +----------------------------+
          | Estaciones IDEAM / Radar  |
          +----------------------------+
                       ↓
            [ Ingesta de datos ]
                       ↓
     [ Limpieza y enriquecimiento de variables ]
                       ↓
      [ Modelo ML: Regresión Lineal / Ridge / Lasso ]
                       ↓
             [ Microservicio FastAPI ]
             ├── /predict → Predicción de caudal
             └── /retrain → Reentrenamiento con nuevos datos




Proyecto basado en **FastAPI + Machine Learning (Lasso/Ridge)** para estimar el caudal (m³/s) de los drenajes urbanos de Cali a partir de datos de lluvia, temperatura e impermeabilidad.

---

## 🚀 1. Requisitos previos

- Python 3.11 o superior  
- Git (opcional)  
- Windows PowerShell / Terminal macOS / Linux Shell  

---

## ⚙️ 2. Clonar el repositorio y crear entorno virtual

```bash
git clone <URL_DEL_REPO>
cd agente_caudales

python -m venv venv
# Activar entorno
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

---

## 📦 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧾 4. Configuración (.env)

Crea un archivo `.env` en la raíz del proyecto con lo siguiente:

```
APP_NAME=agente_caudales
DATA_PATH=app/data/dataset.csv
MODEL_DIR=app/model
HORIZON=3
HORIZON_UNITS=days
```

---

## ▶️ 5. Ejecutar la API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# o
python main.py
```

**URLs útiles**
- Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check → [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🌐 6. Endpoints principales

### 🧠 `/retrain` (POST)
Entrena o reentrena el modelo.

```json
{
  "csv_path": "E:\\pronostico_inundaciones\\agente_caudales\\app\\data\\dataset.csv",
  "horizon": 3,
  "model_type": "lasso"
}
```

### 🔮 `/predict` (POST)
Realiza predicciones del caudal.

```json
{
  "horizon": 3,
  "records": [
    { "fecha": "2025-05-28T00:00:00", "lluvia_mm": 2.4, "temperatura_C": 24.8, "impermeabilidad_pct": 62.0, "caudal_m3s": 1.02, "estacion": "Canal Cañaveralejo" },
    { "fecha": "2025-05-29T00:00:00", "lluvia_mm": 0.0, "temperatura_C": 25.1, "impermeabilidad_pct": 62.0, "caudal_m3s": 0.95, "estacion": "Canal Cañaveralejo" }
  ]
}
```

---

## 📊 7. Validación Holdout (R² / RMSE)

Valida el modelo sobre los últimos días del histórico:

```bash
python eval_holdout.py
```

Genera:
- `app/data/processed/holdout_preds.csv`
- `app/data/processed/holdout_metrics.json`

---

## 📈 8. Gráficas y tablas para el informe

```bash
python figs_informe.py
```

Genera:
- `serie_<ESTACION>.png`
- `dispersion_holdout.png`
- `metricas_por_estacion.csv`

---

## 💻 9. Comandos útiles

**Entrenar con CSV**
```bash
curl -X POST "http://127.0.0.1:8000/retrain" -H "Content-Type: application/json" -d "{\"csv_path\":\"E:\\\\pronostico_inundaciones\\\\agente_caudales\\\\app\\\\data\\\\dataset.csv\",\"horizon\":3,\"model_type\":\"lasso\"}"
```

**Validación y gráficas**
```bash
python eval_holdout.py
python figs_informe.py
```

---

## 🧩 10. Estructura del proyecto

```
agente_caudales/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── data/
│   │   └── dataset.csv
│   ├── model/
│   │   ├── modelo_regresion.pkl
│   │   └── escalador.pkl
│   ├── routes/
│   │   ├── predict_routes.py
│   │   └── train_routes.py
│   ├── schemas/
│   └── services/
│       ├── predict_service.py
│       ├── train_service.py
│       └── feature_service.py
│
├── eval_holdout.py
├── figs_informe.py
├── entrenar_modelo.py
├── requirements.txt
└── .env
```

---

## 🧠 11. Créditos

**Desarrolladores:**  
Equipo *NeuroNautas* — Universidad Santiago de Cali  
Proyecto: *Predicción de caudales pluviales urbanos en el Sur de Cali (2005–2025)*  
Frameworks: FastAPI · scikit-learn · pandas · matplotlib
