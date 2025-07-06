# 🌧️ Sistema Predictivo de Caudales Pluviales Urbanos en Santiago de Cali

---

## 📌 Descripción general

En Santiago de Cali, el manejo de las lluvias intensas representa un desafío crítico. El desbordamiento de los drenajes urbanos por precipitaciones súbitas afecta la movilidad, daña la infraestructura y pone en riesgo la vida de los ciudadanos. La respuesta tradicional, basada en acciones reactivas como limpiar rejillas o bombear agua después de la emergencia, ha demostrado ser costosa, tardía e ineficaz.

Este proyecto propone el diseño de un **prototipo de sistema predictivo**, basado en técnicas de **machine learning** e implementado como un **microservicio FastAPI**, capaz de anticipar los caudales (m³/s) con hasta 6 horas de antelación. Este sistema permitirá transformar datos históricos en conocimiento útil para la gestión preventiva del riesgo por inundaciones urbanas.

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


## 📁 Estructura del proyecto

```plaintext
agente_caudales/
├── app/
│   ├── data/                  # Dataset histórico
│   ├── model/                 # Archivos .pkl del modelo entrenado
│   ├── schemas/               # Validaciones Pydantic
│   ├── utils/                 # Funciones de apoyo (opcional)
│   └── main.py                # Servidor FastAPI
├── entrenar_modelo.py         # Script para entrenamiento del modelo
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto

