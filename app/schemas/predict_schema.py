# app/schemas/predict_schema.py
"""Schemas Pydantic para los endpoints de la API."""
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict
from datetime import datetime


class Registro(BaseModel):
    """Registro individual de datos hidrometeorológicos."""
    fecha: datetime
    lluvia_mm: float = Field(..., ge=0, description="Precipitación (mm)")
    temperatura_C: float = Field(..., description="Temperatura (°C)")
    impermeabilidad_pct: float = Field(
        ..., ge=0, le=100, description="Porcentaje de impermeabilidad")
    caudal_m3s: Optional[float] = Field(
        None, ge=0, description="Caudal observado (m³/s)")
    estacion: str = Field(..., description="Nombre de la estación")


# ── Predicción ──

class PredictRequest(BaseModel):
    """Request para predicción de caudal."""
    horizon: int = Field(1, ge=1, le=24,
                         description="Pasos de 6H a predecir")
    records: List[Registro]


class StationPrediction(BaseModel):
    """Predicción detallada por estación con incertidumbre."""
    estacion: str
    fecha: str
    caudal_pred_m3s: float
    lower_95: float = 0.0
    upper_95: float = 0.0
    horizonte_h: int


class PredictResponse(BaseModel):
    """Respuesta del endpoint de predicción."""
    horizon: int
    horizon_hours: int
    n: int
    predictions: List[StationPrediction] = []
    preds_caudal_m3s: List[float] = []


# ── Reentrenamiento ──

class RetrainRequest(BaseModel):
    """Request para reentrenamiento del modelo."""
    model_config = ConfigDict(protected_namespaces=())
    csv_path: Optional[str] = None
    records: Optional[List[Registro]] = None
    horizon: int = Field(1, ge=1, le=24)
    model_type: str = Field(
        "lasso", pattern="^(lasso|ridge|elasticnet)$")


class RetrainResponse(BaseModel):
    """Respuesta del endpoint de reentrenamiento."""
    model_config = ConfigDict(protected_namespaces=())
    message: str
    model_type: str
    horizon: int
    r2_cv_mean: float
    rmse_cv_mean: float
    mae_cv_mean: float
    n_samples: int
    best_params: dict


# ── Estaciones ──

class StationInfo(BaseModel):
    """Metadata de una estación de monitoreo."""
    nombre: str
    cobertura: str
    cn_number: float
    area_km2: float
    pendiente_media: float
    pct_impermeable: float
    longitud_cauce_km: float
    tiempo_concentracion_h: float
    caudal_max_m3s: float
    lat: float
    lon: float
    station_code: str


class StationsResponse(BaseModel):
    """Lista de estaciones disponibles."""
    n: int
    estaciones: List[StationInfo]


# ── Métricas ──

class StationMetric(BaseModel):
    """Métricas de rendimiento de una estación."""
    estacion: str
    r2: float
    rmse: float


class MetricsResponse(BaseModel):
    """Métricas del modelo actual."""
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    horizon: int
    r2_cv_mean: float
    rmse_cv_mean: float
    por_estacion: List[StationMetric] = []


# ── Forecast Recursivo 48h ──

class ForecastRequest(BaseModel):
    """Request para pronóstico recursivo a 48h."""
    estacion: str = Field(..., description="Nombre de la estación")
    lluvia_mm: float = Field(0.0, ge=0, description="Lluvia esperada por paso (mm)")
    temperatura_C: float = Field(24.0, description="Temperatura (°C)")
    impermeabilidad_pct: float = Field(60.0, ge=0, le=100)
    caudal_previo_m3s: float = Field(0.5, ge=0, description="Caudal previo (m³/s)")
    steps: int = Field(16, ge=1, le=32, description="Pasos de 3h (16=48h)")


class ForecastPoint(BaseModel):
    """Un punto de la serie temporal pronosticada."""
    fecha: str
    hora_adelanto: int
    caudal_pred_m3s: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    """Respuesta del pronóstico recursivo a 48h."""
    estacion: str
    q_max_canal_m3s: float
    n_steps: int
    historico: List[Dict] = []
    pronostico: List[ForecastPoint] = []
