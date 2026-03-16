# app/routes/predict_routes.py
"""Endpoints de predicción y pronóstico recursivo a 48h."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
import pandas as pd

from app.schemas.predict_schema import (
    PredictRequest, PredictResponse, StationPrediction,
    StationsResponse, StationInfo,
    MetricsResponse, StationMetric,
    ForecastRequest, ForecastResponse, ForecastPoint,
)
from app.config import settings
from app.services.train_service import load_model
from app.services.predict_service import (
    make_predictions, make_predictions_with_uncertainty,
    make_recursive_forecast,
)
from app.services.geo_service import get_stations_metadata

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predicciones de caudal (m³/s) con intervalos de incertidumbre al 95%.
    Clipping dual: estadístico (q99) + físico (capacidad del canal).
    """
    try:
        model, meta = load_model(settings.MODEL_DIR)
    except Exception:
        raise HTTPException(500, "Modelo no encontrado. Ejecuta /api/v1/retrain.")

    df = pd.DataFrame([r.model_dump() for r in req.records])
    try:
        preds_flat = make_predictions(model, meta, df, horizon=req.horizon)
        detailed = make_predictions_with_uncertainty(
            model, meta, df, horizon=req.horizon)
    except Exception as e:
        raise HTTPException(400, f"Error construyendo features: {e}")

    station_preds = [StationPrediction(**d) for d in detailed]

    return PredictResponse(
        horizon=req.horizon,
        horizon_hours=req.horizon * 6,
        n=len(preds_flat),
        preds_caudal_m3s=[round(float(v), 4) for v in preds_flat],
        predictions=station_preds,
    )


# ─────────────────────────────────────────────────────────────────────
# Pronóstico Recursivo 48h
# ─────────────────────────────────────────────────────────────────────

@router.post("/forecast-48h", response_model=ForecastResponse)
def forecast_48h(req: ForecastRequest):
    """
    Pronóstico recursivo de caudal a 48 horas en intervalos de 3h.
    Retroalimenta el modelo con sus propias predicciones.
    """
    try:
        model, meta = load_model(settings.MODEL_DIR)
    except Exception:
        raise HTTPException(500, "Modelo no encontrado. Ejecuta /api/v1/retrain.")

    # Cargar dataset histórico
    try:
        df = pd.read_csv(settings.DATA_PATH)
        df["fecha"] = pd.to_datetime(df["fecha"])
    except Exception:
        df = None

    try:
        result = make_recursive_forecast(
            model, meta, df,
            estacion=req.estacion,
            lluvia_mm=req.lluvia_mm,
            temperatura_C=req.temperatura_C,
            impermeabilidad_pct=req.impermeabilidad_pct,
            caudal_previo=req.caudal_previo_m3s,
            steps=req.steps,
        )
    except Exception as e:
        raise HTTPException(400, f"Error en forecast recursivo: {e}")

    return ForecastResponse(
        estacion=result["estacion"],
        q_max_canal_m3s=result["q_max_canal_m3s"],
        n_steps=result["n_steps"],
        historico=result["historico"],
        pronostico=[ForecastPoint(**p) for p in result["pronostico"]],
    )


# ─────────────────────────────────────────────────────────────────────
# Estaciones y métricas
# ─────────────────────────────────────────────────────────────────────

@router.get("/stations", response_model=StationsResponse)
def list_stations():
    """Metadata geoespacial de todas las estaciones."""
    data = get_stations_metadata()
    estaciones = [StationInfo(**s) for s in data]
    return StationsResponse(n=len(estaciones), estaciones=estaciones)


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Métricas del modelo entrenado actual."""
    try:
        _, meta = load_model(settings.MODEL_DIR)
    except Exception:
        raise HTTPException(500, "Modelo no encontrado.")

    metrics_path = Path("app/data/processed/metricas_por_estacion_holdout.csv")
    per_station = []
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        for _, row in mdf.iterrows():
            per_station.append(StationMetric(
                estacion=row["estacion"],
                r2=round(float(row["r2"]), 4),
                rmse=round(float(row["rmse"]), 4),
            ))

    return MetricsResponse(
        model_type=meta.get("model_type", "unknown"),
        horizon=meta.get("horizon", 1),
        r2_cv_mean=round(meta.get("r2_cv_mean", 0.0), 4),
        rmse_cv_mean=round(meta.get("rmse_cv_mean", 0.0), 4),
        por_estacion=per_station,
    )
