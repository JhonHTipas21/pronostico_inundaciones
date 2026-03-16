# app/routes/train_routes.py
"""Endpoint de reentrenamiento del modelo."""
from fastapi import APIRouter, HTTPException

from app.schemas.predict_schema import RetrainRequest, RetrainResponse
from app.config import settings
from app.services.train_service import train_from_df
import pandas as pd

router = APIRouter()


@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    """
    Re-entrena el modelo con datos nuevos o existentes.

    Soporta modelos: ridge, lasso, elasticnet.
    """
    if req.csv_path:
        try:
            df = pd.read_csv(req.csv_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error leyendo CSV: {e}",
            )
    elif req.records:
        df = pd.DataFrame([r.model_dump() for r in req.records])
    else:
        raise HTTPException(
            status_code=400,
            detail="Provee csv_path o records para entrenar.",
        )

    try:
        meta = train_from_df(
            df,
            horizon=req.horizon,
            model_type=req.model_type,
            model_dir=settings.MODEL_DIR,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {e}",
        )

    return RetrainResponse(
        message="Modelo re-entrenado exitosamente",
        model_type=meta["model_type"],
        horizon=meta["horizon"],
        r2_cv_mean=round(meta["r2_cv_mean"], 4),
        rmse_cv_mean=round(meta["rmse_cv_mean"], 4),
        mae_cv_mean=round(meta["mae_cv_mean"], 4),
        n_samples=meta["n_samples"],
        best_params=meta["best_params"],
    )
