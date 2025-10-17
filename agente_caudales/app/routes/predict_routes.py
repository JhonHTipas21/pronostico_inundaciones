from fastapi import APIRouter, HTTPException
from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.config import settings
from app.services.train_service import load_model
from app.services.predict_service import make_predictions
import pandas as pd

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model, meta = load_model(settings.MODEL_DIR)
    except Exception:
        raise HTTPException(500, "Modelo no encontrado: entrena primero /retrain")

    df = pd.DataFrame([r.model_dump() for r in req.records])
    try:
        preds = make_predictions(model, meta, df, horizon=req.horizon)
    except Exception as e:
        raise HTTPException(400, f"Error construyendo features: {e}")
    return PredictResponse(horizon=req.horizon, n=len(preds), preds_caudal_m3s=[float(v) for v in preds])
