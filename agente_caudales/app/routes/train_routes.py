from fastapi import APIRouter, HTTPException
from app.schemas.predict_schema import RetrainRequest
from app.config import settings
from app.services.train_service import train_from_df
import pandas as pd

router = APIRouter()

@router.post("/retrain")
def retrain(req: RetrainRequest):
    if req.csv_path:
        df = pd.read_csv(req.csv_path)
    elif req.records:
        df = pd.DataFrame([r.model_dump() for r in req.records])
    else:
        raise HTTPException(400, "Provee csv_path o records")
    meta = train_from_df(df, horizon=req.horizon, model_type=req.model_type, model_dir=settings.MODEL_DIR)
    return {"message":"ok","meta":meta}
