from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Registro(BaseModel):
    fecha: datetime
    lluvia_mm: float
    temperatura_C: float
    impermeabilidad_pct: float
    caudal_m3s: Optional[float] = None
    estacion: str

class PredictRequest(BaseModel):
    horizon: int = Field(3, ge=1, le=24)
    records: List[Registro]

class PredictResponse(BaseModel):
    horizon: int
    n: int
    preds_caudal_m3s: List[float]

class RetrainRequest(BaseModel):
    csv_path: Optional[str] = None
    records: Optional[List[Registro]] = None
    horizon: int = Field(3, ge=1, le=24)
    model_type: str = Field("lasso", pattern="^(lasso|ridge)$")
