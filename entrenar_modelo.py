# entrenar_modelo.py
import pandas as pd
from app.config import settings
from app.services.train_service import train_from_df

if __name__ == "__main__":
    df = pd.read_csv(settings.DATA_PATH)
    meta = train_from_df(df, horizon=settings.HORIZON, model_type="ridge", model_dir=settings.MODEL_DIR)
    print("Entrenado. Métricas CV:", meta)
