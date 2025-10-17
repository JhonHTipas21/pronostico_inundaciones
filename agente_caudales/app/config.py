from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "agente_caudales")
    DATA_PATH: str = os.getenv("DATA_PATH", "app/data/dataset.csv")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "app/model")
    HORIZON: int = int(os.getenv("HORIZON", "3"))  # 3 o 6
    HORIZON_UNITS: str = os.getenv("HORIZON_UNITS", "days")  # "hours" o "days"
    # si luego pasas a series horarias, cambia a "hours"

settings = Settings()
