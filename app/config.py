# app/config.py
"""Configuración centralizada del proyecto."""
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "agente_caudales")
    DATA_PATH: str = os.getenv("DATA_PATH", "app/data/dataset_6h.csv")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "app/model")
    HORIZON: int = int(os.getenv("HORIZON", "1"))
    HORIZON_UNITS: str = os.getenv("HORIZON_UNITS", "6H")
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    STREAMLIT_URL: str = os.getenv("STREAMLIT_URL", "http://localhost:8501")


settings = Settings()
