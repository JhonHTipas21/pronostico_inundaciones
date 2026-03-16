# app/main.py
"""
FastAPI — Microservicio de predicción de caudales pluviales.
Universidad Santiago de Cali · Proyecto de grado.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.health_routes import router as health_router
from app.routes.predict_routes import router as predict_router
from app.routes.train_routes import router as train_router

app = FastAPI(
    title="Predicción de Caudales Pluviales — Santiago de Cali",
    description=(
        "Microservicio de predicción de caudales pluviales para las cuencas "
        "de los ríos Cañaveralejo, Meléndez, Pance, Lili, Canal Interceptor "
        "Sur y Canal Ciudad Jardín. Modelos: Ridge / Lasso / ElasticNet con "
        "PolynomialFeatures y coeficientes SCS-CN geoespaciales."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite conexión desde Streamlit (localhost:8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ──
app.include_router(health_router, tags=["health"])
app.include_router(predict_router, prefix="/api/v1", tags=["predict"])
app.include_router(train_router, prefix="/api/v1", tags=["train"])
