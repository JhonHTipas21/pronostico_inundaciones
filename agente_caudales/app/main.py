from fastapi import FastAPI
from app.config import settings
from app.routes.health_routes import router as health_router
from app.routes.predict_routes import router as predict_router
from app.routes.train_routes import router as train_router

app = FastAPI(title=settings.APP_NAME)
app.include_router(health_router, tags=["health"])
app.include_router(predict_router, tags=["predict"])
app.include_router(train_router, tags=["train"])
