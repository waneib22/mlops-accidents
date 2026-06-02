from fastapi import FastAPI
from api.base import router as base_router
from api.predict import router as predict_router

app = FastAPI()

app.include_router(base_router)
app.include_router(predict_router)