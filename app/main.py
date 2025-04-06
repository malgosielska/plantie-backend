"""Main module for running the FastAPI application."""

from fastapi import FastAPI
from app.routers.predict import predict_router

app = FastAPI(title="Plantie - Plant Disease Detection API")
app.include_router(predict_router)


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
