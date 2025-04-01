"""Main module for running the FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="Plantie - Plant Disease Detection API")


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
