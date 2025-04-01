"""Main module for running the FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="Plant Disease Detection API")


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
