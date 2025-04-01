"""Main module for running the FastAPI application."""

from fastapi import FastAPI
from data.data_loader import load_data_from_plantvillage_into_generators

app = FastAPI(title="Plantie - Plant Disease Detection API")


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}
    
load_data_from_plantvillage_into_generators("plantvillage_dataset")