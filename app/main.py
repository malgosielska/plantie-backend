"""Main module for running the FastAPI application."""

from fastapi import FastAPI
from data.data_loader import load_data_from_plantvillage_into_generators
from models.plant_disease_recognition_model import create_model

app = FastAPI(title="Plantie - Plant Disease Detection API")


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}

image_size = (224, 224)
train_generator, validation_generator = load_data_from_plantvillage_into_generators("plantvillage_dataset", image_size=image_size, batch_size=32)
model = create_model(train_generator=train_generator, img_size=image_size[0])
print(model.summary())