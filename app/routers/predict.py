"""Module for image prediction using a pre-trained model."""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from app.utils.managements import managements
from app.utils.symptoms import symptoms
from app.utils.class_names import class_names
from app.utils.constants import CLASS_INDICES_PATH, CLASSES_COUNT, MODEL_PATH
from app.model.model_loader import load_class_indices, load_model
from app.utils.images import load_image

predict_router = APIRouter(prefix="/predict", tags=["predict"])


@predict_router.post("/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        model = load_model(MODEL_PATH, CLASSES_COUNT)
        class_indices = load_class_indices(CLASS_INDICES_PATH)

        tensor = load_image(image)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_indices[predicted_idx.item()]
            predicted_names = class_names.get(predicted_class, predicted_class) 
            predicted_symptoms = symptoms.get(predicted_class, "No symptoms available.")
            predicted_management = managements.get(
                predicted_class, "No managements available."
            )
        return JSONResponse(
            media_type="application/json; charset=utf-8",
            content={
                "predicted_names": predicted_names,
                "symptoms": predicted_symptoms,
                "management": predicted_management,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
