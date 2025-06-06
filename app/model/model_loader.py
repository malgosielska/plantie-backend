"""Module for loading a model and class indices"""

import torch
import torchvision.models as models
import json


def load_model(path: str, classes: int):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = models.regnet_y_400mf()
    model.fc = torch.nn.Linear(model.fc.in_features, classes)
    model.load_state_dict(torch.load(path,  map_location=device), strict=False)
    model.eval()
    return model


def load_class_indices(path: str):
    with open(path, "r") as f:
        class_indices = json.load(f)
    return {int(k): v for k, v in class_indices.items()}
