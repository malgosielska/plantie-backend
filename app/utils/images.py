"""Module for image processing utilities."""

import torchvision.transforms as transforms
from PIL import Image
import torch


def transform_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def load_image(image):
    image = transform_image(image)
    image = image.unsqueeze(0)
    return image
