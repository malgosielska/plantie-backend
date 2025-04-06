"""Module for image processing utilities."""

import torchvision.transforms as transforms
from PIL import Image
import torch


def load_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = transform(image).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    image = image.unsqueeze(0)
    return image
