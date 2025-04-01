""" Module for loading data. """

import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator

def load_data_from_plantvillage_into_generators(base_dir: str, image_size=(224, 224), batch_size=32) -> tuple:
    """
        Loads data (images) from the "plantvillage" folder using Keras ImageDataGenerator.

        Parameters:
        - base_dir: Path to the folder containing subfolders with images (e.g. plantvillage/)
        - image_size: The size to which the images will be resized (default is (224, 224))
        - batch_size: The batch size (default is 32)

        Returns:
        - train_generator: Generator for training data
        - valid_generator: Generator for validation data
    """

    train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizacja obrazów
    valid_datagen = ImageDataGenerator(rescale=1./255)  # Normalizacja obrazów

    train_dir = f"{base_dir}/train"
    valid_dir = f"{base_dir}/val"

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        subset="training",
        class_mode="categorical"
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical" 
    )

    return train_generator, valid_generator
