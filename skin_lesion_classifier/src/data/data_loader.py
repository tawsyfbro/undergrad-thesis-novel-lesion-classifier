# src/data/data_loader.py
import os
import cv2
import numpy as np
from typing import Tuple, List
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .preprocessor import ImagePreprocessor


class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_path = self.config['dataset']['path']
        self.num_classes = self.config['dataset']['num_classes']
        self.resize_dims = tuple(self.config['preprocessing']['resize_dims'])

    def load_image(self, image_path: str) -> np.ndarray:

        img = cv2.imread(image_path)
        img = ImagePreprocessor.preprocess_image(img)
        img = cv2.resize(img, self.resize_dims)
        img = img.astype('float32') / 255.0
        return img

    def create_data_generator(self) -> ImageDataGenerator:

        aug_config = self.config['augmentation']
        return ImageDataGenerator(
            rotation_range=aug_config['rotation_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            shear_range=aug_config['shear_range'],
            zoom_range=aug_config['zoom_range'],
            horizontal_flip=aug_config['horizontal_flip'],
            preprocessing_function=self.preprocess_input
        )

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:

        images = []
        labels = []

        for class_idx, class_name in enumerate(os.listdir(self.dataset_path)):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = self.load_image(img_path)
                    images.append(img)
                    labels.append(class_idx)

        return np.array(images), np.array(labels)
