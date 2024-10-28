# src/models/feature_extractor.py
from tensorflow.keras.applications import EfficientNetB7, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import yaml


class FeatureExtractor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model']['feature_extractor']['name']
        self.input_shape = tuple(
            self.config['model']['feature_extractor']['input_shape'])
        self.feature_dim = self.config['model']['feature_extractor']['feature_dim']

    def build_model(self) -> Model:
        """Build and return feature extractor model."""
        if self.model_name == 'efficientnet_b7':
            base_model = EfficientNetB7(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == 'efficientnet_b0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(self.feature_dim, activation='relu')(x)
        x = Dropout(0.5)(x)

        return Model(inputs=base_model.input, outputs=x)
