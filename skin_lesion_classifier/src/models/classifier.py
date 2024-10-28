# src/models/classifier.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.ensemble import RandomForestClassifier
import yaml


class ClassifierModel:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.classifier_type = self.config['model']['classifier']['type']
        self.num_classes = self.config['dataset']['num_classes']

    def build_1d_cnn(self) -> Sequential:

        cnn_params = self.config['model']['classifier']['cnn_params']

        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(1024, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_random_forest(self) -> RandomForestClassifier:

        return RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
