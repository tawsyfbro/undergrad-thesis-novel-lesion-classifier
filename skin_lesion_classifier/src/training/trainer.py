# src/training/trainer.py
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from ..utils.metrics import calculate_metrics
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']

    def prepare_data(self, features: np.ndarray, labels: np.ndarray):
        """Prepare data with SMOTE and train-test split."""
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(features, labels)

        logger.info("Splitting data into train and test sets...")
        return train_test_split(
            X_resampled,
            y_resampled,
            test_size=self.config['training']['validation_split'],
            random_state=42
        )

    def train_cnn(self, model, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train 1D CNN model."""
        logger.info("Starting CNN training...")

        # Reshape data for 1D CNN
        X_train_reshaped = X_train.reshape(
            X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            X_train_reshaped,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test_reshaped, y_test),
            callbacks=self._get_callbacks()
        )

        # Evaluate model
        y_pred = model.predict(X_test_reshaped)
        metrics = calculate_metrics(y_test, np.argmax(y_pred, axis=-1))

        return {
            'history': history.history,
            'metrics': metrics
        }

    def train_random_forest(self, model, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Starting Random Forest training...")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        return {
            'metrics': metrics
        }

    def _get_callbacks(self) -> list:
        """Get training callbacks."""
        return [
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/model_{epoch:02d}.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            )
        ]
